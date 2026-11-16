"""Draft the specified populations, then read their wire stamps in a CPU job."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import pstats
import sys
import time
import traceback

from experiments.numerics_rows.common import check_source, write_json

BASE = Path("/mnt/shared/tessera-measurements")
STUB = Path("/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B")
CACHE = BASE / "glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache/wire"
OPTION_C = Path("/mnt/shared/flux-jobs/tessera-option-c-r1-5c2ef6de-20261115T1023Z/artifact-5c2ef6de")
BF16 = Path("/mnt/shared/flux-jobs/tessera-bf16-only-5c2ef6de-20261115T2027Z/artifact-5c2ef6de")
FAMILY = {"TESSERA_FP8": "TESSERA_E4M3_K1", "TESSERA_NVFP4": "TESSERA_E2M1_K2", "TESSERA_BF16": "TESSERA_BF16_K1"}


def module_key(entry):
    return tuple(json.dumps(entry.get(key), sort_keys=True) for key in
                 ("root", "name", "wire_path", "wire_root", "wire_family", "layer"))


def store_for(entry):
    """Use the served Store interface for checkpoint and canonical dense wires."""
    from experiments.dense_fused_oracle import Store

    store = Store(entry["root"])
    if "wire_path" not in entry:
        return store
    import torch
    from tessera import fused
    from tessera.unit_artifact import parse_unit_metadata

    blob = Path(entry["wire_path"]).read_bytes()
    template = store.schemes[entry.get("scheme_template", entry["name"])]
    if blob[:8] != fused.FUSED_MAGIC:
        meta = parse_unit_metadata(blob)
        name, = [name for name, _ in template["roles"]]
        blob = fused.pack_fused([(name, int(meta.rows), blob)])
    member, = fused.parse_fused(blob)
    meta = parse_unit_metadata(member.blob)
    store.schemes[entry["name"]] = {**template, "grid": meta.grid.name, "body": meta.body.name,
        "plane": meta.manifest.scale_plane.kind.name, "q256": int(meta.manifest.branch.root_q256) // int(meta.grid.arity),
        "rows": int(meta.rows), "columns": int(meta.columns), "roles": [[member.name, int(meta.rows)]], "wire_bytes": len(blob)}
    get = store.get
    store.get = lambda name: torch.frombuffer(bytearray(blob), dtype=torch.uint8) if name == entry["name"] + ".wire_bytes" else get(name)
    return store


def entries(root, *, family=None, structure=None):
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    result = []
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group["scheme"]
        if family and scheme["family"] != family or structure and scheme.get("structure", "dense") != structure:
            continue
        for name in group["targets"]:
            routed = scheme.get("structure") == "routed_moe"
            source = scheme["groups"]["w13"] if routed else scheme
            names = [f"{name}.{e}.{projection}.wire" for e in range(scheme["experts"]) for projection in ("gate_proj", "up_proj", "down_proj")] if routed else [name + ".wire_bytes"]
            paths = sorted({str(root / index[tensor]) for tensor in names})
            if not all(Path(path).is_file() for path in paths):
                raise ValueError(f"missing wire shard for {root}: {name}")
            result.append({"root": str(root), "name": name, "contract_family": FAMILY[scheme["family"]],
                "family": scheme["family"], "structure": "routed_moe" if routed else "dense",
                "layer": int(name.split("layers.")[1].split(".")[0]), "q256": source["q256"],
                "rows": source["rows"], "cols": source["columns"], "roles": source["roles"],
                "experts": scheme.get("experts", 1), "wire_sources": paths,
                "paths_exist": True, "wire_stamp": None, "runs_table": None})
    return result


def arm(name, modules, library, decoder, *, routed=False, selector=None):
    symbol = ("tessera.routed_fused_e2m1.FusedRoutedE2M1MoE.__call__" if library == "e2m1" else
              "tessera.routed_fused.FusedRoutedWindowMoE.__call__") if routed else "tessera.routed_fused_e2m1.dense_forward_quantized" if library == "e2m1" else "tessera::window_gemm_dense" if library == "triton" else "tessera::fused_window_dense"
    return {"schema": "tessera.numerics_population.v1", "arm": name, "sms": None,
            "reviewed": False, "wire_stamps_read": False,
            "modules": [{**entry, "library": library, "pair": [symbol, decoder], "selector_env": selector or {}}
                        for entry in modules]}


def draft(out):
    dense = entries(BASE / "t8-coverage-20260930/stubs/stub-T8D1", family="TESSERA_FP8", structure="dense")
    dense += entries(STUB, family="TESSERA_FP8", structure="dense")
    template, = [entry for entry in dense if entry["root"] == str(STUB) and entry["name"] == "model.language_model.layers.5.mlp.shared_experts.down_proj"]
    for rung in (864, 928):
        name = "model.language_model.layers.7.mlp.shared_experts.down_proj"
        wire = CACHE / (name.replace(".", "__") + f"__TESSERA_E4M3_K1_R{rung}.tessera")
        if not wire.is_file():
            raise ValueError(f"missing canonical wire: {wire}")
        dense.append({**template, "name": name, "scheme_template": template["name"], "layer": 7, "q256": rung,
                      "wire_path": str(wire), "wire_sources": [str(wire)]})
    routed = entries(BASE / "ig1080-base-export-20261009-r2", structure="routed_moe")
    routed += entries(STUB, family="TESSERA_FP8", structure="routed_moe")
    for layer, rung in ((3, 832), (3, 960), (4, 864), (3, 944)):
        template, = [entry for entry in routed if entry["root"] == str(STUB) and entry["layer"] == layer and "wire_root" not in entry]
        paths = [str(CACHE / (f"model__language_model__layers__{layer}__mlp__experts__{e}__{projection}__TESSERA_E4M3_K1_R{rung}.tessera"))
                 for e in range(288) for projection in ("gate_proj", "up_proj", "down_proj")]
        if not all(Path(path).is_file() for path in paths):
            raise ValueError(f"missing canonical routed wires at layer {layer}, R{rung}")
        routed.append({**template, "q256": rung, "wire_root": str(CACHE),
                       "wire_family": {"payload": "TESSERA_E4M3_K1", "rung": f"R{rung}", "family": "TESSERA_FP8"}, "wire_sources": paths})
    populations = [arm("N-E4M3-R", routed, "e4m3mma", "native_routed_window_classes_e4m3mma", routed=True)]
    for library, suffix, decoder, selector in (("e4m3mma", "mma", "native_fused_window_dense_e4m3mma", {}),
        ("e4m3", "f16", "native_fused_window_dense", {"TESSERA_FUSED_E4M3_MMA": "f16"}),
        ("triton", "tri", "native_window_gemm", {"TESSERA_DENSE_FUSED": "0"})):
        populations.append(arm("N-E4M3-D-" + suffix, dense, library, decoder, selector=selector))
    for name, root, library, dense_decoder, routed_decoder in (("N-E2M1", OPTION_C, "e2m1", "native_fused_window_dense_e2m1", "native_routed_fused_window_e2m1"),
        ("N-BF16", BF16, "value", "native_fused_window_dense_bf16", "native_routed_window_classes_bf16")):
        population = arm(name, entries(root, structure="dense"), library, dense_decoder)
        population["modules"] += arm(name, entries(root, structure="routed_moe"), library, routed_decoder, routed=True)["modules"]
        populations.append(population)
    out.mkdir(parents=True, exist_ok=True)
    for population in populations:
        write_json(out / (population["arm"] + ".json"), population)


def read_stamps(read, size, raw_role):
    """Read framing and manifests. This does not verify payload bytes."""
    from experiments.numerics_rows.cases import manifest_stamp, rate_layout
    from tessera import container, fused_frame as frame
    from tessera.manifest import BodyKind, Manifest
    from tessera.unit_artifact import _resolve_window_grid

    def take(offset, length):
        if offset < 0 or length < 0 or offset + length > size:
            raise ValueError("truncated population wire metadata")
        data = read(offset, length)
        if len(data) != length:
            raise ValueError("short population wire metadata read")
        return data

    if take(0, 8) == frame.FUSED_MAGIC:
        magic, version, count = frame._HEADER.unpack(take(0, frame.HEADER_BYTES))
        if version != frame._VERSION or not count:
            raise ValueError("invalid fused population header")
        cursor, heads = frame.HEADER_BYTES, []
        for _ in range(count):
            length, rows, blob_len = frame._MEMBER.unpack(take(cursor, frame.MEMBER_ROW_BYTES))
            cursor += frame.MEMBER_ROW_BYTES
            role = take(cursor, length).decode("utf-8")
            cursor += length
            frame._check_member(role, length, rows, blob_len)
            heads.append((role, rows, blob_len))
        members = []
        for role, rows, length in heads:
            members.append((role, rows, cursor, length))
            cursor += length
        if cursor != size or len({m[0] for m in members}) != len(members):
            raise ValueError("population fused lengths or role names disagree")
    else:
        members = [(raw_role, None, 0, size)]
    records = []
    for role, rows, offset, length in members:
        if length < container.HEADER_BYTES:
            raise ValueError("population member shorter than its header")
        magic, major, minor, header, manifest_bytes, region_bytes = container._HEADER.unpack(take(offset, container.HEADER_BYTES))
        if magic != container.MAGIC or major != container.SCHEMA_MAJOR or minor not in container.SCHEMA_MINORS_READ or header != container.HEADER_BYTES:
            raise ValueError("invalid population member header")
        side = header + manifest_bytes
        if side > length or length - side > region_bytes:
            raise ValueError("population member lengths disagree")
        raw = take(offset + header, manifest_bytes)
        manifest = Manifest.decode(raw, schema_minor=minor)
        if manifest.schema_minor > minor or not any(t.exact_bytes == length - side for t in manifest.terminals):
            raise ValueError("population member has no declared terminal")
        if manifest.body != BodyKind.WINDOW:
            raise ValueError("numerics population needs a WINDOW wire")
        if rows is not None and rows != manifest.geometry.rows:
            raise ValueError("population fused member row count disagrees")
        grid = _resolve_window_grid(manifest)
        records.append({"role": role, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                        "declared_payload_sha256": manifest.payload_digest.hex(),
                        "wire_stamp": manifest_stamp(manifest, grid), **rate_layout(manifest.rates)})
    return records


def wire_locations(store, entry):
    if entry["structure"] != "routed_moe":
        role = store.schemes[entry.get("scheme_template", entry["name"])]["roles"][0][0]
        yield 0, "dense", Path(entry["wire_path"]) if "wire_path" in entry else None, entry["name"] + ".wire_bytes", role
        return
    from experiments.routed_pair_oracle import wire_path

    for expert in range(int(entry["experts"])):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            path = wire_path(entry["wire_root"], entry["layer"], expert, projection, entry["wire_family"]) if "wire_root" in entry else None
            yield expert, projection, path, f"{entry['name']}.{expert}.{projection}.wire", projection


@contextmanager
def wire_reader(store, path, tensor):
    """Use range reads; never request the complete safetensors tensor."""
    if path is not None:
        with path.open("rb") as stream:
            def read(offset, length):
                stream.seek(offset)
                return stream.read(length)
            yield read, path.stat().st_size, str(path)
    else:
        shard = store.index[tensor]
        if shard not in store._open:
            store._open[shard] = store._safe_open(str(Path(store.root) / shard), "pt", device="cpu")
        view = store._open[shard].get_slice(tensor)
        shape = view.get_shape()
        if view.get_dtype() != "U8" or len(shape) != 1:
            raise ValueError(f"population wire needs a uint8 vector: {tensor}")
        yield lambda offset, length: view[offset:offset + length].numpy().tobytes(), shape[0], str(Path(store.root) / shard) + ":" + tensor


def io_counts():
    return {name: int(value) for line in Path("/proc/self/io").read_text().splitlines()
            for name, value in [line.split(":")] if name in ("rchar", "read_bytes")}


def inspect(population, checkpoint=None):
    import torch
    from experiments.dense_fused_oracle import Store

    torch.set_num_threads(1)
    retained, tables = [], {}
    population["duplicate_run_tables"] = []
    population["inspection_scope"] = "framing and manifest only; payload verification remains in C1 and the rows"
    population["wire_stamps_read"] = False
    for index, entry in enumerate(population["modules"]):
        start, before = time.perf_counter(), io_counts()
        path = entry.get("wire_path", entry.get("wire_root", entry["root"]))
        print(f"module-start arm={population['arm']} module={index + 1} name={entry['name']} path={path}", flush=True)
        store, records = Store(entry["root"]), []
        wire_bytes, metadata_bytes = 0, 0
        last_progress = start
        for expert, projection, source, tensor, role in wire_locations(store, entry):
            with wire_reader(store, source, tensor) as (read, size, location):
                def measured(offset, length):
                    nonlocal metadata_bytes
                    data = read(offset, length)
                    metadata_bytes += len(data)
                    return data
                items = read_stamps(measured, size, role)
            wire_bytes += size
            for record in items:
                if record["wire_stamp"]["q256"] != entry["q256"]:
                    raise ValueError(f"wire rung differs from population: {entry['name']}, {expert}, {role}")
                records.append({"expert": expert, "projection": projection, "source": location, **record})
            now = time.perf_counter()
            if now - last_progress >= 5:
                print(f"wire-progress path={location} size={size} elapsed_s={now - start:.3f} metadata_bytes={metadata_bytes}", flush=True)
                last_progress = now
        entry["wire_stamp"] = records
        entry["runs_table"] = [record["runs_table"] for record in records]
        entry["inspection"] = {"elapsed_s": time.perf_counter() - start, "wire_bytes": wire_bytes,
            "metadata_bytes": metadata_bytes, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "io_delta": {key: value - before[key] for key, value in io_counts().items()}}
        print(f"module-done arm={population['arm']} module={index + 1} path={path} "
              + json.dumps(entry["inspection"], sort_keys=True), flush=True)
        if checkpoint:
            checkpoint(population)
        if entry["root"] == str(BASE / "t8-coverage-20260930/stubs/stub-T8D1"):
            key = tuple(sorted({tuple(tuple(run) for run in record["runs_table"]) for record in records}))
            if key in tables:
                population["duplicate_run_tables"].append({"name": entry["name"], "covered_by": tables[key]})
                continue
            tables[key] = entry["name"]
        retained.append(entry)
    population["modules"] = retained
    population["wire_stamps_read"] = True
    return population


def profile_wire(population, out):
    """Measure the metadata path and the prior full-reader work on one wire."""
    import torch
    from experiments.dense_fused_oracle import Store
    from tessera import fused
    from tessera.unit_artifact import parse_unit_artifact

    torch.set_num_threads(1)
    entry = population["modules"][0]
    store = Store(entry["root"])
    _, _, path, tensor, role = next(wire_locations(store, entry))
    records = []
    def stage(name, call):
        start, before = time.perf_counter(), io_counts()
        start_utc = datetime.now(timezone.utc).isoformat()
        print(f"profile-start stage={name}", flush=True)
        profiler = cProfile.Profile()
        value = profiler.runcall(call)
        elapsed = time.perf_counter() - start
        end_utc = datetime.now(timezone.utc).isoformat()
        profiler.dump_stats(str(out / (name + ".prof")))
        with (out / (name + ".profile.txt")).open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(40)
        records.append({"stage": name, "elapsed_s": elapsed, "start_utc": start_utc, "end_utc": end_utc,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "io_delta": {key: value - before[key] for key, value in io_counts().items()}})
        write_json(out / "profile.json", {"source": location, "wire_bytes": size, "stages": records})
        print("profile-done " + json.dumps(records[-1], sort_keys=True), flush=True)
        return value
    with wire_reader(store, path, tensor) as (read, size, location):
        metadata = stage("metadata", lambda: read_stamps(read, size, role))
        blob = stage("full-read", lambda: read(0, size))
    stage("full-sha256", lambda: hashlib.sha256(blob).hexdigest())
    members = [member.blob for member in fused.parse_fused(blob)] if blob[:8] == fused.FUSED_MAGIC else [blob]
    # Keep only each stamp, so one member's expanded planes can be released.
    from experiments.numerics_rows.cases import wire_stamp
    stamps = stage("full-reader", lambda: [wire_stamp(parse_unit_artifact(member, device="cpu")) for member in members])
    if stamps != [item["wire_stamp"] for item in metadata]:
        raise ValueError("metadata stamps differ from the full reader")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("draft", "inspect", "profile"))
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--arm", action="append", help="Inspect only this arm; repeat to select several.")
    args = parser.parse_args()
    if args.mode == "draft":
        draft(args.out)
        return
    check_source(args.source)
    sys.path.insert(0, str(args.source / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    for path in sorted(args.population_dir.glob("*.json")):
        population = json.loads(path.read_text())
        if args.arm and population["arm"] not in args.arm:
            continue
        if args.mode == "profile":
            profile_wire(population, args.out)
            return
        try:
            inspect(population, lambda value: write_json(args.out / path.name, value))
        except Exception as error:
            errors.append({"arm": population["arm"], "error": str(error), "traceback": traceback.format_exc()})
        write_json(args.out / path.name, population)
        write_json(args.out / "errors.json", errors)
    raise SystemExit(3 if errors else 0)


if __name__ == "__main__":
    main()
