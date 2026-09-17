#!/usr/bin/env python
"""The GLM-5.3-Flash routed owner's own request inputs, out of the census wires.

``experiments/bench_native_moe_operator.py`` prices ONE routed owner from its
geometry, one wire per expert-role member, and one safetensors carrying
``source_weight/<unit>``, ``rendered_weight/<unit>``, the mandatory routing bias
and both phases' input/reference tensors.  Nothing produces that set for
GLM-5.3-Flash: the census encodes 864 layer-3 units and writes them under the
producer's own unit names, and a request has to name, per member, the wire and
the record the loader will verify.

This driver emits the half of that set which is a *producer* fact, from
artifacts that already exist, and refuses to invent the other half:

``source`` mode
    the shared bf16 source tensors -- the same bytes for every rate and every
    rank, because A4/A8/A16 quantize one checkpoint -- as ONE safetensors with
    ``source_weight/<unit>`` keys, streamed one unit at a time out of the three
    source shards that hold this layer.

``rate`` mode
    one request member per unit: the wire path (the export's own one-unit blob)
    and a record re-derived from the ACTUAL source slice, the campaign's own
    sealed Hessian commitments and the producer package that wrote the wire.

What it does not produce: the per-phase input/topk/reference tensors, the
mandatory FP32 correction bias, and the per-rank rendered weights.  Those come
from a real vLLM capture and a real render, and a placeholder the harness cannot
tell from a measurement is worse than an absent one, so this driver names them
as remaining inputs in every receipt instead.

The identity is never copied from the campaign's manifest.  It is re-derived
under the producer that wrote the receipts (``historical_producer`` in the
served manifest, whose package seal is checked) and then REQUIRED to equal the
record the export sealed, so an accepted run says "the producer's own identity
factory reproduces this receipt from these bytes" rather than "these bytes were
on disk".
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

ROLE_ORDER = ("w1", "w3", "w2")
#: The source spelling of each role, from the runtime's own table.
PROJECTION_ROLE = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
SOURCE_PREFIX = "source_weight/"
SCHEMA = "tessera.glm_routed_owner_inputs.v1"
#: ``str(dtype)`` -> the safetensors spelling the container writes.
DTYPE_NAMES = {"torch.bfloat16": "BF16", "torch.float32": "F32"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode())


def dump(path: Path, value) -> None:
    """Write one JSON artifact durably, or leave the previous one alone.

    Flush and fsync the candidate, then rename it over the destination and
    fsync the directory, so "the record is written" is true when this returns
    and a reader either sees the old artifact or the complete new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    """Make a rename in ``path`` survive a crash, where the platform allows it."""
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def commit_progress(units: int, phase: str, unit: str | None = None) -> bool:
    """Say how much work is COMMITTED, through PrismaBuild's own channel.

    The worker names its helper in ``PRISMABUILD_ACTION_PROGRESS_HELPER``, and
    that helper is the supported channel -- it knows the declared phase names
    and refuses a typo where the report is made rather than quietly by the
    watcher.  The ten-line file write below it is the documented fallback for
    an action that cannot see the fleet mount (a pinned container), held byte
    for byte against the worker's reader.

    Both are no-ops when this action was not admitted under a phase policy, so
    callers report unconditionally -- and only for work that is already
    durable.  A report is a claim that the unit is on disk, not that a loop
    reached it.
    """
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper:
        import runpy

        commit = runpy.run_path(helper)["commit"]
        commit(units, phase)
        return True
    path = os.environ.get("PRISMABUILD_ACTION_PROGRESS_PATH")
    token = os.environ.get("PRISMABUILD_ACTION_PROGRESS_TOKEN")
    if not path or not token:
        return False
    record = {"schema": "prismabuild.action_progress.v1", "token": token,
              "phase": phase, "units_completed": units, "unit": unit,
              "reported_unix": time.time()}
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return True


# ---------------------------------------------------------------------------
# The plan: what the producer says this layer's units are
# ---------------------------------------------------------------------------
def load_served_manifest(export: Path) -> dict:
    return json.loads((Path(export) / "tessera_serving_manifest.json").read_text())


def bundle_manifest(served: dict, bundle: Path) -> dict:
    """The cached-units bundle the export declares, checked against its hash."""
    from tessera.cached_unit import read_manifest

    manifest = read_manifest(Path(bundle))
    declared = served["cached_units"]
    observed = canonical_sha256(manifest)
    if observed != declared["manifest_sha256"]:
        raise SystemExit(
            f"{bundle}: canonical manifest sha256 {observed} is not the "
            f"{declared['manifest_sha256']} this export declares; the bundle is not the one "
            "these wires were written from")
    if manifest.get("schema") != "tessera.cached_units.v1":
        raise SystemExit(f"{bundle}: not a cached-unit bundle")
    if len(manifest["units"]) != declared["planned_units"]:
        raise SystemExit(f"{bundle}: {len(manifest['units'])} units, not {declared['planned_units']}")
    return manifest


def research_selected_record(served: dict):
    """The research-selected owner block this export ran with, or None.

    A BF16 or above-TP1 E4M3 expert stack has no production builder, so the
    exporter refuses to plan one unless it was given the versioned
    research-selected owner -- and it records that input, with its digest, in
    ``export_identity.options``.  Reading the flag from there is what keeps
    this driver's plan the exporter's plan: a flag guessed here could plan a
    stack the export refused, or refuse one it planned.
    """
    options = served.get("export_identity", {}).get("options") or {}
    return options.get("research_selected_moe")


def layer_units(model: Path, stack: str, plan_entry: dict, *, research_selected: bool) -> dict:
    """The producer's own projection of this layer, through the planner.

    Header shapes come from ``quantizable`` (one header read per shard, never a
    tensor load) and the projection from ``project_expert_plan``, so the unit
    list, its geometry and its source selectors are the exporter's answers
    rather than a second reading of the checkpoint layout here.
    """
    from experiments.export_tessera_serving import project_expert_plan, quantizable

    _shards, dense, packed, routed = quantizable(Path(model))
    projected = project_expert_plan({**dense, **packed, **routed},
                                    json.loads((Path(model) / "config.json").read_text()),
                                    {stack: dict(plan_entry)},
                                    research_selected=research_selected)
    entry = projected["stacks"][stack]
    if entry.get("source_layout") != "unpacked_per_expert":
        raise SystemExit(f"{stack}: planned source layout is not the unpacked per-expert one")
    entry["unit_of"] = {unit["tensor"][: -len(".weight")]: unit for unit in entry["units"]}
    return entry


def member_roster(entry: dict) -> list:
    """The request's member roster, in the owner's own expert-role order.

    The producer's unit order is already expert-major with gate, up, down
    inside each expert (``MOE_GROUPS`` then ``MOE_GROUP_PROJECTIONS``), which is
    the harness's ``ROLE_ORDER`` -- so the roster is checked against that order
    rather than sorted into it.  Sorting member names is what the harness's own
    validator forbids; the order is semantic.
    """
    members = []
    for index, unit in enumerate(entry["units"]):
        expert, role = index // len(ROLE_ORDER), ROLE_ORDER[index % len(ROLE_ORDER)]
        if unit["expert"] != expert or PROJECTION_ROLE[unit["projection"]] != role:
            raise SystemExit(
                f"{unit['tensor']}: the producer plans expert {unit['expert']}/"
                f"{unit['projection']} at roster position {index}, which is {expert}/{role}")
        if unit["source_tensor"] != unit["tensor"]:
            raise SystemExit(f"{unit['tensor']}: an unpacked unit is its own whole source tensor")
        members.append({"unit": unit["tensor"][: -len(".weight")], "expert": expert, "role": role,
                        "projection": unit["projection"], "group": unit["group"],
                        "rows": unit["rows"], "cols": unit["cols"]})
    expected = entry["experts"] * len(ROLE_ORDER)
    if len(members) != expected:
        raise SystemExit(f"{entry['stack']}: {len(members)} member wires, not {expected}")
    return members


# ---------------------------------------------------------------------------
# One safetensors, written without holding the population
# ---------------------------------------------------------------------------
def streamed_safetensors(path: Path, tensors: list, read, *, check, progress=None) -> dict:
    """Write ``tensors`` -- ``(key, torch dtype str, shape, nbytes)`` -- streamed.

    ``safetensors`` has no streaming writer: ``serialize_file`` takes the whole
    population as a mapping, and 864 source slices of this layer are 13.50 GiB (14.50 GB).
    The container is simple and fully determined by the plan -- an 8-byte
    little-endian header length, that many bytes of JSON, then one contiguous
    region whose per-tensor offsets are known before any byte is read -- so the
    header is written from the plan and the payload is copied in one unit at a
    time.  Nothing here is trusted: ``check`` re-reads what was written through
    ``safetensors`` itself and must agree.

    The order is the whole point, and it is the order the artifact's own claims
    need: the candidate is written to a partial file, flushed and fsynced,
    CHECKED in that partial form, and only then renamed over the destination --
    with the directory fsynced.  A destination that already holds a good file
    is therefore never replaced by an unverified candidate, and a failed check
    leaves the previous artifact untouched and the rejected candidate on disk
    as evidence.  ``progress`` is called once, after the rename, so a phase
    watcher is told about work that is durable rather than about work a loop
    reached.  A partial file is not a checkpoint: it is not resumable, and
    nothing here reports it as one.
    """
    import torch

    header, offset = {}, 0
    for key, dtype, shape, length in tensors:
        header[key] = {"dtype": DTYPE_NAMES[dtype], "shape": list(shape),
                       "data_offsets": [offset, offset + length]}
        offset += length
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    blob += b" " * ((-(len(blob) + 8)) % 8)
    partial = Path(str(path) + f".partial-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as stream:
        stream.write(struct.pack("<Q", len(blob)))
        stream.write(blob)
        for key, dtype, shape, length in tensors:
            value = read(key)
            if str(value.dtype) != dtype or list(value.shape) != list(shape):
                raise SystemExit(f"{key}: source tensor is not the plan's {dtype} {list(shape)}")
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
            if raw.nbytes != length:
                raise SystemExit(f"{key}: {raw.nbytes} payload bytes, not {length}")
            stream.write(raw.tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        check(partial, header)
    except BaseException:
        # The destination is untouched and the rejected candidate stays where a
        # reader can look at it.  Say so, because the next step is a person.
        print(f"candidate refused; it is retained at {partial} and {path} was not replaced",
              flush=True)
        raise
    digest = sha256_file(partial)
    os.replace(partial, path)
    fsync_directory(path.parent)
    written = {"path": str(path), "sha256": digest, "device_bytes": offset,
               "header_bytes": 8 + len(blob), "tensors": len(tensors),
               "published_after": "candidate flushed, fsynced and verified in place"}
    if progress is not None:
        progress(len(tensors), str(path))
    return written


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------
def resolve_plan(served: dict, stack: str) -> dict:
    plan = served.get("plan")
    if not isinstance(plan, dict) or stack not in plan:
        raise SystemExit(f"the served manifest has no plan entry for {stack}")
    entry = plan[stack]
    missing = {"grid", "q256"} - set(entry)
    if missing:
        raise SystemExit(f"{stack}: plan entry lacks {sorted(missing)}")
    return entry


def remaining_inputs() -> dict:
    """What a request still needs from a real capture, named rather than faked."""
    return {
        "rendered_weight": "one per rank per rate, from the loader's own PWC render",
        "routing_bias": "the capture's FP32 e_score_correction_bias, fp32 [288]",
        "phase_tensors": ["prefill.input", "prefill.topk_ids", "prefill.topk_weights",
                          "prefill.reference_qdq", "prefill.reference_output",
                          "decode.input", "decode.topk_ids", "decode.topk_weights",
                          "decode.reference_qdq", "decode.reference_output"],
        "routing_capture_sha256": "binds the capture the bias and phase tensors came from",
    }


def preflight_producer_provenance(served: dict, manifest: dict, members: list) -> dict:
    """Which producer wrote these wires, settled BEFORE any output exists.

    ``encoder_source_sha256`` is a seal over the ENTIRE producer package,
    serving files included, so a record re-derived under a newer pin would
    carry that pin's hash and silently re-label bytes the old package wrote.
    These records are re-derived under the package the export itself declares
    (``cached_units.historical_producer``) and are required to equal the
    records the export sealed, so what they carry is that package's seal --
    never this process's.  The one thing this process adds is the encoder
    FIXTURE id, which is behaviour rather than provenance and is what
    ``resumable`` compares; it agrees here, which is why the wire can be
    consumed at all.

    Every part of that is a property of the export and its records, so it is
    checkable before the first record is written.  Refusing here is what keeps
    a refusal from leaving half a rate's records on disk.
    """
    declared = served["cached_units"].get("historical_producer")
    if declared is None:
        raise SystemExit(
            "this export declares no historical producer, so these records cannot be bound "
            "to the package that wrote the wires; refusing to present this process's seal "
            "as the wire's provenance")
    seals = {}
    for member in members:
        identity = manifest["units"][member["unit"]]["identity"]
        pair = (identity["encoder_source_sha256"], identity["encoder_fixture_id"])
        seals.setdefault(pair, []).append(member["unit"])
    if len(seals) != 1:
        raise SystemExit(
            "the export's own records do not agree on one producer: "
            + "; ".join(f"{pair[0][:12]}...x{len(units)} (first {units[0]})"
                        for pair, units in sorted(seals.items())))
    (carried_source, carried_fixture), units = next(iter(seals.items()))
    if carried_source != declared["source_sha256"]:
        raise SystemExit(
            f"{units[0]}: the records carry encoder_source_sha256 {carried_source} and the "
            f"export declares {declared['source_sha256']} for the package that wrote these wires")
    return {"carried_encoder_source_sha256": carried_source,
            "carried_encoder_fixture_id": carried_fixture,
            "sealed_records": len(members)}


def wire_producer_provenance(served: dict, preflight: dict, current: dict) -> dict:
    """The preflight's answer, written into the receipt.  Reports; never refuses."""
    declared = served["cached_units"]["historical_producer"]
    carried_source = preflight["carried_encoder_source_sha256"]
    carried_fixture = preflight["carried_encoder_fixture_id"]
    return {
        "historical_producer": dict(declared),
        "carried_encoder_source_sha256": carried_source,
        "carried_encoder_fixture_id": carried_fixture,
        "sealed_records_agreeing": preflight["sealed_records"],
        "current_process_encoder_source_sha256": current["encoder_source_sha256"],
        "current_process_encoder_fixture_id": current["encoder_fixture_id"],
        "restamped_with_the_current_encoder": False,
        "carried_source_seal_is_this_processes": carried_source == current["encoder_source_sha256"],
        "admission": (
            "the wire's producer is the package named in historical_producer; the record's "
            "encoder_source_sha256 is that package's own seal, reproduced by re-deriving the "
            "identity under it, and no current-pin fingerprint is asserted for these bytes. "
            "What this process contributes is the encoder fixture id, a behavioural identity: "
            "it agrees with the wire's stamped value, which is the compatibility the consumer "
            "needs, and it is not provenance."),
    }


def source_mode(args) -> int:
    """Write the one shared bf16 source tensor file, and prove what it holds."""
    from safetensors import safe_open
    from tessera.cached_unit import tensor_identity

    served = load_served_manifest(args.export)
    stack = args.stack
    plan_entry = resolve_plan(served, stack)
    research = research_selected_record(served)
    entry = layer_units(args.model, stack, plan_entry, research_selected=research is not None)
    members = member_roster(entry)
    manifest = bundle_manifest(served, args.bundle)
    absent = [m["unit"] for m in members if m["unit"] not in manifest["units"]]
    if absent:
        raise SystemExit(f"{len(absent)} member units are absent from the cached bundle, "
                         f"starting with {absent[:3]}")
    sealed = {m["unit"]: manifest["units"][m["unit"]]["identity"]["source"] for m in members}

    index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted({weight_map[f"{m['unit']}.weight"] for m in members})

    def read(unit_key: str):
        tensor = f"{unit_key[len(SOURCE_PREFIX):]}.weight"
        return handles[weight_map[tensor]].get_tensor(tensor)

    tensors = [(f"{SOURCE_PREFIX}{m['unit']}", "torch.bfloat16", (m["rows"], m["cols"]),
                2 * m["rows"] * m["cols"]) for m in members]
    tensors.sort(key=lambda item: item[0])
    observed = {}

    def check(path: Path, header: dict) -> None:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(header):
                raise SystemExit(f"{path}: safetensors roster differs from the header")
            for key in sorted(header):
                value = handle.get_tensor(key)
                if DTYPE_NAMES[str(value.dtype)] != header[key]["dtype"] \
                        or list(value.shape) != header[key]["shape"]:
                    raise SystemExit(f"{key}: read-back dtype/shape differs from the header")
                unit = key[len(SOURCE_PREFIX):]
                identity = tensor_identity(value)
                if identity != sealed[unit]:
                    raise SystemExit(
                        f"{key}: re-read source differs from the bundle's sealed source identity "
                        f"for {unit}")
                observed[unit] = identity["sha256"]
        if len(observed) != len(header):
            raise SystemExit(f"{path}: {len(observed)} tensors re-read, not {len(header)}")

    with ExitStack() as opened:
        handles = {shard: opened.enter_context(
            safe_open(str(Path(args.model) / shard), framework="pt", device="cpu"))
            for shard in shards}
        def progress(count: int, key: str) -> None:
            # Called once, after the candidate was verified in place, renamed
            # and the directory fsynced.  A 13.50 GiB container has no partial
            # durable state to report, so this is the only honest number and
            # the phase allowance has to cover the whole write.
            commit_progress(count, "source", key)
            print(f"source published {count}/{count} {key}", flush=True)

        written = streamed_safetensors(Path(args.out) / "layer3-routed-owner-source.safetensors",
                                       tensors, read, check=check, progress=progress)
    receipt = {
        "schema": SCHEMA, "mode": "source", "layer": args.layer, "stack": stack,
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(args.model).resolve()),
        "model_config_sha256": sha256_file(Path(args.model) / "config.json"),
        "units": len(members), "source_bytes": sum(t[3] for t in tensors),
        "producer_plan": {"schema": "tessera.expert_projection.v1", "stack": stack,
                          "grid": plan_entry["grid"], "q256": plan_entry["q256"],
                          "source_layout": plan_entry["source_layout"],
                          "experts": entry["experts"], "hidden_size": entry["hidden_size"],
                          "intermediate_size": entry["intermediate_size"]},
        "source_shards": shards,
        "bundle_manifest_sha256": served["cached_units"]["manifest_sha256"],
        "research_selected_moe": research,
        "source_tensor_file": written,
        "verified": {"keys": len(members),
                     "tensor_identity_checked_against_the_bundles_sealed_source": len(observed),
                     "read_back_through_safetensors": True},
        "reused_across": ["a4", "a8", "a16", "tp1", "tp2"],
        "remaining_inputs": remaining_inputs(),
    }
    dump(Path(args.out) / "source-receipt.json", receipt)
    print(json.dumps({"schema": SCHEMA, "mode": "source", "status": "source_written",
                      "file": written["path"], "sha256": written["sha256"],
                      "bytes": written["device_bytes"], "units": len(members)}, sort_keys=True),
          flush=True)
    return 0


def identity_factory(served: dict, activation):
    """The producer's own identity factory, H identity established by commitment.

    The campaign's receipts were written by a *historical* producer package --
    that source seal is part of every identity they carry -- so the factory
    comes from that package, loaded under its own seal, exactly as the exporter
    loads it.  Where no historical producer is declared the current package is
    the same factory.
    """
    from tessera.cached_unit import CachedUnitIdentity, unit_input_identity
    from tessera.historical_producer import load_historical_producer

    declared = served["cached_units"].get("historical_producer")
    producer = None
    if declared is not None:
        producer = load_historical_producer(declared["package"], declared["source_sha256"])
    factory = unit_input_identity if producer is None else producer.input_identity

    def derive(weight, unit_name, unit, grid, q256, *, activation=None):
        return factory(weight, unit, grid, q256, activation=activation)

    return CachedUnitIdentity(derive, activation, mode="committed"), producer


def rate_mode(args) -> int:
    """Re-derive every member's sealed identity and bind it to its wire."""
    from safetensors import safe_open
    from tessera.cached_unit import make_unit_record
    from tessera.export import ActivationSource

    served = load_served_manifest(args.export)
    stack = args.stack
    plan_entry = resolve_plan(served, stack)
    research = research_selected_record(served)
    entry = layer_units(args.model, stack, plan_entry, research_selected=research is not None)
    members = member_roster(entry)
    manifest = bundle_manifest(served, args.bundle)
    if not Path(args.source_file).is_file():
        raise SystemExit(f"the shared source file is absent: {args.source_file}")
    # Every provenance claim this rate will make is a property of the export
    # and its records, so it is settled here -- before the first record is
    # written -- rather than after 864 of them are on disk.
    provenance = preflight_producer_provenance(served, manifest, members)

    block = served["activation_aware"]
    settings = {key: value for key, value in block.items() if key not in ("hessian", "note")}
    activation = ActivationSource.from_capture(block["hessian"]["path"], **settings)
    identity_of, producer = identity_factory(served, activation)
    from tessera.control import grid_for_name

    grid = (producer.grid_for_name if producer is not None else grid_for_name)(plan_entry["grid"])
    q256 = plan_entry["q256"]
    wire_dir = Path(args.wire_dir)
    from tessera.cached_unit import encoder_source_sha256 as this_source_seal
    from tessera.encoder_identity import encoder_fixture_id

    records, wire_inputs, rebound, wire_bytes = {}, [], 0, 0
    try:
        with safe_open(str(args.source_file), framework="pt", device="cpu") as source_handle:
            keys = set(source_handle.keys())
            expected_keys = {f"{SOURCE_PREFIX}{m['unit']}" for m in members}
            if keys != expected_keys:
                raise SystemExit(
                    f"{args.source_file}: safetensors roster is not this layer's {len(members)} "
                    f"members (missing {sorted(expected_keys - keys)[:2]}, "
                    f"extra {sorted(keys - expected_keys)[:2]})")
            for member in members:
                unit = member["unit"]
                source = source_handle.get_tensor(f"{SOURCE_PREFIX}{unit}")
                identity = identity_of(source, unit, entry["unit_of"][unit], grid, q256)
                sealed = manifest["units"][unit]
                if identity != sealed["identity"]:
                    differing = sorted(key for key in set(identity) | set(sealed["identity"])
                                       if identity.get(key) != sealed["identity"].get(key))
                    raise SystemExit(
                        f"{unit}: re-derived identity differs from the sealed receipt in "
                        f"{differing}; refusing to emit a record the export did not write")
                blob = (wire_dir / sealed["file"]).read_bytes()
                observed = sha256_bytes(blob)
                if observed != sealed["blob_sha256"] or len(blob) != sealed["blob_bytes"]:
                    raise SystemExit(f"{unit}: wire {sealed['file']} is not the sealed blob")
                record = make_unit_record(blob, identity, filename=sealed["file"])
                record_path = Path(args.out) / args.rate / "records" / f"{unit}.json"
                dump(record_path, record)
                records[unit] = {"record_path": str(record_path),
                                 "record_sha256": sha256_file(record_path),
                                 "blob_sha256": observed, "blob_bytes": len(blob),
                                 "file": sealed["file"],
                                 "wire_path": str(wire_dir / sealed["file"])}
                wire_inputs.append({"unit": unit, "expert": member["expert"], "role": member["role"],
                                    "format": args.format,
                                    "wire_path": str(wire_dir / sealed["file"]),
                                    "wire_record_path": str(record_path)})
                rebound += 1
                wire_bytes += len(blob)
                if rebound % 32 == 0 or rebound == len(members):
                    commit_progress(rebound, "rebind", unit)
                    print(f"rebind {rebound}/{len(members)} {unit}", flush=True)
    finally:
        activation.hessians.close()

    members_path = Path(args.out) / args.rate / "members.json"
    dump(members_path, wire_inputs)
    receipt = {
        "schema": SCHEMA, "mode": "rate", "rate": args.rate, "layer": args.layer, "stack": stack,
        "format": args.format,
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export": str(Path(args.export).resolve()),
        "bundle_manifest": str(Path(args.bundle).resolve()),
        "bundle_manifest_sha256": served["cached_units"]["manifest_sha256"],
        "research_selected_moe": research,
        "source_file": {"path": str(Path(args.source_file).resolve()),
                        "sha256": sha256_file(args.source_file)},
        "historical_producer": served["cached_units"].get("historical_producer"),
        "hessian_identity": served["cached_units"]["hessian_identity"],
        "members": len(wire_inputs), "wire_bytes": wire_bytes,
        "wire_producer_provenance": wire_producer_provenance(
            served, provenance, {"encoder_source_sha256": this_source_seal(),
                                 "encoder_fixture_id": encoder_fixture_id().hex()}),
        "members_file": {"path": str(members_path), "sha256": sha256_file(members_path)},
        "member_records": records,
        "verified": {"identity_rederived_equals_the_sealed_receipt": rebound,
                     "source_tensor_is_the_sealed_source_identity": rebound,
                     "wire_blob_sha256_equals_the_sealed_blob": rebound,
                     "wire_accepted_by_make_unit_record": rebound},
        "remaining_inputs": remaining_inputs(),
    }
    dump(Path(args.out) / args.rate / "rate-receipt.json", receipt)
    print(json.dumps({"schema": SCHEMA, "mode": "rate", "status": "members_written",
                      "rate": args.rate, "members": len(wire_inputs), "wire_bytes": wire_bytes,
                      "members_file": str(members_path)}, sort_keys=True), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "rate"), required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--model", type=Path,
                        default=Path("/mnt/shared/models/GLM-5.3-Flash-BF16"))
    parser.add_argument("--rate", choices=("a4", "a8", "a16"))
    parser.add_argument("--format", help="the owner's format name, e.g. TESSERA_E2M1_K2_R896")
    parser.add_argument("--export", type=Path, required=True,
                        help="the merged first-artifact export whose plan and wires these are")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="the tessera.cached_units.v1 manifest that export declares")
    parser.add_argument("--wire-dir", type=Path,
                        default=Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
                                     "activation-runtime-allocation-20260911/union-a4a8a16-01/cache/wire"))
    parser.add_argument("--source-file", type=Path,
                        default=Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
                                     "routed-owner-inputs-layer3-20260917/"
                                     "layer3-routed-owner-source.safetensors"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.stack = f"model.language_model.layers.{args.layer}.mlp.experts"
    if args.mode == "rate" and (args.rate is None or args.format is None):
        parser.error("--mode rate needs --rate and --format")
    return source_mode(args) if args.mode == "source" else rate_mode(args)


if __name__ == "__main__":
    sys.exit(main())
