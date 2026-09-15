#!/usr/bin/env python3
"""Wire parse and write microbench: the instrument for tessera#502, #503 and #504.

``parent`` runs one child per (repeat, arm), alternating the arm order each
repeat; each arm is a ``src`` tree on its own ``sys.path``. It samples GPU
power beside the children, runs one ``torch.profiler`` child and one stack
sampler child per arm, and compares every digest the arms recorded.

``child`` reads real routed-expert window wires (A8: E4M3 window, CHANNEL)
into RAM, then per wire:

* ``container.parse`` alone (#503: one sha256 over the region, not two);
* ``parse_unit_artifact`` on the device, recording seconds, the storage bytes
  every unit tensor holds, the three step planes' storage (#502), and on CUDA
  the allocator's peak over the parse; the sha256 of every unit tensor's
  values;
* ``build_unit_artifact`` of the parsed unit at the default layout (#504:
  the zero-width COMPLETION plane), with the blob's sha256.

With ``--tcq-weight`` it also encodes an E2M1x2 R896 TCQ unit from a real GLM
expert weight (completion 0 and full) and times ``build_unit_artifact`` at
LADDER and LEGACY, digesting the blob (#504: the device level packer).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench_routed_load import (
    MOE_UNITS,
    PowerSampler,
    _StackSampler,
    _export_torch_profile,
    _load_wires,
    _summarise_stacks,
    _tree_identity,
)


def _tensor_sha(torch, tensor):
    flat = tensor.detach().contiguous().reshape(-1)
    head = f"{tensor.dtype}|{list(tensor.shape)}|".encode()
    return hashlib.sha256(head + flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _unit_tensors(torch, unit):
    names = ([f.name for f in dataclasses.fields(unit)] if dataclasses.is_dataclass(unit)
             else sorted(vars(unit)))
    return {name: getattr(unit, name) for name in names
            if isinstance(getattr(unit, name, None), torch.Tensor)}


def child(args) -> int:
    src = Path(args.src).resolve()
    sys.path.insert(0, str(src))
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import tessera
    from tessera import container
    from tessera.planes import PlaneLayout
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    if not Path(tessera.__file__).resolve().is_relative_to(src):
        raise SystemExit(f"tessera imported from {tessera.__file__}, not {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    layers = [int(x) for x in args.layers.split(",")]
    wires = _load_wires(Path(args.data), layers, args.experts)
    blobs = [(f"L{layer}.E{expert}.{proj}", wires[layer][expert][proj].numpy().tobytes())
             for layer in layers for expert in range(args.experts) for _g, _i, proj in MOE_UNITS]
    del wires
    profiler = None
    if args.profile == "torch":
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        profiler = profile(activities=activities, profile_memory=True)
        profiler.__enter__()
    sampler = _StackSampler() if args.profile == "sample" else None
    if sampler is not None:
        sampler.start()
    t_start = time.time()

    container_seconds = []
    for _key, blob in blobs:
        start = time.perf_counter()
        container.parse(blob)
        container_seconds.append(time.perf_counter() - start)

    units, unit_digests, write_digests = [], {}, {}
    for index, (key, blob) in enumerate(blobs):
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        parsed = parse_unit_artifact(blob, device=device)
        sync()
        parse_seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() - before if cuda else None
        retained = torch.cuda.memory_allocated() - before if cuda else None
        tensors = _unit_tensors(torch, parsed.unit)
        storages = {}
        for tensor in tensors.values():
            storage = tensor.untyped_storage()
            storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        step_planes = {name: tensors[name].untyped_storage().nbytes()
                       for name in ("anchors", "codes", "completion_bits") if name in tensors}
        if index < args.digest_units:
            unit_digests[key] = {name: _tensor_sha(torch, t) for name, t in sorted(tensors.items())}
        sync()
        start = time.perf_counter()
        manifest = parsed.manifest
        rebuilt = build_unit_artifact(parsed.unit, manifest.branch.unit_id, parsed.forests,
                                      manifest.branch.root_q256, parsed.code, fixture_id=None)[2]
        sync()
        write_seconds = time.perf_counter() - start
        write_digests[key] = hashlib.sha256(rebuilt).hexdigest()
        units.append({"key": key, "wire_bytes": len(blob), "parse_seconds": parse_seconds,
                      "write_seconds": write_seconds, "peak_alloc_bytes": peak,
                      "retained_alloc_bytes": retained, "unit_storage_bytes": sum(storages.values()),
                      "step_plane_storage_bytes": step_planes, "body": manifest.body.name,
                      "rewrite_equals_wire": rebuilt == blob})
        del parsed, tensors, rebuilt

    tcq = {}
    if args.tcq_weight:
        from tessera.alphabet import SERIALISABLE_GRIDS
        from tessera.export import DEFAULT_CODE, encode_linear_planes

        grid = {g.name: g for g in SERIALISABLE_GRIDS.values()}["E2M1x2"]
        weight = torch.load(args.tcq_weight, map_location="cpu", weights_only=True)
        weight = weight[: args.tcq_rows].float().to(device)
        for completion in (0, None):
            _e, unit, forests = encode_linear_planes(weight, grid=grid, q256=896, name="u",
                                                     completion=completion)
            label = f"c{'full' if completion is None else completion}"
            inputs = {name: _tensor_sha(torch, t)
                      for name, t in sorted(_unit_tensors(torch, unit).items())}
            for layout in (PlaneLayout.LADDER, PlaneLayout.LEGACY):
                seconds, blob = [], None
                for _ in range(args.write_repeats):
                    sync()
                    start = time.perf_counter()
                    blob = build_unit_artifact(unit, "u", forests, 896, DEFAULT_CODE,
                                               fixture_id=None, layout=layout)[2]
                    sync()
                    seconds.append(time.perf_counter() - start)
                tcq[f"{label}-{layout.name}"] = {
                    "seconds": seconds, "median_seconds": statistics.median(seconds),
                    "blob_sha256": hashlib.sha256(blob).hexdigest(), "blob_bytes": len(blob),
                    "completion_limit": int(unit.completion_limit),
                    "body": unit.body.name, "unit_input_sha256": inputs,
                    "completion_bits_device": str(unit.completion_bits.device)}
    t_end = time.time()
    if sampler is not None:
        sampler.finish(out / "sample.collapsed")
    if profiler is not None:
        profiler.__exit__(None, None, None)
        _export_torch_profile(profiler, out, len(blobs))
    summary = {
        "src": str(src), "tessera_file": tessera.__file__, "device": str(device),
        "host": platform.node(), "torch": torch.__version__, "t_start": t_start, "t_end": t_end,
        "wires": len(blobs), "wire_bytes": sum(len(b) for _k, b in blobs),
        "container_parse_seconds_total": sum(container_seconds),
        "parse_unit_seconds_total": sum(u["parse_seconds"] for u in units),
        "write_seconds_total": sum(u["write_seconds"] for u in units),
        "peak_alloc_bytes_max": max((u["peak_alloc_bytes"] or 0) for u in units) if units else 0,
        "unit_storage_bytes_max": max(u["unit_storage_bytes"] for u in units) if units else 0,
        "step_plane_storage_bytes_total": sum(sum(u["step_plane_storage_bytes"].values())
                                              for u in units),
        "tcq": tcq,
    }
    (out / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    (out / "digests.json").write_text(json.dumps(
        {"units": unit_digests, "writes": write_digests,
         "tcq": {k: {"blob_sha256": v["blob_sha256"], "unit_input_sha256": v["unit_input_sha256"]}
                 for k, v in tcq.items()}}, indent=1, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _run_child(args, src, out, profile="none"):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", "--src", str(src),
           "--data", args.data, "--layers", args.layers, "--experts", str(args.experts),
           "--device", args.device, "--digest-units", str(args.digest_units),
           "--write-repeats", str(args.write_repeats), "--tcq-rows", str(args.tcq_rows),
           "--profile", profile, "--out", str(out)]
    if args.tcq_weight:
        cmd += ["--tcq-weight", args.tcq_weight]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "child.log").open("w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(out), stdout=log, stderr=subprocess.STDOUT).returncode
    return rc, t0, time.time()


def parent(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [(name, Path(src).resolve()) for name, _, src in
            (spec.partition("=") for spec in args.arms.split(","))]
    meta = {"argv": sys.argv, "arms": {n: _tree_identity(s) for n, s in arms},
            "host": platform.node(), "python": sys.executable, "t_start": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    sampler = PowerSampler(out / "power.csv") if args.device == "cuda" else None
    if sampler is not None:
        sampler.start()

    def window(t0, t1):
        return sampler.window(t0, t1) if sampler is not None else None

    runs, failed = [], False
    for rep in range(args.repeats):
        for arm, src in (arms if rep % 2 == 0 else list(reversed(arms))):
            run_dir = out / f"rep{rep}-{arm}"
            rc, t0, t1 = _run_child(args, src, run_dir)
            row = {"rep": rep, "arm": arm, "rc": rc, "t0": t0, "t1": t1}
            if rc == 0:
                row["summary"] = json.loads((run_dir / "summary.json").read_text())
                row["power"] = window(row["summary"]["t_start"], row["summary"]["t_end"])
            else:
                failed = True
                print((run_dir / "child.log").read_text()[-4000:], flush=True)
            runs.append(row)
            print(json.dumps({"rep": rep, "arm": arm, "rc": rc}), flush=True)
            if rc and rep == 0:
                return 1
            (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str))
    if args.profile:
        for arm, src in arms:
            for kind in ("torch", "sample"):
                run_dir = out / f"profile-{kind}-{arm}"
                rc, t0, t1 = _run_child(args, src, run_dir, profile=kind)
                row = {"profile": kind, "arm": arm, "rc": rc, "t0": t0, "t1": t1,
                       "power": window(t0, t1)}
                if kind == "sample":
                    row["stacks"] = _summarise_stacks(run_dir / "sample.collapsed")
                runs.append(row)
                failed = failed or bool(rc)
                (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str))
    if sampler is not None:
        sampler.stop()

    ref_name = arms[0][0]
    ref = json.loads((out / f"rep0-{ref_name}" / "digests.json").read_text())
    verdict = {}
    for arm, _src in arms[1:]:
        got = json.loads((out / f"rep0-{arm}" / "digests.json").read_text())
        mismatches = []
        for section in ("units", "writes", "tcq"):
            for key in sorted(set(ref[section]) | set(got[section])):
                if ref[section].get(key) != got[section].get(key):
                    mismatches.append(f"{section}:{key}")
        verdict[arm] = {"compared": {s: len(ref[s]) for s in ("units", "writes", "tcq")},
                        "mismatch_count": len(mismatches), "mismatches": mismatches[:50]}
    table = {}
    for arm, _src in arms:
        rows = [r["summary"] for r in runs if r.get("arm") == arm and "summary" in r]
        if not rows:
            continue
        med = lambda key: statistics.median(r[key] for r in rows)  # noqa: E731
        table[arm] = {
            "n": len(rows),
            "container_parse_seconds_total_median": med("container_parse_seconds_total"),
            "parse_unit_seconds_total_median": med("parse_unit_seconds_total"),
            "write_seconds_total_median": med("write_seconds_total"),
            "peak_alloc_bytes_max": max(r["peak_alloc_bytes_max"] for r in rows),
            "unit_storage_bytes_max": max(r["unit_storage_bytes_max"] for r in rows),
            "step_plane_storage_bytes_total": rows[0]["step_plane_storage_bytes_total"],
            "tcq_median_seconds": {k: statistics.median(r["tcq"][k]["median_seconds"] for r in rows)
                                   for k in rows[0]["tcq"]},
            "power": [r.get("power") for r in runs if r.get("arm") == arm and "summary" in r],
        }
    meta["t_end"] = time.time()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report = {"failed_children": failed, "compare": verdict, "table": table}
    (out / "compare.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str), flush=True)
    bad = any(v["mismatch_count"] for v in verdict.values())
    return 1 if (failed or bad) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("parent")
    p.add_argument("--arms", required=True, help="name=src[,name=src]; the first is the reference")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    c = sub.add_parser("child")
    c.add_argument("--src", required=True)
    c.add_argument("--profile", default="none", choices=("none", "torch", "sample"))
    for q in (p, c):
        q.add_argument("--data", required=True)
        q.add_argument("--layers", default="3")
        q.add_argument("--experts", type=int, default=8)
        q.add_argument("--device", default="cpu")
        q.add_argument("--digest-units", type=int, default=6)
        q.add_argument("--write-repeats", type=int, default=3)
        q.add_argument("--tcq-weight", default="")
        q.add_argument("--tcq-rows", type=int, default=64)
        q.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return parent(args) if args.mode == "parent" else child(args)


if __name__ == "__main__":
    raise SystemExit(main())
