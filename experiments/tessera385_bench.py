"""tessera#385: the batched LDLQ encoder on the campaign's own units.

Encodes the units ``tessera385_dump_units.py`` (PrismaQuant side) captured --
32 LFM layer-18 expert ``w1`` [1792, 2048] with their routed Hessians, and the
[7168, 2048] dense unit the #283 profile ran -- through the exporter's own
``ActivationSource.for_unit`` -> ``encode_linear`` path (the unbatched arm)
and through ``encode_linears`` at each batch size (the batched arms), at the
campaign's rungs.  Every arm records wall time, params/s, epoch bounds (so the
Netdata power series can be read for exactly that window) and the sha256 of
every unit's blob, so the identity claim is a receipt, not a belief.

``--profile`` re-runs one unbatched and one batched arm under
``torch.profiler`` and writes the kernel table (count, total, mean per kernel
name; the cudaStreamSynchronize count) for both.

``--l2-sweep`` measures the window body's one wide call at four
``TESSERA_WINDOW_L2_BYTES`` budgets in subprocesses (the budget is read at
import), which is the measurement priority 3 of the issue asks for before any
kernel is built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PHASES: list[dict] = []


class phase:
    def __init__(self, name, **extra):
        self.name, self.extra = name, extra

    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self.t0 = time.time()
        self.ru0 = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.synchronize()
        t1 = time.time()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime - self.ru0.ru_utime) + (ru1.ru_stime - self.ru0.ru_stime)
        rec = {"phase": self.name, "start_epoch": self.t0, "end_epoch": t1,
               "wall_s": t1 - self.t0, "cpu_s": cpu, **self.extra}
        self.record = rec
        PHASES.append(rec)
        print(f"[bench] {self.name}: {rec['wall_s']:.3f}s wall, {cpu:.2f}s cpu", flush=True)
        return False


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _family(name: str):
    from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid
    return {
        "BF16_K1": BF16_GRID, "E4M3_K1": E4M3_GRID,
        "E2M1_K2": tuple_grid(E2M1_GRID, 2),
    }[name]


def l2_sweep(args) -> int:
    """One wide window call per budget, in a fresh process each."""
    child = r'''
import json, os, sys, time, torch
from tessera.alphabet import BF16_GRID
from tessera.encode import window_table, grid_vector_table, viterbi_window
from tessera.window_viterbi import _layout, _l2_budget
rows, cols, L, R = %d, %d, 14, 7
g = torch.Generator().manual_seed(0)
t = torch.randn(rows, cols, generator=g).cuda()
w = (torch.rand(rows, cols, generator=g).cuda() + 0.5)
codes = window_table(BF16_GRID, L, sigma=1.0, seed=0, half=16, device="cuda")
vec = grid_vector_table(BF16_GRID, "cuda")[codes.long()]
times = []
for i in range(4):
    torch.cuda.synchronize(); t0 = time.time()
    s, _ = viterbi_window(t, vec, L, R, weights=w)
    torch.cuda.synchronize(); times.append(time.time() - t0)
nmax, width, descs = _layout(t.device, 1 << L, cols, 512)
print(json.dumps({"budget": _l2_budget(t.device), "width": width, "batches": len(descs),
                  "times_s": times, "steady_s": min(times[2:]),
                  "sha": __import__("hashlib").sha256(s.cpu().numpy().tobytes()).hexdigest()}))
''' % (args.l2_rows, args.l2_cols)
    out = []
    for budget in args.l2_budgets:
        env = dict(os.environ)
        if budget:
            env["TESSERA_WINDOW_L2_BYTES"] = str(budget)
        else:
            env.pop("TESSERA_WINDOW_L2_BYTES", None)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr, flush=True)
            return 1
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        rec["requested"] = budget
        rec["start_epoch"], rec["end_epoch"] = t0, time.time()
        out.append(rec)
        print(f"[l2] budget {rec['budget']} width {rec['width']} batches {rec['batches']} "
              f"steady {rec['steady_s']:.3f}s sha {rec['sha'][:12]}", flush=True)
    Path(args.out_dir, "l2_sweep.json").write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="/mnt/shared/tessera-measurements/tessera385-2026-09-06/units")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default="BF16_K1@1792,E2M1_K2@896")
    ap.add_argument("--batch-sizes", default="1,8,16,32")
    ap.add_argument("--expert-count", type=int, default=32)
    ap.add_argument("--dense", action="store_true", help="also the [7168, 2048] unit, B=1 and B=4 copies")
    ap.add_argument("--dense-copies", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-batch", type=int, default=32)
    ap.add_argument("--profile-experts", type=int, default=8,
                    help="units under the profiler (both arms), to bound the trace")
    ap.add_argument("--skip-unbatched", action="store_true")
    ap.add_argument("--only-profile", action="store_true", help="skip the timed arms; profile only")
    ap.add_argument("--profile-cpu", action="store_true",
                    help="record CPU-side events too (the default is CUDA activity only: the "
                         "coset trellis emits millions of aten ops a unit and the CPU event "
                         "table OOM-killed a 40 GB action)")
    ap.add_argument("--l2-sweep", action="store_true")
    ap.add_argument("--l2-budgets", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 4 << 20, 8 << 20, 16 << 20, 32 << 20])
    ap.add_argument("--l2-rows", type=int, default=1792)
    ap.add_argument("--l2-cols", type=int, default=1024)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import tessera
    from tessera.export import ActivationSource, encode_linear, wire_recipe
    from tessera.encoder_identity import encoder_fixture_id
    try:
        from tessera.export import encode_linears
    except ImportError:
        encode_linears = None

    env = {
        "hostname": os.uname().nodename, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0), "tessera_file": tessera.__file__,
        "tessera_git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, cwd=Path(tessera.__file__).parent).stdout.strip(),
        "argv": sys.argv, "started_epoch": time.time(),
        "has_encode_linears": encode_linears is not None,
        "window_l2_bytes": os.environ.get("TESSERA_WINDOW_L2_BYTES"),
    }
    print(f"[bench] env {json.dumps(env, indent=1)}", flush=True)
    if args.l2_sweep:
        rc = l2_sweep(args)
        Path(out_dir, "env.json").write_text(json.dumps(env, indent=1))
        return rc

    with phase("encoder_fixture_id") as p:
        fid = encoder_fixture_id().hex()
    env["encoder_fixture_id"] = fid
    print(f"[bench] encoder_fixture_id {fid}", flush=True)

    from safetensors.torch import load_file
    meta = json.loads(Path(args.units, "units.json").read_text())
    tensors = load_file(str(Path(args.units, "units.safetensors")))
    experts = meta["experts"][: args.expert_count]
    dense = meta["dense_unit"]
    device = "cuda"
    weights = {n: tensors[f"weight/{n}"].to(device) for n in experts + [dense]}
    hessians = {n: tensors[f"hessian/{n}"] for n in experts + [dense]}
    source = ActivationSource(hessians=hessians, provenance=dict(meta["identity"]))

    results: list[dict] = []
    shas: dict[str, dict[str, str]] = {}

    def checkpoint():
        Path(out_dir, "results.json").write_text(json.dumps(
            {"env": env, "phases": PHASES, "arms": results, "blob_sha": shas}, indent=1, default=str))

    def run_arm(tag, family, q256, names, batch, recipe, kwargs, repeat):
        ws = [weights[n] for n in names]
        params = sum(w.numel() for w in ws)
        label = f"{family}@{q256}.{tag}.B{batch}.r{repeat}"
        with phase(label, family=family, q256=q256, batch=batch, units=len(names),
                   params=params, repeat=repeat, arm=tag) as p:
            if batch == 1 and tag == "unbatched":
                blobs = [encode_linear(w, grid=_family(family), q256=q256, name=n,
                                       **kwargs[n]).blob for w, n in zip(ws, names)]
            else:
                blobs = []
                for i in range(0, len(names), batch):
                    chunk = names[i:i + batch]
                    out = encode_linears([weights[n] for n in chunk], grid=_family(family),
                                         q256=q256, names=chunk,
                                         per_unit=[kwargs[n] for n in chunk])
                    blobs.extend(e.blob for e in out)
        rec = dict(p.record)
        rec["params_per_s"] = params / rec["wall_s"]
        rec["blob_sha"] = {n: _sha(b) for n, b in zip(names, blobs)}
        results.append(rec)
        print(f"[bench] {label}: {rec['params_per_s'] / 1e6:.3f} Mparam/s", flush=True)
        checkpoint()
        # identity across arms, per unit
        for n, b in zip(names, blobs):
            key = f"{family}@{q256}"
            seen = shas.setdefault(key, {})
            if n in seen and seen[n] != _sha(b):
                raise SystemExit(f"IDENTITY BROKEN: {key} {n} {tag} B{batch}: {seen[n][:12]} != {_sha(b)[:12]}")
            seen.setdefault(n, _sha(b))
        return rec

    families = []
    for spec in args.families.split(","):
        fam, q = spec.split("@")
        families.append((fam, int(q)))
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    for family, q256 in families:
        grid = _family(family)
        recipe = wire_recipe(grid, q256)
        with phase(f"{family}@{q256}.for_unit", family=family, q256=q256):
            kwargs = {n: source.for_unit(n, weights[n].shape[1], device,
                                         scale_plane=recipe.scale_plane, weight=weights[n])
                      for n in experts + [dense]}
        for r in range(args.repeats if not args.only_profile else 0):
            if not args.skip_unbatched:
                run_arm("unbatched", family, q256, experts, 1, recipe, kwargs, r)
            if encode_linears is not None:
                for b in batch_sizes:
                    run_arm("batched", family, q256, experts, b, recipe, kwargs, r)
        if args.dense and not args.only_profile:
            for r in range(args.repeats):
                run_arm("unbatched", family, q256, [dense], 1, recipe, kwargs, r)
                if encode_linears is not None and args.dense_copies > 1:
                    # copies of one unit: each is encoded independently, so
                    # the throughput is real even though the bytes repeat
                    names = [dense] * args.dense_copies
                    ws = [weights[dense]] * args.dense_copies
                    params = sum(w.numel() for w in ws)
                    label = f"{family}@{q256}.batched.dense_x{args.dense_copies}.r{r}"
                    with phase(label, family=family, q256=q256, batch=args.dense_copies,
                               units=args.dense_copies, params=params, repeat=r,
                               arm="batched_dense_copies") as p:
                        out = encode_linears(ws, grid=grid, q256=q256, names=names,
                                             per_unit=[kwargs[dense]] * args.dense_copies)
                    rec = dict(p.record)
                    rec["params_per_s"] = params / rec["wall_s"]
                    rec["blob_sha"] = {f"{dense}#{i}": _sha(e.blob) for i, e in enumerate(out)}
                    want = shas[f"{family}@{q256}"][dense]
                    if any(_sha(e.blob) != want for e in out):
                        raise SystemExit("IDENTITY BROKEN on the dense copies")
                    results.append(rec)
                    print(f"[bench] {label}: {rec['params_per_s'] / 1e6:.3f} Mparam/s", flush=True)
                    checkpoint()

        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            subset = experts[: args.profile_experts]
            tables = {}
            arms = [("unbatched", 1)]
            if encode_linears is not None:
                arms.append(("batched", args.profile_batch))
            activities = [ProfilerActivity.CUDA] + ([ProfilerActivity.CPU] if args.profile_cpu else [])
            for tag, b in arms:
                with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
                    with phase(f"{family}@{q256}.profile.{tag}.B{b}", family=family, q256=q256,
                               batch=b, units=len(subset), arm=f"profile_{tag}"):
                        if tag == "unbatched":
                            for n in subset:
                                encode_linear(weights[n], grid=grid, q256=q256, name=n, **kwargs[n])
                        else:
                            for i in range(0, len(subset), b):
                                chunk = subset[i:i + b]
                                encode_linears([weights[n] for n in chunk], grid=grid, q256=q256,
                                               names=chunk, per_unit=[kwargs[n] for n in chunk])
                rows = []
                for ev in prof.key_averages():
                    rows.append({"name": ev.key, "count": int(ev.count),
                                 "cuda_total_us": float(getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0.0)) or 0.0),
                                 "cpu_total_us": float(ev.cpu_time_total),
                                 "self_cuda_us": float(getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0.0)) or 0.0)})
                rows.sort(key=lambda r: -r["self_cuda_us"])
                tables[f"{tag}.B{b}"] = rows
                top = [r for r in rows if r["self_cuda_us"] > 0][:12]
                print(f"[profile] {family}@{q256} {tag} B{b}: top device kernels", flush=True)
                for r in top:
                    print(f"    {r['count']:8d}  {r['self_cuda_us'] / 1e6:8.3f}s  "
                          f"{r['self_cuda_us'] / max(1, r['count']):8.1f}us  {r['name'][:80]}", flush=True)
                sync = [r for r in rows if "Synchronize" in r["name"] or "cudaMemcpy" in r["name"]]
                for r in sync:
                    print(f"    sync: {r['count']:8d}  {r['cpu_total_us'] / 1e6:8.3f}s cpu  {r['name']}", flush=True)
            Path(out_dir, f"profile_{family}@{q256}.json").write_text(json.dumps(tables, indent=1))
            checkpoint()

    env["finished_epoch"] = time.time()
    Path(out_dir, "results.json").write_text(json.dumps(
        {"env": env, "phases": PHASES, "arms": results, "blob_sha": shas}, indent=1, default=str))
    print(f"[bench] wrote {out_dir}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
