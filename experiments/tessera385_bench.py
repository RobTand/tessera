"""tessera#385: the batched LDLQ encoder on the campaign's own units.

Encodes the units ``tessera385_dump_units.py`` (PrismaQuant side) captured --
32 LFM layer-18 expert ``w1`` [1792, 2048] with their routed Hessians, and the
[7168, 2048] dense unit the #283 profile ran -- through the exporter's own
``ActivationSource.for_unit`` -> ``encode_linear`` path (the unbatched arm)
and through ``encode_linears`` at each batch size (the batched arms), at the
campaign's rungs.  Every arm records wall time, params/s, epoch bounds (so the
Netdata power series can be read for exactly that window) and the sha256 of
every unit's blob, so the identity claim is a receipt, not a belief.

``--arms`` selects the implementation for separate base/head ABBA processes.
Warmup phases precede measured repeats. ``--profile`` captures one expert by
default (at most eight) per selected arm with CUDA activity only; CPU-event
tables can exhaust host memory on the production coset trellis. Profiles are
separate from timing.

``--l2-sweep`` measures the window body's one wide call at explicit
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
CHECKPOINT = None


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(package):
    # Actual loaded source, including dirty/untracked code and packaged data;
    # Git HEAD alone identifies neither a PB snapshot nor an installed wheel.
    root = Path(package.__file__).resolve().parent
    files = {str(p.relative_to(root)): _file_sha(p)
             for p in sorted(root.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts
             and p.suffix not in (".pyc", ".pyo")}
    return {"root": str(root), "files_sha256": files,
            "tree_sha256": _sha(json.dumps(files, sort_keys=True).encode()),
            "harness_sha256": _file_sha(__file__)}


def _memory():
    import torch
    # Linux's current RSS is distinct from getrusage's lifetime high-water RSS.
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": rss,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved()}


class phase:
    def __init__(self, name, **extra):
        extra.setdefault("kind", "preparation")
        self.name, self.extra = name, extra

    def __enter__(self):
        import torch
        if CHECKPOINT:
            CHECKPOINT(active={"phase": self.name, "start_epoch": time.time(), **self.extra})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.memory0 = _memory()
        self.t0 = time.time()
        self.monotonic0 = time.perf_counter()
        self.ru0 = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, *exc):
        import torch
        if exc[0] is None:
            torch.cuda.synchronize()
        t1 = time.time()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime - self.ru0.ru_utime) + (ru1.ru_stime - self.ru0.ru_stime)
        rec = {"phase": self.name, "start_epoch": self.t0, "end_epoch": t1,
               "wall_s": time.perf_counter() - self.monotonic0, "cpu_s": cpu,
               "cpu_user_s": ru1.ru_utime - self.ru0.ru_utime,
               "cpu_system_s": ru1.ru_stime - self.ru0.ru_stime,
               "memory_start": self.memory0, "memory_end": _memory(),
               "status": "failed" if exc[0] else "complete",
               "error": str(exc[1]) if exc[0] else None, **self.extra}
        self.record = rec
        PHASES.append(rec)
        if CHECKPOINT:
            CHECKPOINT()
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
        _atomic_json(Path(args.out_dir, "l2_sweep.json"), out)
        print(f"[l2] budget {rec['budget']} width {rec['width']} batches {rec['batches']} "
              f"steady {rec['steady_s']:.3f}s sha {rec['sha'][:12]}", flush=True)
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
    ap.add_argument("--warmup-repeats", type=int, default=1,
                    help="full workload warmups per arm; recorded separately from measurements")
    ap.add_argument("--arms", choices=("auto", "unbatched", "batched", "both"), default="auto",
                    help="auto preserves available arms; explicit batched/both requires encode_linears")
    ap.add_argument("--run-label", default="", help="external ABBA arm/position, recorded verbatim")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-batch", type=int, default=32)
    ap.add_argument("--profile-experts", type=int, default=1,
                    help="1..8 units per selected arm; use 8 with --profile-batch 8 for real B8")
    ap.add_argument("--profile-input-columns", type=int, default=0,
                    help="profile-only leading input columns; slices weights/Hessians before preparation")
    ap.add_argument("--skip-unbatched", action="store_true")
    ap.add_argument("--only-profile", action="store_true", help="skip the timed arms; profile only")
    ap.add_argument("--profile-cpu", action="store_true",
                    help="retired: fails early because production CPU-event capture exhausted a 40 GB action")
    ap.add_argument("--l2-sweep", action="store_true")
    ap.add_argument("--l2-budgets", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 4 << 20, 8 << 20, 16 << 20, 32 << 20])
    ap.add_argument("--l2-rows", type=int, default=1792)
    ap.add_argument("--l2-cols", type=int, default=1024)
    args = ap.parse_args()
    if args.expert_count < 1 or args.dense_copies < 1 or args.repeats < 1 or args.warmup_repeats < 0:
        ap.error("expert-count, dense-copies and repeats must be positive; warmup-repeats must be nonnegative")
    if args.only_profile and not args.profile:
        ap.error("--only-profile requires --profile")
    if not 1 <= args.profile_experts <= 8 or args.profile_batch < 1:
        ap.error("profile-experts must be between 1 and 8; profile-batch must be positive")
    if args.profile_input_columns and (not args.only_profile or args.profile_input_columns < 32
                                       or args.profile_input_columns % 32):
        ap.error("profile-input-columns requires only-profile and a positive multiple of 32")
    if args.profile_cpu:
        ap.error("--profile-cpu is unsafe for production experts (unbounded CPU event table); use CUDA-only --profile")
    if args.skip_unbatched and args.arms in ("unbatched", "both"):
        ap.error("--skip-unbatched conflicts with --arms unbatched/both")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    if not batch_sizes or min(batch_sizes) < 1:
        ap.error("batch-sizes must contain positive integers")
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
    if args.arms in ("batched", "both") and encode_linears is None:
        ap.error("selected arms require encode_linears, absent from this source snapshot")
    selected_arms = []
    if args.arms != "batched" and not args.skip_unbatched:
        selected_arms.append("unbatched")
    if args.arms != "unbatched" and encode_linears is not None:
        selected_arms.append("batched")
    if not selected_arms:
        ap.error("no available arms selected")

    env = {
        "hostname": os.uname().nodename, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0), "tessera_file": tessera.__file__,
        "tessera_git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, cwd=Path(tessera.__file__).parent).stdout.strip(),
        "argv": sys.argv, "started_epoch": time.time(),
        "has_encode_linears": encode_linears is not None,
        "window_l2_bytes": os.environ.get("TESSERA_WINDOW_L2_BYTES"),
        "source": _source_identity(tessera), "run_label": args.run_label,
        "config": vars(args), "selected_arms": selected_arms,
        "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version, "cuda": torch.version.cuda,
        "native_threads": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "status": "running",
    }
    results: list[dict] = []
    shas: dict[str, dict[str, str]] = {}

    def checkpoint(active=None):
        _atomic_json(out_dir / "results.json",
                     {"env": env, "phases": PHASES, "arms": results,
                      "blob_sha": shas, "active_phase": active})

    global CHECKPOINT
    CHECKPOINT = checkpoint
    checkpoint()
    print(f"[bench] source {env['source']['tree_sha256']} run {args.run_label!r}", flush=True)
    if args.l2_sweep:
        rc = l2_sweep(args)
        env["source_unchanged"] = _source_identity(tessera) == env["source"]
        env["status"] = "complete" if rc == 0 and env["source_unchanged"] else "failed"
        env["finished_epoch"] = time.time()
        _atomic_json(out_dir / "env.json", env)
        checkpoint()
        if not env["source_unchanged"]:
            raise SystemExit("source bytes changed during L2 sweep; receipt is invalid")
        return rc

    with phase("encoder_fixture_id") as p:
        fid = encoder_fixture_id().hex()
    env["encoder_fixture_id"] = fid
    checkpoint()
    print(f"[bench] encoder_fixture_id {fid}", flush=True)

    from safetensors.torch import load_file
    input_paths = [Path(args.units, n).resolve() for n in ("units.json", "units.safetensors")]
    env["inputs"] = {str(p): {"sha256": _file_sha(p), "size_bytes": p.stat().st_size}
                     for p in input_paths}
    checkpoint()
    with phase("load_inputs", kind="preparation"):
        meta = json.loads(input_paths[0].read_text())
        tensors = load_file(str(input_paths[1]))
    experts = meta["experts"][: args.expert_count]
    if len(experts) != args.expert_count:
        ap.error("expert-count exceeds captured expert population")
    dense = meta["dense_unit"]
    names_needed = experts + ([dense] if args.dense else [])
    device = "cuda"
    with phase("resident_weights", kind="preparation"):
        weights = {n: tensors[f"weight/{n}"].to(device) for n in names_needed}
    hessians = {n: tensors[f"hessian/{n}"] for n in names_needed}
    if args.profile_input_columns:
        cols = args.profile_input_columns
        if any(cols > weights[n].shape[1] for n in names_needed):
            ap.error("profile-input-columns exceeds a captured unit input dimension")
        weights = {n: w[:, :cols].contiguous() for n, w in weights.items()}
        hessians = {n: h[:cols, :cols].contiguous() for n, h in hessians.items()}
        env["profile_input_derivation"] = {
            "operation": "leading input columns and matching Hessian principal submatrix",
            "columns": cols, "scope": "profiler evidence only; not the full timing workload"}
    source = ActivationSource(hessians=hessians, provenance=dict(meta["identity"]))
    env["input_identity"] = meta["identity"]
    env["workload"] = {n: {"shape": list(weights[n].shape), "dtype": str(weights[n].dtype)}
                       for n in names_needed}
    checkpoint()

    def run_arm(tag, family, q256, names, batch, recipe, kwargs, repeat, kind="measured", workload="experts"):
        ws = [weights[n] for n in names]
        params = sum(w.numel() for w in ws)
        label = f"{family}@{q256}.{workload}.{kind}.{tag}.B{batch}.r{repeat}"
        with phase(label, family=family, q256=q256, batch=batch, units=len(names),
                   params=params, repeat=repeat, arm=tag, kind=kind, workload=workload) as p:
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
            if len(blobs) != len(names):
                raise RuntimeError(f"encoder returned {len(blobs)} blobs for {len(names)} units")
        rec = dict(p.record)
        rec["params_per_s"] = params / rec["wall_s"]
        rec["blob_sha"] = {n: _sha(b) for n, b in zip(names, blobs)}
        rec["blob_sha_ordered"] = [{"name": n, "sha256": _sha(b)} for n, b in zip(names, blobs)]
        results.append(rec)
        print(f"[bench] {label}: {rec['params_per_s'] / 1e6:.3f} Mparam/s", flush=True)
        # identity across arms, per unit
        for n, b in zip(names, blobs):
            key = f"{family}@{q256}"
            seen = shas.setdefault(key, {})
            if n in seen and seen[n] != _sha(b):
                rec["identity_status"] = "failed"
                checkpoint()
                raise SystemExit(f"IDENTITY BROKEN: {key} {n} {tag} B{batch}: {seen[n][:12]} != {_sha(b)[:12]}")
            seen.setdefault(n, _sha(b))
        rec["identity_status"] = "consistent_with_observed_blobs"
        checkpoint()
        return rec

    families = []
    for spec in args.families.split(","):
        fam, q = spec.split("@")
        families.append((fam, int(q)))

    for family, q256 in families:
        grid = _family(family)
        recipe = wire_recipe(grid, q256)
        with phase(f"{family}@{q256}.for_unit", family=family, q256=q256, kind="preparation"):
            kwargs = {n: source.for_unit(n, weights[n].shape[1], device,
                                         scale_plane=recipe.scale_plane, weight=weights[n])
                      for n in names_needed}
        timing_arms = ([("unbatched", 1)] if "unbatched" in selected_arms else [])
        if "batched" in selected_arms:
            timing_arms.extend(("batched", b) for b in batch_sizes)
        if not args.only_profile:
            for tag, b in timing_arms:
                for r in range(args.warmup_repeats):
                    run_arm(tag, family, q256, experts, b, recipe, kwargs, r, kind="warmup")
        for r in range(args.repeats if not args.only_profile else 0):
            for tag, b in timing_arms:
                run_arm(tag, family, q256, experts, b, recipe, kwargs, r)
        if args.dense and not args.only_profile:
            dense_arms = ([("unbatched", 1)] if "unbatched" in selected_arms else [])
            if "batched" in selected_arms:
                dense_arms.append(("batched", args.dense_copies))
            for kind, repeats in (("warmup", args.warmup_repeats), ("measured", args.repeats)):
                for r in range(repeats):
                    for tag, b in dense_arms:
                        run_arm(tag, family, q256, [dense] * b, b, recipe, kwargs, r,
                                kind=kind, workload="dense_copies" if b > 1 else "dense")

        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            subset = experts[: args.profile_experts]
            tables = {}
            arms = [("unbatched", 1)] if "unbatched" in selected_arms else []
            if "batched" in selected_arms:
                arms.append(("batched", args.profile_batch))
            activities = [ProfilerActivity.CUDA]
            for tag, b in arms:
                for r in range(args.warmup_repeats):
                    run_arm(tag, family, q256, subset, b, recipe, kwargs, r, kind="profile_warmup")
                with phase(f"{family}@{q256}.profile.{tag}.B{b}", family=family, q256=q256,
                           batch=b, effective_batch=min(b, len(subset)), units=len(subset),
                           arm=f"profile_{tag}", kind="profile", activities=["CUDA"],
                           includes_profiler_finalization=True):
                    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
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
                _atomic_json(out_dir / f"profile_{family}@{q256}.json", tables)
                del prof
                checkpoint()

    env["source_unchanged"] = _source_identity(tessera) == env["source"]
    env["inputs_unchanged"] = all(_file_sha(p) == env["inputs"][str(p)]["sha256"] for p in input_paths)
    env["finished_epoch"] = time.time()
    env["status"] = "complete" if env["source_unchanged"] and env["inputs_unchanged"] else "invalid_provenance"
    checkpoint()
    if env["status"] != "complete":
        raise SystemExit("source or input bytes changed during measurement; receipt is invalid")
    print(f"[bench] wrote {out_dir}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
