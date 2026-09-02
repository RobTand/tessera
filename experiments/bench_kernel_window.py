"""Time the fused window decoder against the paths it would replace.

Five arms, each a separate `--arm` so a contended box can be re-measured one
piece at a time:

``bandwidth``  the box's own achievable copy rate, measured, so every GB/s
               below has a measured denominator rather than the 273 GB/s
               spec number.
``decode``     per-unit tile decode: the pure-torch reader (eager and
               ``torch.compile``) against the fused kernel, on the real reach
               checkpoint, both L2-hot on one unit and DRAM-cold by rotating
               through every unit of the checkpoint.
``gemv``       the decode-phase question: ``x @ W.T`` at M in 1..8 for every
               Linear shape of Qwen3-0.6B and Qwen3-4B (and an assumed 27B
               list), fused against BF16 cuBLAS and against the resident-FP8
               path the lane serves today (per-token dynamic quant plus
               ``torch._scaled_mm``).  Cold by rotation.
``ablate``     where the GEMV's time goes, by deleting one term at a time
               from a copy of the kernel: the wire read, the value gather,
               the activation load and FMA.
``profile``    a ``torch.profiler`` trace of one pass of each kernel.

Every arm samples ``nvidia-smi`` power alongside its timed loop and prints the
mean and max next to the number, because on GB10 utilisation is
non-diagnostic and power against the ~140 W envelope is what says whether the
box was loaded (CLAUDE.md principle 15).

Usage
-----
    export PYTHONPATH=$PWD/src TMPDIR=/home/rob/tmp \
           TRITON_CACHE_DIR=/home/rob/.triton-cache
    python experiments/bench_kernel_window.py --arm all \
        --out /home/rob/tessera-runs/kernel-window
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E4M3_GRID              # noqa: E402
from tessera import kernel_window as kw             # noqa: E402

REACH = Path("/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook")

#: The pure-torch window reader as it stood when this kernel was written,
#: snapshotted by path so a concurrent edit to the serving tree cannot move
#: the baseline under the measurement.
TORCH_BASELINE = Path("/home/rob/tmp/kernel-window/window_torch_baseline.py")

#: Rotate over at least this many wire bytes per timed arm so that a call
#: reads DRAM rather than L2.  256 MB is comfortably past any GB10 L2.
COLD_BYTES = 256e6

#: `(name, rows, cols, per-layer count)` per model, one entry per *unit* --
#: the granularity the lane issues a GEMV at, since q/k/v are three members of
#: one fused module and each member is its own Tessera unit.
MODELS = {
    "Qwen3-0.6B": (28, [
        ("q_proj", 2048, 1024), ("k_proj", 1024, 1024), ("v_proj", 1024, 1024),
        ("o_proj", 1024, 2048), ("gate_proj", 3072, 1024), ("up_proj", 3072, 1024),
        ("down_proj", 1024, 3072),
    ]),
    "Qwen3-4B": (36, [
        ("q_proj", 4096, 2560), ("k_proj", 1024, 2560), ("v_proj", 1024, 2560),
        ("o_proj", 2560, 4096), ("gate_proj", 9728, 2560), ("up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
    ]),
    # No 27B config is on this box; this is an ASSUMED dense 27B list (hidden
    # 5120, intermediate 17408, 64 layers, 40 q heads / 8 kv heads at head_dim
    # 128).  The projection is bytes over bandwidth, so it rescales with the
    # true shapes without re-measuring.
    "27B-assumed": (64, [
        ("q_proj", 5120, 5120), ("k_proj", 1024, 5120), ("v_proj", 1024, 5120),
        ("o_proj", 5120, 5120), ("gate_proj", 17408, 5120), ("up_proj", 17408, 5120),
        ("down_proj", 5120, 17408),
    ]),
}


# --- power ------------------------------------------------------------------


class Power:
    """Samples ``nvidia-smi`` power in a thread for the life of the process.

    ``window()`` returns the mean/max/min watts over a wall-clock interval, so
    each timed loop below carries the power the box drew while it ran.
    """

    def __init__(self, period: float = 0.12):
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._period = period
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self._samples.append((time.time(), float(out.splitlines()[0])))
            except Exception:
                pass
            self._stop.wait(self._period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict:
        w = [p for t, p in self._samples if t0 <= t <= t1]
        if not w:
            return {"mean_w": None, "max_w": None, "n": 0}
        return {"mean_w": round(sum(w) / len(w), 1), "max_w": round(max(w), 1),
                "min_w": round(min(w), 1), "n": len(w)}


POWER: Power | None = None


def bench(fn, n: int = 200, warm: int = 20, seconds: float = 1.5) -> dict:
    """Milliseconds per call by CUDA events, with the power drawn while timing.

    The loop runs at least ``seconds`` of wall clock so the power sampler has
    something to average; a fast kernel just gets more iterations.
    """
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.time()
    total_ms, total_n = 0.0, 0
    while True:
        start.record()
        for _ in range(n):
            fn()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
        total_n += n
        if time.time() - t0 >= seconds:
            break
    t1 = time.time()
    result = {"us": round(total_ms / total_n * 1000, 3), "iters": total_n}
    if POWER is not None:
        result.update(POWER.window(t0, t1))
    return result


# --- units ------------------------------------------------------------------


def real_units(limit=None):
    """Every unit of the reach checkpoint, prepared on the device."""
    from safetensors import safe_open

    from tessera.fused import parse_fused
    from tessera.unit_artifact import parse_unit_artifact

    out = []
    with safe_open(str(REACH / "model.safetensors"), framework="pt") as handle:
        keys = sorted(k for k in handle.keys() if k.endswith(".wire_bytes"))
        for key in keys:
            blob = bytes(handle.get_tensor(key).numpy().tobytes())
            for member in parse_fused(blob):
                parsed = parse_unit_artifact(member.blob, device="cuda")
                name = f"{key[: -len('.wire_bytes')]}:{member.name}"
                out.append((name, parsed))
                if limit and len(out) >= limit:
                    return out
    return out


def synth(rows: int, cols: int, rate: int = 4, window_bits: int = 14, seed: int = 0):
    """A window unit of the given shape from random bits.

    A real unit's states are the Viterbi's, but the kernel's access pattern
    depends only on the *distribution* of states over the table, and a
    trellis-coded stream's states are close to uniform by construction (that
    is what the code is for).  Random bits give exactly uniform states and the
    same gather footprint, at no encoding cost -- which is why the 4B and 27B
    shapes below are synthesised rather than Viterbi-encoded.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8,
                         device="cuda", generator=g)
    codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8,
                          device="cuda", generator=g)
    scale = torch.rand(rows, device="cuda", generator=g) * 0.01 + 0.001
    return kw.prepare_window_unit(body, (rate,) * cols, window_bits, codes,
                                  E4M3_GRID, scale, device="cuda")


class Rotor:
    """Cycles a list so consecutive calls touch different memory."""

    def __init__(self, items):
        self.items = list(items)
        self.i = 0

    def next(self):
        item = self.items[self.i]
        self.i = (self.i + 1) % len(self.items)
        return item


def cold_units(rows, cols, target=COLD_BYTES, **kw_):
    """Enough distinct copies of one shape's wire to overflow L2."""
    first = synth(rows, cols, seed=0, **kw_)
    n = max(2, int(target // max(1, first.wire_bytes)) + 1)
    units = [first]
    for i in range(1, n):
        u = kw.PreparedWindowUnit(
            first.plane_words.clone(), first.wire_bytes, first.offsets, first.rates,
            first.initial, first.code_table, first.value_table, first.row_scale,
            first.rows, first.cols, first.window_bits, first.max_rate,
        )
        units.append(u)
    return Rotor(units)


def cold_tensors(make, nbytes, target=COLD_BYTES):
    n = max(2, int(target // max(1, nbytes)) + 1)
    return Rotor([make() for _ in range(n)])


# --- arms -------------------------------------------------------------------


def arm_bandwidth(out: dict):
    """The measured copy rate, both directions of a 1 GiB buffer."""
    n = 1 << 30
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    r = bench(lambda: dst.copy_(src), n=5, warm=3, seconds=2.0)
    r["GB_per_s_rw"] = round(2 * n / (r["us"] * 1e-6) / 1e9, 1)
    out["bandwidth"] = r
    print(f"copy_ 1 GiB: {r['us']:.1f} us  -> {r['GB_per_s_rw']} GB/s read+write"
          f"  ({r.get('mean_w')} W mean, {r.get('max_w')} W max)")
    del src, dst
    torch.cuda.empty_cache()


def _torch_baseline(parsed):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_wtb", TORCH_BASELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    unit = parsed.unit
    native = torch.tensor(parsed.grid.native, dtype=torch.uint8, device="cuda")
    return mod.prepare_window(unit.body_bits.cuda(), unit.rates, unit.window_bits,
                              unit.window_codes.cuda(), "cuda", code_map=native)


def arm_decode(out: dict, quick: bool = False):
    units = real_units(limit=8 if quick else None)
    print(f"{len(units)} real units")
    rows = []

    # -- L2-hot, one unit at a time, three representative shapes plus the
    #    torch reader on the same unit.
    seen = set()
    for name, parsed in units:
        shape = tuple(parsed.unit.body_bits.shape)
        if shape in seen:
            continue
        seen.add(shape)
        p = kw.prepare_from_parsed(parsed, device="cuda")
        moved = p.wire_bytes + p.rows * p.cols
        fused = bench(lambda p=p: p.decode(), n=50, seconds=1.5)
        fused["GB_per_s"] = round(moved / (fused["us"] * 1e-6) / 1e9, 1)
        base = _torch_baseline(parsed)
        eager = bench(lambda base=base: base.decode(), n=5, warm=2, seconds=1.5)
        compiled_fn = torch.compile(lambda b: b.decode(), dynamic=False)
        comp = bench(lambda base=base: compiled_fn(base), n=10, warm=3, seconds=1.5)
        rows.append({
            "unit": name, "rows": p.rows, "cols": p.cols,
            "wire_bytes": p.wire_bytes, "tile_bytes": p.rows * p.cols,
            "fused_hot": fused, "torch_eager": eager, "torch_compiled": comp,
            "speedup_vs_eager": round(eager["us"] / fused["us"], 1),
            "speedup_vs_compiled": round(comp["us"] / fused["us"], 1),
        })
        print(f"  {p.rows}x{p.cols}: fused {fused['us']:.2f} us "
              f"({fused['GB_per_s']} GB/s, {fused.get('mean_w')} W) | "
              f"torch eager {eager['us']:.1f} | compiled {comp['us']:.1f} us"
              f"  -> {rows[-1]['speedup_vs_eager']}x / "
              f"{rows[-1]['speedup_vs_compiled']}x")
        del base, compiled_fn

    # -- DRAM-cold: rotate through every unit of the checkpoint.
    prepared = [kw.prepare_from_parsed(p, device="cuda") for _, p in units]
    total_wire = sum(p.wire_bytes for p in prepared)
    total_tile = sum(p.rows * p.cols for p in prepared)
    rot = Rotor(prepared)
    cold = bench(lambda: rot.next().decode(), n=len(prepared), seconds=3.0)
    per_unit_moved = (total_wire + total_tile) / len(prepared)
    cold["GB_per_s"] = round(per_unit_moved / (cold["us"] * 1e-6) / 1e9, 1)
    cold["units"] = len(prepared)
    cold["plane_MB"] = round(total_wire / 1e6, 1)
    out["decode"] = {"hot": rows, "cold_rotation": cold}
    print(f"  cold rotation over {len(prepared)} units ({total_wire/1e6:.0f} MB of "
          f"plane): {cold['us']:.2f} us/unit, {cold['GB_per_s']} GB/s, "
          f"{cold.get('mean_w')} W mean / {cold.get('max_w')} W max")
    print(f"  whole-checkpoint decode: {cold['us'] * len(prepared) / 1000:.2f} ms")


def _fp8_pair(rows, cols):
    tile = torch.randint(0, 200, (rows, cols), dtype=torch.uint8,
                         device="cuda").view(torch.float8_e4m3fn)
    scale = torch.rand(rows, 1, device="cuda") * 0.01 + 0.001
    return tile, scale


def arm_gemv(out: dict, models=("Qwen3-0.6B", "Qwen3-4B", "27B-assumed"),
             batches=(1, 2, 4, 8), quick: bool = False):
    shapes = {}
    for model in models:
        for _name, r, c in MODELS[model][1]:
            shapes.setdefault((r, c), []).append(model)
    table = []
    for (r, c), used_by in sorted(shapes.items()):
        rot = cold_units(r, c)
        one = rot.items[0]
        wire = one.wire_bytes
        bf16 = cold_tensors(
            lambda: torch.randn(r, c, dtype=torch.bfloat16, device="cuda"), r * c * 2)
        fp8 = cold_tensors(lambda: _fp8_pair(r, c), r * c)
        entry = {"rows": r, "cols": c, "used_by": used_by,
                 "wire_bytes": wire, "bf16_bytes": r * c * 2, "fp8_bytes": r * c,
                 "replicas": len(rot.items), "M": {}}
        for m in (batches if not quick else (1,)):
            x = torch.randn(m, c, dtype=torch.bfloat16, device="cuda")

            fused = bench(lambda: rot.next().gemv(x), n=64, seconds=1.5)
            fused["GB_per_s"] = round(wire / (fused["us"] * 1e-6) / 1e9, 1)

            def bf16_call():
                w = bf16.next()
                return torch.nn.functional.linear(x, w)
            b = bench(bf16_call, n=64, seconds=1.5)
            b["GB_per_s"] = round(r * c * 2 / (b["us"] * 1e-6) / 1e9, 1)

            def fp8_pair_call():
                w, ws = fp8.next()
                xq, xs = kw._fp8_per_token(x)
                return torch._scaled_mm(xq, w.t(), scale_a=xs, scale_b=ws.t(),
                                        out_dtype=torch.bfloat16)

            def fp8_mm_only(w=None, ws=None):
                w, ws = fp8.next()
                return torch._scaled_mm(xq0, w.t(), scale_a=xs0, scale_b=ws.t(),
                                        out_dtype=torch.bfloat16)

            xq0, xs0 = kw._fp8_per_token(x)
            try:
                fp = bench(fp8_pair_call, n=64, seconds=1.5)
                fp["GB_per_s"] = round(r * c / (fp["us"] * 1e-6) / 1e9, 1)
                fmm = bench(fp8_mm_only, n=64, seconds=1.5)
            except Exception as exc:                      # pragma: no cover
                fp = {"us": None, "error": str(exc)[:120]}
                fmm = {"us": None}
            entry["M"][m] = {"fused_gemv": fused, "bf16_linear": b,
                             "fp8_quant_plus_mm": fp, "fp8_mm_only": fmm}
            print(f"  {r}x{c} M={m}: fused {fused['us']:.2f} us "
                  f"({fused['GB_per_s']} GB/s of wire, {fused.get('mean_w')} W) | "
                  f"bf16 {b['us']:.2f} | fp8 quant+mm "
                  f"{fp['us'] if fp['us'] is None else round(fp['us'], 2)} | "
                  f"fp8 mm only "
                  f"{fmm['us'] if fmm.get('us') is None else round(fmm['us'], 2)}")
        table.append(entry)
        del rot, bf16, fp8
        torch.cuda.empty_cache()

    # per-token totals over each model's Linears
    lookup = {(e["rows"], e["cols"]): e for e in table}
    totals = {}
    for model, (layers, units) in MODELS.items():
        if model not in models:
            continue
        per = {}
        for m in (batches if not quick else (1,)):
            f = sum(lookup[(r, c)]["M"][m]["fused_gemv"]["us"] for _n, r, c in units)
            b = sum(lookup[(r, c)]["M"][m]["bf16_linear"]["us"] for _n, r, c in units)
            p8 = [lookup[(r, c)]["M"][m]["fp8_quant_plus_mm"]["us"] for _n, r, c in units]
            per[m] = {
                "fused_us_per_token": round(f * layers, 1),
                "bf16_us_per_token": round(b * layers, 1),
                "fp8_resident_us_per_token": (None if any(v is None for v in p8)
                                              else round(sum(p8) * layers, 1)),
                "launches_per_token": len(units) * layers,
            }
        totals[model] = {"layers": layers, "per_M": per}
        print(f"  {model}: " + ", ".join(
            f"M={m} fused {v['fused_us_per_token']} us/token vs bf16 "
            f"{v['bf16_us_per_token']} vs fp8 {v['fp8_resident_us_per_token']}"
            for m, v in per.items()))
    out["gemv"] = {"shapes": table, "models": totals}


def arm_ablate(out: dict):
    """Where the GEMV's time goes, by deleting one term at a time."""
    import triton
    import triton.language as tl
    from tessera.kernel_window import _span_of, _state_of

    @triton.jit
    def _abl(x_ptr, words_ptr, offset_ptr, rate_ptr, init_ptr, value_ptr, scale_ptr,
             out_ptr, rows, cols, m,
             window: tl.constexpr, LANES: tl.constexpr, VEC: tl.constexpr,
             MBLK: tl.constexpr, KB: tl.constexpr, SPLIT: tl.constexpr,
             MODE: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        lane = tl.arange(0, LANES)
        v = tl.arange(0, VEC)
        mi = tl.arange(0, MBLK)
        j = tl.arange(0, KB)
        live_m = mi < m
        base = pid_n * (LANES * VEC) + lane * VEC
        n = base[:, None] + v[None, :]
        live = n < rows
        scale = tl.load(scale_ptr + n, mask=live, other=0.0)
        acc = tl.zeros((MBLK, LANES, VEC), dtype=tl.float32)
        chunk = tl.cdiv(cols, SPLIT * KB) * KB
        start = pid_k * chunk
        stop = tl.minimum(start + chunk, cols)
        for k0 in range(start, stop, KB):
            kk = k0 + j
            lk = kk < cols
            offset = tl.load(offset_ptr + kk, mask=lk, other=0)
            rate = tl.load(rate_ptr + kk, mask=lk, other=1).to(tl.int64)
            init = tl.load(init_ptr + kk, mask=lk, other=0).to(tl.int64)
            anchor = offset[:, None] + (base[None, :].to(tl.int64) + 1) * rate[:, None]
            ok = lk[:, None] & (base[None, :] < rows)
            if MODE == 1 or MODE == 4:           # no wire read
                span = anchor * 2654435761
            else:
                span = _span_of(words_ptr, anchor // 8, ok)
            code = base[None, :, None].to(tl.int64) + v[None, None, :].to(tl.int64)
            state = _state_of(span[:, :, None], (anchor % 8)[:, :, None], code,
                              v[None, None, :].to(tl.int64), rate[:, None, None],
                              init[:, None, None], window)
            if MODE == 2 or MODE == 4:           # no value gather
                value = (state & 0xFF).to(tl.float32)
            else:
                value = tl.load(value_ptr + state, mask=ok[:, :, None],
                                other=0.0).to(tl.float32)
            if MODE == 3:                        # no activation load, no FMA
                acc += tl.sum(value, axis=0)[None, :, :]
            else:
                xv = tl.load(x_ptr + mi[:, None] * cols + kk[None, :],
                             mask=live_m[:, None] & lk[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(xv[:, :, None, None] * value[None, :, :, :], axis=1)
        if MODE == 5:                            # no atomic: plain store
            tl.store(out_ptr + mi[:, None, None] * rows + n[None, :, :],
                     acc * scale[None, :, :],
                     mask=live_m[:, None, None] & live[None, :, :])
        else:
            tl.atomic_add(out_ptr + mi[:, None, None] * rows + n[None, :, :],
                          acc * scale[None, :, :],
                          mask=live_m[:, None, None] & live[None, :, :])

    rows_, cols_ = 1024, 3072
    p = synth(rows_, cols_, seed=11)
    x = torch.randn(1, cols_, dtype=torch.bfloat16, device="cuda")
    lanes = 32
    split = kw._gemv_split(rows_, lanes)
    res = {}
    for mode, label in ((0, "full"), (1, "no wire read"), (2, "no value gather"),
                        (4, "neither wire nor gather"), (5, "no split-K atomic"),
                        (3, "no activation load / FMA")):
        o = torch.zeros((1, rows_), dtype=torch.float32, device="cuda")

        def run(mode=mode, o=o):
            o.zero_()
            _abl[(triton.cdiv(rows_, lanes * 8), split)](
                x, p.plane_words, p.offsets, p.rates, p.initial, p.value_table,
                p.row_scale, o, rows_, cols_, 1, window=p.window_bits, LANES=lanes,
                VEC=8, MBLK=1, KB=8, SPLIT=split, MODE=mode, num_warps=2)
        r = bench(run, n=64, seconds=1.5)
        res[label] = r
        print(f"  {label:28s} {r['us']:7.2f} us   ({r.get('mean_w')} W)")
    full = res["full"]["us"]
    for label, r in res.items():
        if label != "full":
            r["share_of_full_removed"] = round((full - r["us"]) / full, 3)
    out["ablate_gemv"] = {"shape": [rows_, cols_], "split": split, "arms": res}


def arm_profile(out: dict, outdir: Path):
    units = real_units(limit=1)
    p = kw.prepare_from_parsed(units[0][1], device="cuda")
    x = torch.randn(1, p.cols, dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        p.decode()
        p.gemv(x)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(50):
            p.decode()
        for _ in range(50):
            p.gemv(x)
        torch.cuda.synchronize()
    trace = outdir / "trace_kernel_window.json"
    prof.export_chrome_trace(str(trace))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    (outdir / "profile_kernel_window.txt").write_text(table)
    print(table)
    out["profile"] = {"trace": str(trace), "unit": units[0][0],
                      "rows": p.rows, "cols": p.cols}


# --- driver -----------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    choices=["all", "bandwidth", "decode", "gemv", "ablate", "profile"])
    ap.add_argument("--out", type=Path, default=Path("/home/rob/tessera-runs/kernel-window"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="Qwen3-0.6B,Qwen3-4B,27B-assumed")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    result = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__,
              "concurrent_gpu_pids": subprocess.run(
                  ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                   "--format=csv,noheader"], capture_output=True,
                  text=True).stdout.strip().splitlines()}
    print(f"{len(result['concurrent_gpu_pids'])} process(es) on the card at start")

    global POWER
    with Power() as power:
        POWER = power
        time.sleep(0.5)
        idle = power.window(time.time() - 0.5, time.time())
        result["idle_power"] = idle
        print(f"idle draw before the first arm: {idle}")
        if args.arm in ("all", "bandwidth"):
            arm_bandwidth(result)
        if args.arm in ("all", "decode"):
            arm_decode(result, quick=args.quick)
        if args.arm in ("all", "gemv"):
            arm_gemv(result, models=tuple(args.models.split(",")), quick=args.quick)
        if args.arm in ("all", "ablate"):
            arm_ablate(result)
        if args.arm in ("all", "profile"):
            arm_profile(result, args.out)
        POWER = None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = args.out / f"bench-{args.arm}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    main()
