#!/usr/bin/env python
"""Sweep the three encoder-fit caps behind tessera issues #66, #67 and #68.

Each cap becomes a swept axis, and every arm reports the objective the fit
itself minimises plus wall-clock, so a coordinator run at scale can read off
where each cap binds:

* axis A (#66, ``_lloyd_levels`` pass cap): Lloyd levels for ``levels``
  masses at each value of ``--lloyd-iters``; objective is the
  nearest-level SSE on the FULL source.
* axis B (#67, ``samples[::4]`` stride): Lloyd levels fit on every
  ``--strides`` subsample of the source; objective is the same full-source
  nearest-level SSE, so a strided fit that spends anchors on sampling
  geometry reads worse than the full-sample fit.
* axis C (#68, ``_fit_lut`` swap cap): the 16-entry E4M3 table fit at each
  value of ``--swaps``; objective is the plane's own ``_lut_cost``.

Two paths.  The default is a CPU smoke on synthetic data (a sorted Gaussian
sample for A/B, lognormal scale targets with heteroscedastic weights for C).
``--device`` moves the torch-side work (axis C, and the SSE scoring) and
``--input`` points at a ``torch.save``d tensor of real values -- flattened,
finite-filtered, and used as the Lloyd source (sorted) and as the LUT
targets -- so the same script runs the coordinator's at-scale arm with e.g.::

    PYTHONPATH=src python experiments/encoder_fit_cap_sweep.py \
        --device cuda --input /path/to/half_scales.pt --out sweep.json

``_lloyd_levels`` itself is pure Python and always runs on CPU; ``--device``
only affects the torch scoring and axis C.  Nothing here encodes a model or
serves anything: every number is a fit-time objective.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tessera.alphabet import GAUSSIAN_SOURCE, _lloyd_levels  # noqa: E402
from tessera.encode import _fit_lut, _lut_cost  # noqa: E402


def _parse_ints(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def _nearest_level_sse(source: torch.Tensor, levels: list[float]) -> float:
    """Full-source nearest-level SSE, on ``--device``."""
    table = torch.tensor(levels, dtype=torch.float64, device=source.device)
    gap = (source.to(torch.float64).unsqueeze(1) - table.unsqueeze(0)).abs().amin(dim=1)
    return float((gap * gap).sum())


def _load_input(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("targets", "scales", "weight", "values", "data"):
            if key in obj and torch.is_tensor(obj[key]):
                obj = obj[key]
                break
        else:
            raise SystemExit(f"--input {path}: dict without a tensor entry")
    flat = torch.as_tensor(obj).flatten().float()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        raise SystemExit(f"--input {path}: no finite values")
    return flat


def axis_lloyd_iters(source_t: tuple[float, ...], scored: torch.Tensor,
                     levels: int, iters: list[int]) -> list[dict]:
    """Axis A (#66): Lloyd pass cap -> full-source SSE + wall-clock."""
    rows = []
    prev = None
    for it in iters:
        t0 = time.perf_counter()
        lv = _lloyd_levels(source_t, levels, it)
        secs = time.perf_counter() - t0
        sse = _nearest_level_sse(scored, lv)
        row = {"iters": it, "sse": sse, "secs": secs}
        if prev is not None:
            row["delta_vs_prev"] = prev - sse
        rows.append(row)
        prev = sse
    return rows


def axis_stride(source_t: tuple[float, ...], scored: torch.Tensor,
                levels: int, strides: list[int], iters: int) -> list[dict]:
    """Axis B (#67): fit stride -> full-source SSE + wall-clock.

    ``iters`` is held fixed so the stride is the only moving part; pass a
    large value to take #66's cap out of the picture.
    """
    rows = []
    for stride in strides:
        sub = source_t[::stride] or source_t
        t0 = time.perf_counter()
        lv = _lloyd_levels(sub, levels, iters)
        secs = time.perf_counter() - t0
        rows.append({
            "stride": stride,
            "fit_points": len(sub),
            "sse": _nearest_level_sse(scored, lv),
            "secs": secs,
        })
    return rows


def synthetic_lut(n: int, seed: int, device: torch.device,
                  entries: int = 16) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Lognormal scale targets with heteroscedastic weights (issue #68's shape).

    ``global_scale`` places the largest target in E4M3's top binade, the way
    ``_pack_scales_lut`` does, so the candidate window is the production one.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.exp(torch.randn(n, generator=gen) * 1.0 + 0.5).clamp(0.02, 7.0)
    weights = torch.exp(torch.randn(n, generator=gen) * 1.5)
    return targets.to(device), weights.to(device), float(2.0 ** -6)


def axis_swaps(targets: torch.Tensor, weights: torch.Tensor, global_scale: float,
               swaps: list[int], entries: int = 16) -> list[dict]:
    """Axis C (#68): swap-pass cap -> plane cost + wall-clock."""
    rows = []
    for sw in swaps:
        t0 = time.perf_counter()
        _, table = _fit_lut(targets, weights, global_scale, entries, swaps=sw)
        secs = time.perf_counter() - t0
        rows.append({
            "swaps": sw,
            "cost": float(_lut_cost(targets, weights, table)),
            "secs": secs,
        })
    return rows


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    t_all = time.perf_counter()

    if args.input:
        raw = _load_input(args.input)
        source_f = tuple(sorted(float(v) for v in raw.tolist()))
        lut_targets = raw.abs().to(device)
        lut_weights = torch.ones_like(lut_targets)
        lut_global = float(raw.abs().max() / 448.0) or float(2.0 ** -6)
        source_note = f"input file {args.input} ({raw.numel()} finite values)"
    else:
        source_f = GAUSSIAN_SOURCE(args.source_count)
        lut_targets, lut_weights, lut_global = synthetic_lut(
            args.lut_n, args.seed, device)
        source_note = f"synthetic GAUSSIAN_SOURCE({args.source_count})"

    scored = torch.tensor(source_f, dtype=torch.float64, device=device)
    iters = _parse_ints(args.lloyd_iters)
    strides = _parse_ints(args.strides)
    swaps = _parse_ints(args.swaps)

    out: dict = {
        "source": source_note,
        "device": str(device),
        "levels": args.levels,
        "lloyd_budget_for_stride_axis": args.stride_iters,
    }
    print(f"[axis A] Lloyd pass cap; source: {source_note}; levels={args.levels}")
    out["axis_a_lloyd_iters"] = axis_lloyd_iters(source_f, scored, args.levels, iters)
    for row in out["axis_a_lloyd_iters"]:
        extra = f"  d={row['delta_vs_prev']:+.6e}" if "delta_vs_prev" in row else ""
        print(f"  iters={row['iters']:>4}  sse={row['sse']:.6e}  secs={row['secs']:.3f}{extra}")

    print(f"[axis B] fit stride at fixed iters={args.stride_iters}")
    out["axis_b_stride"] = axis_stride(source_f, scored, args.levels,
                                       strides, args.stride_iters)
    for row in out["axis_b_stride"]:
        print(f"  stride={row['stride']:>3}  fit_points={row['fit_points']:>6}  "
              f"sse={row['sse']:.6e}  secs={row['secs']:.3f}")

    print(f"[axis C] LUT swap cap; n={lut_targets.numel()} seed={args.seed}")
    out["axis_c_swaps"] = axis_swaps(lut_targets, lut_weights, lut_global,
                                     swaps, entries=args.entries)
    for row in out["axis_c_swaps"]:
        print(f"  swaps={row['swaps']:>3}  cost={row['cost']:.6e}  secs={row['secs']:.3f}")

    out["total_secs"] = time.perf_counter() - t_all
    print(f"total_secs={out['total_secs']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input", default=None,
                    help="torch.save'd tensor of real values; default: synthetic")
    ap.add_argument("--source-count", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--lloyd-iters", default="5,10,20,40,80,160")
    ap.add_argument("--stride-iters", type=int, default=200)
    ap.add_argument("--strides", default="1,2,4,8,16")
    ap.add_argument("--lut-n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entries", type=int, default=16)
    ap.add_argument("--swaps", default="0,1,2,3,5,9")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: 1024-point source, 512 LUT targets")
    return ap


def main(argv: list[str] | None = None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.smoke:
        args.source_count = min(args.source_count, 1024)
        args.lut_n = min(args.lut_n, 512)
        args.stride_iters = min(args.stride_iters, 60)
        if args.lloyd_iters == ap.get_default("lloyd_iters"):
            args.lloyd_iters = "5,10,20,40,60"
        if args.swaps == ap.get_default("swaps"):
            args.swaps = "0,1,2,5"
    return run(args)


if __name__ == "__main__":
    main()
