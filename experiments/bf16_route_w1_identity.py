"""Is the library's BF16 route the same experiment W1 ran?  One tensor, both ways.

W1's ``tessera16_alphabet_floor.py`` measured the alphabet floor with a BF16
grid it built in the script: the finite normal bf16 values over a 32-binade
window, 8192 codes, snapped by ``torch.cdist(...).argmin`` in float32 -- and
it ran the encoder **in memory**, because no such grid could be serialised.
The library's grid is the whole of bf16, 65536 codes, snapped exactly in
float64, and it goes through the real wire.  Three differences, and this
measures each so that none of them is assumed away:

1. **Grid width.**  Every table quantile at the shipping (L, sigma) lies deep
   inside W1's window, so the two grids should snap to identical *values*.
   Asserted entry by entry.
2. **The snap.**  ``cdist`` computes ``x^2 + y^2 - 2xy``, not ``|x - y|``.  At
   8192 codes and float32 that mis-picks a handful of entries whose quantile
   sits within a rounding of a bf16 midpoint.  This counts them, and then
   **encodes the same tensor under both tables** so the difference is priced
   in error rather than argued about.
3. **Wire vs memory.**  W1's BF16 arm was ``encode_unit`` ->
   ``reconstruct_unit``; this one is ``encode_linear_planes`` -> bytes ->
   ``read_unit_artifact``, at the bytes actually written.  If those disagree
   the exporter is wrong, so they are compared directly.

Run::

    PYTHONPATH=src:experiments python experiments/bf16_route_w1_identity.py \\
      --out /mnt/shared/tessera-runs/bf16/w1_identity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tessera.encode as tessera_encode  # noqa: E402
from tessera.alphabet import BF16_GRID, GAUSSIAN_SOURCE, PayloadGrid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit, grid_vector_table, window_table  # noqa: E402
from tessera.export import (  # noqa: E402
    DEFAULT_CODE, DEFAULT_SCALE_REFIT, DEFAULT_TRELLIS_WEIGHTING,
    E4M3_WINDOW_BITS, _plan_for, encode_linear_planes)
from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: E402
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

GLM_SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
GLM_ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
#: W1's grid, verbatim (``tessera16_alphabet_floor.bf16_grid``).
W1_EXP_LO, W1_EXP_HI = -29, 2
W1_SIGMA = 1.0


def w1_grid(lo: int = W1_EXP_LO, hi: int = W1_EXP_HI) -> PayloadGrid:
    mags = [(1.0 + m / 128.0) * 2.0 ** e for e in range(lo, hi + 1) for m in range(128)]
    values = tuple([+v for v in mags] + [-v for v in mags])
    return PayloadGrid(f"BF16e{lo}:{hi}", values)


def cdist_table(grid: PayloadGrid, window_bits: int, sigma: float, seed: int = 0):
    """W1's snap: nearest by ``torch.cdist`` in float32, ties to the lower code."""
    size = 1 << window_bits
    quantiles = torch.tensor(GAUSSIAN_SOURCE(size, float(sigma)))
    generator = torch.Generator().manual_seed(int(seed))
    points = quantiles[torch.randperm(size, generator=generator)].reshape(-1, 1).float()
    vectors = grid_vector_table(grid).float()
    return torch.cdist(points, vectors).argmin(dim=1).to(torch.int32)


def encode_memory(w, grid, q256: int, window_bits: int):
    """W1's in-memory arm: the exporter's own settings, no artifact.

    ``export.encode_linear_planes`` resolves the recipe and then calls
    ``encode_unit``; a grid outside ``SERIALISABLE_GRIDS`` cannot reach the
    serialiser, so this is that call with the recipe's fields spelled out --
    ``BF16_RECIPE``'s, which are ``E4M3_RECIPE``'s but for the alphabet.
    """
    rates, forest = _plan_for(grid, q256, w.shape[1], BodyKind.WINDOW, W1_SIGMA)
    return encode_unit(
        w, forest, rates, DEFAULT_CODE, completion=0, scale_refit=DEFAULT_SCALE_REFIT,
        span=1, scale_plane=ScalePlaneKind.CHANNEL,
        trellis_weighting=DEFAULT_TRELLIS_WEIGHTING, body=BodyKind.WINDOW,
        window_bits=window_bits, window_seed=0, window_sigma=None,
        channel_sigma=W1_SIGMA,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--rungs", type=int, nargs="+", default=[1024, 1280, 1536, 1792])
    ap.add_argument("--window-bits", type=int, default=E4M3_WINDOW_BITS)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="w1_identity.json")
    a = ap.parse_args()

    out = {"args": vars(a)}
    L, sigma = a.window_bits, W1_SIGMA
    wide, narrow = BF16_GRID, w1_grid()

    # 1. the two grids snap to the same values
    t_wide = window_table(wide, L, sigma=sigma, seed=0, half=16)
    t_narrow = window_table(narrow, L, sigma=sigma, seed=0, half=16)
    v_wide = grid_vector_table(wide)[t_wide.long()].reshape(-1)
    v_narrow = grid_vector_table(narrow)[t_narrow.long()].reshape(-1)
    out["grid_width"] = {
        "library_codes": wide.size, "w1_codes": narrow.size,
        "entries": int(v_wide.numel()),
        "value_mismatches": int((v_wide != v_narrow).sum()),
        "reach": float(v_wide.abs().max()),
        "distinct_values": int(v_wide.unique().numel()),
    }

    # 2. the snap: exact float64 against W1's float32 cdist, on W1's own grid
    t_cdist = cdist_table(narrow, L, sigma)
    v_cdist = grid_vector_table(narrow)[t_cdist.long()].reshape(-1)
    differ = (v_cdist != v_narrow).nonzero().reshape(-1)
    rne = torch.tensor(GAUSSIAN_SOURCE(1 << L, sigma))
    generator = torch.Generator().manual_seed(0)
    points = rne[torch.randperm(1 << L, generator=generator)].float()
    out["snap"] = {
        "entries": int(v_wide.numel()),
        "cdist_vs_exact_mismatches": int(differ.numel()),
        "exact_vs_bf16_rne_mismatches": int(
            (points.to(torch.bfloat16).float() != v_wide).sum()),
        "cdist_vs_bf16_rne_mismatches": int(
            (points.to(torch.bfloat16).float() != v_cdist).sum()),
        "examples": [
            {"point": float(points[i]), "cdist": float(v_cdist[i]),
             "exact": float(v_narrow[i])}
            for i in differ[:5].tolist()
        ],
    }
    print(json.dumps({k: out[k] for k in ("grid_width", "snap")}, indent=1), flush=True)

    # 3. does the difference matter?  Encode the tensor under each table.
    index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
    name = (f"model.language_model.layers.{a.layer}.mlp.experts.0.{a.proj}.weight")
    with safe_open(f"{GLM_SRC}/{index[name]}", framework="pt") as f:
        w = f.get_tensor(name).contiguous().cuda().float()
    blob = torch.load(
        f"{GLM_ACT}/model__language_model__layers__{a.layer}__mlp__experts.pt",
        map_location="cpu", weights_only=False,
    )
    xa = blob["inputs"].float()
    x_ev = xa[xa.shape[0] - a.eval_rows:].contiguous().cuda()
    del xa, blob
    y = x_ev @ w.T
    ny, nw = y.norm(), w.norm()
    out["tensor"] = {"name": name, "shape": list(w.shape), "eval_rows": a.eval_rows}
    out["arms"] = {}
    print(f"\n{'arm':<34} {'bpp':>8} {'wt':>9} {'out':>9} {'out_bf16':>9} {'s':>5}",
          flush=True)

    def score(arm, hat, bpp, secs):
        fold = hat.to(torch.bfloat16).float()
        row = {
            "bpp": bpp,
            "wt": float((hat - w).norm() / nw),
            "out": float((x_ev @ hat.T - y).norm() / ny),
            "out_bf16": float((x_ev @ fold.T - y).norm() / ny),
            "secs": secs,
        }
        out["arms"][arm] = row
        print(f"{arm:<34} {bpp:8.4f} {row['wt']:9.5f} {row['out']:9.5f} "
              f"{row['out_bf16']:9.5f} {secs:5.0f}", flush=True)
        return row

    original = tessera_encode._window_table_cpu

    for q in a.rungs:
        started = time.time()
        exported, unit, forests = encode_linear_planes(
            w, grid=wide, q256=q, name=name, window_bits=L, verify=True)
        secs = time.time() - started
        hat_wire = read_unit_artifact(exported.blob, device=w.device)
        hat_memory = reconstruct_unit(unit, forests, None)
        row = score(f"library wire R={q // 256}", hat_wire, float(exported.bpp), secs)
        row["wire_equals_memory"] = bool(torch.equal(hat_wire, hat_memory))
        del hat_wire, hat_memory
        torch.cuda.empty_cache()

        # The same encode with W1's cdist table, on W1's grid: the only change.
        def patched(grid, window_bits, sig, seed, half, _o=original):
            if grid.size > 256 and window_bits == L:
                return cdist_table(grid, window_bits, sig, seed)
            return _o(grid, window_bits, sig, seed, half)

        # W1's arm: its grid is outside SERIALISABLE_GRIDS, so it cannot go
        # through the exporter -- it is ``encode_unit`` -> ``reconstruct_unit``,
        # exactly as W1 ran it, with the cdist table patched in.  Priced at the
        # library wire's bytes, which is the same count: both grids are two
        # bytes a code.
        tessera_encode._window_table_cpu = patched
        try:
            started = time.time()
            unit = encode_memory(w, narrow, q, L)
            hat = reconstruct_unit(unit, narrow, None)
            score(f"W1 grid + cdist table R={q // 256}", hat,
                  float(exported.bpp), time.time() - started)
            del hat, unit
        finally:
            tessera_encode._window_table_cpu = original
        torch.cuda.empty_cache()

    ratios = {}
    for q in a.rungs:
        lib = out["arms"].get(f"library wire R={q // 256}")
        w1 = out["arms"].get(f"W1 grid + cdist table R={q // 256}")
        if lib and w1:
            ratios[q // 256] = {k: lib[k] / w1[k] for k in ("wt", "out", "out_bf16")}
    out["library_over_w1"] = ratios
    print("\nlibrary / W1 (below 1.0 is the library ahead):")
    print(json.dumps(ratios, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
