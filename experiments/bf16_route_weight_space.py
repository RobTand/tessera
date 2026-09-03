"""The number PrismaQuant's allocator will see: BF16 vs E4M3 at matched bytes.

W1's alphabet-floor measurement asked whether the ceiling above ~6 bpp was the
trellis's or the alphabet's, and answered: the alphabet's.  It ran the BF16 arm
**in memory** and priced it analytically, because the grid was not
serialisable.  It is now, so this runs **both arms through the real wire** --
``encode_linear_planes`` -> bytes -> ``read_unit_artifact`` -- and prices each
at the bytes actually written, including the second byte the BF16 code plane
costs per table entry.  Nothing here is analytic.

**The comparison is at matched bytes, per unit, and it is stated as a
crossover.**  At one rung the two arms are not the same size: BF16's table is
two bytes an entry where E4M3's is one, which is +0.03 bpp on a 2048x4096 GLM
expert and +0.25 bpp on a 1024x1024 Qwen Linear -- the wide alphabet's whole
overhead, and it is charged.  So each unit gets:

  * both arms at R = 4..7, at their own measured bpp;
  * the **crossover rate**: the lowest R at which the BF16 arm's error is
    below the E4M3 arm's *at the same R*, which is the conservative reading
    (BF16 is paying more there), with the bpp penalty printed beside it;
  * whether BF16 at R also beats E4M3 at **R+1**, which costs a whole extra
    bit -- the question an allocator holding a byte budget actually asks.

**Two error axes, because the allocator's cost is H-weighted.**  ``wt`` is the
plain relative Frobenius error; ``h`` weights each input column by its
activation second moment -- for the GLM experts by the real captured
activations (``out`` is then the same quantity measured directly, as
``|x(W - Wh)^T| / |xW^T|``), for the dense Qwen Linears by the diagonal H the
stock census captured.

**References on the same rows.**  EXL3 K=4/5/6/8 reconstructions for the GLM
experts, the per-channel FP8 RTN LS-refit floor (8 bpp -- the error no E4M3
tile at any rate goes below) for both sets, and for Qwen the production NVFP4
GPTQ+JSO export at 4.5 bpp.  The GLM set has no served NVFP4 arm here: the
NVFP4 route's comparison is W4A4 and lives in its own receipt, and quoting a
weight-only NVFP4 number beside a W16A16 one would be the comparator bug this
project has already made once.

Stages::

    PYTHONPATH=src:experiments TMPDIR=/home/rob/tmp \\
      TRITON_CACHE_DIR=/home/rob/.triton-cache \\
      python experiments/bf16_route_weight_space.py --stage glm \\
      --out /mnt/shared/tessera-runs/bf16/weight_space_glm.json

    ... --stage dense --out /mnt/shared/tessera-runs/bf16/weight_space_dense.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID, E4M3_GRID  # noqa: E402
from tessera.export import (  # noqa: E402
    BF16_CHANNEL_SIGMA,
    E4M3_WINDOW_BITS,
    encode_linear_planes,
)
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

E4M3_MAX = 448.0
#: The GLM captures and the EXL3 reconstructions, on the shared mount.
GLM_SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
GLM_ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
EXL3 = "/home/rob/dq-runs/exl3-ref"
#: The dense set: Qwen3-0.6B, its diagonal H, and the production NVFP4 export.
DENSE_SRC = "/home/rob/models/Qwen3-0.6B"
DENSE_H = "/mnt/shared/tessera-runs/bf16/refs/h_diag.pt"
DENSE_NVFP4 = "/mnt/shared/tessera-runs/bf16/refs/qwen3-0.6b-nvfp4"
DENSE_UNITS = [
    "model.layers.2.mlp.down_proj",
    "model.layers.2.mlp.gate_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.27.mlp.down_proj",
]
RUNGS = [1024, 1280, 1536, 1792]                  # R = 4, 5, 6, 7


def geomean(xs) -> float:
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def e4m3_rtn(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def ls_refit(w: torch.Tensor, scale: torch.Tensor, steps: int, dim: int = 1):
    """The per-channel FP8 RTN floor's scale: least squares onto its own codes.

    The same alternation ``tessera8_bounds`` runs, restated here so this
    script has no import from a measurement that may move: with the codes
    fixed the row's error is quadratic in its scale and the optimum is
    ``<w, u> / <u, u>``.
    """
    for _ in range(steps):
        u = e4m3_rtn(w / scale)
        num = (w * u).sum(dim=dim, keepdim=True)
        den = (u * u).sum(dim=dim, keepdim=True).clamp_min(1e-30)
        scale = torch.where(den > 0, num / den, scale)
    return scale


def fp8_floor(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().amax(dim=1, keepdim=True) / E4M3_MAX
    scale = ls_refit(w, scale, 6, dim=1)
    return e4m3_rtn(w / scale) * scale


def wire(w: torch.Tensor, grid, q256: int, name: str, window_bits: int,
         window_sigma: "float | None" = None):
    """Encode to real bytes and read them back.  ``(hat, bpp, secs)``.

    ``window_sigma=None`` takes the recipe's, which since #48 is **per rung**
    on BF16 (``export._window_sigma_for``): the body's reach in row-RMS grows
    as ``sqrt(R)`` instead of staying at the R=4 value.  Every BF16 arm at
    R != 4 therefore writes different bytes than it did before that change,
    and the receipts measured under the pinned spread -- the six-expert R=8
    table in issue #18 among them -- reproduce with
    ``--pinned-reach``, which passes ``BF16_CHANNEL_SIGMA`` here.
    """
    started = time.time()
    exported, _unit, _forests = encode_linear_planes(
        w, grid=grid, q256=q256, name=name, window_bits=window_bits, verify=True,
        **({} if window_sigma is None else {"window_sigma": float(window_sigma)}),
    )
    hat = read_unit_artifact(exported.blob, device=w.device)
    return hat, float(exported.bpp), time.time() - started


def pinned_reach(a, grid):
    """``BF16_CHANNEL_SIGMA`` under ``--pinned-reach`` on BF16, else ``None``.

    The pre-#48 wire tied the window table's spread to the row scale, so the
    body's reach was the R=4 value at every rung.  The flag reproduces the
    receipts measured under it; without it the BF16 arms carry the recipe's
    per-rung reach, which is what ships.  E4M3 is unaffected either way -- its
    recipe still leaves the spread unset.
    """
    return BF16_CHANNEL_SIGMA if (a.pinned_reach and grid.name == "BF16") else None


def crossover(res: dict, rungs) -> dict:
    """The lowest R where BF16 beats E4M3, and what it cost to get there."""
    out = {}
    for axis in ("wt", "h"):
        first = None
        for q in rungs:
            b, e = res.get(f"bf16_q{q}"), res.get(f"e4m3_q{q}")
            if b is None or e is None:
                continue
            if first is None and b[axis] < e[axis]:
                first = {
                    "rate": q / 256,
                    "bpp_bf16": b["bpp"], "bpp_e4m3": e["bpp"],
                    "bpp_penalty": b["bpp"] - e["bpp"],
                    "ratio_same_rate": b[axis] / e[axis],
                }
                nxt = res.get(f"e4m3_q{q + 256}")
                if nxt is not None:
                    first["beats_e4m3_one_rung_up"] = b[axis] < nxt[axis]
                    first["ratio_one_rung_up"] = b[axis] / nxt[axis]
        out[axis] = first
    return out


# ------------------------------------------------------------------- GLM


def stage_glm(a):
    index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"args": vars(a), "tensors": {}}
    lines: list[str] = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
        Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")

    for layer in a.layers:
        blob = torch.load(
            f"{GLM_ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False,
        )
        xa = blob["inputs"].float()
        x_ev = xa[xa.shape[0] - a.eval_rows:].contiguous().cuda()
        del xa, blob
        # The H the out-space error is weighted by, on the held-out rows.
        h = (x_ev * x_ev).sum(dim=0)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{GLM_SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            tname = f"L{layer}.{proj}"
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            res = {"rows": w.shape[0], "cols": w.shape[1]}
            log(f"\n== {tname} {tuple(w.shape)}")
            log(f"    {'arm':<28} {'bpp':>7} {'wt':>9} {'h':>9} {'out':>9} "
                f"{'out_bf16':>9} {'s':>5}")

            def rec(arm, hat, bpp, secs=0.0):
                e = hat - w
                fold = hat.to(torch.bfloat16).float()
                r = {
                    "bpp": bpp,
                    "wt": float(e.norm() / nw),
                    "h": float(math.sqrt(
                        float(((e * e).sum(0) * h).sum() / ((w * w).sum(0) * h).sum()))),
                    "out": float((x_ev @ hat.T - y).norm() / ny),
                    "out_bf16": float((x_ev @ fold.T - y).norm() / ny),
                    "secs": secs,
                }
                res[arm] = r
                log(f"    {arm:<28} {bpp:7.4f} {r['wt']:9.5f} {r['h']:9.5f} "
                    f"{r['out']:9.5f} {r['out_bf16']:9.5f} {secs:5.0f}")
                return r

            for k in a.exl3:
                p = Path(EXL3) / f"L{layer}_{proj}_K{k}.pt"
                if p.exists():
                    rec(f"EXL3 K={k}", torch.load(p, map_location="cuda").float(), k + 0.0117)
            rec("FP8 RTN LS-refit", fp8_floor(w), 8.0 + 32 / w.shape[1])
            for q in a.rungs:
                for label, grid in (("e4m3", E4M3_GRID), ("bf16", BF16_GRID)):
                    hat, bpp, secs = wire(w, grid, q, tname, a.window_bits,
                                          pinned_reach(a, grid))
                    rec(f"{label}_q{q}", hat, bpp, secs)
                    del hat
                    torch.cuda.empty_cache()
            res["crossover"] = crossover(res, a.rungs)
            log(f"    crossover {json.dumps(res['crossover'])}")
            out["tensors"][tname] = res
            Path(a.out).write_text(json.dumps(out, indent=1))
            del w, y
            torch.cuda.empty_cache()
        del x_ev
        torch.cuda.empty_cache()
    summarise(out, log, a.rungs)
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(f"\nwrote {a.out}")


# ----------------------------------------------------------------- dense


def open_all(directory: str):
    index = {}
    for path in sorted(glob.glob(directory + "/*.safetensors")):
        handle = safe_open(path, framework="pt")
        for key in handle.keys():
            index[key] = handle
    return index


def stage_dense(a):
    H = {k: v.cuda().float() for k, v in torch.load(DENSE_H).items()}
    src = open_all(DENSE_SRC)
    nv = open_all(DENSE_NVFP4)
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    out = {"args": vars(a), "tensors": {}}
    lines: list[str] = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
        Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")

    def nvfp4(name, rows, cols):
        pk = nv[name + ".weight_packed"].get_tensor(name + ".weight_packed").cuda()
        s = nv[name + ".weight_scale"].get_tensor(name + ".weight_scale").cuda().float()
        g = nv[name + ".weight_global_scale"].get_tensor(name + ".weight_global_scale").cuda().float()
        lo, hi = (pk & 0xF).long(), (pk >> 4).long()
        dq = lambda t: e2m1[t & 7] * torch.where(t >= 8, -1.0, 1.0)
        return (torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
                * (s / g).repeat_interleave(16, dim=1))

    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        rows, cols = w.shape
        h = H[name]
        nw = w.norm()
        res = {"rows": rows, "cols": cols}
        log(f"\n== {name} {tuple(w.shape)}")
        log(f"    {'arm':<28} {'bpp':>7} {'wt':>9} {'h':>9} {'wt_bf16':>9} {'s':>5}")

        def rec(arm, hat, bpp, secs=0.0):
            e = hat - w
            fold = hat.to(torch.bfloat16).float()
            r = {
                "bpp": bpp,
                "wt": float(e.norm() / nw),
                "h": float(math.sqrt(
                    float(((e * e).sum(0) * h).sum() / ((w * w).sum(0) * h).sum()))),
                "wt_bf16": float((fold - w).norm() / nw),
                "secs": secs,
            }
            res[arm] = r
            log(f"    {arm:<28} {bpp:7.4f} {r['wt']:9.5f} {r['h']:9.5f} "
                f"{r['wt_bf16']:9.5f} {secs:5.0f}")
            return r

        rec("FP8 RTN LS-refit", fp8_floor(w), 8.0 + 32 / cols)
        if name + ".weight_packed" in nv:
            rec("NVFP4 GPTQ+JSO (production)", nvfp4(name, rows, cols), 4.5)
        for q in a.rungs:
            for label, grid in (("e4m3", E4M3_GRID), ("bf16", BF16_GRID)):
                hat, bpp, secs = wire(w, grid, q, name, a.window_bits,
                                      pinned_reach(a, grid))
                rec(f"{label}_q{q}", hat, bpp, secs)
                del hat
                torch.cuda.empty_cache()
        res["crossover"] = crossover(res, a.rungs)
        log(f"    crossover {json.dumps(res['crossover'])}")
        out["tensors"][name] = res
        Path(a.out).write_text(json.dumps(out, indent=1))
        del w
        torch.cuda.empty_cache()
    summarise(out, log, a.rungs)
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(f"\nwrote {a.out}")


def summarise(out, log, rungs):
    arms: dict[str, list] = {}
    for res in out["tensors"].values():
        for arm, r in res.items():
            if isinstance(r, dict) and "bpp" in r:
                arms.setdefault(arm, []).append(r)
    n = len(out["tensors"])
    keys = ("wt", "h", "out", "out_bf16", "wt_bf16")
    out["summary"] = {}
    log(f"\n== geomean over {n} tensors")
    log(f"    {'arm':<28} {'bpp':>7} " + " ".join(f"{k:>9}" for k in keys))
    for arm, rs in arms.items():
        if len(rs) != n:
            continue
        row = {"bpp": sum(r["bpp"] for r in rs) / n}
        for k in keys:
            if all(k in r for r in rs):
                row[k] = geomean([r[k] for r in rs])
        out["summary"][arm] = row
        log(f"    {arm:<28} {row['bpp']:7.4f} "
            + " ".join(f"{row.get(k, float('nan')):9.5f}" for k in keys))
    out["crossover"] = {
        name: res.get("crossover") for name, res in out["tensors"].items()
    }
    log("\n== crossover (lowest R where BF16 beats E4M3 at the same R)")
    for name, cross in out["crossover"].items():
        for axis, hit in (cross or {}).items():
            log(f"    {name:<44} {axis}: " + (
                "never in the measured range" if hit is None else
                f"R={hit['rate']:.0f} at +{hit['bpp_penalty']:.3f} bpp, "
                f"{hit['ratio_same_rate']:.4f}x same rate"
                + (f", {hit['ratio_one_rung_up']:.4f}x vs E4M3 one rung up"
                   if "ratio_one_rung_up" in hit else "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["glm", "dense"], default="glm")
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--units", nargs="+", default=DENSE_UNITS)
    ap.add_argument("--rungs", type=int, nargs="+", default=RUNGS)
    ap.add_argument("--window-bits", type=int, default=E4M3_WINDOW_BITS)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--pinned-reach", action="store_true",
                    help="encode BF16 at the pre-#48 spread (reach pinned to "
                         "the R=4 value at every rung) to reproduce older receipts")
    ap.add_argument("--out", default="weight_space.json")
    a = ap.parse_args()
    {"glm": stage_glm, "dense": stage_dense}[a.stage](a)


if __name__ == "__main__":
    main()
