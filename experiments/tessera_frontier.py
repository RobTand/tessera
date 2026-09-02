"""One frontier, on the production encoder, across both tiles.

Every Tessera arm here is ``encode_linear`` -> bytes -> ``read_unit_artifact``
-- the exporter's own encoder and the reader's own decode, priced at the
bytes actually written -- so a point on this chart is a point the exporter
can ship.  The arms span the grammar's axes on the six GLM-5.3-Flash expert
tensors, scored on the same held-out rows with the same three legs as every
earlier harness (weight-only output space, executed W?A4 under the served
NVFP4 activation quantiser, executed W?A8 under the vLLM FP8 per-token
quantiser):

    grid   x body            x scale plane   x rate
    E2M1x2   TCQ (span 2)      LUT16           q256 ladder
    E4M3     WINDOW (L bits)   CHANNEL         (per position)

plus the comparators the earlier harnesses established: EXL3 K=2..8 quantised
fresh on the same Hessian rows (W4A16), per-channel FP8 RTN (the E4M3 floor),
and whatever ``tessera_vs_exl3_followups.json`` / ``tessera8_targets.json``
hold for NVFP4 and Gridbook, copied by name.  Ratios against EXL3 are taken
at matched bpp by log-linear interpolation between its rungs, on every leg.

    PYTHONPATH=src:experiments:/home/rob/prismaquant python experiments/tessera_frontier.py \\
        --layers 5 20 42 --window-bits 12 --out experiments/results/tessera_frontier.json
"""
from __future__ import annotations

import argparse, json, math, subprocess, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid          # noqa: E402
from tessera.export import DEFAULT_SCALE_REFIT, encode_linear          # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind                  # noqa: E402
from tessera.unit_artifact import read_unit_artifact                   # noqa: E402
from tessera8_targets import ACT, EXL3, SRC                            # noqa: E402

GRIDS = {"E2M1x2": tuple_grid(E2M1_GRID, 2), "E4M3": E4M3_GRID}
TCQ, WINDOW = BodyKind.TCQ, BodyKind.WINDOW
LUT, CHANNEL = ScalePlaneKind.LUT, ScalePlaneKind.CHANNEL

# (label, grid, q256 per position, body, plane).  Window widths come from the
# CLI.  Bytes: TCQ span 2 over LUT16 spends q/256 + 0.25 (label, per weight on
# the pair grid; 0.5 on E4M3) + 0.25 (nibble); a window over LUT16 spends
# q/256 + 0.25 + table; a window over CHANNEL spends q/256 + 16/cols + table.
ARMS = [
    ("E2M1x2 TCQ span2 LUT16 q512  (2.5)",  "E2M1x2", 512,  TCQ,    LUT),
    ("E2M1x2 TCQ span2 LUT16 q640  (3.0)",  "E2M1x2", 640,  TCQ,    LUT),
    ("E2M1x2 TCQ span2 LUT16 q768  (3.5)",  "E2M1x2", 768,  TCQ,    LUT),
    ("E2M1x2 TCQ span2 LUT16 q896  (4.0)",  "E2M1x2", 896,  TCQ,    LUT),
    ("E2M1x2 window LUT16 q576  (2.5)",     "E2M1x2", 576,  WINDOW, LUT),
    ("E2M1x2 window LUT16 q704  (3.0)",     "E2M1x2", 704,  WINDOW, LUT),
    ("E2M1x2 window LUT16 q832  (3.5)",     "E2M1x2", 832,  WINDOW, LUT),
    ("E2M1x2 window LUT16 q960  (4.0)",     "E2M1x2", 960,  WINDOW, LUT),
    ("E4M3 TCQ span2 LUT16 q832  (4.0)",    "E4M3",   832,  TCQ,    LUT),
    ("E4M3 TCQ span2 LUT16 q1088 (5.0)",    "E4M3",   1088, TCQ,    LUT),
    ("E4M3 TCQ span2 CHANNEL q896  (4.0)",  "E4M3",   896,  TCQ,    CHANNEL),
    ("E4M3 TCQ span2 CHANNEL q1152 (5.0)",  "E4M3",   1152, TCQ,    CHANNEL),
    ("E4M3 window LUT16 q960  (4.0)",       "E4M3",   960,  WINDOW, LUT),
    ("E4M3 window LUT16 q1216 (5.0)",       "E4M3",   1216, WINDOW, LUT),
    ("E4M3 window CHANNEL q768  (3.0)",     "E4M3",   768,  WINDOW, CHANNEL),
    ("E4M3 window CHANNEL q1024 (4.0)",     "E4M3",   1024, WINDOW, CHANNEL),
    ("E4M3 window CHANNEL q1280 (5.0)",     "E4M3",   1280, WINDOW, CHANNEL),
    ("E4M3 window CHANNEL q1536 (6.0)",     "E4M3",   1536, WINDOW, CHANNEL),
]

COPY_FROM = {
    "experiments/results/tessera_vs_exl3_followups.json": ("Gridbook", "FP8-CB", "FP4-CB", "NVFP4"),
    "experiments/results/tessera8_targets.json": ("NVFP4", "FP8-CB", "FP4-CB"),
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def encoder_digest() -> str:
    """A content hash of the encoder this run actually executed.

    Every frontier JSON so far records ``git: unknown``, because these runs
    happen on a box that holds an rsync of the working tree and no checkout --
    and a result whose tree cannot be named is a result that cannot be
    reproduced.  A digest over the sources needs no repository: two runs that
    print the same twelve characters ran the same encoder, wherever the bytes
    came from.
    """
    import hashlib
    root = Path(__file__).resolve().parent.parent / "src" / "tessera"
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def exl3_at(rows: "dict[float, dict]", bpp: float, leg: str) -> "float | None":
    """EXL3's leg at ``bpp`` by log-linear interpolation between its rungs."""
    pts = sorted((b, r[leg]) for b, r in rows.items())
    if not pts:
        return None
    if bpp <= pts[0][0]:
        (b0, e0), (b1, e1) = pts[0], pts[1] if len(pts) > 1 else pts[0]
    elif bpp >= pts[-1][0]:
        (b0, e0), (b1, e1) = pts[-2] if len(pts) > 1 else pts[-1], pts[-1]
    else:
        for (b0, e0), (b1, e1) in zip(pts, pts[1:]):
            if b0 <= bpp <= b1:
                break
    if b1 == b0:
        return e0
    t = (bpp - b0) / (b1 - b0)
    return math.exp(math.log(e0) + t * (math.log(e1) - math.log(e0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--window-bits", type=int, nargs="+", default=[12])
    ap.add_argument("--arms", nargs="*", default=None, help="substring filters on arm labels")
    ap.add_argument("--refit", type=int, default=DEFAULT_SCALE_REFIT)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--slice", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    ap.add_argument("--out", default="experiments/results/tessera_frontier.json")
    a = ap.parse_args()

    from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

    arms = [arm for arm in ARMS if not a.arms or any(f in arm[0] for f in a.arms)]
    out_path = Path(a.out)
    out = {"args": vars(a), "git": git_hash(), "encoder_digest": encoder_digest(),
           "experts": {}}
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    copied = {}
    for path, prefixes in COPY_FROM.items():
        p = Path(path)
        if not p.exists():
            log(f"    COPY_FROM {p.name}: absent")
            continue
        src = json.load(open(p))
        # Both source files key their comparators as ``arms[name] -> [row, ...]``
        # with the tensor named inside each row; this script keys everything as
        # ``experts[tensor][arm]``.  Reading ``.get("experts")`` here returned an
        # empty dict from both files and copied nothing -- silently, because a
        # missing comparator looks exactly like a comparator that was not asked
        # for.  Transpose, and *say* how many arms crossed over, so the next
        # silent zero is visible in the log.
        arms = src.get("arms")
        if not isinstance(arms, dict):
            log(f"    COPY_FROM {p.name}: no 'arms' object; copied 0")
            continue
        # And the two sources do not even agree with each other: one names the
        # tensor inside every row, the other leaves rows positional against the
        # file's own top-level ``tensors`` list.  Take the name where it is
        # written and the position where it is not.
        order = src.get("tensors") or []
        n = 0
        for name, rows in arms.items():
            if not any(name.startswith(pre) for pre in prefixes):
                continue
            if not isinstance(rows, list):
                continue
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or "out" not in row:
                    continue
                tname_src = row.get("tensor") or (order[i] if i < len(order) else None)
                if tname_src is None:
                    continue
                v = {k: row[k] for k in row if k != "tensor"}
                copied.setdefault(tname_src, {})[name] = dict(v, copied_from=p.name)
                n += 1
        log(f"    COPY_FROM {p.name}: {n} rows over "
            f"{len({a for t in copied.values() for a in t})} arms")

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    dev = "cuda"
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        if a.slice:
            x_fit, x_ev = x_fit[:, :a.slice[1]].contiguous(), x_ev[:, :a.slice[1]].contiguous()
        g = select_mse_grid_input_global_scale([x_fit])
        xq4 = nvfp4_activation_qdq_served(x_ev, g).float()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        del x_fit, xa
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            if a.slice:
                w = w[:a.slice[0], :a.slice[1]].contiguous()
            tname = f"L{layer}.{proj}"
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            res = {}
            exl3_rows = {}
            log(f"\n== {tname} {tuple(w.shape)}  git {out['git']}")
            log(f"    {'arm':<46} {'bpp':>6} {'wt':>8} {'out':>8} {'a4':>8} {'a8':>8} {'s':>5} | vs EXL3 out/a4/a8")

            def rec(arm, hat, bpp, secs=0.0, exl3=False):
                r = {"bpp": bpp, "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny), "secs": secs}
                if exl3:
                    exl3_rows[bpp] = r
                ratios = ""
                if exl3_rows and not exl3:
                    rr = {}
                    for leg in ("out", "a4", "a8"):
                        ref = exl3_at(exl3_rows, bpp, leg)
                        rr[leg] = None if ref is None else r[leg] / ref
                    r["vs_exl3"] = rr
                    # As served: this arm's executed leg against EXL3's W4A16 weight leg.
                    ref_out = exl3_at(exl3_rows, bpp, "out")
                    r["served_vs_exl3_w4a16"] = {leg: (None if ref_out is None else r[leg] / ref_out)
                                                 for leg in ("a4", "a8")}
                    ratios = " | " + " ".join(f"{rr[l]:.3f}" if rr[l] else "  -  " for l in ("out", "a4", "a8"))
                res[arm] = r
                log(f"    {arm:<46} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} "
                    f"{r['a4']:8.5f} {r['a8']:8.5f} {secs:5.0f}{ratios}")

            if not a.slice:
                for Kx in a.exl3:
                    p = Path(EXL3) / f"L{layer}_{proj}_K{Kx}.pt"
                    if p.exists():
                        rec(f"EXL3 K={Kx} (LDLQ, W4A16)", torch.load(p, map_location=dev).float(),
                            Kx + 0.011723, exl3=True)
            # The E4M3 floor: per-channel FP8 RTN, amax/448 per row (8 bpp).
            s = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
            hat8 = (w / s).to(torch.float8_e4m3fn).float() * s
            rec("FP8 per-channel RTN (E4M3 floor, W8A8)", hat8, 8.0 + 32 / w.shape[1])

            def wire(arm, **kw):
                t0 = time.time()
                unit = encode_linear(w, name=tname, scale_refit=a.refit, **kw)
                hat = read_unit_artifact(unit.blob, device=dev)
                rec(arm, hat, 8 * len(unit.blob) / w.numel(), time.time() - t0)

            for label, gname, q, body, plane in arms:
                grid = GRIDS[gname]
                if body is TCQ:
                    wire(label, grid=grid, q256=q, body=TCQ, scale_plane=plane, span=2)
                else:
                    for L in a.window_bits:
                        wire(f"{label} L={L}", grid=grid, q256=q, body=WINDOW,
                             window_bits=L, scale_plane=plane)
            for cname, v in copied.get(tname, {}).items():
                res[cname] = v
            out["experts"][tname] = res
            out_path.write_text(json.dumps(out, indent=1))
            out_path.with_suffix(".log").write_text("\n".join(lines) + "\n")
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()

    # Geometric means over tensors, per arm, with ratios vs EXL3 at matched bpp.
    log("\n== geomeans over tensors")
    names = {}
    for tname, res in out["experts"].items():
        for arm, v in res.items():
            names.setdefault(arm, []).append(v)
    summary = {}
    for arm, lst in names.items():
        n = len(lst)
        g = lambda k: math.exp(sum(math.log(v[k]) for v in lst) / n)
        row = {"n": n, "bpp": sum(v["bpp"] for v in lst) / n,
               "wt": g("wt"), "out": g("out"), "a4": g("a4"), "a8": g("a8")}
        if all("vs_exl3" in v for v in lst):
            row["vs_exl3"] = {leg: math.exp(sum(math.log(v["vs_exl3"][leg]) for v in lst) / n)
                              for leg in ("out", "a4", "a8") if all(v["vs_exl3"].get(leg) for v in lst)}
        summary[arm] = row
        rr = row.get("vs_exl3", {})
        log(f"    {arm:<46} {row['bpp']:6.3f} {row['wt']:8.5f} {row['out']:8.5f} "
            f"{row['a4']:8.5f} {row['a8']:8.5f} n={n}"
            + (" | " + " ".join(f"{rr[l]:.3f}" for l in ("out", "a4", "a8") if l in rr) if rr else ""))
    out["summary"] = summary
    out_path.write_text(json.dumps(out, indent=1))
    out_path.with_suffix(".log").write_text("\n".join(lines) + "\n")
    log("\nDONE")


if __name__ == "__main__":
    main()
