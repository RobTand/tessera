"""The window body against the shipping TCQ body on the TRUE wire.

Every arm here is ``encode_linear`` -> bytes -> ``read_unit_artifact``: the
exporter's own settings (span 2, LUT16 plane, refit 4, scale-weighted trellis)
for the TCQ body, and the same plane and refit for the window body, priced at
the bytes actually written (manifest, table and all).  ``tessera_bitshift_*``
measured the construction under an experiment's planes; this measures the
wire, mixed-rate schedules over one table included, on the six GLM-5.3-Flash
expert tensors against the same held-out rows.

Matched bytes: the span-2 label is one bit per super-symbol of two
positions, so on the pair grid E2M1x2 the TCQ wire spends ``q/256 + 0.25``
(label) ``+ 0.25`` (LUT16 nibble) bits per weight and on the arity-1 E4M3 grid
``q/256 + 0.5 + 0.25``; the window body spends ``q/256 + 0.25`` plus its
``2^L``-byte table.  A TCQ rung ``q`` is therefore compared with window rung
``q + 64`` on E2M1x2 and ``q + 128`` on E4M3, at 3.0/3.5/4.0 and 4.0/5.0 bpp.

    PYTHONPATH=src:experiments:/home/rob/prismaquant python experiments/tessera_window_wire.py \
        --layers 5 20 42 --window-bits 12
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid          # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian          # noqa: E402
from tessera.export import DEFAULT_SCALE_REFIT, encode_linear          # noqa: E402
from tessera.manifest import BodyKind                                  # noqa: E402
from tessera.unit_artifact import read_unit_artifact                   # noqa: E402
from tessera8_targets import ACT, EXL3, SRC                            # noqa: E402

GRIDS = {"E2M1x2": tuple_grid(E2M1_GRID, 2), "E4M3": E4M3_GRID}
# TCQ rung -> window rung at the same bytes (the span-2 label's 0.25 b/wt).
RUNGS = {"E2M1x2": [(640, 704), (768, 832), (896, 960)],
         "E4M3": [(832, 960), (1088, 1216)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--grids", nargs="+", default=["E2M1x2", "E4M3"])
    ap.add_argument("--window-bits", type=int, nargs="+", default=[12])
    ap.add_argument("--rungs-json", default=None, help="override RUNGS as JSON")
    ap.add_argument("--refit", type=int, default=DEFAULT_SCALE_REFIT)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--slice", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    ap.add_argument("--no-tcq", action="store_true")
    ap.add_argument("--ldlq-sigma", type=float, nargs="*", default=None,
                    help="also run each window arm with LDLQ at these Hessian regularisers; "
                         "H is x_fit^T x_fit, the same fit rows the plane and the NVFP4 input "
                         "scale are fit on, and the score stays the held-out eval rows")
    ap.add_argument("--ldlq-block", type=int, default=128)
    ap.add_argument("--refit-metric", default=None,
                    help="run the window arms again with the row-scale refit under this error: "
                         "hessian | h^ALPHA")
    ap.add_argument("--out", default="experiments/results/tessera_window_wire.json")
    a = ap.parse_args()
    rungs = json.loads(a.rungs_json) if a.rungs_json else RUNGS

    from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

    out_path = Path(a.out)
    out = {"args": vars(a), "experts": {}}
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

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
        # The Hessian for the activation-aware encoder arms, from the SAME fit
        # rows every other fitted quantity here uses; the score below is the
        # held-out tail, so no arm is graded on rows it was fit on.
        H = None
        if a.ldlq_sigma or a.refit_metric:
            H = (x_fit.double().T @ x_fit.double()).float() / x_fit.shape[0]
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
            log(f"\n== {tname} {tuple(w.shape)}")
            log(f"    {'arm':<52} {'bpp':>6} {'wt':>8} {'out':>8} {'a4':>8} {'a8':>8} {'s':>6}")

            def rec(arm, hat, bpp, secs=0.0):
                r = {"bpp": bpp, "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny), "secs": secs}
                res[arm] = r
                log(f"    {arm:<52} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} "
                    f"{r['a4']:8.5f} {r['a8']:8.5f} {secs:6.0f}")

            if not a.slice:
                for Kx in a.exl3:
                    p = Path(EXL3) / f"L{layer}_{proj}_K{Kx}.pt"
                    if p.exists():
                        rec(f"EXL3 K={Kx} (LDLQ, W4A16)", torch.load(p, map_location=dev).float(),
                            Kx + 0.011723)

            def wire(arm, **kw):
                t0 = time.time()
                unit = encode_linear(w, name=tname, scale_refit=a.refit, **kw)
                hat = read_unit_artifact(unit.blob, device=dev)
                rec(arm, hat, 8 * len(unit.blob) / w.numel(), time.time() - t0)

            for gname in a.grids:
                grid = GRIDS[gname]
                for q_tcq, q_win in rungs[gname]:
                    if not a.no_tcq:
                        wire(f"{gname} TCQ q{q_tcq} (exporter default)", grid=grid, q256=q_tcq)
                    for L in a.window_bits:
                        wire(f"{gname} window q{q_win} L={L}", grid=grid, q256=q_win,
                             body=BodyKind.WINDOW, window_bits=L)
                        if H is None or H.shape[0] != w.shape[1]:
                            continue
                        metric = None
                        if a.refit_metric == "hessian":
                            metric = H
                        elif a.refit_metric:
                            hd = H.diagonal()
                            metric = (hd / hd.mean()).pow(float(a.refit_metric.removeprefix("h^")))
                        # Each lever alone and the two together, so the cross
                        # term is visible rather than assumed.
                        combos = [(sigma, None) for sigma in (a.ldlq_sigma or [])]
                        if metric is not None:
                            combos.append((None, metric))
                            combos += [(sigma, metric) for sigma in (a.ldlq_sigma or [])]
                        for sigma, m in combos:
                            ldl = None
                            if sigma is not None:
                                ldl = block_ldl(regularize_hessian(H, sigma_reg=sigma), a.ldlq_block)
                            tag = ("" if sigma is None else f" LDLQ{sigma}") + \
                                  ("" if m is None else f" refit-{a.refit_metric}")
                            wire(f"{gname} window q{q_win} L={L}{tag}", grid=grid, q256=q_win,
                                 body=BodyKind.WINDOW, window_bits=L, ldl=ldl,
                                 ldl_block=a.ldlq_block, refit_metric=m)
            out["experts"][tname] = res
            out_path.write_text(json.dumps(out, indent=1))
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()
    log("\nDONE")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
