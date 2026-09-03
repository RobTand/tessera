#!/usr/bin/env python
"""Issue #50's ceiling: what the LUT plane's landing costs, and how much of it
any sixteen-entry table could ever get back.

`#35` instrumented the metric-aware LUT refit (`refit_diagnostics`) and found
the *landing* -- `_fit_lut`'s separable model plus nearest-in-linear assignment
into sixteen E4M3 entries -- taking back 24-91% of whatever step it is handed,
several times more than the Jacobi->Gauss-Seidel step fix was worth.  Issue #50
asks for the ceiling before any optimiser is built: run the refit with the
landing *disabled* and read the six-unit geomean.

This measures that ceiling **post hoc at fixed codes**, which needs no encoder
change and is exact rather than iterative-in-the-encoder:

1. Encode each unit exactly as the shipping arm does (`encode_linear_planes`).
2. Recover, from the *stock tensors the runtime would read*, the two factors
   the reconstruction is a product of: the per-position code value `U`
   (E2M1 nibble -> value) and the per-16 block scale `S` (E4M3 byte x global).
   `S * U == stock_dequant(...)` exactly, and the run asserts it.
3. Re-solve the plane `S` at those fixed codes, under **the same objective the
   arm's own refit minimises** (full H, or the diagonal `h^alpha`), under four
   successively weaker constraints.  Every step is an exact coordinate move on
   a convex quadratic, so every arm below is monotone and the `free` arm is the
   global continuous optimum, not a step towards it.

| arm | the plane may be | answers |
|---|---|---|
| `landed` | what the encoder shipped | the reference |
| `free` | any real number per block | **the ceiling** -- issue #50's question |
| `free-e4m3` | any in-range E4M3 value per block | the ceiling a *scale byte* allows (no table at all, 8 bits/block) |
| `oracle-assign` | one of the encoder's own 16 entries, chosen on the true objective | the *assignment* half of the landing, at zero byte cost |
| `oracle-table` | 16 E4M3 entries and an assignment, both chosen on the true objective | the whole prize a perfect table fit could win, at the same bytes |

`oracle-table` is an oracle, not an encoder path: it runs at fixed codes, after
the alternation, and nothing in `src/` calls it.  It exists to size the prize
the way issue `#4`'s oracle did, before anyone builds the optimiser.

The score that decides is out-space on held-out rows, the same one the `#35`
receipt used.  `hfit` -- the fit-row quadratic the refit is provably monotone
in -- is carried beside it, because these two arms move in opposite directions
whenever the plane overfits the fit rows and a receipt quoting one alone is
choosing its answer.

    PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      python experiments/lut_landing_ceiling.py --units ... --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import SERIALISABLE_GRIDS                     # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian        # noqa: E402
from tessera.encode import (                                        # noqa: E402
    E4M3_NORMAL_BYTES, LUT_ENTRIES, e4m3_positive_values, refit_diagnostics)
from tessera.export import (                                        # noqa: E402
    DEFAULT_CODE, encode_linear_planes, wire_recipe)
from tessera.manifest import ScalePlaneKind                         # noqa: E402
from tessera.stock import (                                         # noqa: E402
    _nvfp4_values, materialize_stock, stock_dequant)


# ---------------------------------------------------------------- the metric

class Metric:
    """The refit's own error, as an operator -- full H or a diagonal `h`.

    `cost(E) = sum (E M) . E`.  For the full Hessian that is `E H E^T`, whose
    square root over `W H W^T` is exactly the `hfit` column; for a 1-D metric
    it is the diagonal-weighted squared error.  The oracles below only ever
    need `mul`, the per-block curvature, and a panel update, so one class
    covers both and no arm silently optimises a different error than the
    encoder it is being compared to.
    """

    def __init__(self, metric: torch.Tensor):
        self.ndim = int(metric.ndim)
        if self.ndim == 1:
            self.h = metric
        else:
            self.H = metric

    def mul(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.h if self.ndim == 1 else X @ self.H

    def cost(self, E: torch.Tensor) -> float:
        return float((self.mul(E) * E).sum())

    def block_curvature(self, Ub: torch.Tensor, half: int) -> torch.Tensor:
        """`A[r, b] = u_rb M u_rb^T` -- the diagonal 16x16 blocks of M only."""
        rows, nb, _ = Ub.shape
        if self.ndim == 1:
            return (Ub * Ub * self.h.reshape(1, nb, half)).sum(dim=2)
        cols = nb * half
        Hd = torch.diagonal(self.H.reshape(nb, half, nb, half), dim1=0, dim2=2).permute(2, 0, 1)
        return torch.einsum("rbi,bij,rbj->rb", Ub, Hd, Ub)

    def panel_update(self, G: torch.Tensor, delta: torch.Tensor, Ubb: torch.Tensor,
                     lo: int, hi: int) -> None:
        """`G -= M (delta * u_b)` in place -- the block's move, through M."""
        if self.ndim == 1:
            G[:, lo:hi] -= delta.unsqueeze(1) * Ubb * self.h[lo:hi]
        else:
            G.sub_((delta.unsqueeze(1) * Ubb) @ self.H[lo:hi, :])


# ------------------------------------------------------------- the oracles

def _grad(W: torch.Tensor, U: torch.Tensor, S: torch.Tensor, M: Metric, half: int):
    E = W - S.repeat_interleave(half, dim=1) * U
    return M.mul(E), M.cost(E)


def solve_plane_exact(W, U, S0, M, half, chunk=128):
    """The continuous optimum in closed form, per row -- no iteration at all.

    Rows are independent under the metric, so each row's plane is the solution
    of `M_r c = v_r` with `M_r[b,d] = u_rb M u_rd^T` and `v_r[b] = u_rb M w_r`.
    A block whose codes are all zero has a null row and column; it keeps the
    scale it has and leaves the system.  This exists to certify the coordinate
    descent beside it: a ceiling claimed from an iteration that had not
    converged would understate the prize, and the two agreeing to five
    decimals is what rules that out.
    """
    rows, nb = S0.shape
    cols = nb * half
    S = S0.clone()
    A = M.block_curvature(U.reshape(rows, nb, half), half)
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        Ub = U[r0:r1].reshape(r1 - r0, nb, half)
        if M.ndim == 1:
            # Separable: the system is diagonal, so the solve IS the per-block
            # closed form.  Written out rather than special-cased away, so the
            # two metrics travel the same path.
            Bv = (W[r0:r1].reshape(r1 - r0, nb, half) * Ub
                  * M.h.reshape(1, nb, half)).sum(dim=2)
            a = A[r0:r1]
            S[r0:r1] = torch.where(a > 0, Bv / a.clamp_min(1e-30), S[r0:r1])
            continue
        Hb = M.H.reshape(nb, half, cols)
        P = torch.einsum("rbi,bic->rbc", Ub, Hb)                 # [chunk, nb, cols]
        Mm = torch.einsum("rbdi,rdi->rbd", P.reshape(r1 - r0, nb, nb, half), Ub)
        v = torch.einsum("rbc,rc->rb", P, W[r0:r1])
        dead = A[r0:r1] <= 0
        if bool(dead.any()):
            # A block with all-zero codes contributes nothing to the
            # reconstruction, so its scale is unidentified: take it out of the
            # system (zero row and column, unit diagonal) and hand it back the
            # scale the encoder shipped, which is what the refit does too.
            Mm = torch.where(dead.unsqueeze(1) | dead.unsqueeze(2),
                             torch.zeros_like(Mm), Mm)
            Mm = Mm + torch.diag_embed(dead.to(W.dtype))
            v = torch.where(dead, S0[r0:r1], v)
        try:
            S[r0:r1] = torch.linalg.solve(Mm, v.unsqueeze(2)).squeeze(2)
        except Exception:
            # A rank-deficient metric leaves null directions in the plane; the
            # minimum-norm solution is still a minimiser of the objective.
            S[r0:r1] = torch.linalg.lstsq(Mm, v.unsqueeze(2)).solution.squeeze(2)
        del P, Mm, v
    E = W - S.repeat_interleave(half, dim=1) * U
    return S, M.cost(E)


def solve_plane(W, U, S0, M, half, cand=None, sweeps=400, tol=1e-13, revert=False):
    """Exact block-coordinate descent on the plane at fixed codes.

    `cand is None` -- each block takes its exact continuous minimiser given
    the others, which is the same closed form the encoder's refit uses
    (`s + (G . u) / A`); sweeping it to convergence on a convex quadratic
    reaches the global continuous optimum.  `cand` given -- each block takes
    whichever of those values minimises the true quadratic, which is the
    curvature-weighted, cross-block-aware assignment issue #50 asks about
    (`Delta = -2 d g + d^2 A` is exact for a single block's move).

    `revert` holds the oracle to the encoder's own refusal: a block whose
    coordinate optimum is a **non-positive** scale keeps the scale it has
    (`_refit_scales_lut`'s `valid`, and the measured reason for it -- the
    trellis runs on `work / scale`, so handing a half a collapsed scale forces
    the next pass to spend its shared path on that half's enormous normalised
    residual, and the alternation stops being monotone in true SSE).  Without
    this the oracle is free to collapse exactly the blocks the encoder refuses
    to, which on `L2.mlp.down_proj` -- the one unit of the six with any reverts
    at all -- is most of its apparent prize.  The mask is recomputed at the top
    of every sweep, which is where the refit recomputes it too.

    Returns `(S, cost, sweeps_used)`.  Monotone by construction: every move is
    the minimiser of the true objective along one coordinate.
    """
    rows, nb = S0.shape
    Ub = U.reshape(rows, nb, half)
    A = M.block_curvature(Ub, half)
    S = S0.clone()
    G, cost = _grad(W, U, S, M, half)
    used = 0
    for sweep in range(sweeps):
        before = cost
        if revert:
            gs = ((G.reshape(rows, nb, half) * Ub).sum(dim=2))
            pin = ~((A > 0) & (S + gs / A.clamp_min(1e-30) > 0))
            del gs
        for b in range(nb):
            lo, hi = b * half, (b + 1) * half
            Ubb = Ub[:, b, :]
            g = (G[:, lo:hi] * Ubb).sum(dim=1)
            a = A[:, b]
            live = (a > 0) if not revert else ((a > 0) & ~pin[:, b])
            if cand is None:
                d = torch.where(live, g / a.clamp_min(1e-30), torch.zeros_like(g))
            else:
                dc = cand.reshape(1, -1) - S[:, b:b + 1]              # [rows, ncand]
                delta_cost = -2.0 * dc * g.unsqueeze(1) + dc * dc * a.unsqueeze(1)
                pick = delta_cost.argmin(dim=1)
                d = torch.where(live, dc.gather(1, pick.unsqueeze(1)).squeeze(1),
                                torch.zeros_like(g))
                d = torch.where(delta_cost.gather(1, pick.unsqueeze(1)).squeeze(1) < 0.0,
                                d, torch.zeros_like(d))
            if not bool((d != 0).any()):
                continue
            S[:, b] = S[:, b] + d
            M.panel_update(G, d, Ubb, lo, hi)
        # Recompute from scratch each sweep: the panel updates are exact but
        # accumulate float error over nb of them, and a monotone claim has to
        # be made on a recomputed cost, not a running one.
        G, cost = _grad(W, U, S, M, half)
        used = sweep + 1
        if before - cost <= tol * abs(before):
            break
    return S, cost, used


def entry_pass(W, U, idx, table, M, cand, half):
    """One exact pass over the sixteen table ENTRIES, on the true quadratic.

    Every block assigned to entry `k` moves by the same `d = v - table[k]`, so
    the cost change is exactly `-2 d sum_k g + d^2 (D_k M D_k)` with `D_k` the
    code values at those positions and zero elsewhere -- one quadratic in `d`,
    minimised over the finite E4M3 candidate set.  Cross-block terms *inside*
    the moved set are in `D_k M D_k`; that is the term `_fit_lut`'s separable
    model drops.
    """
    rows, nb = idx.shape
    table = table.clone()
    moved = 0
    for k in range(table.numel()):
        mask = (idx == k)
        if not bool(mask.any()):
            continue
        Dk = U * mask.repeat_interleave(half, dim=1)
        G, _ = _grad(W, U, table[idx], M, half)
        num = float((G * Dk).sum())
        den = float((M.mul(Dk) * Dk).sum())
        if den <= 0.0:
            continue
        dc = cand - table[k]
        delta_cost = -2.0 * dc * num + dc * dc * den
        j = int(delta_cost.argmin())
        if float(delta_cost[j]) < 0.0:
            table[k] = cand[j]
            moved += 1
    return table, moved


def oracle_table(W, U, idx0, table0, M, cand, half, rounds=8, tol=1e-13,
                 floor=None):
    """Alternate exact assignment and exact entry choice, both on the true
    quadratic, to a local optimum of the joint (table, assignment) problem.

    This is the *oracle* for issue #50's proposed fix, not the fix: it runs at
    fixed codes after the alternation, and it is free to spend as much compute
    as it likes.  Its number is the prize an in-encoder optimiser would be
    chasing, so a small number closes the issue the way `#4`'s oracle did.
    """
    table, idx = table0.clone(), idx0.clone()
    S, cost, _ = solve_plane(W, U, table[idx], M, half, cand=table, tol=tol)
    if floor is not None and floor[1] < cost:
        # The assignment oracle's own point is a FEASIBLE point of this larger
        # problem (same sixteen entries, optimal assignment).  Starting below
        # it is the difference between an oracle and a worse search: without
        # this the "bigger" oracle can report a higher cost than the smaller
        # one and the table would read as if re-choosing entries hurt.
        S, cost = floor[0].clone(), floor[1]
        idx = (S.reshape(-1, 1) - table.reshape(1, -1)).abs().argmin(dim=1).reshape(idx.shape)
    idx = (S.reshape(-1, 1) - table.reshape(1, -1)).abs().argmin(dim=1).reshape(idx.shape)
    for r in range(rounds):
        before = cost
        table, moved = entry_pass(W, U, idx, table, M, cand, half)
        S, cost, _ = solve_plane(W, U, table[idx], M, half, cand=table, tol=tol)
        idx = (S.reshape(-1, 1) - table.reshape(1, -1)).abs().argmin(dim=1).reshape(idx.shape)
        if moved == 0 or before - cost <= tol * abs(before):
            break
    return table[idx], cost, r + 1


# ------------------------------------------------------------------- driver

def rel(num: torch.Tensor, den: torch.Tensor) -> float:
    return float(num.norm() / den.norm())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--h", default="/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts", default="/mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--no-oracle-table", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["control", "jacobi", "gs"],
                    choices=["control", "jacobi", "gs"],
                    help="which encoder arms to measure the landing of.  The control is the "
                         "SERVED default objective and is always the drift control, so it runs "
                         "first and last whatever else is asked for.")
    ap.add_argument("--free-cd-sweeps", type=int, default=400,
                    help="cap on the continuous arm's coordinate descent.  The reported ceiling "
                         "is the lower of that and the exact per-row solve, so a low cap costs "
                         "nothing but the CD's value as an independent check -- which is why the "
                         "certification table is read off the Qwen run, where the cap is not hit.")
    ap.add_argument("--oracle-table-starts", type=int, default=2, choices=[1, 2],
                    help="1: the encoder's own table only.  2: also a table fit to the "
                         "continuous optimum, keeping whichever reaches the lower true cost.")
    ap.add_argument("--oracle-table-rounds", type=int, default=8)
    ap.add_argument("--source", default="qwen", choices=["qwen", "glm"],
                    help="qwen: the six dense Qwen3-0.6B units of the #35 receipt, with the "
                         "captured full Hessian.  glm: the six GLM-5.3-Flash expert tensors of "
                         "the LUT plane's own decision gate, with H built from the same fit rows "
                         "the plane is fit on.  A dense-Qwen result does not transfer to experts "
                         "and the reverse is equally false, so the gate needs both.")
    ap.add_argument("--glm-layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--glm-projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="/mnt/shared/tessera-runs/ldlq-lut/lut_landing_ceiling.json")
    a = ap.parse_args()

    grid = next(g for g in SERIALISABLE_GRIDS.values() if g.name == a.grid)
    recipe = wire_recipe(grid, a.q256)
    if recipe.scale_plane is not ScalePlaneKind.LUT:
        raise SystemExit(f"{a.grid} q{a.q256} is a {recipe.scale_plane.name} plane; "
                         "issue #50 is about the LUT plane's landing")
    dev = "cuda"
    if a.source == "qwen":
        payload = torch.load(a.h, map_location="cpu", weights_only=False)
        acts = torch.load(a.acts, map_location="cpu", weights_only=False)
        Hall, prov = payload["H"], payload["provenance"]
        names = a.units or sorted(acts["x"])

        def cases():
            with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
                for name in names:
                    W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
                    yield (name, W, Hall[name].to(dev, torch.float32),
                           acts["x"][name].to(dev, torch.float32))
    else:
        # The GLM six: expert 0 of three layers, two projections each -- the
        # exact set `tessera_window_wire.py` runs, so the landed arm here can
        # be checked against the published LUT-plane receipt.  H and the score
        # rows come from one activation capture split fit/eval, the same split
        # that receipt used, so no arm is graded on rows it was fit on.
        # `tessera8_targets` owns these two paths, but importing it drags in
        # `prismaquant` for activation quantisers this run does not use.  Read
        # the assignments instead of executing the module: one source of truth
        # for the paths, no dependency this measurement does not have.
        import ast as _ast
        _src = (Path(__file__).resolve().parent / "tessera8_targets.py").read_text()
        _consts = {t.id: n.value.value
                   for n in _ast.parse(_src).body if isinstance(n, _ast.Assign)
                   for t in n.targets
                   if isinstance(t, _ast.Name) and isinstance(n.value, _ast.Constant)}
        ACT, SRC = _consts["ACT"], _consts["SRC"]
        index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
        prov = {"source": f"{ACT} (expert inputs)", "fit_tokens": "capture - eval_rows",
                "eval_tokens": a.eval_rows}

        def cases():
            for layer in a.glm_layers:
                blob = torch.load(
                    f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                    map_location="cpu", weights_only=False)
                xa = blob["inputs"].float()
                n_fit = xa.shape[0] - a.eval_rows
                x_fit = xa[:n_fit].contiguous().to(dev)
                x_ev = xa[n_fit:].contiguous().to(dev)
                H = (x_fit.double().T @ x_fit.double()).float() / x_fit.shape[0]
                del x_fit, xa
                for proj in a.glm_projs:
                    name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
                    with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                        W = f.get_tensor(name).contiguous().to(dev).float()
                    yield (f"L{layer}.{proj}", W, H, x_ev)
                del H, x_ev
                torch.cuda.empty_cache()
    out = {"args": vars(a), "provenance": prov, "units": {}}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"wire: {grid.name} q256={a.q256} -> body {recipe.body.name} plane "
        f"{recipe.scale_plane.name} span {recipe.span} L={recipe.window_bits}")
    log(f"H from {prov['source']}  fit {prov['fit_tokens']} tok  eval {prov['eval_tokens']} tok")

    if True:
        for name, W, H, X in cases():
            Y = X @ W.T
            h = H.diagonal().clone()
            hn = h / h.mean()
            den_w, den_h = W.norm(), float(((W * W).sum(0) * hn).sum())
            den_hf = float(((W @ H) * W).sum())
            rows, cols = W.shape
            half = 16
            nb = cols // half
            res: dict = {}
            log(f"\n== {name} {tuple(W.shape)}  eval rows {X.shape[0]}  blocks/row {nb}")
            log(f"    {'arm':<52} {'out':>9} {'plain':>9} {'hwt':>9} {'hfit':>9} {'s':>6}")

            def score(arm, What, secs=0.0, extra=None):
                E = What - W
                r = {"out": rel(X @ E.T, Y), "plain": float(E.norm() / den_w),
                     "hweighted": math.sqrt(float(((E * E).sum(0) * hn).sum()) / den_h),
                     "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf), "secs": secs}
                if extra:
                    r.update(extra)
                res[arm] = r
                log(f"    {arm:<52} {r['out']:9.5f} {r['plain']:9.5f} "
                    f"{r['hweighted']:9.5f} {r['hfit']:9.5f} {secs:6.1f}")
                return r

            Lf = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
            hmet = hn.pow(a.alpha)
            arms = {
                f"control [LDLQ {a.sigma}/{a.block} + refit h^{a.alpha}]":
                    (dict(ldl=Lf, ldl_block=a.block, refit_metric=hmet), hmet),
                f"LDLQ {a.sigma}/{a.block} + refit full-H (Jacobi)":
                    (dict(ldl=Lf, ldl_block=a.block, refit_metric=H), H),
                f"LDLQ {a.sigma}/{a.block} + refit full-H (Gauss-Seidel)":
                    (dict(ldl=Lf, ldl_block=a.block, refit_metric=H, refit_gauss_seidel=True), H),
            }
            want = {"control": list(arms)[0], "jacobi": list(arms)[1], "gs": list(arms)[2]}
            keep = [want[k] for k in ("control", "jacobi", "gs") if k in a.arms]
            if want["control"] not in keep:
                keep.insert(0, want["control"])
            arms = {k: v for k, v in arms.items() if k in keep}
            order = keep + [keep[0] + " REPEAT"]

            for label in order:
                kw, metric = arms[label.removesuffix(" REPEAT")]
                t0 = time.time()
                with refit_diagnostics() as diag:
                    _, unit, forests = encode_linear_planes(
                        W, grid=grid, q256=a.q256, name=name, verify=False, **kw)
                secs = time.time() - t0
                st = materialize_stock(unit, forests, DEFAULT_CODE)
                What = stock_dequant(st).to(dev).float()

                # The two factors, from the tensors the runtime reads.  U is
                # the E2M1 nibble's value; S is the E4M3 scale byte times the
                # global.  A product identity, asserted, not assumed: if this
                # were off by a factor the oracles below would be solving a
                # different problem and every ratio would be fiction.
                packed = st["weight_packed"]
                nib = torch.empty(rows, cols, dtype=torch.uint8, device=packed.device)
                nib[:, 0::2] = packed & 0xF
                nib[:, 1::2] = packed >> 4
                U = _nvfp4_values(nib).to(dev)
                gl = 1.0 / float(st["weight_global_scale"].reshape(-1)[0])
                S = (st["weight_scale"].float() * gl).to(dev)
                if not torch.equal((S.repeat_interleave(half, dim=1) * U), What):
                    raise SystemExit(f"{name} {label}: S * U != stock_dequant -- "
                                     "the factor split is wrong, every oracle below is void")
                sha = hashlib.sha256(What.cpu().numpy().tobytes()).hexdigest()

                base = score(label, What, secs, extra={
                    "sha256": sha,
                    "refit_landed": [d["landed"] for d in diag],
                    "refit_continuous": [d["continuous"] for d in diag],
                    "refit_before": [d["before"] for d in diag],
                    "refit_stepped": [d["stepped"] for d in diag],
                    "refit_reverted": [d["reverted"] for d in diag],
                })
                if label.endswith(" REPEAT"):
                    first = res[label.removesuffix(" REPEAT")]
                    log(f"    -- drift control: bytes "
                        f"{'IDENTICAL' if sha == first['sha256'] else 'DIFFER'}  "
                        f"out {first['out']:.6f} -> {base['out']:.6f} "
                        f"({base['out'] / first['out'] - 1.0:+.4%})")
                    res["_drift"] = {"bytes_identical": sha == first["sha256"],
                                     "out_first": first["out"], "out_last": base["out"]}
                    continue

                M = Metric(metric)
                table = unit.scale_lut.to(dev).view(torch.float8_e4m3fn).float() * \
                    float(unit.scale_global)
                idx = (S.reshape(-1, 1) - table.reshape(1, -1)).abs().argmin(dim=1).reshape(rows, nb)
                if not torch.equal(table[idx], S):
                    raise SystemExit(f"{name} {label}: the table does not reproduce the plane")
                # The E4M3 grid the wire can store a scale byte on, times the
                # unit's own global -- the candidate set for both the
                # unrestricted-byte arm and the entry search.  In-range only:
                # a candidate the encoder's own `_fit_lut` could not have
                # chosen would make the oracle answer a different question.
                gv = e4m3_positive_values(dev) * float(unit.scale_global)
                Sf0, _ = solve_plane_exact(W, U, S, M, half)
                span_lo = min(float(S.min()), float(Sf0[Sf0 > 0].min()) if bool((Sf0 > 0).any())
                              else float(S.min()))
                span_hi = max(float(S.max()), float(Sf0.max()))
                lo_i = max(int((gv < span_lo).sum()) - 1, 0)
                hi_i = min(int((gv <= span_hi).sum()) + 1, gv.numel())
                cand = gv[lo_i:hi_i].contiguous()
                del Sf0

                t0 = time.time()
                Sf, c_cd, nf = solve_plane(W, U, S, M, half, cand=None,
                                           sweeps=a.free_cd_sweeps)
                Sx, c_ex = solve_plane_exact(W, U, S, M, half)
                cf = c_cd
                if c_ex < cf:
                    Sf, cf = Sx, c_ex
                score(f"  free plane (continuous)  [{label}]",
                      Sf.repeat_interleave(half, dim=1) * U, time.time() - t0,
                      extra={"sweeps": nf, "fit_cost": cf, "fit_cost_cd": c_cd,
                             "fit_cost_exact": c_ex,
                             "exact_over_cd": c_ex / c_cd if c_cd else float("nan"),
                             "negatives": int((Sf <= 0).sum())})
                del Sx

                # A discrete coordinate descent finds a LOCAL optimum, so a
                # bound read off one start is not a bound.  Every restricted
                # arm below runs from the landed plane AND from the continuous
                # optimum snapped into its own candidate set, and keeps the
                # lower true cost.
                def restricted(tag, cset, extra=None, revert=False):
                    t = time.time()
                    best = None
                    for start in (S, cset[(Sf.reshape(-1, 1) - cset.reshape(1, -1))
                                          .abs().argmin(dim=1)].reshape(S.shape)):
                        Sc, cc, nc = solve_plane(W, U, start, M, half, cand=cset,
                                                 revert=revert)
                        if best is None or cc < best[1]:
                            best = (Sc, cc, nc)
                    Sc, cc, nc = best
                    score(f"  {tag:<24} [{label}]",
                          Sc.repeat_interleave(half, dim=1) * U, time.time() - t,
                          extra={"sweeps": nc, "fit_cost": cc, **(extra or {})})
                    return Sc, cc

                restricted("free per-block E4M3", cand, {"candidates": int(cand.numel())})
                Sa, ca = restricted("oracle assign (own 16)", table)
                # The same three, under the encoder's non-positive-target
                # refusal.  On five of the six Qwen units the refit reverts
                # nothing and these are the same arms; on the sixth they are
                # the honest ones.
                t0 = time.time()
                Sr, cr, nr = solve_plane(W, U, S, M, half, cand=None, revert=True,
                                         sweeps=a.free_cd_sweeps)
                score(f"  free plane, revert rule  [{label}]",
                      Sr.repeat_interleave(half, dim=1) * U, time.time() - t0,
                      extra={"sweeps": nr, "fit_cost": cr,
                             "pinned": int((Sr == S).all(dim=0).sum())})
                restricted("assign, revert rule", table, revert=True)

                if not a.no_oracle_table:
                    t0 = time.time()
                    best = None
                    starts = [table] if a.oracle_table_starts == 1 else [
                        table, cand[(Sf.reshape(-1, 1) - cand.reshape(1, -1))
                                    .abs().argmin(dim=1)].reshape(S.shape)]
                    for t0_table in starts:
                        if t0_table.ndim == 2:      # a full plane -> fit a table to it
                            from tessera.encode import _fit_lut
                            Ab = M.block_curvature(U.reshape(rows, nb, half), half)
                            tb, tv = _fit_lut(t0_table.reshape(-1), Ab.reshape(-1),
                                              float(unit.scale_global), LUT_ENTRIES)
                            t0_table = tv
                        i0 = (S.reshape(-1, 1) - t0_table.reshape(1, -1)).abs() \
                            .argmin(dim=1).reshape(rows, nb)
                        St, ct, nt = oracle_table(W, U, i0, t0_table, M, cand, half,
                                                  rounds=a.oracle_table_rounds,
                                                  floor=(Sa, ca))
                        if best is None or ct < best[1]:
                            best = (St, ct, nt)
                    St, ct, nt = best
                    score(f"  oracle table+assign (16) [{label}]",
                          St.repeat_interleave(half, dim=1) * U, time.time() - t0,
                          extra={"rounds": nt, "fit_cost": ct})

            out["units"][name] = res
            Path(a.out).write_text(json.dumps(out, indent=1))
            del W, X, Y, Lf
            torch.cuda.empty_cache()

    arms_all = set.intersection(*[{k for k in v if not k.startswith("_")}
                                  for v in out["units"].values()])

    def geo(arm, field):
        return math.exp(sum(math.log(out["units"][u][arm][field]) for u in out["units"])
                        / len(out["units"]))

    log("\n== geomean over units -- out-space (held-out) and hfit (the fit rows' own "
        "quadratic, the one the refit is monotone in)")
    log(f"    {'arm':<58} {'out':>9} {'hfit':>9}")
    rows_g = sorted(((arm, geo(arm, "out"), geo(arm, "hfit")) for arm in arms_all),
                    key=lambda r: r[0])
    for arm, go, gh in rows_g:
        log(f"    {arm:<58} {go:9.5f} {gh:9.5f}")
    out["geomean_out"] = {arm: go for arm, go, _ in rows_g}
    out["geomean_hfit"] = {arm: gh for arm, _, gh in rows_g}

    log("\n== the ceiling, per encoder arm: what the landing costs and what is reachable")
    ceil = {}
    for label in [k for k in arms_all if not k.startswith("  ")]:
        base_o, base_h = geo(label, "out"), geo(label, "hfit")
        row = {"landed_out": base_o, "landed_hfit": base_h}
        for tag, pretty in (("free plane (continuous)", "free"),
                            ("free per-block E4M3", "free-e4m3"),
                            ("oracle assign (own 16)", "oracle-assign"),
                            ("oracle table+assign (16)", "oracle-table")):
            arm = f"  {tag:<24} [{label}]"
            if arm in arms_all:
                row[pretty] = {"out": geo(arm, "out"), "out_x": geo(arm, "out") / base_o,
                               "hfit": geo(arm, "hfit"), "hfit_x": geo(arm, "hfit") / base_h}
        ceil[label] = row
        log(f"\n    {label}")
        log(f"      landed                     out {base_o:.5f}          hfit {base_h:.5f}")
        for pretty in ("free", "free-e4m3", "oracle-assign", "oracle-table"):
            if pretty in row:
                r = row[pretty]
                log(f"      {pretty:<24}   out {r['out']:.5f} ({r['out_x']:.4f}x)  "
                    f"hfit {r['hfit']:.5f} ({r['hfit_x']:.4f}x)")
    out["ceiling"] = ceil

    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
