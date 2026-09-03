#!/usr/bin/env python
"""Issue #50: how much of the LUT refit's landing loss can a sixteen-entry
table get back -- a matched pair on the landing alone, at the encoder's own
states, plus the end-state ladder up to the continuous ceiling.

`#35`'s instrument split each metric-aware LUT refit three ways and found the
*landing* -- `_fit_lut`'s separable model plus nearest-in-linear assignment
into sixteen E4M3 entries -- taking back 24-91% of the step it was handed.
The issue reads that as recoverable: the fit drops every cross-block term of
the true quadratic, so a fit that keeps them should return most of it.  This
script sizes the prize before anyone builds the optimiser.

**The matched pair.**  Every metric-aware refit call the encoder makes is
captured (its inputs: codes, plane, table, metric -- the state the encoder was
actually in) and replayed offline twice on identical inputs:

* the encoder's own refit -- step, line search, ``_fit_lut``, nearest-in-linear
  (this reproduces the sink's ``landed`` exactly, and the run checks it); and
* the same refit with the landing made **cross-block aware**: starting from the
  plane the encoder landed on, each block is re-assigned to the table entry
  minimising the TRUE quadratic given every other block (ICM, gradient field
  carried block to block exactly as the Gauss-Seidel step carries it), then
  each of the sixteen entries is moved on the E4M3 grid by an exact coordinate
  step whose parabola keeps every cross-block term, alternated to convergence.

Same inputs, same continuous target, same sixteen-entry budget, same grid, same
global: the landing is the only treatment.  Every oracle step is provably
monotone in the fit quadratic and the run asserts it after every sweep.  The
recoverable fraction of a pass's landing loss is ``(landed - coupled) /
(landed - continuous)`` -- the issue's own frame -- and is also given against
the exact per-row joint minimiser (``free``), which is the true ceiling: the
sink's ``continuous`` is one line-searched step, not the minimiser.

**The end state.**  On E2M1x2 there is no release, so the final plane IS the
trailing refit's landed output (the sink's trailing ``landed`` equals
``hfit^2 * W H W^T`` to seven digits on all twelve rows of `qwen_lut_gs.json`).
So the trailing pass's ladder -- landed, coupled-assign, coupled-table,
free-e4m3 (any E4M3 per block, 8 bits), free (any real) -- is scored on
``out`` and ``hfit`` like any arm, and the two wire-representable rungs are
re-materialised through ``materialize_stock``/``stock_dequant`` and compared
``torch.equal``, so "same bytes" is proved rather than assumed.

**Pre-registered bar, written before the first run.**  The encoder-side
coupled landing gets built only if the trailing-pass ``coupled-table`` beats
``landed`` on the Gauss-Seidel arm's six-unit ``out`` geomean by more than
**1.38%** -- the margin `#35`'s promotion rule used.  Below it, #50 closes with
the number and no code ships.

**Two things the issue does not ask, carried anyway.**  (1) Under the served
``h^1.0`` default the metric is diagonal, the separable model is exact and
nearest-in-linear is the exact minimiser, so its landing has no cross-block
loss at all -- the replay on the control arm checks that ICM moves nothing.
(2) At the control's final codes, ONE full-Hessian refit of the plane (the
encoder's own, then with the coupled landing) is scored too: if that beats the
Gauss-Seidel arm, the alternation under full H is not where the plane's gain
lives.  Both are labelled for what they are.

**Scored two ways, labelled.**  ``out`` is the held-out activation-space
relative error (the `#35` receipt's deciding column; a screen, not a serve).
``hfit`` is ``sqrt(E H E^T / W H W^T)`` on the fit rows -- the quadratic every
refit here is monotone in.  fp32 throughout; the ``free`` solve in fp64.

Held: six Qwen3-0.6B units, E2M1x2 at `q256=896`, LDLQ 1.0/32, one process,
the served default first and again last, encoder-arm digests checked against
`qwen_lut_gs.json` so these encodes are the receipt's across processes.

    ssh sparklina 'export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache; \
      cd /mnt/shared/ts50-lut-landing && PYTHONPATH=src \
      /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u experiments/lut_landing_oracle.py \
      --out /mnt/shared/tessera-runs/ldlq-lut/qwen_lut_landing_oracle.json'
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tessera.encode as enc                                            # noqa: E402
from tessera.alphabet import SERIALISABLE_GRIDS                       # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian           # noqa: E402
from tessera.decode import decode_codes_mixed, dequantize, unit_scale_field  # noqa: E402
from tessera.encode import (                                           # noqa: E402
    E4M3_NORMAL_BYTES, _lut_values, e4m3_positive_values, refit_diagnostics)
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.manifest import ScalePlaneKind                            # noqa: E402
from tessera.stock import materialize_stock, stock_dequant             # noqa: E402

BAR = 0.0138   # #35's promotion margin: build only past it.  Pre-registered.

# ---- capture every metric-aware refit call's inputs, without touching src/.
_ORIG_REFIT = enc._refit_scales_lut_metric
CAPTURE: "list | None" = None


def _capturing_refit(work, units, half, table_bytes, index, effective, global_scale,
                     metric, gauss_seidel=False, **kw):
    if CAPTURE is not None:
        CAPTURE.append(dict(
            work=work.detach().clone(), units=units.detach().clone(), half=half,
            table_bytes=table_bytes.clone(), index=index.clone(),
            effective=effective.clone(), global_scale=global_scale,
            metric=metric, gauss_seidel=gauss_seidel))
    return _ORIG_REFIT(work, units, half, table_bytes, index, effective, global_scale,
                       metric, gauss_seidel=gauss_seidel, **kw)


enc._refit_scales_lut_metric = _capturing_refit


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Quadratic:
    """``L(C) = sum_r (w_r - C_r U_r) M (w_r - C_r U_r)^T`` over per-block scales,
    ``M`` a full ``[cols, cols]`` Hessian or a ``[cols]`` diagonal.

    ``Ub`` is ``[rows, nb, half]`` -- block ``b``'s unscaled codes -- and every
    method keeps the gradient field ``G = (W - C U) M`` current so a block or
    an entry moves at the cost of its own sixteen columns through ``M``.
    """

    def __init__(self, W, U, metric, half):
        self.W, self.U, self.half = W.float(), U.float(), half
        self.rows, self.cols = W.shape
        self.nb = self.cols // half
        self.Ub = self.U.reshape(self.rows, self.nb, half)
        self.diag = metric.ndim == 1
        self.M = metric.to(self.W.dtype).to(self.W.device)
        if self.diag:
            h = self.M.reshape(1, self.nb, half)
            self.A = (self.Ub * self.Ub * h).sum(dim=2)
        else:
            Hd = torch.diagonal(self.M.reshape(self.nb, half, self.nb, half),
                                dim1=0, dim2=2).permute(2, 0, 1)
            self.A = torch.einsum("rbi,bij,rbj->rb", self.Ub, Hd, self.Ub)

    def apply(self, X):
        return X * self.M if self.diag else X @ self.M

    def field(self, C):
        return C.repeat_interleave(self.half, dim=1)

    def recon(self, C):
        return self.field(C) * self.U

    def cost(self, C) -> float:
        E = self.W - self.recon(C)
        return float((self.apply(E) * E).sum())

    def grad_field(self, C):
        return self.apply(self.W - self.recon(C))

    def _push(self, G, P, lo, hi):
        """``G - P M`` for a ``P`` supported on columns ``lo:hi``."""
        if self.diag:
            G = G.clone()
            G[:, lo:hi] -= P * self.M[lo:hi]
            return G
        return G - P @ self.M[lo:hi, :]

    def icm_sweep(self, C, I, table, G):
        """One sweep: each block to the table entry minimising the true quadratic
        GIVEN every other block where it stands now -- nearest-in-linear to the
        conditional optimum, exact for a parabola -- with ``G`` carrying every
        move already made; blocks with a non-positive conditional optimum are
        held, as the encoder holds them.  Returns ``(C, I, G, moved)``."""
        moved = 0
        half = self.half
        for b in range(self.nb):
            lo, hi = b * half, (b + 1) * half
            Ubb = self.Ub[:, b, :]
            A = self.A[:, b]
            s = C[:, b] + (G[:, lo:hi] * Ubb).sum(dim=1) / A.clamp_min(1e-30)
            j = (s[:, None] - table[None, :]).abs().argmin(dim=1)
            # The encoder's revert rule, kept: a block whose conditional
            # optimum is non-positive holds its scale (``valid`` in
            # ``_refit_scales_lut_metric``), so the pair differs in the
            # coupling alone and never in un-doing the revert leg.
            ok = (A > 0) & (s > 0)
            new = torch.where(ok, table[j], C[:, b])
            j = torch.where(ok, j, I[:, b])
            d = new - C[:, b]
            changed = d != 0
            if bool(changed.any()):
                moved += int(changed.sum())
                G = self._push(G, d.unsqueeze(1) * Ubb, lo, hi)
                C = C.clone(); C[:, b] = new
                I = I.clone(); I[:, b] = j
        return C, I, G, moved

    def entry_pass(self, C, I, table, table_bytes, G, grid_values, grid_bytes, tol):
        """One coordinate pass over the sixteen entries, cross-block terms kept.

        Moving entry ``k`` by ``delta`` moves every block assigned to it, so the
        change in the quadratic is exactly ``Q_k delta^2 + 2 g_k delta`` with
        ``Q_k = sum_r V_rk M V_rk^T`` (``V_rk`` the union of the row's codes on
        the blocks assigned to ``k``) and ``g_k = -sum (G * V_k)``.  The best
        DISTINCT E4M3 value under that parabola is taken exactly."""
        moved = 0
        for k in range(table.numel()):
            mask = (I == k)
            if not bool(mask.any()):
                continue
            Vk = (mask.unsqueeze(2) * self.Ub).reshape(self.rows, self.cols)
            VkM = self.apply(Vk)
            Q = float((VkM * Vk).sum())
            g = -float((G * Vk).sum())
            if Q <= 0:
                continue
            used = torch.isin(grid_bytes, table_bytes) & (grid_bytes != table_bytes[k])
            delta = grid_values - table[k]
            dl = Q * delta * delta + 2.0 * g * delta
            dl = torch.where(used, torch.full_like(dl, float("inf")), dl)
            best = int(dl.argmin())
            if float(dl[best]) < -tol:
                step = float(delta[best])
                C = torch.where(mask, C + step, C)
                G = G - step * VkM
                table = table.clone(); table[k] = grid_values[best]
                table_bytes = table_bytes.clone(); table_bytes[k] = grid_bytes[best]
                moved += 1
        return C, table, table_bytes, G, moved

    def free_solve(self, C0, chunk=256):
        """The exact per-row joint minimiser, ``M_r^{-1} v_r`` in fp64.  A block
        with zero curvature (all-zero codes) has no scale and keeps its own."""
        if self.diag:
            B = (self.W.reshape(self.rows, self.nb, self.half) * self.Ub
                 * self.M.reshape(1, self.nb, self.half)).sum(dim=2)
            return torch.where(self.A > 0, B / self.A.clamp_min(1e-30), C0)
        out = torch.empty_like(C0)
        H64 = self.M.double().reshape(self.nb, self.half, self.cols)
        for r0 in range(0, self.rows, chunk):
            r1 = min(r0 + chunk, self.rows)
            ch = r1 - r0
            Ub = self.Ub[r0:r1].double()                                   # [ch, nb, half]
            # ``Z_r`` is block-structured (block ``b`` lives on its own sixteen
            # columns), so ``Z H`` is one panel product per block and ``Z H Z^T``
            # reads the result back through the same panels.
            ZH = torch.einsum("rbi,bij->rbj", Ub, H64)                    # [ch, nb, cols]
            M = torch.einsum("rbcj,rcj->rbc", ZH.reshape(ch, self.nb, self.nb, self.half), Ub)
            v = (ZH @ self.W[r0:r1].double().unsqueeze(2)).squeeze(2)      # [ch, nb]
            dead = self.A[r0:r1] <= 0
            eye = torch.eye(self.nb, dtype=torch.float64, device=Ub.device)
            M = torch.where(dead.unsqueeze(2) | dead.unsqueeze(1), torch.zeros_like(M), M)
            M = M + dead.double().unsqueeze(2) * eye
            v = torch.where(dead, C0[r0:r1].double(), v)
            out[r0:r1] = torch.linalg.solve(M, v.unsqueeze(2)).squeeze(2).float()
        return out


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--h", default="/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts", default="/mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--reference", default="/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_gs.json",
                    help="#35's sweep JSON; the encoder arms' digests are checked against it")
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grid = grid_by_name(a.grid)
    recipe = wire_recipe(grid, a.q256)
    if recipe.scale_plane is not ScalePlaneKind.LUT:
        raise SystemExit("this is a LUT-plane question")
    payload = torch.load(a.h, map_location="cpu", weights_only=False)
    acts = torch.load(a.acts, map_location="cpu", weights_only=False)
    Hall, prov = payload["H"], payload["provenance"]
    ref = json.load(open(a.reference)) if a.reference and Path(a.reference).exists() else None
    units = a.units or sorted(acts["x"])
    dev = "cuda"
    out = {"args": vars(a), "provenance": prov, "bar": BAR, "units": {}}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"wire: {grid.name} q256={a.q256} -> body {recipe.body.name} plane "
        f"{recipe.scale_plane.name} span {recipe.span}")
    log(f"H from {prov['source']}  fit {prov['fit_tokens']} tok  eval {prov['eval_tokens']} tok")
    log(f"pre-registered bar: trailing coupled-table beats landed on the GS arm's out geomean by > {BAR:.2%}")

    e4m3 = e4m3_positive_values(dev)                                     # [119], unscaled
    grid_bytes = torch.arange(E4M3_NORMAL_BYTES[0], E4M3_NORMAL_BYTES[1] + 1,
                              dtype=torch.long, device=dev)

    CONTROL = "control [LDLQ 1.0/32 + refit h^1.0]"
    JAC = "LDLQ 1.0/32 + refit full-H (Jacobi)"
    GS = "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"
    REF_NAMES = {CONTROL: "LDLQ 1.0/32 + refit h^1.0", JAC: "LDLQ 1.0/32 + refit full-H",
                 GS: "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"}
    LADDER = ["coupled-assign", "coupled-table", "free-e4m3", "free"]

    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        for name in units:
            W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
            H = Hall[name].to(dev, torch.float32)
            X = acts["x"][name].to(dev, torch.float32)
            Y = X @ W.T
            rows, cols = W.shape
            half = 16
            nb = cols // half
            den_w = float(W.norm())
            den_hf = float(((W @ H) * W).sum())
            den_y = float(Y.norm())
            L = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
            h1 = H.diagonal() / H.diagonal().mean()
            res: dict = {}
            log(f"\n== {name} {tuple(W.shape)}  eval rows {X.shape[0]}  blocks/row {nb}")
            log(f"    {'arm':<66} {'out':>8} {'plain':>8} {'hfit':>8} {'cost':>12}")

            def score(What):
                E = What - W
                return {"out": float((X @ E.T).norm() / den_y),
                        "plain": float(E.norm() / den_w),
                        "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf)}

            def show(label, r, extra=""):
                log(f"    {label:<66} {r['out']:8.5f} {r['plain']:8.5f} {r['hfit']:8.5f} "
                    f"{r.get('cost', float('nan')):12.5e} {extra}")

            def coupled(q, C, I, T, Tb, gvals, tag):
                """ICM on the given table to convergence, then the entry pass
                alternated with ICM to convergence.  Returns two states and
                their costs, asserting monotone descent throughout."""
                G = q.grad_field(C)
                prev = q.cost(C)
                sweeps = moves = 0
                for _ in range(a.rounds):
                    C, I, G, moved = q.icm_sweep(C, I, T, G)
                    now = q.cost(C)
                    assert now <= prev * (1 + 1e-4) + 1e-12, f"{tag}: ICM raised {prev} -> {now}"
                    sweeps += 1; moves += moved
                    stop = moved == 0 or now > prev * (1 - 1e-7)
                    prev = now
                    if stop:
                        break
                assign = dict(C=C, I=I, T=T, Tb=Tb, cost=prev, sweeps=sweeps, moves=moves)
                rounds = emoves = bmoves = 0
                for _ in range(a.rounds):
                    C, T, Tb, G, em = q.entry_pass(C, I, T, Tb, G, gvals, grid_bytes, 1e-10 * prev)
                    now = q.cost(C)
                    assert now <= prev * (1 + 1e-4) + 1e-12, f"{tag}: entry pass raised {prev} -> {now}"
                    C, I, G, bm = q.icm_sweep(C, I, T, G)
                    now2 = q.cost(C)
                    assert now2 <= now * (1 + 1e-4) + 1e-12, f"{tag}: ICM raised {now} -> {now2}"
                    rounds += 1; emoves += em; bmoves += bm
                    stop = (em == 0 and bm == 0) or now2 > prev * (1 - 1e-7)
                    prev = now2
                    if stop:
                        break
                table = dict(C=C, I=I, T=T, Tb=Tb, cost=prev, rounds=rounds,
                             entry_moves=emoves, moves=bmoves)
                return assign, table

            def free_e4m3(q, C, gvals, tag):
                I = (C[:, :, None] - gvals[None, None, :]).abs().argmin(dim=2)
                G = q.grad_field(C)
                prev = q.cost(C)
                sweeps = 0
                for _ in range(a.rounds):
                    C, I, G, moved = q.icm_sweep(C, I, gvals, G)
                    now = q.cost(C)
                    assert now <= prev * (1 + 1e-4) + 1e-12, f"{tag}: free-e4m3 raised {prev} -> {now}"
                    sweeps += 1
                    stop = moved == 0 or now > prev * (1 - 1e-7)
                    prev = now
                    if stop:
                        break
                return dict(C=C, cost=prev, sweeps=sweeps)

            def replay_pass(state, p, tag):
                """The matched pair on one captured refit call."""
                with refit_diagnostics() as diag:
                    nb_, ni_, ne_ = _ORIG_REFIT(
                        state["work"], state["units"], state["half"], state["table_bytes"],
                        state["index"], state["effective"], state["global_scale"],
                        state["metric"], gauss_seidel=state["gauss_seidel"])
                d = dict(diag[0])
                q = Quadratic(state["work"], state["units"], state["metric"], state["half"])
                C_enc = ne_.reshape(rows, nb)
                T_enc = _lut_values(nb_, state["global_scale"])
                I_enc = ni_.to(torch.long).reshape(rows, nb)
                assert torch.equal(T_enc[I_enc], C_enc)
                S0 = state["effective"].reshape(rows, nb)
                if q.diag:
                    # Under a 1-D metric the sink records the parabola without
                    # its constant (``refit_diagnostics``); shift it into the
                    # true-cost frame so every column below is one quantity.
                    const = q.cost(S0) - d["before"]
                    for k in ("before", "stepped", "continuous", "landed"):
                        d[k] += const
                assert close(q.cost(S0), d["before"]), (q.cost(S0), d["before"])
                assert close(q.cost(C_enc), d["landed"]), (q.cost(C_enc), d["landed"])
                # fp32 sums of ~1e7 terms with heavy cancellation: the same
                # quantity computed twice can differ by ~5e-5 relative on
                # L2.down_proj.  Recorded, so the noise floor is on the record.
                d["replay_rel_discrepancy"] = max(
                    abs(q.cost(S0) / d["before"] - 1.0), abs(q.cost(C_enc) / d["landed"] - 1.0))
                gvals = e4m3 * state["global_scale"]
                assign, table = coupled(q, C_enc.clone(), I_enc.clone(), T_enc.clone(),
                                        nb_.to(torch.long).clone(), gvals, f"{tag} p{p}")
                Cf = q.free_solve(C_enc)
                cf = q.cost(Cf)
                assert cf <= table["cost"] * (1 + 1e-4), (cf, table["cost"])
                rec = {k: d[k] for k in ("before", "stepped", "continuous", "landed", "reverted",
                                         "candidate", "replay_rel_discrepancy")}
                rec.update(coupled_assign=assign["cost"], coupled_table=table["cost"], free=cf,
                           assign_moves=assign["moves"], assign_sweeps=assign["sweeps"],
                           table_rounds=table["rounds"], table_entry_moves=table["entry_moves"],
                           table_block_moves=table["moves"],
                           free_nonpositive=int((Cf <= 0).sum()), blocks=int(Cf.numel()))
                return rec, q, dict(landed=(C_enc, I_enc, T_enc, nb_.to(torch.long)),
                                    assign=assign, table=table, free=Cf, gvals=gvals)

            def materialised(unit, forests, q, C, I, Tb):
                order = torch.argsort(Tb)
                rank = torch.empty_like(order); rank[order] = torch.arange(order.numel(), device=dev)
                u2 = dataclasses.replace(
                    unit, scale_lut=Tb[order].to(torch.uint8),
                    scale_refine=rank[I].reshape(-1).to(torch.uint8))
                What2 = stock_dequant(materialize_stock(u2, forests, DEFAULT_CODE)).to(dev).float()
                assert torch.equal(What2, q.recon(C)), "the oracle plane did not survive the wire"
                return What2

            def ladder(label, unit, forests, q, planes, prefix="  "):
                """Score the trailing-pass ladder on out/hfit; prove the two
                wire-representable rungs through the wire."""
                C, I, T, Tb = planes["landed"]
                for rung, st in (("coupled-assign", planes["assign"]), ("coupled-table", planes["table"])):
                    What = materialised(unit, forests, q, st["C"], st["I"], st["Tb"])
                    r = score(What); r["cost"] = st["cost"]; r["sha256"] = sha(What)
                    r["table_bytes"] = sorted(int(b) for b in st["Tb"].tolist())
                    res[f"{rung} [{label}]"] = r
                    show(f"{prefix}{rung:<16} [{label}]", r)
                fe = free_e4m3(q, planes["table"]["C"].clone(), planes["gvals"], label)
                r = score(q.recon(fe["C"])); r["cost"] = fe["cost"]; r["sweeps"] = fe["sweeps"]
                res[f"free-e4m3 [{label}]"] = r
                show(f"{prefix}{'free-e4m3':<16} [{label}]", r, f"{fe['sweeps']} sweeps")
                Cf = planes["free"]
                r = score(q.recon(Cf)); r["cost"] = q.cost(Cf)
                r["nonpositive_scales"] = int((Cf <= 0).sum()); r["blocks"] = int(Cf.numel())
                res[f"free [{label}]"] = r
                show(f"{prefix}{'free':<16} [{label}]", r, f"{r['nonpositive_scales']} of {r['blocks']} <= 0")

            def encode(label, objective, gs):
                global CAPTURE
                kw = dict(ldl=L, ldl_block=a.block)
                kw["refit_metric"] = H if objective == "hessian" else h1
                if gs:
                    kw["refit_gauss_seidel"] = True
                CAPTURE = []
                t0 = time.time()
                with refit_diagnostics() as diag:
                    _, unit, forests = encode_linear_planes(
                        W, grid=grid, q256=a.q256, name=name, verify=False, **kw)
                secs = time.time() - t0
                states, CAPTURE = CAPTURE, None
                st = materialize_stock(unit, forests, DEFAULT_CODE)
                What = stock_dequant(st).to(dev).float()
                r = score(What)
                r["secs"] = secs
                r["sha256"] = sha(What)
                r["refit"] = [dict(d) for d in diag]
                if ref is not None and name in ref["units"]:
                    want = ref["units"][name][REF_NAMES[label.removesuffix(" REPEAT")]]["sha256"]
                    r["matches_reference"] = (want == r["sha256"])
                codes = decode_codes_mixed(unit, forests, DEFAULT_CODE)
                U = dequantize(codes, torch.ones(rows, cols, device=dev), grid)
                Sf = unit_scale_field(unit, rows, cols)
                assert torch.equal(U * Sf, What), "S*U is not the stock reconstruction"
                assert len(states) == len(diag) == unit.scale_refit
                assert torch.equal(states[-1]["units"], U), "the trailing capture is not the final codes"
                mark = "" if r.get("matches_reference", True) else "  !! DIFFERS from reference"
                show(label, r, f"{secs:6.1f}s{mark}")
                res[label] = r
                return unit, forests, U, states

            def per_pass(label, unit, forests, U, states):
                recs = []
                q = planes = None
                for p, state in enumerate(states):
                    rec, q, planes = replay_pass(state, p, label)
                    recs.append(rec)
                res[label]["replay"] = recs
                # The trailing pass is the wire.  Take the plane from the unit
                # itself, and record whether the replay reproduced it bit for bit.
                Sf = unit_scale_field(unit, rows, cols)
                C_w = Sf[:, ::half].contiguous()
                I_w = unit.scale_refine.to(torch.long).reshape(rows, nb)
                Tb_w = unit.scale_lut.to(torch.long)
                T_w = _lut_values(unit.scale_lut, unit.scale_global)
                C_r, _, _, Tb_r = planes["landed"]
                same = bool(torch.equal(C_r, C_w) and torch.equal(Tb_r, Tb_w))
                res[label]["replay_matches_wire"] = same
                if not q.diag:   # the full-H quadratic IS hfit^2 * W H W^T; the diagonal one is not
                    assert close(q.cost(C_w), res[label]["hfit"] ** 2 * den_hf, 1e-4)
                if not same:
                    log(f"    !! replay of the trailing refit differs from the wire's plane; "
                        f"ladder recomputed from the wire")
                    gv = planes["gvals"]
                    assign, table = coupled(q, C_w.clone(), I_w.clone(), T_w.clone(),
                                            Tb_w.clone(), gv, f"{label} wire")
                    planes = dict(landed=(C_w, I_w, T_w, Tb_w), assign=assign, table=table,
                                  free=q.free_solve(C_w), gvals=gv)
                ladder(label, unit, forests, q, planes)
                return q, planes

            # ---- the control, first
            unit, forests, U, states = encode(CONTROL, "h^1.0", False)
            per_pass(CONTROL, unit, forests, U, states)
            # (2) one full-H refit at the control's final codes: the encoder's
            # own landing, then the coupled one, then the ceiling.
            last = states[-1]
            with refit_diagnostics() as diag:
                nb_, ni_, ne_ = _ORIG_REFIT(
                    last["work"], U, last["half"], unit.scale_lut, unit.scale_refine,
                    unit_scale_field(unit, rows, cols)[:, ::half].reshape(-1).contiguous(),
                    unit.scale_global, H, gauss_seidel=True)
            qH = Quadratic(last["work"], U, H, half)
            C_sw = ne_.reshape(rows, nb); I_sw = ni_.to(torch.long).reshape(rows, nb)
            T_sw = _lut_values(nb_, unit.scale_global)
            SWAP = "control codes + one full-H GS refit"
            What = materialised(unit, forests, qH, C_sw, I_sw, nb_.to(torch.long))
            r = score(What); r["cost"] = qH.cost(C_sw); r["refit"] = [dict(diag[0])]; r["sha256"] = sha(What)
            res[SWAP] = r
            show(SWAP, r, f"encoder landing; continuous {diag[0]['continuous']:.5e}")
            gv = e4m3 * unit.scale_global
            assign, table = coupled(qH, C_sw.clone(), I_sw.clone(), T_sw.clone(),
                                    nb_.to(torch.long).clone(), gv, SWAP)
            Cf = qH.free_solve(C_sw)
            ladder(SWAP, unit, forests, qH,
                   dict(landed=(C_sw, I_sw, T_sw, nb_.to(torch.long)), assign=assign,
                        table=table, free=Cf, gvals=gv))
            # ---- the two full-H arms
            for label, obj, gs in ((JAC, "hessian", False), (GS, "hessian", True)):
                unit, forests, U, states = encode(label, obj, gs)
                per_pass(label, unit, forests, U, states)
            # ---- the control again, last
            encode(CONTROL + " REPEAT", "h^1.0", False)
            same = res[CONTROL]["sha256"] == res[CONTROL + " REPEAT"]["sha256"]
            log(f"    -- drift control: bytes {'IDENTICAL' if same else 'DIFFER'}  "
                f"out {res[CONTROL]['out']:.6f} -> {res[CONTROL + ' REPEAT']['out']:.6f}")
            res["_drift"] = {"bytes_identical": same}
            # ---- per-pass table for this unit
            log(f"    -- per pass: fractions of the pass's starting cost; 'recov' = share of the")
            log(f"       landing loss (landed - continuous) the coupled landing gets back; 'vs free'")
            log(f"       the same against the exact joint minimiser")
            log(f"    {'arm':<8} {'p':>2} {'step':>8} {'landing':>8} {'assign':>8} {'table':>8} "
                f"{'recov':>7} {'vs free':>8} {'freegap':>8} {'moves':>6}")
            for label, short in ((CONTROL, "h^1.0"), (JAC, "jacobi"), (GS, "gs")):
                for p, rec in enumerate(res[label]["replay"]):
                    b, s, c, ld = rec["before"], rec["stepped"], rec["continuous"], rec["landed"]
                    asg, tb, fr = rec["coupled_assign"], rec["coupled_table"], rec["free"]
                    land = ld - c
                    log(f"    {short:<8} {p:2d} {(b - s) / b:8.3%} {land / b:8.3%} "
                        f"{(ld - asg) / b:8.3%} {(ld - tb) / b:8.3%} "
                        f"{((ld - tb) / land if land > 0 else float('nan')):7.1%} "
                        f"{((ld - tb) / (ld - fr) if ld > fr else float('nan')):8.1%} "
                        f"{(ld - fr) / b:8.3%} {rec['assign_moves'] + rec['table_block_moves']:6d}")
            out["units"][name] = res
            del W, H, X, Y, L
            torch.cuda.empty_cache()
            Path(a.out).write_text(json.dumps(out, indent=1))

    # ---- geomeans and the pre-registered verdict
    names = list(out["units"])
    arms = sorted(set.intersection(*[{k for k in v if not k.startswith("_")} for v in out["units"].values()]))

    def geo(arm, field):
        return math.exp(sum(math.log(out["units"][u][arm][field]) for u in names) / len(names))

    log("\n== geomean over units")
    log(f"    {'arm':<80} {'out':>8} {'hfit':>8}")
    for arm in sorted(arms, key=lambda x: geo(x, "out")):
        log(f"    {arm:<80} {geo(arm, 'out'):8.5f} {geo(arm, 'hfit'):8.5f}")
    out["geomean"] = {arm: {"out": geo(arm, "out"), "hfit": geo(arm, "hfit")} for arm in arms}

    SWAP = "control codes + one full-H GS refit"
    log("\n== the end-state ladder, per encoder arm (ratios to that arm's landed)")
    for base in (CONTROL, JAC, GS, SWAP):
        lo, lh = geo(base, "out"), geo(base, "hfit")
        log(f"    {base}")
        log(f"      {'landed':<16} out {lo:.5f}            hfit {lh:.5f}")
        for k in LADDER:
            arm = f"{k} [{base}]"
            go, gh = geo(arm, "out"), geo(arm, "hfit")
            log(f"      {k:<16} out {go:.5f} ({go / lo:.4f}x)  hfit {gh:.5f} ({gh / lh:.4f}x)")

    landed = geo(GS, "out")
    oracle = geo(f"coupled-table [{GS}]", "out")
    free = geo(f"free [{GS}]", "out")
    ctl = geo(CONTROL, "out")

    def pooled(label, num, den):
        """Sum over units and passes of (num - den) style fractions: cost-weighted."""
        tot_n = tot_d = 0.0
        for u in names:
            for rec in out["units"][u][label]["replay"]:
                tot_n += num(rec); tot_d += den(rec)
        return tot_n / tot_d if tot_d else float("nan")

    verdict = {
        "gs_landed_out": landed, "gs_coupled_table_out": oracle, "gs_free_out": free,
        "control_landed_out": ctl,
        "oracle_gain_out": 1.0 - oracle / landed,
        "ceiling_gain_out": 1.0 - free / landed,
        "recoverable_fraction_out_geomean": ((landed - oracle) / (landed - free)) if landed > free else float("nan"),
        "recoverable_fraction_trailing_cost_gs_mean": sum(
            (r["landed"] - r["coupled_table"]) / (r["landed"] - r["continuous"])
            for u in names for r in out["units"][u][GS]["replay"][-1:]) / len(names),
        "recoverable_fraction_all_passes_gs_pooled": pooled(
            GS, lambda r: r["landed"] - r["coupled_table"], lambda r: r["landed"] - r["continuous"]),
        "recoverable_fraction_all_passes_jacobi_pooled": pooled(
            JAC, lambda r: r["landed"] - r["coupled_table"], lambda r: r["landed"] - r["continuous"]),
        "bar": BAR, "clears_bar": (1.0 - oracle / landed) > BAR,
        "full_h_coupled_vs_control_out": oracle / ctl,
        "swap_landed_vs_control_out": geo(SWAP, "out") / ctl,
        "swap_coupled_table_vs_control_out": geo(f"coupled-table [{SWAP}]", "out") / ctl,
        "swap_coupled_table_vs_gs_landed_out": geo(f"coupled-table [{SWAP}]", "out") / landed,
    }
    out["verdict_issue_50"] = verdict
    log("\n== pre-registered verdict (GS arm, out geomean)")
    log(json.dumps(verdict, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out.replace(".json", ".log")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
