#!/usr/bin/env python
"""Issue #50: how much of the LUT refit's landing loss can ANY sixteen-entry
table get back -- measured as an oracle at the wire's own final codes.

`#35`'s instrument split each metric-aware LUT refit three ways and found the
*landing* -- `_fit_lut`'s separable model plus nearest-in-linear assignment
into sixteen E4M3 entries -- taking back 24-91% of the step it was handed.
The issue reads that as recoverable: the fit drops every cross-block term of
the true quadratic, so a fit that keeps them should return most of it.  This
script sizes the prize before anyone builds the optimiser.

**Where it measures.**  At the END of the encode, at the codes the wire holds.
On E2M1x2 there is no release, so the final plane IS the trailing refit's
landed output and the sink's trailing ``landed`` equals ``hfit^2 * (W H W^T)``
to seven digits (checked, all twelve rows of `qwen_lut_gs.json`).  So every
oracle below re-solves the plane at fixed codes under the same objective the
arm's own refit minimised, and each step is provably monotone in that
objective, which the run asserts after every sweep.

| arm | the plane may be | answers |
|---|---|---|
| ``landed`` | what the encoder shipped | the reference |
| ``oracle-assign`` | one of the encoder's own 16 entries, each block chosen on the TRUE quadratic given the others (ICM) | the assignment half of the landing, at zero byte cost |
| ``oracle-table`` | 16 distinct E4M3 entries AND an assignment, both chosen on the true quadratic (ICM alternated with an exact per-entry coordinate step onto the E4M3 grid) | what a cross-block-aware table fit could win, at the same bytes |
| ``free-e4m3`` | any in-range E4M3 value per block | the ceiling a scale BYTE allows -- no table, 8 bits a block |
| ``free`` | any real per block: the exact per-row joint minimiser ``M_r^{-1} v_r`` | issue #50's ceiling, "the landing disabled" |

``oracle-assign`` and ``oracle-table`` are re-materialised through the wire's
own ``materialize_stock`` / ``stock_dequant`` and compared ``torch.equal`` to
the oracle's reconstruction, so the two arms that claim "same bytes" are
proved representable rather than assumed.  ``free-e4m3`` and ``free`` are not
planes the wire can hold and are scored from the reconstruction alone.

**Pre-registered bar, written before the first run.**  The encoder-side
coupled landing gets built only if ``oracle-table`` beats ``landed`` on the
Gauss-Seidel arm's six-unit ``out`` geomean by more than **1.38%** -- the
margin `#35`'s promotion rule used, the span the two refit objectives were
found to occupy.  Below it, #50 closes with the number and no code ships.
Whatever the bar says, the receipt carries the comparison the issue does not
ask: full-H + a perfect table against the served ``h^1.0`` default's landed
number, because if that loses the landing is not what stands between the
full-H arm and the default.

**Scored two ways, labelled.**  ``out`` is the held-out activation-space
relative error (the `#35` receipt's deciding column; a screen, not a serve).
``hfit`` is ``sqrt(E H E^T / W H W^T)`` on the fit rows, the quadratic the
refit is provably monotone in.  ``plain`` is unweighted weight space.
Everything in fp32; the ``free`` solve in fp64.

Held: six Qwen3-0.6B units, E2M1x2 at `q256=896`, LDLQ 1.0/32, one process,
the served default first and again last.  The three encoder arms are byte-
checked against `qwen_lut_gs.json`'s digests, so this run's encodes are the
receipt's encodes across processes and a code change.

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

from tessera.alphabet import SERIALISABLE_GRIDS                       # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian           # noqa: E402
from tessera.decode import decode_codes_mixed, dequantize, unit_scale_field  # noqa: E402
from tessera.encode import (                                           # noqa: E402
    E4M3_NORMAL_BYTES, _lut_values, e4m3_positive_values, refit_diagnostics)
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.manifest import ScalePlaneKind                            # noqa: E402
from tessera.stock import materialize_stock, stock_dequant             # noqa: E402

BAR = 0.0138   # #35's promotion margin: build only past it.  Pre-registered.


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Quadratic:
    """``L(C) = sum_r (w_r - C_r U_r) H (w_r - C_r U_r)^T`` over per-block scales.

    ``Ub`` is ``[rows, nb, half]`` -- block ``b``'s unscaled codes -- and every
    method keeps the gradient field ``G = (W - C U) H`` current so a block or
    an entry moves at the cost of its own sixteen columns through H.
    """

    def __init__(self, W, U, H, half):
        self.W, self.U, self.H, self.half = W, U, H, half
        self.rows, self.cols = W.shape
        self.nb = self.cols // half
        self.Ub = U.reshape(self.rows, self.nb, half)
        Hd = torch.diagonal(H.reshape(self.nb, half, self.nb, half), dim1=0, dim2=2).permute(2, 0, 1)
        self.A = torch.einsum("rbi,bij,rbj->rb", self.Ub, Hd, self.Ub)     # block curvatures

    def field(self, C):
        return C.repeat_interleave(self.half, dim=1)

    def recon(self, C):
        return self.field(C) * self.U

    def cost(self, C) -> float:
        E = self.W - self.recon(C)
        return float(((E @ self.H) * E).sum())

    def grad_field(self, C):
        return (self.W - self.recon(C)) @ self.H

    def icm_sweep(self, C, I, table, G):
        """One sweep: each block to the table entry minimising the true quadratic
        GIVEN every other block where it stands now.  Nearest-in-linear to the
        conditional optimum is exact for a parabola; ``G`` carries every move
        already made, which is the whole difference from the encoder's landing.
        Returns ``(C, I, G, moved)``.  With ``table`` the full E4M3 grid this is
        the ``free-e4m3`` arm; with the unit's sixteen it is ``oracle-assign``."""
        moved = 0
        half = self.half
        for b in range(self.nb):
            lo, hi = b * half, (b + 1) * half
            Ubb = self.Ub[:, b, :]
            A = self.A[:, b]
            s = C[:, b] + (G[:, lo:hi] * Ubb).sum(dim=1) / A.clamp_min(1e-30)
            j = (s[:, None] - table[None, :]).abs().argmin(dim=1)
            new = torch.where(A > 0, table[j], C[:, b])
            j = torch.where(A > 0, j, I[:, b])
            d = new - C[:, b]
            changed = d != 0
            if bool(changed.any()):
                moved += int(changed.sum())
                G = G - (d.unsqueeze(1) * Ubb) @ self.H[lo:hi, :]
                C = C.clone(); C[:, b] = new
                I = I.clone(); I[:, b] = j
        return C, I, G, moved

    def entry_pass(self, C, I, table, table_bytes, G, grid_values, grid_bytes, tol):
        """One coordinate pass over the sixteen entries, cross-block terms kept.

        Moving entry ``k`` by ``delta`` moves every block assigned to it, so the
        change in the quadratic is exactly ``Q_k delta^2 + 2 g_k delta`` with
        ``Q_k = sum_r V_rk H V_rk^T`` (``V_rk`` the union of the row's codes on
        the blocks assigned to ``k``) and ``g_k = -sum (G * V_k)``.  The best
        DISTINCT in-range E4M3 value under that parabola is taken exactly."""
        moved = 0
        for k in range(table.numel()):
            mask = (I == k)
            if not bool(mask.any()):
                continue
            Vk = (mask.unsqueeze(2) * self.Ub).reshape(self.rows, self.cols)
            VkH = Vk @ self.H
            Q = float((VkH * Vk).sum())
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
                G = G - step * VkH
                table = table.clone(); table[k] = grid_values[best]
                table_bytes = table_bytes.clone(); table_bytes[k] = grid_bytes[best]
                moved += 1
        return C, table, table_bytes, G, moved

    def free_solve(self, C0, chunk=256):
        """The exact per-row joint minimiser, ``M_r^{-1} v_r`` in fp64.  A block
        with zero curvature (all-zero codes) has no scale and keeps its own."""
        out = torch.empty_like(C0)
        H64 = self.H.double()
        for r0 in range(0, self.rows, chunk):
            r1 = min(r0 + chunk, self.rows)
            Ub = self.Ub[r0:r1].double()                                   # [ch, nb, half]
            ch = r1 - r0
            # ``Z_r`` is block-structured (block ``b`` lives on its own sixteen
            # columns), so ``Z H`` is one panel product per block and ``Z H Z^T``
            # reads the result back through the same panels.
            ZH = torch.einsum("rbi,bij->rbj", Ub, H64.reshape(self.nb, self.half, self.cols))
            M = torch.einsum("rbcj,rcj->rbc", ZH.reshape(ch, self.nb, self.nb, self.half), Ub)
            v = (ZH @ self.W[r0:r1].double().unsqueeze(2)).squeeze(2)      # [ch, nb]
            dead = self.A[r0:r1] <= 0
            eye = torch.eye(self.nb, dtype=torch.float64, device=Ub.device)
            M = torch.where(dead.unsqueeze(2) | dead.unsqueeze(1), torch.zeros_like(M), M)
            M = M + dead.double().unsqueeze(2) * eye
            v = torch.where(dead, C0[r0:r1].double(), v)
            out[r0:r1] = torch.linalg.solve(M, v.unsqueeze(2)).squeeze(2).float()
        return out


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
    ap.add_argument("--rounds", type=int, default=30)
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
    log(f"pre-registered bar: oracle-table beats landed on the GS arm's out geomean by > {BAR:.2%}")

    grid_values = e4m3_positive_values(dev)                              # [119]
    grid_bytes = torch.arange(E4M3_NORMAL_BYTES[0], E4M3_NORMAL_BYTES[1] + 1,
                              dtype=torch.long, device=dev)

    CONTROL = "control [LDLQ 1.0/32 + refit h^1.0]"
    ARMS = [
        ("LDLQ 1.0/32 + refit full-H (Jacobi)", dict(objective="hessian", gs=False)),
        ("LDLQ 1.0/32 + refit full-H (Gauss-Seidel)", dict(objective="hessian", gs=True)),
    ]
    REF_NAMES = {CONTROL: "LDLQ 1.0/32 + refit h^1.0",
                 ARMS[0][0]: "LDLQ 1.0/32 + refit full-H",
                 ARMS[1][0]: "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"}

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
            log(f"    {'arm':<62} {'out':>8} {'plain':>8} {'hfit':>8} {'cost':>12}")

            def score(What):
                E = What - W
                return {"out": float((X @ E.T).norm() / den_y),
                        "plain": float(E.norm() / den_w),
                        "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf)}

            def show(label, r, extra=""):
                log(f"    {label:<62} {r['out']:8.5f} {r['plain']:8.5f} {r['hfit']:8.5f} "
                    f"{r.get('cost', float('nan')):12.5e} {extra}")

            def encode(label, objective, gs):
                kw = dict(ldl=L, ldl_block=a.block)
                kw["refit_metric"] = H if objective == "hessian" else h1
                if gs:
                    kw["refit_gauss_seidel"] = True
                t0 = time.time()
                with refit_diagnostics() as diag:
                    _, unit, forests = encode_linear_planes(
                        W, grid=grid, q256=a.q256, name=name, verify=False, **kw)
                secs = time.time() - t0
                st = materialize_stock(unit, forests, DEFAULT_CODE)
                What = stock_dequant(st).to(dev).float()
                r = score(What)
                r["secs"] = secs
                r["sha256"] = sha(What)
                r["refit"] = [dict(d) for d in diag]
                if ref is not None and name in ref["units"]:
                    want = ref["units"][name][REF_NAMES[label]]["sha256"]
                    r["matches_reference"] = (want == r["sha256"])
                # The two factors the wire's reconstruction is a product of.
                codes = decode_codes_mixed(unit, forests, DEFAULT_CODE)
                U = dequantize(codes, torch.ones(rows, cols, device=dev), grid)
                Sf = unit_scale_field(unit, rows, cols)
                assert torch.equal(U * Sf, What), "S*U is not the stock reconstruction"
                C0 = Sf[:, ::half].contiguous()
                T = _lut_values(unit.scale_lut, unit.scale_global)
                I0 = unit.scale_refine.to(torch.long).reshape(rows, nb)
                assert torch.equal(T[I0], C0), "index/table do not give the plane"
                q = Quadratic(W, U, H, half)
                r["cost"] = q.cost(C0)
                # The fit quadratic through the same expression as the sink.
                r["cost_is_hfit"] = abs(r["cost"] / (r["hfit"] ** 2 * den_hf) - 1.0)
                if diag:
                    r["cost_is_trailing_landed"] = abs(r["cost"] / diag[-1]["landed"] - 1.0)
                mark = "" if r.get("matches_reference", True) else "  !! DIFFERS from reference"
                show(label, r, f"{secs:6.1f}s{mark}")
                res[label] = r
                return unit, forests, q, C0, I0, T

            def materialised(unit, forests, q, C, I, T, Tb):
                """Prove a (table, index) plane is the wire's: rebuild the unit
                with sorted bytes and remapped index, materialise, compare."""
                order = torch.argsort(Tb)
                rank = torch.empty_like(order); rank[order] = torch.arange(order.numel(), device=dev)
                u2 = dataclasses.replace(
                    unit, scale_lut=Tb[order].to(torch.uint8),
                    scale_refine=rank[I].reshape(-1).to(torch.uint8))
                What2 = stock_dequant(materialize_stock(u2, forests, DEFAULT_CODE)).to(dev).float()
                assert torch.equal(What2, q.recon(C)), "the oracle plane did not survive the wire"
                return What2

            def oracles(label, unit, forests, q, C0, I0, T):
                base = q.cost(C0)
                Tb = unit.scale_lut.to(torch.long).clone()
                T = T.clone()
                # -- oracle-assign: ICM over the unit's own sixteen, to convergence
                C, I, G = C0.clone(), I0.clone(), q.grad_field(C0)
                prev, sweeps, moved_total = base, 0, 0
                for _ in range(a.rounds):
                    C, I, G, moved = q.icm_sweep(C, I, T, G)
                    now = q.cost(C)
                    assert now <= prev * (1 + 1e-6) + 1e-9, f"ICM raised the cost {prev} -> {now}"
                    sweeps += 1; moved_total += moved
                    if moved == 0 or now > prev * (1 - 1e-7):
                        prev = now; break
                    prev = now
                What = materialised(unit, forests, q, C, I, T, Tb)
                r = score(What); r["cost"] = prev; r["sweeps"] = sweeps; r["blocks_moved"] = moved_total
                r["sha256"] = sha(What)
                res[f"oracle-assign [{label}]"] = r
                show(f"  oracle-assign (own 16, ICM)      [{label}]", r,
                     f"{sweeps} sweeps, {moved_total} moves")
                # -- oracle-table: alternate the entry step with ICM
                rounds, entry_moves, block_moves = 0, 0, 0
                for _ in range(a.rounds):
                    C, T, Tb, G, em = q.entry_pass(C, I, T, Tb, G, grid_values, grid_bytes, 1e-10 * prev)
                    now = q.cost(C)
                    assert now <= prev * (1 + 1e-6) + 1e-9, f"entry pass raised the cost {prev} -> {now}"
                    C, I, G, bm = q.icm_sweep(C, I, T, G)
                    now2 = q.cost(C)
                    assert now2 <= now * (1 + 1e-6) + 1e-9, f"ICM raised the cost {now} -> {now2}"
                    rounds += 1; entry_moves += em; block_moves += bm
                    if (em == 0 and bm == 0) or now2 > prev * (1 - 1e-7):
                        prev = now2; break
                    prev = now2
                What = materialised(unit, forests, q, C, I, T, Tb)
                r = score(What); r["cost"] = prev; r["rounds"] = rounds
                r["entry_moves"] = entry_moves; r["blocks_moved"] = block_moves
                r["table_bytes"] = sorted(int(b) for b in Tb.tolist())
                r["table_bytes_before"] = sorted(int(b) for b in unit.scale_lut.tolist())
                r["sha256"] = sha(What)
                res[f"oracle-table [{label}]"] = r
                show(f"  oracle-table (16 E4M3 + ICM)     [{label}]", r,
                     f"{rounds} rounds, {entry_moves} entry moves, {block_moves} block moves")
                # -- free-e4m3: every block free on the whole positive normal grid
                Cg, Ig, Gg = C.clone(), torch.zeros_like(I), q.grad_field(C)
                Ig = (Cg[:, :, None] - grid_values[None, None, :]).abs().argmin(dim=2)
                pg, sweeps = prev, 0
                for _ in range(a.rounds):
                    Cg, Ig, Gg, moved = q.icm_sweep(Cg, Ig, grid_values, Gg)
                    now = q.cost(Cg)
                    assert now <= pg * (1 + 1e-6) + 1e-9, f"free-e4m3 raised the cost {pg} -> {now}"
                    sweeps += 1
                    if moved == 0 or now > pg * (1 - 1e-7):
                        pg = now; break
                    pg = now
                r = score(q.recon(Cg)); r["cost"] = pg; r["sweeps"] = sweeps
                res[f"free-e4m3 [{label}]"] = r
                show(f"  free-e4m3 (any E4M3 per block)   [{label}]", r, f"{sweeps} sweeps")
                # -- free: the exact joint minimiser
                Cf = q.free_solve(C0)
                pf = q.cost(Cf)
                assert pf <= pg * (1 + 1e-6) + 1e-9, f"the exact solve is above free-e4m3: {pg} -> {pf}"
                r = score(q.recon(Cf)); r["cost"] = pf
                r["nonpositive_scales"] = int((Cf <= 0).sum()); r["blocks"] = int(Cf.numel())
                res[f"free [{label}]"] = r
                show(f"  free (exact per-row solve)       [{label}]", r,
                     f"{r['nonpositive_scales']} of {r['blocks']} scales <= 0")

            first = encode(CONTROL, "h^1.0", False)
            oracles(CONTROL, *first)
            for label, kw in ARMS:
                got = encode(label, kw["objective"], kw["gs"])
                oracles(label, *got)
            encode(CONTROL + " REPEAT", "h^1.0", False)
            same = res[CONTROL]["sha256"] == res[CONTROL + " REPEAT"]["sha256"]
            log(f"    -- drift control: bytes {'IDENTICAL' if same else 'DIFFER'}  "
                f"out {res[CONTROL]['out']:.6f} -> {res[CONTROL + ' REPEAT']['out']:.6f}")
            res["_drift"] = {"bytes_identical": same}
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
    log(f"    {'arm':<75} {'out':>8} {'hfit':>8}")
    for arm in sorted(arms, key=lambda x: geo(x, "out")):
        log(f"    {arm:<75} {geo(arm, 'out'):8.5f} {geo(arm, 'hfit'):8.5f}")
    out["geomean"] = {arm: {"out": geo(arm, "out"), "hfit": geo(arm, "hfit")} for arm in arms}

    log("\n== the ceiling and the prize, per encoder arm (ratios to that arm's landed)")
    ladder = ["oracle-assign", "oracle-table", "free-e4m3", "free"]
    for enc in [CONTROL] + [x[0] for x in ARMS]:
        lo, lh = geo(enc, "out"), geo(enc, "hfit")
        log(f"    {enc}")
        log(f"      {'landed':<16} out {lo:.5f}            hfit {lh:.5f}")
        for k in ladder:
            arm = f"{k} [{enc}]"
            go, gh = geo(arm, "out"), geo(arm, "hfit")
            log(f"      {k:<16} out {go:.5f} ({go / lo:.4f}x)  hfit {gh:.5f} ({gh / lh:.4f}x)")
    gs = ARMS[1][0]
    landed = geo(gs, "out")
    oracle = geo(f"oracle-table [{gs}]", "out")
    free = geo(f"free [{gs}]", "out")
    ctl = geo(CONTROL, "out")
    verdict = {
        "gs_landed_out": landed, "gs_oracle_table_out": oracle, "gs_free_out": free,
        "control_landed_out": ctl,
        "oracle_gain": 1.0 - oracle / landed,
        "ceiling_gain": 1.0 - free / landed,
        "recoverable_fraction_out": ((landed - oracle) / (landed - free)) if landed > free else float("nan"),
        "recoverable_fraction_hfit_sq": (
            (geo(gs, "hfit") ** 2 - geo(f"oracle-table [{gs}]", "hfit") ** 2)
            / (geo(gs, "hfit") ** 2 - geo(f"free [{gs}]", "hfit") ** 2)),
        "bar": BAR, "clears_bar": (1.0 - oracle / landed) > BAR,
        "full_h_oracle_vs_control_out": oracle / ctl,
    }
    out["verdict_issue_50"] = verdict
    log("\n== pre-registered verdict (GS arm, out geomean)")
    log(json.dumps(verdict, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out.replace(".json", ".log")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
