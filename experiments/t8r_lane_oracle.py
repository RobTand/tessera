#!/usr/bin/env python3
"""Three-way numerics oracle for a routed window MoE stack on real wires:
the fused lane (``routed_fused.FusedRoutedWindowMoE``, tessera#640/#701), the
compact Triton adapter over the SAME bundles (``native_window_moe_from_
bundles``), and a reference that never touches the bundles.

WHY A THIRD LEG.  Fused and compact read one ``PackedWindowMoeBundles``: the
same repacked words, run tables, column permutation and TP row-cut start
state (``compact_prep``).  A defect in any of those is identical in both and
invisible to fused - compact.  The reference therefore parses the FULL wire
through the materialising reader (``scheme.parse_tessera_expert_blob``),
decodes it with Tessera's reference decoder (``stock.materialize_stock`` ->
``decode.materialize_fp8``: e4m3 bytes and fp32 row scales), and cuts this
rank's slice out of the dense matrix with plain indexing.

WHAT IS COMPARED, per layer (rank ``--rank`` of ``--tp``):

1. One-hot extraction (every expert, both projections, both lanes).  Row k of
   x is e_k, quantised by the runtime's own per-token E4M3 op; every output
   element is one decoded weight through the E4M3 epilogue
   ``bf16(((w * q) * a_scale) * w_scale)``, which torch reproduces bitwise
   (``tests/fused_bound.one_hot_expected``).  A wrong column map, run offset,
   rate or start state shows as a non-bitwise element; mismatches are
   attributed by column rate (the reference unit's own per-column rates),
   by output row (the first ceil(L/R) rows are where a start state matters)
   and by expert.
2. The routed forward at each M in ``--ms`` and each routing in ``--routes``,
   with the served SwiGLU clamp:
   * ``ref_exact``: fp64 GEMMs on the contract's quantised operands (per-token
     dynamic E4M3 of x and of the activation, ``torch.ops._C.dynamic_per_
     token_scaled_fp8_quant``), no bf16 rounding of any intermediate, fp64
     top-k sum.
   * ``ref_lane``: the same, with the lane's own dtypes and order: gemm1 to
     bf16, silu/mul in fp32 then bf16, per-route bf16, fixed-order fp32 top-k
     sum, bf16 out.
   * The NOISE FLOOR is ``ref_lane - ref_exact``: what the lane's own
     accumulation dtype costs on these inputs.
   * Stage checks (teacher-forced, identical quantised operands on every
     side): gate/up per route vs fp64, down per route (top_k = 1 routing, so
     each route is its own row) on the reference's bf16 activation vs fp64,
     each held per element to the dtype-derived bound of ``tests/
     fused_bound.dense_bound``, and summarised per expert.
   * The fused forward against its own staged composition, bitwise.

Run inside the serving image through ``experiments/t8r_lane_oracle_action.sh``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from bench_routed_load import MOE_UNITS, _load_wires, _schemes, _target  # noqa: E402

U_ACC = 2.0 ** -23
U32 = 2.0 ** -24
U64 = 2.0 ** -53
L_WINDOW = 14


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gamma(n, u):
    return n * u / (1.0 - n * u)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(a, b):
    """``a`` against ``b`` (the comparison target), flattened, in fp64."""
    import torch

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    d = a - b
    nb = float(b.norm())
    na = float(a.norm())
    return {"max_abs": float(d.abs().max()) if d.numel() else 0.0,
            "rel_frob": float(d.norm()) / nb if nb > 0 else (0.0 if float(d.norm()) == 0 else math.inf),
            "cosine": float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 1.0,
            "max_abs_ref": float(b.abs().max()) if b.numel() else 0.0,
            "n": int(d.numel())}


def bf16_ulp(v):
    import torch

    return torch.exp2(torch.floor(torch.log2(v.clamp(min=2.0 ** -126))) - 7)


def fp8_quant_rows(a):
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    a = a.contiguous()
    out = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=a.device)
    scale = torch.empty((a.shape[0], 1), dtype=torch.float32, device=a.device)
    if a.shape[0]:
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, a, scale, None)
    return out, scale


def stage_bound(r, sigma, k, n_mul):
    """``fused_bound.dense_bound`` from precomputed r / sigma (S = 1, rounded to bf16)."""
    e_acc = (gamma(k + 3, U_ACC) + gamma(k, U64)) * sigma
    e_pre = e_acc + gamma(n_mul, U32) * (r.abs() + e_acc)
    return e_pre + 0.5 * bf16_ulp(r.abs() + e_pre)


def within(out, r, bound):
    d = (out.double() - r).abs()
    ratio = d / bound.clamp(min=1e-300)
    return {"violations": int((d > bound).sum()), "elements": int(d.numel()),
            "max_diff_over_bound": float(ratio.max()) if d.numel() else 0.0}


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------

class Layer:
    pass


def build_layer(args, data, layer_idx, device):
    import torch
    from tessera.native_window_moe import native_window_moe_from_bundles
    from tessera.routed_fused import FusedRoutedWindowMoE, fused_routed_window_supported
    from tessera.serving import moe_route

    L = Layer()
    L.index = layer_idx
    t0 = time.time()
    wires = _load_wires(data, [layer_idx], args.experts)[layer_idx]
    declared = _schemes(data, [layer_idx], args.experts)[layer_idx]
    L.declared = declared
    L.read_s = time.time() - t0
    t0 = time.time()
    intake = moe_route._RankLocalPackedIntake(declared, _target(layer_idx), device, args.rank, args.tp)
    if not intake.compact:
        raise SystemExit("the intake did not take the compact window path")
    lengths = (torch.zeros(args.experts, 2, dtype=torch.long), torch.zeros(args.experts, dtype=torch.long))
    for e in range(args.experts):
        for group, index, proj in MOE_UNITS:
            wire = wires[e][proj]
            intake.load(group, index, e, wire, device=device)
            if group == "w13":
                lengths[0][e, index] = wire.numel()
            else:
                lengths[1][e] = wire.numel()
    plans, roles = intake.plans, intake.roles
    bundles = intake.finish(*lengths)
    del intake
    torch.cuda.synchronize()
    L.load_s = time.time() - t0
    L.bundles = bundles
    L.support = fused_routed_window_supported(bundles.gate, bundles.up, bundles.down)
    L.fused = FusedRoutedWindowMoE.from_bundles(bundles.gate, bundles.up, bundles.down, activation="silu")
    L.compact = native_window_moe_from_bundles(bundles.down, gate=bundles.gate, up=bundles.up,
                                               activation="silu")
    L.served_adapter = type(bundles.adapter()).__name__
    L.runs = {p: [[int(v) for v in row] for row in b.runs_all.reshape(args.experts, -1, 4)[0].tolist()]
              for p, b in (("gate", bundles.gate), ("up", bundles.up), ("down", bundles.down))}
    L.has_init = {p: int(b.has_init.sum()) for p, b in (("gate", bundles.gate), ("up", bundles.up),
                                                           ("down", bundles.down))}
    # --- the reference: full wire -> materialising reader -> dense decode -> cut
    t0 = time.time()
    L.cut = {}
    cut_of = {}
    for group, index, proj in MOE_UNITS:
        role = roles[group][index]
        name = str(role["roles"][0][0])
        plan = plans[group]
        rs = plan.role(name)
        whole = bool(rs.is_whole)
        axis = plan.axis
        lo, hi = (None, None) if whole else (int(rs.lo), int(rs.hi))
        L.cut[proj] = {"role": name, "axis": str(axis), "lo": lo, "hi": hi, "whole": whole}
        cut_of[proj] = (role, axis, lo, hi, whole)
    inter = int(declared["intermediate_size"])
    half = inter // args.tp
    want = {"gate_proj": ("row", args.rank * half, (args.rank + 1) * half),
            "up_proj": ("row", args.rank * half, (args.rank + 1) * half),
            "down_proj": ("col", args.rank * half, (args.rank + 1) * half)}
    for proj, (ax, lo, hi) in want.items():
        c = L.cut[proj]
        if args.tp > 1 and (not str(c["axis"]).startswith(ax[0]) or (c["lo"], c["hi"]) != (lo, hi)):
            raise SystemExit(f"layer {layer_idx} {proj}: plan cut {c} is not the TP{args.tp} rank "
                             f"{args.rank} contract {ax}[{lo}:{hi}]")
    from tessera.serving.scheme import parse_tessera_expert_blob
    from tessera.stock import materialize_stock

    W, S, R = {}, {}, {}
    for group, index, proj in MOE_UNITS:
        role, axis, lo, hi, whole = cut_of[proj]
        for e in range(args.experts):
            blob = wires[e][proj].contiguous().numpy().tobytes()
            parsed = parse_tessera_expert_blob(blob, role, f"{_target(layer_idx)} {proj} expert {e}",
                                               device=device)
            if len(parsed) != 1:
                raise SystemExit(f"{proj} expert {e}: {len(parsed)} roles")
            pu = parsed[0][1]
            tiles = materialize_stock(pu.unit, pu.forests, pu.code)
            w = tiles["weight"].to(device)
            s = tiles["weight_scale"].to(device).reshape(-1).float()
            rates = getattr(pu.unit, "rates", None)
            rates = torch.tensor([int(r) for r in rates], dtype=torch.int8) if rates is not None else None
            if rates is not None and int(rates.numel()) != int(tiles["weight"].shape[1]):
                rates = None          # not a per-column table; attribution falls back to "-1"
            if not whole:
                if str(axis).startswith("r"):
                    w, s = w[lo:hi], s[lo:hi]
                else:
                    w = w[:, lo:hi]
                    rates = rates[lo:hi] if rates is not None else None
            if proj not in W:
                W[proj] = torch.empty((args.experts,) + tuple(w.shape), dtype=torch.float8_e4m3fn,
                                      device=device)
                S[proj] = torch.empty((args.experts, int(s.numel())), dtype=torch.float32, device=device)
                R[proj] = torch.empty((args.experts, int(w.shape[1])), dtype=torch.int8)
            W[proj][e] = w.view(torch.float8_e4m3fn) if w.dtype != torch.float8_e4m3fn else w
            S[proj][e] = s
            if rates is not None:
                R[proj][e] = rates
            else:
                R[proj][e] = -1
            del parsed, pu, tiles, w, s
        torch.cuda.synchronize()
    L.W, L.S, L.rates = W, S, R
    L.ref_s = time.time() - t0
    del wires
    return L


# ---------------------------------------------------------------------------
# 1. one-hot extraction
# ---------------------------------------------------------------------------

def one_hot(args, L, device):
    import torch

    fused, b = L.fused, L.bundles
    E = args.experts
    inter_l = int(b.down.cols)
    hidden = int(b.down.rows)
    out = {}
    nb = args.onehot_chunk
    for stage in ("gate_up", "down"):
        k_dim = hidden if stage == "gate_up" else inter_l
        x = torch.eye(k_dim, device=device).bfloat16()
        xq, a = fp8_quant_rows(x)
        a = a.reshape(-1).contiguous().float()
        hot = xq.float().diagonal()
        projs = ("gate_proj", "up_proj") if stage == "gate_up" else ("down_proj",)
        acc = {(lane, p): {"mismatch": 0, "elements": 0, "max_ulps": 0.0,
                           "by_rate": {}, "start_rows_mismatch": 0, "experts_with_mismatch": [],
                           "d2": 0.0, "r2": 0.0}
               for lane in ("fused", "compact") for p in projs}
        for e0 in range(0, E, nb):
            e1 = min(E, e0 + nb)
            n = e1 - e0
            if stage == "gate_up":
                ids = torch.arange(e0, e1, device=device, dtype=torch.int32).expand(k_dim, n).contiguous()
                rw = torch.ones(k_dim, n, device=device)
                gu = fused.gate_up(xq, ids, rw, a_scale=a, preserve=True)            # [K, n, 2I]
                got = {("fused", "gate_proj"): gu[..., :inter_l], ("fused", "up_proj"): gu[..., inter_l:]}
                got[("compact", "gate_proj")] = b.gate(xq, ids, rw, a_scale=a, preserve=True,
                                                       apply_router_weight_on_input=False)
                got[("compact", "up_proj")] = b.up(xq, ids, rw, a_scale=a, preserve=True,
                                                   apply_router_weight_on_input=False)
            else:
                xq_all = xq.repeat(n, 1).contiguous()
                a_all = a.repeat(n).contiguous()
                ids = torch.arange(e0, e1, device=device, dtype=torch.int32).repeat_interleave(k_dim).reshape(-1, 1)
                rw = torch.ones(n * k_dim, 1, device=device)
                f = fused.down_routes(xq_all, ids, rw, a_scale=a_all, route_input=True, round_routes=True)
                c = b.down(xq_all, ids, rw, a_scale=a_all, route_input=True,
                           apply_router_weight_on_input=False, round_routes=True)
                got = {("fused", "down_proj"): f.reshape(n, k_dim, hidden).permute(1, 0, 2),
                       ("compact", "down_proj"): c.reshape(n, k_dim, hidden).permute(1, 0, 2)}
            for j, e in enumerate(range(e0, e1)):
                for p in projs:
                    w = L.W[p][e].float()                               # [rows, K]
                    s = L.S[p][e]
                    want = (((w.t() * hot[:, None]) * a[:, None]) * s[None, :]).bfloat16()   # [K, rows]
                    exact = (w.t().double() * s.double()[None, :])       # the weight itself, fp64
                    rates = L.rates[p][e].to(device).long()              # [K]
                    for lane in ("fused", "compact"):
                        g = got[(lane, p)][:, j, :]
                        st = acc[(lane, p)]
                        bad = g != want
                        nbad = int(bad.sum())
                        st["mismatch"] += nbad
                        st["elements"] += int(bad.numel())
                        gd = g.double()
                        wd = want.double()
                        if nbad:
                            ulps = ((gd - wd).abs() / bf16_ulp(wd.abs())).max()
                            st["max_ulps"] = max(st["max_ulps"], float(ulps))
                            if len(st["experts_with_mismatch"]) < 32:
                                first = bad.nonzero()[0].tolist()
                                st["experts_with_mismatch"].append(
                                    {"expert": e, "count": nbad, "first_col_row": first,
                                     "got": float(gd[first[0], first[1]]),
                                     "want": float(wd[first[0], first[1]])})
                            st["start_rows_mismatch"] += int(bad[:, :4].sum())
                        per_col = bad.sum(dim=1)                             # [K]
                        for r in rates.unique().tolist():
                            m = rates == r
                            key = str(r)
                            br = st["by_rate"].setdefault(key, {"cols": 0, "mismatch": 0})
                            br["cols"] += int(m.sum())
                            br["mismatch"] += int(per_col[m].sum())
                        # effective weight (extraction / (hot * a)) against the exact weight
                        eff = gd / (hot.double() * a.double())[:, None]
                        st["d2"] += float(((eff - exact) ** 2).sum())
                        st["r2"] += float((exact ** 2).sum())
            del got
            torch.cuda.synchronize()
        for key, st in acc.items():
            st["rel_frob_effective_vs_exact_weight"] = math.sqrt(st.pop("d2") / st.pop("r2"))
            out[f"{key[0]}:{key[1]}"] = st
    return out


# ---------------------------------------------------------------------------
# 2. the forward
# ---------------------------------------------------------------------------

def make_inputs(m, experts, top_k, hidden, route, seed, device):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, hidden, generator=g)
    # A post-RMSNorm hidden state carries a few massive channels; they set the
    # per-token E4M3 scale, which is what this exercises.
    outliers = torch.randperm(hidden, generator=g)[:16]
    x[:, outliers] *= 30.0
    x = x * torch.exp(0.5 * torch.randn(m, 1, generator=g))       # per-token magnitude spread
    x = x.bfloat16()
    if route == "spread":
        ids = torch.rand(m, experts, generator=g).topk(top_k, dim=1).indices
    else:
        ids = torch.randperm(experts, generator=g)[:top_k].expand(m, top_k)
    scores = torch.sigmoid(torch.randn(m, top_k, generator=g))
    w = scores / scores.sum(dim=1, keepdim=True)
    return (x.to(device), ids.to(torch.int32).contiguous().to(device),
            w.float().contiguous().to(device), outliers.tolist())


def forward_case(args, L, m, route, device):
    import torch
    from tessera.native_window_moe import _silu_and_mul

    fused, compact, b = L.fused, L.compact, L.bundles
    E, K = args.experts, args.top_k
    hidden, inter_l = int(b.down.rows), int(b.down.cols)
    lim = args.swiglu_limit
    x, ids, w, outliers = make_inputs(m, E, K, hidden, route, 7000 + m + (0 if route == "spread" else 1), device)
    T, Rn = m, m * K
    rec = {"m": m, "route": route, "experts_hit": int(torch.unique(ids).numel()),
           "outlier_channels": outliers}
    with torch.inference_mode():
        out_f = fused(x, ids, w, swiglu_limit=lim, apply_router_weight_on_input=False)
        out_f2 = fused(x, ids, w, swiglu_limit=lim, apply_router_weight_on_input=False)
        out_c = compact(x, ids, w, swiglu_limit=lim, apply_router_weight_on_input=False)
        gu_f = fused.gate_up(x, ids, w, preserve=True)                              # [T, K, 2I]
        act_f = _silu_and_mul(gu_f[..., :inter_l].reshape(Rn, inter_l),
                              gu_f[..., inter_l:].reshape(Rn, inter_l), clamp_limit=lim)
        staged_f = fused.down_routes(act_f, ids, w, route_input=True, round_routes=True)
        g_c = b.gate(x, ids, w, preserve=True, apply_router_weight_on_input=False).reshape(Rn, inter_l)
        u_c = b.up(x, ids, w, preserve=True, apply_router_weight_on_input=False).reshape(Rn, inter_l)
    rec["fused_deterministic"] = bool(torch.equal(out_f, out_f2))
    rec["fused_equals_staged_composition"] = int((out_f != staged_f).sum())
    # ---- reference, stage 1
    xq, s1 = fp8_quant_rows(x)
    A = xq.double() * s1.double()
    route_e = ids.reshape(-1).long()
    route_t = torch.arange(T, device=device).repeat_interleave(K)
    g_ex = torch.zeros(Rn, inter_l, dtype=torch.float64, device=device)
    u_ex = torch.zeros_like(g_ex)
    sg = torch.zeros_like(g_ex)
    su = torch.zeros_like(g_ex)
    experts = torch.unique(route_e).tolist()
    idx_of = {}
    for e in experts:
        idx = (route_e == e).nonzero().reshape(-1)
        idx_of[e] = idx
        Ar = A.index_select(0, route_t.index_select(0, idx))
        for proj, r, s in (("gate_proj", g_ex, sg), ("up_proj", u_ex, su)):
            Wd = L.W[proj][e].double() * L.S[proj][e].double()[:, None]
            r.index_copy_(0, idx, Ar @ Wd.t())
            s.index_copy_(0, idx, Ar.abs() @ Wd.abs().t())
    b_g = stage_bound(g_ex, sg, hidden, 2)
    b_u = stage_bound(u_ex, su, hidden, 2)
    g_f = gu_f[..., :inter_l].reshape(Rn, inter_l)
    u_f = gu_f[..., inter_l:].reshape(Rn, inter_l)
    rec["stage_gate_up_bound"] = {
        "fused_gate": within(g_f, g_ex, b_g), "fused_up": within(u_f, u_ex, b_u),
        "compact_gate": within(g_c, g_ex, b_g), "compact_up": within(u_c, u_ex, b_u)}
    g_l = g_ex.to(torch.bfloat16)
    u_l = u_ex.to(torch.bfloat16)
    act_l = _silu_and_mul(g_l, u_l, clamp_limit=lim)                          # [R, I] bf16
    gc = torch.clamp(g_ex, max=lim) if lim is not None else g_ex
    uc = torch.clamp(u_ex, min=-lim, max=lim) if lim is not None else u_ex
    act_ex = torch.nn.functional.silu(gc) * uc                                 # fp64
    # ---- reference, stage 2 (exact path and lane path)
    aq_ex, s2_ex = fp8_quant_rows(act_ex.float())
    aq_l, s2_l = fp8_quant_rows(act_l)
    A2_ex = aq_ex.double() * s2_ex.double()
    A2_l = aq_l.double() * s2_l.double()
    wr = w.reshape(-1).double()[:, None]
    d_ex = torch.zeros(Rn, hidden, dtype=torch.float64, device=device)
    d_l = torch.zeros_like(d_ex)
    sd = torch.zeros_like(d_ex)
    for e in experts:
        idx = idx_of[e]
        Wd = L.W["down_proj"][e].double() * L.S["down_proj"][e].double()[:, None]
        d_ex.index_copy_(0, idx, (A2_ex.index_select(0, idx) @ Wd.t()) * wr.index_select(0, idx))
        Al = A2_l.index_select(0, idx)
        d_l.index_copy_(0, idx, (Al @ Wd.t()) * wr.index_select(0, idx))
        sd.index_copy_(0, idx, (Al.abs() @ Wd.abs().t()) * wr.index_select(0, idx).abs())
    b_d = stage_bound(d_l, sd, inter_l, 3)
    with torch.inference_mode():
        ids1 = ids.reshape(-1, 1).contiguous()
        w1 = w.reshape(-1, 1).contiguous()
        dr_f = fused.down_routes(act_l, ids1, w1, route_input=True, round_routes=True)   # [R, H]
        dr_c = b.down(act_l, ids1, w1, route_input=True, apply_router_weight_on_input=False,
                      round_routes=True)
    rec["stage_down_bound"] = {"fused": within(dr_f, d_l, b_d), "compact": within(dr_c, d_l, b_d)}
    ref_exact = d_ex.reshape(T, K, hidden).sum(1)
    routes_bf = d_l.to(torch.bfloat16).float().reshape(T, K, hidden)
    acc = torch.zeros(T, hidden, dtype=torch.float32, device=device)
    for j in range(K):
        acc = acc + routes_bf[:, j, :]
    ref_lane = acc.to(torch.bfloat16)
    rec["forward"] = {
        "noise_floor(ref_lane-ref_exact)": metrics(ref_lane, ref_exact),
        "fused-ref_exact": metrics(out_f, ref_exact),
        "compact-ref_exact": metrics(out_c, ref_exact),
        "fused-ref_lane": metrics(out_f, ref_lane.double()),
        "compact-ref_lane": metrics(out_c, ref_lane.double()),
        "fused-compact": metrics(out_f, out_c.double()),
    }
    # ---- per expert (teacher-forced stages)
    per_expert = []
    for e in experts:
        idx = idx_of[e]
        row = {"expert": e, "routes": int(idx.numel())}
        for name, lane, ref, floor_ref in (
                ("gate", (g_f, g_c), g_ex, g_l), ("up", (u_f, u_c), u_ex, u_l),
                ("down", (dr_f, dr_c), d_l, d_l.to(torch.bfloat16))):
            r = ref.index_select(0, idx)
            fl = metrics(floor_ref.index_select(0, idx), r)
            f = metrics(lane[0].index_select(0, idx), r)
            c = metrics(lane[1].index_select(0, idx), r)
            fc = metrics(lane[0].index_select(0, idx), lane[1].index_select(0, idx).double())
            row[name] = {"floor_rel": fl["rel_frob"], "fused_rel": f["rel_frob"], "compact_rel": c["rel_frob"],
                         "fused_compact_rel": fc["rel_frob"], "fused_cos": f["cosine"],
                         "compact_cos": c["cosine"], "fused_max_abs": f["max_abs"],
                         "compact_max_abs": c["max_abs"], "floor_max_abs": fl["max_abs"]}
        per_expert.append(row)
    rec["per_expert"] = per_expert
    worst = {}
    for name in ("gate", "up", "down"):
        for lane in ("fused", "compact"):
            ratios = [(r[name][f"{lane}_rel"] / max(r[name]["floor_rel"], 1e-300), r["expert"])
                      for r in per_expert]
            ratios.sort(reverse=True)
            worst[f"{lane}:{name}"] = {"max_rel_over_floor": ratios[0][0], "expert": ratios[0][1],
                                       "median_rel_over_floor": ratios[len(ratios) // 2][0]}
    rec["per_expert_worst"] = worst
    del g_ex, u_ex, sg, su, d_ex, d_l, sd, A, A2_ex, A2_l
    torch.cuda.empty_cache()
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--ms", default="1,16,2048")
    ap.add_argument("--routes", default="spread,concentrated")
    ap.add_argument("--swiglu-limit", type=float, default=10.0,
                    help="the model's swiglu_limit (GLM-5.3 config: 10.0); <= 0 means none")
    ap.add_argument("--onehot-chunk", type=int, default=8)
    ap.add_argument("--no-onehot", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.swiglu_limit is not None and args.swiglu_limit <= 0:
        args.swiglu_limit = None
    import torch
    import tessera

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cuda", torch.cuda.current_device())
    data = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = {"tessera_file": tessera.__file__, "torch": torch.__version__,
            "routed_fused_env": os.environ.get("TESSERA_ROUTED_FUSED"),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "device": torch.cuda.get_device_name(device), "args": vars(args)}
    for layer_idx in [int(v) for v in args.layers.split(",")]:
        log(f"layer {layer_idx}: building")
        L = build_layer(args, data, layer_idx, device)
        rec = dict(head)
        rec.update({"layer": layer_idx, "read_s": L.read_s, "load_s": L.load_s, "ref_s": L.ref_s,
                    "q256": {g: int(L.declared["groups"][g]["q256"]) for g in ("w13", "w2")},
                    "support_reason": L.support, "served_adapter": L.served_adapter,
                    "fused_launch_pair": list(L.fused.launch_pair),
                    "compact_launch_pair": list(L.compact.launch_pair),
                    "runs": L.runs, "has_init_experts": L.has_init, "cut": L.cut,
                    "ref_rates_present": {p: bool((L.rates[p] >= 0).all()) for p in L.rates},
                    "peak_allocated_after_build": int(torch.cuda.max_memory_allocated())})
        log(f"layer {layer_idx}: read {L.read_s:.1f}s load {L.load_s:.1f}s ref {L.ref_s:.1f}s "
            f"runs {L.runs} support={L.support!r} served={L.served_adapter}")
        if not args.no_onehot:
            t0 = time.time()
            rec["one_hot"] = one_hot(args, L, device)
            log(f"layer {layer_idx}: one-hot {time.time() - t0:.1f}s " + json.dumps(
                {k: (v["mismatch"], v["elements"], v["max_ulps"]) for k, v in rec["one_hot"].items()}))
        rec["cases"] = []
        for route in args.routes.split(","):
            for m in [int(v) for v in args.ms.split(",")]:
                t0 = time.time()
                case = forward_case(args, L, m, route, device)
                case["seconds"] = time.time() - t0
                rec["cases"].append(case)
                fwd = case["forward"]
                log(f"layer {layer_idx} {route} M={m}: " + json.dumps(
                    {k: round(v["rel_frob"], 7) for k, v in fwd.items()})
                    + f" staged_diff={case['fused_equals_staged_composition']}"
                    + " stage_violations=" + json.dumps(
                        {k: v["violations"] for k, v in {**case["stage_gate_up_bound"],
                                                          **{f"down_{a}": b for a, b in case["stage_down_bound"].items()}}.items()}))
        rec["peak_allocated"] = int(torch.cuda.max_memory_allocated())
        path = out_dir / f"{args.label}-L{layer_idx}.json"
        path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        log(f"layer {layer_idx}: wrote {path}")
        del L
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
