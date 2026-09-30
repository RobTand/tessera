"""Numerics oracle for the fused E2M1 family of ``routed_fused_window.cu``.

    python3 experiments/t4_code/fused_e2m1_check.py --out DIR [--quick]

Builds the E2M1 library (``-DTESSERA_ROUTED_FUSED_FP4=1``, sm_121a) straight
from the source, encodes synthetic weights with the window body over the
LUT16 plane (``tessera.export.encode_linear``, L = 14) at one-run and two-run
tables, packs each unit with the compact loader
(``compact_prep.prepare_window_lut_compact``), stacks the experts, and runs
every launch the library has against a float64 reference built from the
reader's own tuple codes and scale bytes (``decode._decode_window``, the LUT;
the decode oracle ``e2m1_decode_oracle.py`` ties those to the artifact bit
for bit):

* ``onehot``: each activation row has one nonzero, so every output is one
  exact product; the kernel must equal the reference's single fp32 rounding
  sequence (``bf16(fp32(a w) * ratio [* rw])``) exactly -- this pins every
  row, column, scale group and tile position;
* ``random``: dense activations through ``scaled_fp4_quant``; the error
  against the float64 sum must stay within one bf16 rounding plus the fp32
  accumulation bound ``K * 2^-23 * sum |a w| * ratio``;
* the same at the TP2 rank-1 row cut (a carried start state), at partial
  superblocks, at two-run tables, and at the dense launch with and without a
  K split; and a CUDA-graph replay of each launch must equal its eager call
  bit for bit.

Writes ``DIR/fused_e2m1_check.json``; exits nonzero on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tessera import routed_fused as rf                                          # noqa: E402
from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact  # noqa: E402
from tessera.decode import _decode_window                                       # noqa: E402
from tessera.export import encode_linear                                        # noqa: E402
from tessera.lane_planes import lut_scale_bytes                                 # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind                           # noqa: E402
from tessera.unit_artifact import parse_unit_artifact                          # noqa: E402

from tessera.alphabet import E2M1_GRID, tuple_grid                              # noqa: E402

GRID = tuple_grid(E2M1_GRID, 2)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)
SRC = ROOT / "src" / "tessera" / "serving" / "csrc" / "routed_fused_window.cu"


def build(out: Path):
    from torch.utils.cpp_extension import load

    b = out / "build"
    b.mkdir(parents=True, exist_ok=True)
    return load(name="tessera_routed_fused_e2m1_dev", sources=[str(SRC)], build_directory=str(b),
                extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-DTESSERA_ROUTED_FUSED_FP8=0",
                                   "-DTESSERA_ROUTED_FUSED_MMA8=0", "-DTESSERA_ROUTED_FUSED_FP4=1",
                                   "-gencode", "arch=compute_121a,code=sm_121a"],
                verbose=False)


# ----------------------------------------------------------------------------- units
def encode(rows, cols, q256, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(rows, cols, generator=g) * 0.02).to(dev)
    return encode_linear(w, grid=GRID, q256=q256, body=BodyKind.WINDOW,
                         scale_plane=ScalePlaneKind.LUT, window_bits=14).blob


def ref_weight(blob, cut, dev):
    """float64 [rows, cols]: e2m1(code) * e4m3(lut[nibble]) -- the unit global excluded."""
    unit = parse_unit_artifact(blob, device=dev).unit
    steps, cols = unit.body_bits.shape
    n = steps * 2
    r0, r1 = cut if cut is not None else (0, n)
    codes = _decode_window(unit, GRID, torch.int64)
    nib = torch.stack([codes >> 4, codes & 15], dim=1).reshape(n, cols)[r0:r1]
    lut16 = lut_scale_bytes(unit.scale_lut, dev)
    idx = unit.scale_refine.to(dev).reshape(n, cols // int(unit.half)).long()[r0:r1]
    sf = lut16[idx].view(torch.float8_e4m3fn).double().repeat_interleave(int(unit.half), dim=1)
    return E2M1.to(dev)[nib] * sf


def chunk_desc4(perm, n_lo, cols):
    """int32 [cols / 64, 4]: (hi mask cols 0-31, hi mask cols 32-63, low-rate columns before, 0)."""
    hi = torch.zeros(cols, dtype=torch.bool)
    hi[perm.cpu().long()[n_lo:]] = True
    hi = hi.view(cols // 64, 64).to(torch.int64)
    sh = torch.arange(32, dtype=torch.int64)
    m0 = (hi[:, :32] << sh).sum(1)
    m1 = (hi[:, 32:] << sh).sum(1)
    lo = 64 - hi.sum(1)
    lo_before = torch.cumsum(lo, 0) - lo
    d = torch.stack([m0, m1, lo_before, torch.zeros_like(lo)], dim=1)
    d = torch.where(d >= 2 ** 31, d - 2 ** 32, d)
    return d.to(torch.int32)


def stack(blobs, cut, dev):
    """The per-expert units of one projection, stacked as the launch takes them."""
    units = []
    for b in blobs:
        wire = parse_compact_wire(b, device=dev, name="w")
        units.append(prepare_window_lut_compact(wire, device=dev, **({"rows": cut} if cut else {})))
    u0 = units[0]
    rows, cols = int(u0.rows), int(u0.cols)
    pairs, descs, inits, has = [], [], [], []
    for u in units:
        pair, why = rf.run_pair(u.rep.runs, cols)
        if pair is None:
            raise RuntimeError(f"run_pair refused: {why}")
        pairs.append(pair.to(dev).to(torch.int32))
        descs.append(chunk_desc4(u.rep.perm, int(pair[2]), cols))
        init = u.permuted_start_state()
        has.append(init is not None)
        inits.append(init.to(torch.int32) if init is not None else torch.zeros(cols, dtype=torch.int32, device=dev))
    tws = {int(u.rep.tile_words) for u in units}
    if len(tws) != 1:
        raise RuntimeError(f"experts disagree on tile_words: {sorted(tws)}")
    wmax = max(u.rep.words.numel() for u in units)
    wmax = -(-wmax // 4) * 4
    words = torch.zeros(len(units), wmax, dtype=torch.int32, device=dev)
    for e, u in enumerate(units):
        words[e, :u.rep.words.numel()] = u.rep.words.reshape(-1)
    pmax = max(u.scale_plane.numel() for u in units)
    pmax = -(-pmax // 16) * 16
    plane = torch.zeros(len(units), pmax, dtype=torch.uint8, device=dev)
    for e, u in enumerate(units):
        plane[e, :u.scale_plane.numel()] = u.scale_plane.reshape(-1)
    return {
        "rows": rows, "cols": cols, "tile_words": tws.pop(),
        "words": words, "codes": torch.stack([u.codes for u in units]).contiguous(),
        "init": torch.stack(inits).contiguous(),
        "has_init": torch.tensor([int(h) for h in has], dtype=torch.int32, device=dev),
        "plane": plane, "lut": torch.stack([u.scale_lut.reshape(16) for u in units]).contiguous(),
        "global": torch.tensor([u.global_scale for u in units], dtype=torch.float64),
        "runs": torch.stack(pairs).contiguous(), "desc": torch.stack(descs).to(dev).contiguous(),
        "slot_words": rf.slot_words_for_pair(pairs[0].cpu()),
        "wref": [ref_weight(b, cut, dev) for b in blobs],
        "has_any_init": any(has), "runs_table": [int(v) for v in pairs[0].tolist()],
    }


# ----------------------------------------------------------------------------- activations
def quantize(x, dev):
    """``scaled_fp4_quant`` at the tensor's own global: (codes, sfa, A_deq float64, gs)."""
    from tessera.kernel_a4 import a4_quantize_activation

    amax = float(x.float().abs().max())
    gs = 448.0 * 6.0 / max(amax, 1e-12)
    codes, sfa = a4_quantize_activation(x, torch.tensor(gs, dtype=torch.float32, device=dev))
    return codes.contiguous(), sfa.view(torch.uint8).contiguous(), a_deq(codes, sfa), gs


def a_deq(codes, sfa):
    c = codes.to(torch.int64)
    nib = torch.stack([c & 15, c >> 4], dim=2).reshape(c.shape[0], -1)
    return E2M1.to(codes.device)[nib] * sfa.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)


def onehot_x(rows, cols, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.zeros(rows, cols, dtype=torch.bfloat16)
    k = torch.randint(0, cols, (rows,), generator=g)
    v = (torch.rand(rows, generator=g) * 3 + 0.5) * torch.where(torch.rand(rows, generator=g) < 0.5, -1.0, 1.0)
    x[torch.arange(rows), k] = v.to(torch.bfloat16)
    return x.to(dev)


# ----------------------------------------------------------------------------- comparisons
def bf16_ulp(v):
    a = v.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def compare(got, ref, bound, exact):
    """got bf16, ref float64 (pre-rounding), bound float64 accumulation bound."""
    g = got.double()
    err = (g - ref).abs()
    if exact:
        want = ref.float().to(torch.bfloat16).double()   # the reference's one rounding sequence
        bad = int((g != want).sum())
        return {"mismatch": bad, "max_err": float(err.max()) if err.numel() else 0.0, "ok": bad == 0}
    tol = bf16_ulp(ref) + bound * (1 + 2.0 ** -7) + bf16_ulp(ref + bound) * 0
    excess = err - tol
    bad = int((excess > 0).sum())
    rel = err / (bf16_ulp(ref) + bound).clamp_min(1e-30)
    return {"mismatch": bad, "max_err": float(err.max()) if err.numel() else 0.0,
            "max_err_over_tol": float(rel.max()) if rel.numel() else 0.0, "ok": bad == 0}


def graph_equal(fn, out):
    """Capture ``fn`` (which writes ``out``) and replay it: the same bits as eager."""
    fn()
    torch.cuda.synchronize()
    eager = out.clone()
    out.zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    return bool(torch.equal(out.view(torch.int16), eager.view(torch.int16)))


# ----------------------------------------------------------------------------- the routed launches
def routing(counts, top_k, seed, dev):
    """expert ids with exactly ``counts[e]`` routes each, shuffled over [T, top_k]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.cat([torch.full((c,), e, dtype=torch.int64) for e, c in enumerate(counts)])
    P = ids.numel()
    assert P % top_k == 0
    ids = ids[torch.randperm(P, generator=g)]
    rw = torch.rand(P, generator=g).float() + 0.25
    E = len(counts)
    cnt = torch.tensor(counts, dtype=torch.int32)
    offsets = torch.zeros(E + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(cnt, 0)
    order = torch.argsort(ids, stable=True)
    item_off = torch.zeros(E + 1, dtype=torch.int32)
    item_off[1:] = torch.cumsum((cnt + rf.BM - 1) // rf.BM, 0)
    return {"ids": ids.to(dev), "offsets": offsets.to(dev), "flat_sorted": order.to(torch.int32).to(dev),
            "rw_sorted": rw[order].contiguous().to(dev), "rw": rw.to(dev), "item_off": item_off.to(dev),
            "T": P // top_k, "top_k": top_k, "P": P}


def run_routed(lib, mode, xq, sfa, s0, s1, ratio0, ratio1, R, a_row_mode, mul_weight, out, counter):
    dev = xq.device
    lib.routed_fused_forward_fp4(
        int(mode), xq, sfa,
        s0["words"], s1["words"], s0["codes"], s1["codes"], s0["init"], s1["init"],
        s0["has_init"], s1["has_init"], s0["plane"], s1["plane"], s0["lut"], s1["lut"],
        ratio0, ratio1, s0["runs"], s1["runs"], s0["desc"], s1["desc"],
        int(s0["rows"]), int(s0["tile_words"]), int(s0["slot_words"]),
        R["offsets"], R["flat_sorted"], R["rw_sorted"], R["item_off"], counter,
        int(R["top_k"]), int(a_row_mode), bool(mul_weight), float("inf"), out,
        int(torch.cuda.get_device_properties(dev).multi_processor_count))


def routed_case(lib, gate, up, down, R, kind, dev, seed):
    """gate/up (mode 1 and mode 0) then down (mode 2) against the reference."""
    res = {}
    E = len(gate["wref"])
    K1, I = gate["cols"], gate["rows"]
    T, top_k, P = R["T"], R["top_k"], R["P"]
    x = onehot_x(T, K1, seed, dev) if kind == "onehot" else torch.randn(T, K1, dtype=torch.bfloat16, device=dev)
    xq, sfa, A, gs = quantize(x, dev)
    rg = (gate["global"] / gs).float().to(dev)
    ru = (up["global"] / gs).float().to(dev)
    counter = torch.zeros(4, dtype=torch.int32, device=dev)
    ids = R["ids"]
    tok = torch.arange(P, device=dev) // top_k
    # reference accumulators per route (float64), and the |a w| sums
    accg = torch.zeros(P, I, dtype=torch.float64, device=dev)
    accu = torch.zeros_like(accg)
    absg = torch.zeros_like(accg)
    absu = torch.zeros_like(accg)
    for e in range(E):
        sel = (ids == e).nonzero().reshape(-1)
        a = A[tok[sel]]
        accg[sel] = a @ gate["wref"][e].T
        accu[sel] = a @ up["wref"][e].T
        absg[sel] = a.abs() @ gate["wref"][e].abs().T
        absu[sel] = a.abs() @ up["wref"][e].abs().T
    ratg = rg.double()[ids].unsqueeze(1)
    ratu = ru.double()[ids].unsqueeze(1)
    exact = kind == "onehot"
    eps = K1 * 2.0 ** -23
    # mode 1: route-preserved [T, top_k, 2I]
    out1 = torch.empty(T, top_k, 2 * I, dtype=torch.bfloat16, device=dev)
    f1 = lambda: (counter[0:1].zero_(),
                  run_routed(lib, 1, xq, sfa, gate, up, rg, ru, R, 0, False, out1, counter[0:1]))
    f1()
    torch.cuda.synchronize()
    o = out1.reshape(P, 2 * I)
    if exact:
        refg = (accg.float() * rg[ids].unsqueeze(1)).double()
        refu = (accu.float() * ru[ids].unsqueeze(1)).double()
    else:
        refg, refu = accg * ratg, accu * ratu
    res["mode1_gate"] = compare(o[:, :I], refg, absg * ratg * eps, exact)
    res["mode1_up"] = compare(o[:, I:], refu, absu * ratu * eps, exact)
    res["mode1_graph"] = graph_equal(f1, out1)
    # mode 0: SwiGLU into sorted position [P, I]; reference from the kernel's
    # own rounded gate/up (mode 1 is checked above), in fp32 as the kernel does it
    out0 = torch.empty(P, I, dtype=torch.bfloat16, device=dev)
    f0 = lambda: (counter[1:2].zero_(),
                  run_routed(lib, 0, xq, sfa, gate, up, rg, ru, R, 0, False, out0, counter[1:2]))
    f0()
    torch.cuda.synchronize()
    gf = o[:, :I].float()
    uf = o[:, I:].float()
    act = (gf / (1.0 + torch.exp(-gf))) * uf
    want0 = act.to(torch.bfloat16)[R["flat_sorted"].long()]
    d0 = (out0.float() - want0.float()).abs()
    ulp0 = bf16_ulp(want0.double()).float()
    res["mode0"] = {"max_err_ulps": float((d0 / ulp0).max()), "ok": bool((d0 <= ulp0).all())}
    res["mode0_graph"] = graph_equal(f0, out0)
    # mode 2: down over the route-indexed activation (a_row_mode 2), weighted
    K2, H = down["cols"], down["rows"]
    xa = onehot_x(P, K2, seed + 1, dev) if exact else torch.randn(P, K2, dtype=torch.bfloat16, device=dev)
    xq2, sfa2, A2, gs2 = quantize(xa, dev)
    rd = (down["global"] / gs2).float().to(dev)
    accd = torch.zeros(P, H, dtype=torch.float64, device=dev)
    absd = torch.zeros_like(accd)
    for e in range(E):
        sel = (ids == e).nonzero().reshape(-1)
        accd[sel] = A2[sel] @ down["wref"][e].T
        absd[sel] = A2[sel].abs() @ down["wref"][e].abs().T
    out2 = torch.empty(P, H, dtype=torch.bfloat16, device=dev)
    f2 = lambda: (counter[2:3].zero_(),
                  run_routed(lib, 2, xq2, sfa2, down, down, rd, rd, R, 2, True, out2, counter[2:3]))
    f2()
    torch.cuda.synchronize()
    rw = R["rw"]
    if exact:
        ref2 = ((accd.float() * rd[ids].unsqueeze(1)) * rw.unsqueeze(1)).double()
    else:
        ref2 = accd * rd.double()[ids].unsqueeze(1) * rw.double().unsqueeze(1)
    res["mode2"] = compare(out2, ref2, absd * rd.double()[ids].unsqueeze(1) * rw.double().unsqueeze(1)
                           * K2 * 2.0 ** -23, exact)
    res["mode2_graph"] = graph_equal(f2, out2)
    # the served chain: mode 0's sorted activation into the down launch at
    # a_row_mode 1 (A row = sorted position), output by route, weighted
    xq3, sfa3, A3, gs3 = quantize(out0, dev)
    rd3 = (down["global"] / gs3).float().to(dev)
    pos_e = torch.repeat_interleave(torch.arange(E, device=dev),
                                    (R["offsets"][1:] - R["offsets"][:-1]).long())
    acc3 = torch.zeros(P, H, dtype=torch.float64, device=dev)
    abs3 = torch.zeros_like(acc3)
    for e in range(E):
        sel = (pos_e == e).nonzero().reshape(-1)
        acc3[sel] = A3[sel] @ down["wref"][e].T
        abs3[sel] = A3[sel].abs() @ down["wref"][e].abs().T
    scale3 = rd3.double()[pos_e].unsqueeze(1) * R["rw_sorted"].double().unsqueeze(1)
    ref3 = torch.empty_like(acc3)
    bnd3 = torch.empty_like(acc3)
    fs = R["flat_sorted"].long()
    ref3[fs] = acc3 * scale3
    bnd3[fs] = abs3 * scale3 * K2 * 2.0 ** -23
    out3 = torch.empty(P, H, dtype=torch.bfloat16, device=dev)
    f3 = lambda: (counter[3:4].zero_(),
                  run_routed(lib, 2, xq3, sfa3, down, down, rd3, rd3, R, 1, True, out3, counter[3:4]))
    f3()
    torch.cuda.synchronize()
    res["chain_mode0_mode2"] = compare(out3, ref3, bnd3, False)
    res["chain_graph"] = graph_equal(f3, out3)
    # the dequantised activation is the input the ratio assumes (x ~= A / gs)
    res["a_side_rel_err"] = float(((A / gs) - x.double()).norm() / x.double().norm().clamp_min(1e-30))
    res["ok"] = all(v["ok"] if isinstance(v, dict) else bool(v) for k, v in res.items()
                    if k != "a_side_rel_err")
    return res


def dense_case(lib, role, M, k_split, kind, dev, seed):
    K, N = role["cols"], role["rows"]
    x = onehot_x(M, K, seed, dev) if kind == "onehot" else torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    xq, sfa, A, gs = quantize(x, dev)
    ratio = (role["global"] / gs).float().to(dev)
    acc = A @ role["wref"][0].T
    absacc = A.abs() @ role["wref"][0].abs().T
    exact = kind == "onehot"
    ref = (acc.float() * ratio[0]).double() if exact else acc * float(ratio[0])
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    partial = torch.empty(k_split * M * N if k_split > 1 else 0, dtype=torch.float32, device=dev)

    def f():
        counter.zero_()
        lib.dense_forward_fp4(xq, sfa, role["words"], role["codes"], role["init"], role["has_init"],
                              role["plane"], role["lut"], ratio, role["runs"], role["desc"],
                              int(N), int(role["tile_words"]), int(role["slot_words"]), counter, int(k_split),
                              partial, out, int(torch.cuda.get_device_properties(dev).multi_processor_count))
    f()
    torch.cuda.synchronize()
    r = compare(out, ref, absacc * float(ratio[0]) * K * 2.0 ** -23, exact)
    r["graph"] = graph_equal(f, out)
    r["ok"] = r["ok"] and r["graph"]
    return r


def dense_refuses(lib, role, k_split, dev):
    """A split past the cap (an item of one chunk) is refused by name."""
    K, N = role["cols"], role["rows"]
    xq, sfa, _, gs = quantize(torch.randn(8, K, dtype=torch.bfloat16, device=dev), dev)
    ratio = (role["global"] / gs).float().to(dev)
    out = torch.empty(8, N, dtype=torch.bfloat16, device=dev)
    partial = torch.empty(k_split * 8 * N, dtype=torch.float32, device=dev)
    try:
        lib.dense_forward_fp4(xq, sfa, role["words"], role["codes"], role["init"], role["has_init"],
                              role["plane"], role["lut"], ratio, role["runs"], role["desc"],
                              int(N), int(role["tile_words"]), int(role["slot_words"]),
                              torch.zeros(1, dtype=torch.int32, device=dev), int(k_split), partial, out, 1)
    except RuntimeError as exc:
        return {"ok": "two K chunks" in str(exc), "message": str(exc)[:300]}
    return {"ok": False, "message": "accepted"}


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    lib = build(out)
    report = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "tessera_head": os.environ.get("TESSERA_HEAD"),
              "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
              "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
              "build_secs": round(time.time() - t0, 1), "cases": []}
    path = out / "fused_e2m1_check.json"
    # Shapes: gate/up I = 1280 rows (two tiles, the second partial) over H = 512;
    # down H = 512 rows over I = 1280 (20 chunks).  Four experts, routes 1, 37,
    # 64 and 130 (a single route, partial and whole superblocks), top_k 2.
    E, H, I = 4, 512, 1280
    counts = [1, 37, 64, 130]
    q256s = [512, 448, 384, 960, 1024, 128, 576] if not a.quick else [512, 448]
    failures = 0
    for q in q256s:
        for cut_name in (("whole", "rank1") if q in (448, 512) else ("whole",)):
            case = {"q256": q, "cut": cut_name}
            try:
                seed = 1000 * q
                gb = [encode(I, H, q, seed + e, dev) for e in range(E)]
                ub = [encode(I, H, q, seed + 100 + e, dev) for e in range(E)]
                db = [encode(H, I, q, seed + 200 + e, dev) for e in range(E)]
                cut_gu = (I // 2, I) if cut_name == "rank1" else None
                gate, up = stack(gb, cut_gu, dev), stack(ub, cut_gu, dev)
                down = stack(db, None, dev)          # rank 1 of the down cuts columns, not rows
                case.update(runs_gate=gate["runs_table"], runs_down=down["runs_table"],
                            has_init=gate["has_any_init"], tile_words=gate["tile_words"],
                            slot_words=gate["slot_words"])
                R = routing(counts, 2, q, dev)
                for kind in ("onehot", "random"):
                    case[f"routed_{kind}"] = routed_case(lib, gate, up, down, R, kind, dev, seed)
                if cut_name == "whole":
                    role = stack([db[0]], None, dev)
                    nk = role["cols"] // 64            # one FP4 instruction's K per chunk
                    for M in (1, 7, 64, 100, 300):
                        # 1, a middle split, and the cap (every item two chunks)
                        for ks in (1, 3, nk // 2):
                            for kind in ("onehot", "random"):
                                case[f"dense_M{M}_S{ks}_{kind}"] = dense_case(lib, role, M, ks, kind, dev, seed + M)
                    case["dense_split_cap_refused"] = dense_refuses(lib, role, nk // 2 + 1, dev)
                case["ok"] = all(v.get("ok", True) for v in case.values() if isinstance(v, dict))
            except Exception as exc:  # noqa: BLE001 -- recorded, counted as a failure
                import traceback
                case["error"] = f"{type(exc).__name__}: {exc}"
                case["traceback"] = traceback.format_exc()[-4000:]
                case["ok"] = False
            failures += 0 if case["ok"] else 1
            report["cases"].append(case)
            print(json.dumps({k: (v if not isinstance(v, dict) else v.get("ok")) for k, v in case.items()
                              if k != "traceback"}), flush=True)
            if not case["ok"]:
                for k, v in case.items():
                    if isinstance(v, dict) and not v.get("ok", True):
                        print("   FAIL", k, json.dumps(v)[:600], flush=True)
                if "traceback" in case:
                    print(case["traceback"], flush=True)
            report["failures"] = failures
            report["secs"] = round(time.time() - t0, 1)
            path.write_text(json.dumps(report, indent=1))
    print(f"failures={failures} secs={report['secs']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
