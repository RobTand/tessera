"""The fused window kernel's E2M1 family (``tessera.routed_fused_e2m1``).

The library runs the Tessera-4 wire (the E2M1x2 window body over the LUT16
scale plane) on the block-scaled FP4 instruction.  The oracle is the reader's
own decode: each expert's weight is ``e2m1(code) * ue4m3(lut[nibble])``
from ``decode._decode_window`` and the unit's LUT, and each activation is the
dequantised ``scaled_fp4_quant`` output the kernel consumes.  Held here:

* one-hot activations make every output one exact product, so the kernel must
  equal the reference's own rounding sequence bit for bit -- that pins every
  row, column, scale group and chunk position of every launch;
* dense activations stay within one bf16 rounding plus the fp32 accumulation
  bound ``K * 2^-23 * sum |a w| * ratio``, stage by stage on the exact input
  each stage consumes (gate/up, the SwiGLU epilogue, down, the served chain);
* a TP2 rank-1 row cut (a carried start state) computes the whole unit's rows
  bit for bit; two runs are bitwise equal; a CUDA-graph replay equals eager;
* the dense identity at M = 1..300 with and without a K split, and its split
  cap refused by name; rows that end inside a 256-row block (32, 128, 384:
  GLM-5.3's DSA indexer ``weights_proj`` and ``wk``, and a block and a half)
  write only their own rows, and a rank's row cut of a dense unit computes
  the whole unit's rows;
* the refusals by name, the chunk descriptors against a brute-force count,
  and the scope: the library is not reachable from ``tessera.serving``.

GPU cases need sm_121 (the instruction exists on the architecture-specific
target only) and the runtime's ``scaled_fp4_quant``; they run through
PrismaBuild inside the pinned serving image
(``experiments/routed_fused_tests.sh``).
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import routed_fused as rf                   # noqa: E402
from tessera import routed_fused_e2m1 as fe              # noqa: E402
from tessera.errors import GrammarError                  # noqa: E402


def _sm121() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)


gpu = pytest.mark.skipif(not _sm121(), reason="the block-scaled FP4 instruction is sm_121a")

E, H, I = 4, 512, 1280
COUNTS = [1, 37, 64, 130]          # a single route, partial and whole superblocks
TOP_K = 2
L = 14
#: q256 128 (rate 1), 448 (rates 3/4), 512 (rate 2), 960 (7/8), 1024 (rate
#: 8): both one-run extremes, an even rate, and two two-run tables, one at
#: the largest slot.
Q256 = [128, 448, 512, 960, 1024]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)


# ----------------------------------------------------------------------------- CPU
def _brute_desc(perm: torch.Tensor, n_lo: int, cols: int) -> torch.Tensor:
    out = torch.zeros(cols // 64, 4, dtype=torch.int64)
    hi = set(perm.tolist()[n_lo:])
    lo_before = 0
    for kc in range(cols // 64):
        m0 = sum(1 << c for c in range(32) if 64 * kc + c in hi)
        m1 = sum(1 << c for c in range(32) if 64 * kc + 32 + c in hi)
        out[kc] = torch.tensor([m0, m1, lo_before, 0])
        lo_before += sum(1 for c in range(64) if 64 * kc + c not in hi)
    return torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)


@pytest.mark.parametrize("n_lo", [0, 1, 100, 255, 256])
def test_chunk_desc_is_the_brute_force_count(n_lo):
    g = torch.Generator().manual_seed(n_lo)
    perms = torch.stack([torch.randperm(256, generator=g) for _ in range(3)])
    got = fe.chunk_desc(perms, n_lo, 256)
    assert got.shape == (3, 4, 4) and got.dtype == torch.int32
    for e in range(3):
        assert torch.equal(got[e], _brute_desc(perms[e], n_lo, 256)), e


def test_every_pair_the_library_decodes_fits_sm121():
    """Every one-run rate and adjacent two-run pair at both launch shapes."""
    for lo in range(rf.RATE_MIN, rf.RATE_MAX + 1):
        for hi in ((lo, lo + 1) if lo < rf.RATE_MAX else (lo,)):
            pair = torch.tensor([lo, 0, 32, 0, hi, 32, 32, 0] if hi != lo else [lo, 0, 64, 0, 0, 64, 0, 0],
                                dtype=torch.int32)
            sw = rf.slot_words_for_pair(pair)
            for mode in (0, 2):
                assert fe.smem_bytes(mode, sw) <= rf.SM121_MAX_DYNAMIC_SMEM, (lo, hi, mode)
    assert fe.smem_bytes(0, 16) == 93_648


def test_the_dense_split_keeps_two_chunks_per_item():
    for cols in (256, 512, 1280, 4096):
        s = fe.dense_split_max(cols)
        assert (cols // fe.BK) // s >= 2 and (cols // fe.BK) // (s + 1) < 2, cols


def test_only_the_e2m1_library_takes_the_architecture_specific_target():
    base = rf._cflags("sm_121", True, True)
    assert not any("FP4" in f for f in base) and not any("121a" in f for f in base)
    fp4 = rf._cflags("sm_121", False, False, True)
    assert "-DTESSERA_ROUTED_FUSED_FP4=1" in fp4 and any("sm_121a" in f for f in fp4)


def test_the_library_is_not_reachable_from_serving():
    """No route loads a window-body E2M1 stack yet (``ROUTES['TESSERA_NVFP4']``
    admits the TCQ span-2 body), so the library is not a serving extension:
    no ``native_extensions`` entry, and no serving module reaches its load
    site.  The route change that admits the window body flips this test."""
    from tessera.serving import ext
    from tessera.serving.scheme import ROUTES

    from test_serving_native_extensions import SRC, _serving_modules, scan_jit_extension_loads

    assert ROUTES["TESSERA_NVFP4"]["body"] == "TCQ"
    sites = scan_jit_extension_loads(SRC, _serving_modules())
    assert sites and fe.MODULE_NAME not in {s["name"] for s in sites}
    assert fe.MODULE_NAME not in {e["module_name_prefix"] for e in ext.NATIVE_EXTENSIONS}
    source = (SRC / "tessera" / "routed_fused_e2m1.py").read_text()
    assert f'name="{fe.MODULE_NAME}"' in source


# ----------------------------------------------------------------------------- GPU fixtures
def _encode(rows, cols, q256, seed):
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear
    from tessera.manifest import BodyKind, ScalePlaneKind

    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(rows, cols, generator=g) * 0.02).cuda()
    return encode_linear(w, grid=tuple_grid(E2M1_GRID, 2), q256=q256, body=BodyKind.WINDOW,
                         scale_plane=ScalePlaneKind.LUT, window_bits=L).blob


def _ref_weight(blob, cut):
    """float64 ``[rows, cols]``: e2m1(code) * ue4m3(lut[nibble]), the global excluded."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.decode import _decode_window
    from tessera.lane_planes import lut_scale_bytes
    from tessera.unit_artifact import parse_unit_artifact

    unit = parse_unit_artifact(blob, device="cuda").unit
    steps, cols = unit.body_bits.shape
    n = 2 * steps
    r0, r1 = cut if cut is not None else (0, n)
    codes = _decode_window(unit, tuple_grid(E2M1_GRID, 2), torch.int64)
    nib = torch.stack([codes >> 4, codes & 15], dim=1).reshape(n, cols)[r0:r1]
    lut16 = lut_scale_bytes(unit.scale_lut, "cuda")
    idx = unit.scale_refine.cuda().reshape(n, cols // int(unit.half)).long()[r0:r1]
    sf = lut16[idx].view(torch.float8_e4m3fn).double().repeat_interleave(int(unit.half), dim=1)
    return E2M1.cuda()[nib] * sf


def _unit(blob, **cut):
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact

    return prepare_window_lut_compact(parse_compact_wire(blob, device="cuda", name="w"), device="cuda", **cut)


def _bundle(blobs, part, **cut):
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    axis = WindowUnitAxis(len(blobs), [part], family="e2m1")
    for e, b in enumerate(blobs):
        axis.put(part, e, _unit(b, **cut))
    s = axis.finish()[part]
    return prepare_grouped_window_gemm_from_soa(
        words_all=s["words"], table_all=s["table"], codes_all=s["codes"], native_all=s["native"],
        scale_all=s["scale"], runs_all=s["runs"], init_all=s["init"], has_init=s["has_init"],
        word_off=s["word_off"], tile_words=s["tile_words"], total_words=s["total_words"],
        run_off=s["run_off"], perm_all=s["perm"], rows=s["rows"], cols=s["cols"],
        experts=len(blobs), window_bits=s["window_bits"], family="e2m1",
        scale_plane_all=s["scale_plane"], scale_lut_all=s["scale_lut"], global_all=s["global_scale"])


@functools.lru_cache(maxsize=None)
def _blobs(q256):
    seed = 1000 * q256
    return ([_encode(I, H, q256, seed + e) for e in range(E)],
            [_encode(I, H, q256, seed + 100 + e) for e in range(E)],
            [_encode(H, I, q256, seed + 200 + e) for e in range(E)])


@functools.lru_cache(maxsize=None)
def _stack(q256, cut=None):
    """(gate, up, down bundles, per-expert float64 reference weights).  With
    ``cut`` it is a TP rank's shard: gate/up rows and down columns ``cut``
    (the down reference is then not built)."""
    gb, ub, db = _blobs(q256)
    if cut is None:
        return ((_bundle(gb, "gate_proj"), _bundle(ub, "up_proj"), _bundle(db, "down_proj")),
                ([_ref_weight(b, None) for b in gb], [_ref_weight(b, None) for b in ub],
                 [_ref_weight(b, None) for b in db]))
    return ((_bundle(gb, "gate_proj", rows=cut), _bundle(ub, "up_proj", rows=cut),
             _bundle(db, "down_proj", cols=cut)), None)


def _routing(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.full((c,), e, dtype=torch.int64) for e, c in enumerate(COUNTS)])
    ids = ids[torch.randperm(ids.numel(), generator=g)].reshape(-1, TOP_K)
    rw = (torch.rand(ids.shape, generator=g) + 0.25).float()
    return ids.cuda(), rw.cuda()


def _x(rows, cols, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        return torch.randn(rows, cols, generator=g).to(torch.bfloat16).cuda()
    x = torch.zeros(rows, cols, dtype=torch.bfloat16)
    k = torch.randint(0, cols, (rows,), generator=g)
    v = (torch.rand(rows, generator=g) * 3 + 0.5) * torch.where(torch.rand(rows, generator=g) < 0.5, -1.0, 1.0)
    x[torch.arange(rows), k] = v.to(torch.bfloat16)
    return x.cuda()


def _gs(x):
    return 448.0 * 6.0 / max(float(x.float().abs().max()), 1e-12)


def _a_deq(x, gs):
    """The activation the kernel multiplies: ``scaled_fp4_quant(x, gs)`` dequantised."""
    from tessera.kernel_a4 import a4_quantize_activation

    codes, sfa = a4_quantize_activation(x, torch.tensor(gs, dtype=torch.float32, device="cuda"))
    c = codes.to(torch.int64)
    nib = torch.stack([c & 15, c >> 4], dim=2).reshape(c.shape[0], -1)
    return E2M1.cuda()[nib] * sfa.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)


def _per_expert(w, a, expert_of_row):
    """float64 ``(a_r @ w_e(r).T, |a_r| @ |w_e(r)|.T)`` per row ``r``."""
    acc = torch.zeros(a.shape[0], w[0].shape[0], dtype=torch.float64, device="cuda")
    absacc = torch.zeros_like(acc)
    for e in range(len(w)):
        sel = (expert_of_row == e).nonzero().reshape(-1)
        acc[sel] = a[sel] @ w[e].T
        absacc[sel] = a[sel].abs() @ w[e].abs().T
    return acc, absacc


def _ulp(v):
    return torch.exp2(torch.floor(torch.log2(v.abs().clamp_min(2.0 ** -126))) - 7)


def _check(got, ref, bound, exact, what):
    """``exact``: the reference's one rounding sequence, bit for bit; else
    within one bf16 rounding plus the accumulation ``bound``."""
    g = got.double()
    if exact:
        want = ref.float().to(torch.bfloat16).double()
        bad = int((g != want).sum())
        assert bad == 0, f"{what}: {bad} of {g.numel()} differ"
        return
    err = (g - ref).abs()
    tol = _ulp(ref) + bound * (1 + 2.0 ** -7)
    bad = int((err > tol).sum())
    assert bad == 0, f"{what}: {bad} of {g.numel()} exceed the bound (max err/tol {float((err / tol).max()):.3f})"


def _moe(q256, cut=None, gs13=1.0, gs2=1.0):
    (gate, up, down), _ = _stack(q256, cut)
    return fe.FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=gs13, gs2=gs2)


# ----------------------------------------------------------------------------- the routed launches
@gpu
@pytest.mark.parametrize("kind", ["onehot", "random"])
@pytest.mark.parametrize("q256", Q256)
def test_gate_up_against_the_decode(q256, kind):
    ids, rw = _routing(q256)
    x = _x(ids.shape[0], H, kind, q256)
    gs = _gs(x)
    moe = _moe(q256, gs13=gs)
    out = moe.gate_up(x, ids, rw)
    torch.cuda.synchronize()
    _, (wg, wu, _wd) = _stack(q256)
    a = _a_deq(x, gs)
    flat = ids.reshape(-1)
    tok = torch.arange(flat.numel(), device="cuda") // TOP_K
    o = out.reshape(-1, 2 * I)
    for half, (w, bundle) in enumerate(((wg, moe.gate), (wu, moe.up))):
        acc, absacc = _per_expert(w, a[tok], flat)
        ratio = (bundle.global_all / torch.tensor(gs, dtype=torch.float32, device="cuda"))[flat].unsqueeze(1)
        exact = kind == "onehot"
        ref = (acc.float() * ratio).double() if exact else acc * ratio.double()
        _check(o[:, half * I:(half + 1) * I], ref, absacc * ratio.double() * H * 2.0 ** -23, exact,
               f"{'gate' if half == 0 else 'up'} q256={q256}")


@gpu
@pytest.mark.parametrize("q256", [448, 512])
def test_the_rank1_row_cut_is_the_whole_units_rows(q256):
    """TP2 rank 1 holds rows I/2..I of gate and up, starting mid-stream from
    the carried state: its route-preserved output is the whole unit's right
    halves, bit for bit."""
    ids, rw = _routing(q256 + 1)
    x = _x(ids.shape[0], H, "random", q256 + 1)
    whole = _moe(q256, gs13=_gs(x)).gate_up(x, ids, rw)
    cut = _moe(q256, cut=(I // 2, I), gs13=_gs(x))
    assert bool(cut.gate.has_init.any()), "the cut must start mid-stream"
    part = cut.gate_up(x, ids, rw)
    torch.cuda.synchronize()
    h = I // 2
    assert torch.equal(part[..., :h].view(torch.int16), whole[..., h:I].view(torch.int16))
    assert torch.equal(part[..., h:].view(torch.int16), whole[..., I + h:].view(torch.int16))


def _mode0(moe, x, ids, rw):
    """The gate/up launch with the SwiGLU epilogue into route-sorted rows."""
    routing = moe._routing(ids, rw)
    xq, sfa = moe._quantized(x, moe.gs13)
    act = torch.empty((routing.routes, I), dtype=torch.bfloat16, device="cuda")
    moe._launch(0, xq, sfa, routing, a_row_mode=0, mul_weight=False, limit=float("inf"), out=act, counter=0)
    return act, routing


@gpu
@pytest.mark.parametrize("q256", [128, 448, 1024])
def test_the_swiglu_epilogue_is_the_silu_of_its_own_gate_up(q256):
    ids, rw = _routing(q256 + 2)
    x = _x(ids.shape[0], H, "random", q256 + 2)
    moe = _moe(q256, gs13=_gs(x))
    gu = moe.gate_up(x, ids, rw).reshape(-1, 2 * I).float()
    act, routing = _mode0(moe, x, ids, rw)
    torch.cuda.synchronize()
    g, u = gu[:, :I], gu[:, I:]
    want = ((g / (1.0 + torch.exp(-g))) * u).to(torch.bfloat16)[routing.flat_sorted.long()]
    d = (act.float() - want.float()).abs()
    assert bool((d <= _ulp(want.double()).float()).all()), float(d.max())


def _down_ref(wd, a, experts_of_row, scale):
    acc, absacc = _per_expert(wd, a, experts_of_row)
    return acc * scale, absacc * scale * I * 2.0 ** -23


def _reduced_check(out, route_ref, route_bound, tokens, what):
    """``out[t] = bf16(sum_j bf16(y[t, j]))`` in fixed order: within the
    token's rounding plus each route's rounding and bound."""
    r = route_ref.reshape(tokens, TOP_K, -1)
    b = route_bound.reshape(tokens, TOP_K, -1)
    ref = r.sum(1)
    tol = _ulp(ref) + (_ulp(r) + b * (1 + 2.0 ** -7)).sum(1)
    err = (out.double() - ref).abs()
    bad = int((err > tol).sum())
    assert bad == 0, f"{what}: {bad} exceed (max err/tol {float((err / tol).max()):.3f})"


@gpu
@pytest.mark.parametrize("q256", [128, 448, 960])
def test_down_routes_against_the_decode(q256):
    ids, rw = _routing(q256 + 3)
    xa = _x(ids.numel(), I, "random", q256 + 3)
    gs2 = _gs(xa)
    moe = _moe(q256, gs2=gs2)
    out = moe.down_routes(xa, ids, rw)
    torch.cuda.synchronize()
    _, (_, _, wd) = _stack(q256)
    flat = ids.reshape(-1)
    ratio = (moe.down.global_all.double() / gs2)[flat].unsqueeze(1) * rw.reshape(-1, 1).double()
    ref, bound = _down_ref(wd, _a_deq(xa, gs2), flat, ratio)
    _reduced_check(out, ref, bound, ids.shape[0], f"down q256={q256}")


@gpu
@pytest.mark.parametrize("q256", [448, 1024])
def test_the_served_chain_down_reads_the_sorted_activation(q256):
    """``__call__``: gate/up with SwiGLU into sorted rows, re-quantised at
    ``gs2``, down at A row = sorted position, weighted, reduced per token --
    checked on the exact activation the down launch consumes."""
    ids, rw = _routing(q256 + 4)
    x = _x(ids.shape[0], H, "random", q256 + 4)
    probe = _moe(q256, gs13=_gs(x))
    act, routing = _mode0(probe, x, ids, rw)
    gs2 = _gs(act)
    moe = _moe(q256, gs13=_gs(x), gs2=gs2)
    out = moe(x, ids, rw)
    torch.cuda.synchronize()
    _, (_, _, wd) = _stack(q256)
    flat_sorted = routing.flat_sorted.long()
    expert_sorted = ids.reshape(-1)[flat_sorted]
    ratio = ((moe.down.global_all.double() / gs2)[expert_sorted] * rw.reshape(-1)[flat_sorted].double()).unsqueeze(1)
    ref_s, bound_s = _down_ref(wd, _a_deq(act, gs2), expert_sorted, ratio)
    ref = torch.empty_like(ref_s)
    bound = torch.empty_like(bound_s)
    ref[flat_sorted], bound[flat_sorted] = ref_s, bound_s
    _reduced_check(out, ref, bound, ids.shape[0], f"chain q256={q256}")


@gpu
@pytest.mark.parametrize("q256", [128, 448, 960, 1024])
def test_graph_replay_and_a_second_run_are_bitwise_eager(q256):
    ids, rw = _routing(q256 + 5)
    x = _x(ids.shape[0], H, "random", q256 + 5)
    moe = _moe(q256, gs13=_gs(x), gs2=4.0)
    eager = moe(x, ids, rw)
    again = moe(x, ids, rw)
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int16), again.view(torch.int16))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        moe(x, ids, rw)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = moe(x, ids, rw)
    for _ in range(2):
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


# ----------------------------------------------------------------------------- the dense identity
@gpu
@pytest.mark.parametrize("kind", ["onehot", "random"])
@pytest.mark.parametrize("m", [1, 7, 100, 300])
def test_dense_forward_against_the_decode(m, kind):
    """A down unit as a dense Linear: unsplit, a middle split and the cap
    (every item two chunks), and a graph replay of each."""
    _, _, db = _blobs(448)
    w = _ref_weight(db[0], None)
    x = _x(m, I, kind, m)
    gs = _gs(x)
    role = fe.prepare_dense_role(_unit(db[0]), gs)
    a = _a_deq(x, gs)
    exact = kind == "onehot"
    ref = (a @ w.T).float().mul(role.ratio[0]).double() if exact else (a @ w.T) * float(role.ratio[0])
    bound = (a.abs() @ w.abs().T) * float(role.ratio[0]) * I * 2.0 ** -23
    for k_split in (1, 3, fe.dense_split_max(I)):
        out = fe.dense_forward(role, x, k_split=k_split)
        torch.cuda.synchronize()
        _check(out, ref, bound, exact, f"dense M={m} S={k_split}")
        captured = torch.empty_like(out)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fe.dense_forward(role, x, k_split=k_split, out=captured)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fe.dense_forward(role, x, k_split=k_split, out=captured)
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), out.view(torch.int16)), k_split


@gpu
@pytest.mark.parametrize("kind", ["onehot", "random"])
@pytest.mark.parametrize("rows", [32, 128, 384])
def test_dense_rows_that_end_inside_a_block(rows, kind):
    """A projection whose rows end inside the last 256-row block: the launch
    decodes the whole block from the wire's padded tile, and every row it
    writes is the projection's own -- the rest of ``out`` and of the next
    row's columns are untouched (``out`` is a column slice of a wider
    buffer, filled with a sentinel)."""
    blob = _encode(rows, H, 448, 7000 + rows)
    w = _ref_weight(blob, None)
    cap = fe.dense_split_max(H)
    for m in (1, 100):
        x = _x(m, H, kind, rows + m)
        gs = _gs(x)
        role = fe.prepare_dense_role(_unit(blob), gs)
        a = _a_deq(x, gs)
        exact = kind == "onehot"
        ref = (a @ w.T).float().mul(role.ratio[0]).double() if exact else (a @ w.T) * float(role.ratio[0])
        bound = (a.abs() @ w.abs().T) * float(role.ratio[0]) * H * 2.0 ** -23
        for k_split in (1, cap):
            wide = torch.full((m, rows + 64), -7.0, dtype=torch.bfloat16, device="cuda")
            out = wide[:, :rows]
            fe.dense_forward(role, x, k_split=k_split, out=out)
            torch.cuda.synchronize()
            _check(out, ref, bound, exact, f"dense rows={rows} M={m} S={k_split}")
            assert bool((wide[:, rows:] == -7.0).all()), f"rows={rows} S={k_split} wrote past its rows"
            captured = torch.full_like(wide, -7.0)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fe.dense_forward(role, x, k_split=k_split, out=captured[:, :rows])
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fe.dense_forward(role, x, k_split=k_split, out=captured[:, :rows])
            captured.fill_(-7.0)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured.view(torch.int16), wide.view(torch.int16)), (rows, k_split)


@gpu
@pytest.mark.parametrize("cut", [(0, 128), (256, 384)])
def test_a_dense_row_cut_is_the_whole_units_rows(cut):
    """A column-parallel dense module's rank holds a row cut of the unit (the
    second from a carried start state): its output is the whole unit's rows
    of the cut bit for bit, one-hot and random, unsplit and at the cap."""
    _, _, db = _blobs(448)
    cap = fe.dense_split_max(I)
    for kind in ("onehot", "random"):
        x = _x(100, I, kind, 9000 + cut[0])
        gs = _gs(x)
        whole = fe.prepare_dense_role(_unit(db[0]), gs)
        part = fe.prepare_dense_role(_unit(db[0], rows=cut), gs)
        for k_split in (1, cap):
            want = fe.dense_forward(whole, x, k_split=k_split)[:, cut[0]:cut[1]].contiguous()
            got = fe.dense_forward(part, x, k_split=k_split)
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (cut, kind, k_split)


@gpu
def test_a_split_past_the_cap_is_refused_by_name():
    """The host wrapper and the library each refuse a split that leaves an
    item one K chunk (a descriptor slot could be rewritten before it is read)."""
    from tessera.kernel_a4 import a4_quantize_activation

    _, _, db = _blobs(448)
    role = fe.prepare_dense_role(_unit(db[0]), 1.0)
    x = _x(8, I, "random", 0)
    cap = fe.dense_split_max(I)
    with pytest.raises(GrammarError, match="two K chunks"):
        fe.dense_forward(role, x, k_split=cap + 1)
    codes, sfa = a4_quantize_activation(x, role.gs)
    out = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
    partial = torch.empty((cap + 1) * 8 * H, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="two K chunks"):
        fe._ext().dense_forward_fp4(codes, sfa.view(torch.uint8), role.words, role.codes, role.init,
                                    role.has_init, role.plane, role.lut, role.ratio, role.runs, role.desc,
                                    H, role.tile_words, role.slot_words,
                                    torch.zeros(1, dtype=torch.int32, device="cuda"), cap + 1, partial, out, 1)


# ----------------------------------------------------------------------------- refusals
@gpu
def test_the_stack_refusals_name_their_reason(monkeypatch):
    (gate, up, down), _ = _stack(448)
    assert fe.fused_routed_e2m1_supported(gate, up, down) is None
    assert "family" in fe.fused_routed_e2m1_supported(
        gate, up, _replace(down, family="e4m3"))
    assert "window_bits" in fe.fused_routed_e2m1_supported(
        gate, _replace(up, window_bits=12), down)
    monkeypatch.setenv(rf.ENV_TOGGLE, "0")
    assert "disabled" in fe.fused_routed_e2m1_supported(gate, up, down)
    monkeypatch.delenv(rf.ENV_TOGGLE)
    _, _, db = _blobs(448)
    unit = _replace(_unit(db[0]), rows=48)
    assert "multiple of 32" in fe.dense_role_reason(unit)
    with pytest.raises(GrammarError, match="multiple of 32"):
        fe.prepare_dense_role(unit, 1.0)
    with pytest.raises(GrammarError, match="one static scalar"):
        fe.FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=torch.ones(2), gs2=1.0)
    with pytest.raises(GrammarError, match="finite positive"):
        fe.FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=0.0, gs2=1.0)
    with pytest.raises(GrammarError, match="silu"):
        fe.FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=1.0, gs2=1.0, activation="gelu")


def _replace(bundle, **changes):
    import dataclasses

    return dataclasses.replace(bundle, **changes)
