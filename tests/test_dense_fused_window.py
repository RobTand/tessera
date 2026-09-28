"""The fused window kernel's DENSE identity (``tessera.routed_fused.dense_forward``;
contract v43, the dense follow-up to tessera#640).

A dense Linear is the E = 1, top-1, unweighted case of the routed lane, so the
same kernel serves it with identity routing and, for small M, a split over K
into an fp32 partial workspace that a fixed-order reduce sums before the one
bf16 rounding.  What this pins, against the same per-role definition oracle
the grouped and fused routed tests use (``test_window_gemm_grouped.Expert``)
and against the Triton window GEMM the module keeps as its other lane:

* parity for both window families at M = 1, 3, 64, 65, 200 and 1536, and at
  the first M the bandwidth model runs in one pass, so both the split-K regime
  (S > 1) and the one-pass regime (S = 1) are measured;
* a TP row cut's start state (``has_init``);
* the real GLM dense role shapes (down 4096x6144, gate/up 12288x4096 and
  their TP2 halves 4096x3072 and 6144x4096) at M = 1, 3, 64, 512 and 2048,
  held per element to a derived bound against the fp64 reference of the same
  quantised inputs and against the Triton lane;
* two-run bitwise equality (no atomics) and CUDA-graph replay against eager;
* the served module: ``prepare_dense_native_module`` decides the fused lane,
  stamps ``tessera::fused_window_dense`` under the family's decoder, serves a
  two-role module into column slices of one output, counts its 16-bit tables
  in the residency accounting, and keeps the Triton lane -- naming the reason
  -- under ``TESSERA_DENSE_FUSED=0`` or for a role the predicate refuses;
* the predicate's refusals by name, and the split-K model's two regimes.

The lane is a CUDA kernel JIT-built on first use; every GPU case here runs
through PrismaBuild inside the pinned serving image
(``experiments/routed_fused_tests.sh``).
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import routed_fused as rf                    # noqa: E402
from tessera import window_gemm as wg                     # noqa: E402
from tessera.errors import GrammarError                   # noqa: E402
from tessera.grammar import bresenham_rate_schedule       # noqa: E402

import fused_bound as fb                                  # noqa: E402
from test_window_gemm_grouped import Expert, _quant  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

L = 14
# One role: two 128-row N blocks, eight 32-column K steps (split-K up to 8).
ROWS, COLS = 256, 256
M_CASES = [1, 3, 64, 65, 200, 1536]
FAMILIES = ["value", "e4m3"]
#: The rungs the mixed-rate tests read (tessera#694): one-rate wires at every
#: rate the kernel decodes and two-rate schedules at the GLM rungs --
#: q256 832 (3.25: rates 3/4), 928 (3.625), 1088 (4.25: 4/5), 1152 (4.5) --
#: plus the two-rate 7/8 and 1/2 extremes.  256 columns realise every one of
#: them exactly (``bresenham_rate_schedule`` refuses a rung it cannot).
Q256_CASES = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 384, 832, 928, 1088, 1152, 1920]


def _sched(cols, q256):
    """The grammar's Bresenham schedule for ``q256`` over ``cols`` columns
    (cap 8: the packer's whole range, so the value family's rungs are read too)."""
    from fractions import Fraction

    return bresenham_rate_schedule(Fraction(q256, 256), cols, cap=8)


def _init(cols, seed):
    return torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(seed),
                         dtype=torch.int32)


def _role(family, *, rows=ROWS, cols=COLS, seed=700, init=None, rates=None,
          arithmetic=None, quantizer="native"):
    """One dense role at rate 4 (unless ``rates``), as the Triton lane freezes
    it: the definition-side ``Expert`` and its ``PreparedWindowGemm``."""
    expert = Expert(rows, cols, rates or (4,) * cols, seed, family=family, init=init)
    arith = arithmetic or ("folded" if family == "value" else "epilogue")
    bundle = wg.prepare_window_gemm(
        expert.unit, block_m=64, block_n=64, block_k=64,
        quantizer=quantizer if family == "e4m3" else None, arithmetic=arith)
    return expert, bundle


def _inputs(family, m, cols, seed):
    """``(x bf16, the A operand the route hands the kernel, a_scale or None)``."""
    x = torch.randn(m, cols, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()
    if family == "e4m3":
        xq, a = _quant(x)
        return x, xq.contiguous(), a.reshape(-1).contiguous().float()
    return x, x.contiguous(), None


def _a64(family, xq, a):
    """The scaled fp64 A operand: ``xq * a_scale`` for E4M3, the bf16 x for value."""
    return xq.double() * a.double()[:, None] if family == "e4m3" else xq.double()


def _bound(expert, family, xq, a, s):
    """``(r, bound)``: the definition in fp64 -- the decoded weights, the row
    scale (folded before the dot for the value family, on the accumulator for
    E4M3) and the per-token activation scale -- and the per-element bound on
    ``|kernel - r|`` for a K split of ``s`` (``fused_bound.dense_bound``; the
    model is stated there and in ``test_dense_forward_on_the_glm_role_shapes``)."""
    return fb.dense_bound(family, _a64(family, xq, a), fb.fp64_weight(expert, family),
                          expert.cols, s)


def _split(role, m):
    """The K split the kernel runs at ``m`` rows (the bandwidth model)."""
    sms = rf._sm_count(torch.cuda.current_device())
    return rf.dense_k_split(m, role.rows, role.cols, sms, tile_words=role.tile_words)


def _fused(role, xq, a, out=None, counter=None):
    m = int(xq.shape[0])
    if out is None:
        out = torch.empty(m, role.rows, dtype=torch.bfloat16, device="cuda")
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    rf.dense_forward(role, xq, a, out, counter)
    return out


def _triton(bundle, xq, a):
    return bundle(xq, a) if a is not None else bundle(xq)


def _within(fused, oracle, what, *, triton=None):
    """``fused`` within the derived bound of the fp64 reference per element
    and, given the ``triton`` lane's output over the same wire (one pass,
    ``S = 1``: its bound is at most the fused one), within twice the bound of
    it -- both lanes sit inside the bound.  Prints the worst ratios."""
    r, bound = oracle
    ratio = fb.check_within(fused, r, bound, f"{what}: fused vs the fp64 reference")
    line = f"DENSE-BOUND {what} fused/bound={ratio:.4f}"
    if triton is not None:
        t_ratio = fb.check_within(triton, r, bound, f"{what}: Triton vs the fp64 reference")
        pair = fb.check_within(fused, triton.double(), bound, f"{what}: fused vs Triton", scale=2.0)
        line += f" triton/bound={t_ratio:.4f} pair/(2*bound)={pair:.4f}"
    print(line)
    return ratio


# --- the kernel against the definition and the Triton lane ------------------------

@cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("m", M_CASES)
def test_dense_forward_matches_the_definition_and_the_triton_lane(family, m):
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    assert (role.rows, role.cols, role.fp8) == (ROWS, COLS, family == "e4m3")
    _x, xq, a = _inputs(family, m, COLS, 900 + m)
    fused = _fused(role, xq, a)
    assert fused.shape == (m, ROWS) and fused.dtype == torch.bfloat16
    # The Triton lane computes the same function of the same wire in another
    # accumulation order: parity is bound-limited, not bitwise.
    s = _split(role, m)
    _within(fused, _bound(expert, family, xq, a, s), f"{family} M={m} S={s}",
            triton=_triton(bundle, xq, a))


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_the_k_split_model_picks_both_regimes_and_both_are_exact(family):
    """``dense_k_split`` splits K when fewer items than SMs exist (decode) and
    runs one pass once every SM has an item (prefill); parity holds in both,
    at the first one-pass M this device reaches rather than a fixed M."""
    sms = rf._sm_count(torch.cuda.current_device())
    n_blocks = ROWS // rf.BN
    split = rf.dense_k_split(1, ROWS, COLS, sms)
    assert 1 < split <= COLS // rf.BK, (split, sms)
    one_pass_m = rf.BM * -(-sms // n_blocks)
    assert -(-one_pass_m // rf.BM) * n_blocks >= sms
    assert rf.dense_k_split(one_pass_m, ROWS, COLS, sms) == 1
    assert rf.dense_k_split(0, ROWS, COLS, sms) == 1
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    for m in (1, one_pass_m):
        _x, xq, a = _inputs(family, m, COLS, 77 + m)
        s = _split(role, m)
        _within(_fused(role, xq, a), _bound(expert, family, xq, a, s),
                f"{family} M={m} (S={s})")


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_dense_forward_reads_a_row_cuts_start_state(family):
    """A TP row shard of a column-parallel module starts its decode inside
    the wire (``has_init``); the kernel reads the same start state the
    definition oracle does."""
    expert, bundle = _role(family, init=_init(COLS, 41), seed=710)
    assert bundle.has_init
    role = rf.prepare_dense_role(bundle)
    assert int(role.has_init.item()) == 1
    for m in (1, 65):
        _x, xq, a = _inputs(family, m, COLS, 300 + m)
        s = _split(role, m)
        _within(_fused(role, xq, a), _bound(expert, family, xq, a, s),
                f"{family} M={m} S={s} with a start state", triton=_triton(bundle, xq, a))


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_dense_two_runs_are_bitwise_equal(family):
    """Both regimes: the split-K reduce sums the partials in a fixed order and
    the one-pass epilogue rounds once; no atomics anywhere."""
    _expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    for m in (1, 200, 1536):
        _x, xq, a = _inputs(family, m, COLS, 21 + m)
        first = _fused(role, xq, a)
        for _ in range(3):
            assert torch.equal(_fused(role, xq, a), first), (family, m)


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_dense_forward_captures_and_replays_against_eager(family):
    """The work counter is zeroed INSIDE the captured region and the partial
    workspace is a graph-pool allocation, so a replay starts a fresh work
    list; two replays equal the eager forward bitwise, and new inputs copied
    into the static buffers replay to the new answer."""
    _expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    m = 40                                   # split-K regime on any device
    _x, xq, a = _inputs(family, m, COLS, 31)
    eager = _fused(role, xq, a)
    out = torch.empty_like(eager)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _fused(role, xq, a, out, counter)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fused(role, xq, a, out, counter)
    for _ in range(2):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    _x2, xq2, a2 = _inputs(family, m, COLS, 32)
    xq.copy_(xq2)
    if a is not None:
        a.copy_(a2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, _fused(role, xq2, a2))


# --- the GLM dense role shapes -----------------------------------------------------

#: The dense Linears of GLM-5.3 as the kernel sees each one (rows x cols, cols
#: = K): the whole down and gate/up projections, and their TP2 halves -- the
#: down projection is row-parallel, so a rank holds a column cut (K / 2); the
#: gate/up projection is column-parallel, so a rank holds a row cut, which
#: starts its decode inside the wire (``has_init``).  6144 is the K > 4096 case.
GLM_ROLE_SHAPES = [
    pytest.param("dense_down", 4096, 6144, False, id="dense_down-4096x6144"),
    pytest.param("dense_gate_up", 12288, 4096, False, id="dense_gate_up-12288x4096"),
    pytest.param("dense_down_tp2_column_cut", 4096, 3072, False, id="dense_down_tp2-4096x3072"),
    pytest.param("dense_gate_up_tp2_row_cut", 6144, 4096, True, id="dense_gate_up_tp2-6144x4096"),
]
GLM_M_CASES = [1, 3, 64, 512, 2048]
# The bound model lives in ``tests/fused_bound.py`` (shared with the routed and
# grouped tests); these names keep this file's statements readable.
_gamma, _bf16_ulp, _fp64_weight = fb.gamma, fb.bf16_ulp, fb.fp64_weight


def _glm_bound(family, a64, w64, k, s):
    """``(r, bound)``: the fp64 reference and the per-element bound on
    ``|fused - r|`` stated in the test's docstring (``fused_bound.dense_bound``)."""
    return fb.dense_bound(family, a64, w64, k, s)


@cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("role_name,rows,cols,row_cut", GLM_ROLE_SHAPES)
def test_dense_forward_on_the_glm_role_shapes(role_name, rows, cols, row_cut, family):
    """The fused lane at the real GLM dense shapes, rate 4 in every column
    (the ``[[4, 0, cols, 0]]`` run table), against the fp64 reference of the
    same quantised inputs and against the Triton lane, per output element.

    THE BOUND (derived from the dtypes; nothing fitted).  The kernel sums K
    exact fp32 products (bf16 x bf16 and e4m3 x e4m3 both are) -- as S fp32
    partials summed in a fixed order when it splits K, S =
    ``routed_fused.dense_k_split(m, rows, cols, sms)`` -- charging one fp32
    ulp per accumulation step so a truncating tensor-core adder is covered;
    the E4M3 family then applies two round-to-nearest fp32 multiplies
    ``(acc * a_scale) * w_scale``; the result is rounded once to bf16
    (round-to-nearest).  With ``Sigma = sum_k |a_k w_k|`` over the scaled
    operands in fp64 and ``gamma(n, u) = n u / (1 - n u)``:

        E_acc = gamma(K + S + 2, 2^-23) * Sigma + gamma(K, 2^-53) * Sigma
        E_pre = E_acc                                      (value, folded)
        E_pre = E_acc + gamma(2, 2^-24) * (|r| + E_acc)    (e4m3, epilogue)
        |fused - r| <= E_pre + ulp_bf16(|r| + E_pre) / 2

    ``gamma(2, 2^-24)`` is the stated ``2^-23 |y|`` for the two epilogue
    multiplies; the ``2^-53`` term is the fp64 reference's own dot; the half
    ulp is taken at ``|r| + E_pre``, the largest magnitude the pre-rounding
    fp32 value can have, so a straddled binade cannot undershoot it.

    THE TRITON COMPARISON.  The Triton lane sums the same K exact products in
    one pass (no split, S = 1) with the same epilogue, so its own bound is at
    most the fused one, and ``|fused - triton| <= 2 E_pre + ulp_bf16(|r| +
    E_pre)`` -- one bf16 ulp plus ``2 gamma Sigma`` (plus the two epilogue
    terms on E4M3) -- asserted per element and reported in bf16 ulps of y.
    """
    init = _init(cols, 4200 + rows // 128) if row_cut else None
    expert, bundle = _role(family, rows=rows, cols=cols, seed=720 + rows // 128 + cols // 32,
                           init=init)
    assert bool(bundle.has_init) == row_cut
    assert rf.fused_dense_window_supported(bundle) is None
    role = rf.prepare_dense_role(bundle)
    assert (role.rows, role.cols, int(role.has_init.item())) == (rows, cols, int(row_cut))
    sms = rf._sm_count(torch.cuda.current_device())
    w64 = _fp64_weight(expert, family)
    failures = []
    for m in GLM_M_CASES:
        s = rf.dense_k_split(m, rows, cols, sms)
        _x, xq, a = _inputs(family, m, cols, 1300 + m)
        a64 = xq.double() * a.double()[:, None] if family == "e4m3" else xq.double()
        r, bound = _glm_bound(family, a64, w64, cols, s)
        fused = _fused(role, xq, a)
        triton = _triton(bundle, xq, a)
        assert fused.shape == triton.shape == (m, rows)
        d_fused = (fused.double() - r).abs()
        d_triton = (triton.double() - r).abs()
        d_ft = (fused.double() - triton.double()).abs()
        bound_ft = 2.0 * bound
        ulp_y = _bf16_ulp(r.abs())
        row_unit = _bf16_ulp(r.abs().amax(dim=-1, keepdim=True))
        facts = {
            "fused_over_bound": float((d_fused / bound).max()),
            "triton_over_bound": float((d_triton / bound).max()),
            "fused_vs_triton_over_bound": float((d_ft / bound_ft).max()),
            "fused_vs_triton_ulps_of_y": float((d_ft / ulp_y).max()),
            "fused_vs_triton_ulps_of_row_max": float((d_ft / row_unit).max()),
            "violations": (int((d_fused > bound).sum()), int((d_triton > bound).sum()),
                           int((d_ft > bound_ft).sum())),
        }
        print(f"GLM-ROLE-SHAPE {role_name} {rows}x{cols} K={cols} {family} M={m} S={s} "
              f"fused/bound={facts['fused_over_bound']:.4f} "
              f"triton/bound={facts['triton_over_bound']:.4f} "
              f"fused-vs-triton/bound={facts['fused_vs_triton_over_bound']:.4f} "
              f"fused-vs-triton max {facts['fused_vs_triton_ulps_of_y']:.2f} bf16 ulps of y "
              f"({facts['fused_vs_triton_ulps_of_row_max']:.2f} of the row max) "
              f"violations(fused,triton,pair)={facts['violations']}")
        if any(facts["violations"]):
            failures.append((m, s, facts))
    assert not failures, f"{role_name} {rows}x{cols} {family}: bound exceeded at {failures}"


# --- the served module -----------------------------------------------------------

def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.decode"),
            pytest.importorskip("tessera.alphabet"))


def _scheme(family, rows, cols, roles, wire_bytes, q256=1024):
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8

    e4m3 = family == "e4m3"
    return {"family": TESSERA_FP8 if e4m3 else TESSERA_BF16, "grid": "E4M3" if e4m3 else "BF16",
            "body": "WINDOW", "plane": "CHANNEL", "q256": q256, "rows": rows, "columns": cols,
            "wire_bytes": wire_bytes, "roles": [[n, r] for n, r in roles]}


def _encode_module(family, roles, cols, q256=1024, seed=0):
    """Encode ``roles`` = [(name, rows)] on the family's grid at ``q256``;
    return the container blob, its scheme and the module's exact reference
    weight in fp64 ``[rows, cols]``: the materialised E4M3 bytes times their
    fp32 row scale (the product ``stock_dequant`` takes in fp32, exact here),
    or the folded BF16 tile."""
    fused, export, stock, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    blobs, refs = [], []
    for i, (name, rows) in enumerate(roles):
        w = torch.randn(rows, cols, device="cuda") * 0.02
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        grid = alphabet.E4M3_GRID if family == "e4m3" else alphabet.BF16_GRID
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=grid, q256=q256, name=name, verify=False)
        if family == "e4m3":
            tiles = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
            refs.append(tiles["weight"].to("cuda").double()
                        * tiles["weight_scale"].to("cuda").double().reshape(-1, 1))
        else:
            refs.append(decode.materialize_bf16_folded(unit, forests, export.DEFAULT_CODE)
                        .to("cuda").double())
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(family, sum(r for _, r in roles), cols, roles, len(blob), q256)
    return blob, scheme, torch.cat(refs)


def _whole_plan(declared):
    from tessera.serving.sharding import plan_shard

    roles = [(str(n), int(r)) for n, r in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    return plan_shard("test", roles=roles, columns=columns,
                      out_partitions=[r for _, r in roles], in_size=columns,
                      tp_rank=0, tp_size=1, input_size=columns, output_size=rows)


def _module(blob, scheme):
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import parse_compact_blob_for_scheme, validate_tessera_scheme

    declared = validate_tessera_scheme(scheme, "test")
    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    return prepare_dense_native_module(compact, _whole_plan(declared),
                                       family=scheme["family"], device="cuda")


def _served(module, family, xq, x, a):
    return module.apply(xq, a) if family == "e4m3" else module.apply(x)


def _module_bound(family, ref_w, xq, x, a):
    """``(r, bound)`` for a served module over the fp64 weight ``ref_w``: each
    role's rows run as one dense launch whose K split is at most
    ``cols // BK`` (``dense_k_split``'s range), so ``S = cols // BK`` bounds
    every role on the fused lane, and the Triton lane's one pass (``S = 1``)
    with it."""
    cols = int(ref_w.shape[1])
    a64 = _a64(family, xq, a) if family == "e4m3" else x.double()
    return fb.dense_bound(family, a64, ref_w, cols, cols // rf.BK)


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_the_served_module_takes_the_fused_lane_and_serves_two_roles(family, monkeypatch):
    """The lane is decided once per module at load; a two-role (gate/up)
    module is served by one op into column slices of one output, its launch
    pair is the fused one under the family's decoder, and the Triton lane
    built over the same bytes under the opt-out gives the same answer within
    a bf16 straddle and names the reason it was kept."""
    from tessera.serving import telemetry
    from tessera.serving.native_window import LANE_FUSED, LANE_TRITON
    from tessera.serving.scheme import FUSED_WINDOW_DENSE_SYMBOL, WINDOW_GEMM_SYMBOL

    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE, raising=False)
    roles = [("gate_proj", 256), ("up_proj", 128)]
    blob, scheme, ref_w = _encode_module(family, roles, cols=256, seed=5)
    module = _module(blob, scheme)
    assert module.lane == LANE_FUSED and module.lane_reason is None
    decoder = (telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE if family == "e4m3"
               else telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED)
    assert module.launch_pair == (FUSED_WINDOW_DENSE_SYMBOL, decoder)
    assert (module.symbol, module.decoder) == module.launch_pair
    assert module.role_names == ("gate_proj", "up_proj")
    named = dict(module.named_tensors())
    for index, facts in enumerate(module.layout_facts()):
        assert named[f"roles.{index}.fused_table16"].shape == (1, rf.TABLE_ENTRIES)
        assert named[f"roles.{index}.fused_table16"].dtype == torch.int16
        # An encoded unit carries a start register whole or cut (all zero for
        # a whole unit -- ``has_history`` false -- and the kernel reads it
        # through the same flag the Triton bundle sets, ``has_init``).
        flag = int(named[f"roles.{index}.fused_has_init"].item())
        assert flag in (0, 1) and flag >= int(facts.has_history)
        assert named[f"roles.{index}.init_perm"].shape == (256,)
    for m in (1, 5, 64, 129):
        x, xq, a = _inputs(family, m, 256, 500 + m)
        got = _served(module, family, xq, x, a)
        assert got.shape == (m, 384) and got.dtype == torch.bfloat16
        _within(got, _module_bound(family, ref_w, xq, x, a),
                f"{family} module M={m} vs the materialised reference")
    monkeypatch.setenv(rf.ENV_TOGGLE_DENSE, "0")
    twin = _module(blob, scheme)
    assert twin.lane == LANE_TRITON
    assert twin.launch_pair == (WINDOW_GEMM_SYMBOL, (
        telemetry.DECODER_NATIVE_WINDOW_GEMM if family == "e4m3"
        else telemetry.DECODER_NATIVE_WINDOW_GEMM_FOLDED))
    assert twin.lane_reason == f"role 'gate_proj': disabled by {rf.ENV_TOGGLE_DENSE}=0"
    for m in (1, 129):
        x, xq, a = _inputs(family, m, 256, 600 + m)
        _within(_served(module, family, xq, x, a), _module_bound(family, ref_w, xq, x, a),
                f"{family} module M={m} over the same bytes",
                triton=_served(twin, family, xq, x, a))
    # The fused lane's own storage beyond the shared bundles: one composed
    # 16-bit table and one int32 flag per role.
    assert module.packed_bytes() - twin.packed_bytes() == len(roles) * (rf.TABLE_ENTRIES * 2 + 4)
    assert not any(name.endswith(("fused_table16", "fused_has_init"))
                   for name, _ in twin.named_tensors())


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_a_module_the_predicate_refuses_keeps_the_triton_lane_and_says_why(family, monkeypatch):
    from tessera.serving.native_window import LANE_TRITON
    from tessera.serving.scheme import WINDOW_GEMM_SYMBOL

    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE, raising=False)
    blob, scheme, ref_w = _encode_module(family, [("weight", 192)], cols=256, seed=9)
    module = _module(blob, scheme)
    assert module.lane == LANE_TRITON
    assert module.lane_reason == "role 'weight': 192 rows; the dense identity needs a multiple of 128"
    assert module.launch_pair[0] == WINDOW_GEMM_SYMBOL
    x, xq, a = _inputs(family, 7, 256, 61)
    _within(_served(module, family, xq, x, a), _module_bound(family, ref_w, xq, x, a),
            f"{family} 192-row module on the Triton lane")


# --- the predicate ---------------------------------------------------------------

@cuda
def test_the_predicate_refuses_by_name():
    _e, ok = _role("value")
    assert rf.fused_dense_window_supported(ok) is None
    _e, ok8 = _role("e4m3")
    assert rf.fused_dense_window_supported(ok8) is None
    _e, short = _role("value", rows=192)
    assert rf.fused_dense_window_supported(short) == \
        "192 rows; the dense identity needs a multiple of 128"
    _e, narrow = _role("value", cols=64)
    assert rf.fused_dense_window_supported(narrow) == \
        f"64 columns; the kernel needs a multiple of {rf.BK} and at least {rf.MIN_COLS}"
    # two rates (the two bracketing a root) are read since contract v45 (#694)
    _e, mixed = _role("value", rates=tuple(3 if c % 2 else 4 for c in range(COLS)))
    assert rf.fused_dense_window_supported(mixed) is None
    # three rates are not a grammar schedule; the kernel reads one run pair
    _e, three = _role("value", rates=tuple((2, 3, 4)[c % 3] for c in range(COLS)))
    assert "3 runs" in rf.fused_dense_window_supported(three)
    # a column order that is not the packer's stable (rate, column) sort
    shuffled = torch.tensor([1, 0] + list(range(2, COLS)), dtype=mixed.perm.dtype,
                            device=mixed.perm.device)
    assert "ascending column order" in rf.fused_dense_window_supported(
        dataclasses.replace(mixed, perm=shuffled))
    assert "not a permutation" in rf.fused_dense_window_supported(
        dataclasses.replace(mixed, perm=torch.zeros_like(mixed.perm)))
    # a tile stride the run table does not produce
    assert "tile_words" in rf.fused_dense_window_supported(
        dataclasses.replace(mixed, tile_words=int(mixed.tile_words) + 16))
    _e, epilogue = _role("value", arithmetic="epilogue")
    assert rf.fused_dense_window_supported(epilogue) == \
        "arithmetic 'epilogue'; the fused identity serves 'folded' for value"
    _e, unattested = _role("e4m3", quantizer=None)
    assert rf.fused_dense_window_supported(unattested) == \
        "the role was prepared without the native activation quantizer"
    with pytest.raises(GrammarError, match="refuses this role"):
        rf.prepare_dense_role(short)


def test_the_opt_out_is_read_before_any_role_fact(monkeypatch):
    monkeypatch.setenv(rf.ENV_TOGGLE_DENSE, "0")
    assert rf.fused_dense_window_enabled() is False
    assert rf.fused_dense_window_supported(object()) == f"disabled by {rf.ENV_TOGGLE_DENSE}=0"
    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE)
    assert rf.fused_dense_window_enabled() is True


def test_the_k_split_model_is_the_bandwidth_model():
    """Pure arithmetic: one pass once every SM has an item; otherwise the
    integer minimiser of ``wire * sms / min(S * items0, sms) + 2 S M N 4``
    over ``1 .. min(K / 32, ceil(sms / items0))``, restated here."""
    sms = 48

    def restated(m, rows, cols):
        items0 = -(-m // rf.BM) * (rows // rf.BN)
        if m <= 0 or items0 >= sms:
            return 1
        wire = rows * cols // 2
        return min(range(1, min(cols // rf.BK, -(-sms // items0)) + 1),
                   key=lambda s: wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4)

    assert rf.dense_k_split(0, 256, 4096, sms) == 1
    assert rf.dense_k_split(8192, 256, 4096, sms) == 1
    assert rf.dense_k_split(1, 6144, 4096, sms) == 1          # 48 blocks: every SM busy
    for m, rows, cols in ((1, 256, 4096), (1, 4096, 2048), (8, 4096, 4096), (64, 2048, 4096),
                          (200, 4096, 6144), (1, 12288, 4096), (3, 128, 128)):
        got = rf.dense_k_split(m, rows, cols, sms)
        assert got == restated(m, rows, cols), (m, rows, cols, got)
        assert 1 <= got <= cols // rf.BK
        # the wire is the role's own words per tile: rate 4 restates the default
        assert rf.dense_k_split(m, rows, cols, sms, tile_words=64 * cols) == got
    assert rf.dense_k_split(1, 256, 4096, sms) > 1
    # the wire's bytes move the optimum where the split is not capped by
    # ceil(sms / items0): at M = 32 x 256 x 4096 (2 items, cap 24) rate 1
    # (16 words per column per tile), rate 4 and rate 8 pick three splits
    light = rf.dense_k_split(32, 256, 4096, sms, tile_words=16 * 4096)
    heavy = rf.dense_k_split(32, 256, 4096, sms, tile_words=128 * 4096)
    assert light < rf.dense_k_split(32, 256, 4096, sms) < heavy, (light, heavy)
    # More rows means more items and never a larger split at the same M.
    assert rf.dense_k_split(1, 2048, 4096, sms) <= rf.dense_k_split(1, 256, 4096, sms)


# --- the launch identity is published -------------------------------------------

def test_the_dense_identity_is_a_published_launch_of_both_window_routes():
    from tessera.serving import bf16_route, fp8_route, telemetry
    from tessera.serving.native_window import (DENSE_LANES, FUSED_WINDOW_DENSE_DECODER,
                                               LANE_FUSED, LANE_TRITON)
    from tessera.serving.scheme import (FUSED_WINDOW_DENSE_SYMBOL, STRUCTURE_DENSE, TESSERA_BF16,
                                        TESSERA_FP8, WINDOW_GEMM_SYMBOL, launch_pairs)

    assert FUSED_WINDOW_DENSE_SYMBOL == "tessera::fused_window_dense"
    assert DENSE_LANES[LANE_FUSED] == (FUSED_WINDOW_DENSE_SYMBOL, FUSED_WINDOW_DENSE_DECODER)
    assert DENSE_LANES[LANE_TRITON][0] == WINDOW_GEMM_SYMBOL
    assert FUSED_WINDOW_DENSE_DECODER == {
        "epilogue": telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE,
        "folded": telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED}
    assert {telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE,
            telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED} <= telemetry.DECODERS
    for module, route, decoder in (
            (fp8_route, TESSERA_FP8, telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE),
            (bf16_route, TESSERA_BF16, telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED)):
        assert module.DENSE_FUSED_LAUNCH == (FUSED_WINDOW_DENSE_SYMBOL, decoder)
        assert module.DENSE_LAUNCHES == (module.DENSE_LAUNCH, module.DENSE_FUSED_LAUNCH)
        for mode in ("resident", "streamed"):
            for regime in ("decode", "batch"):
                assert set(module.DENSE_LAUNCHES) == launch_pairs(
                    route, structure=STRUCTURE_DENSE, regime=regime, mode=mode), (route, regime, mode)


# --- every rate, and the two-rate schedules (tessera#694) -------------------------

@cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("q256", Q256_CASES)
def test_dense_forward_decodes_every_rate_exactly(family, q256):
    """One-hot rows through the real kernel: each output element is one
    decoded weight through the epilogue, so the fused output must equal the
    torch restatement of that arithmetic BITWISE at every (row, column) --
    which catches a wrong column map, run offset, or field position at any
    rate, in either run, with and without a start state.  ``S`` is whatever
    the bandwidth model picks for ``M = cols`` rows, so the split path's
    partials and reduce are read too (a split adds exact zeros only)."""
    rates = _sched(COLS, q256)
    assert set(rates) <= set(rf.RATES) and len(set(rates)) in (1, 2)
    for seed, init in ((7000 + q256, None), (7100 + q256, _init(COLS, 7200 + q256))):
        expert, bundle = _role(family, rates=rates, seed=seed, init=init)
        assert rf.fused_dense_window_supported(bundle) is None, (family, q256)
        role = rf.prepare_dense_role(bundle)
        assert role.tile_words == 16 * sum(rates) == int(bundle.tile_words)
        _x, xq, a, hot = fb.one_hot_inputs(family, COLS, _quant)
        got = _fused(role, xq, a)
        want = fb.one_hot_expected(expert, family, hot, a)
        bad = (got != want)
        assert not bool(bad.any()), (
            f"{family} q256={q256} rates {sorted(set(rates))} init={init is not None}: "
            f"{int(bad.sum())} of {bad.numel()} one-hot products differ; first at "
            f"{bad.nonzero()[0].tolist()} (column, row)")


@cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("q256", [768, 832, 928, 1088, 1152, 2048])
@pytest.mark.parametrize("cols", [128, COLS])
def test_dense_forward_at_every_rate_is_within_the_derived_bound(family, q256, cols):
    """Random inputs at the GLM rungs and the extremes, K = 128 (the smallest
    the kernel takes) and 256, M in the split and one-pass regimes: the fused
    output is within ``fused_bound.dense_bound`` of the fp64 reference per
    element, and within twice it of the Triton lane (follow-up 3 of #693:
    the fused-vs-Triton statistic is a pass criterion with a derived limit)."""
    rates = _sched(cols, q256)
    expert, bundle = _role(family, cols=cols, rates=rates, seed=7300 + q256 + cols)
    role = rf.prepare_dense_role(bundle)
    sms = rf._sm_count(torch.cuda.current_device())
    w64 = fb.fp64_weight(expert, family)
    for m in (1, 64, 200):
        s = rf.dense_k_split(m, ROWS, cols, sms, tile_words=role.tile_words)
        _x, xq, a = _inputs(family, m, cols, 7400 + m + q256)
        a64 = xq.double() * a.double()[:, None] if family == "e4m3" else xq.double()
        r, bound = fb.dense_bound(family, a64, w64, cols, s)
        fused = _fused(role, xq, a)
        what = f"{family} q256={q256} K={cols} M={m} S={s}"
        ratio = fb.check_within(fused, r, bound, f"{what}: fused vs the fp64 reference")
        triton = _triton(bundle, xq, a)
        fb.check_within(triton, r, bound, f"{what}: Triton vs the fp64 reference")
        pair = fb.check_within(fused, triton.double(), bound, f"{what}: fused vs Triton", scale=2.0)
        print(f"RATE-BOUND {what} fused/bound={ratio:.4f} pair/(2*bound)={pair:.4f}")


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_a_row_stride_that_is_only_even_takes_the_unsplit_path(family):
    """Follow-up 4 of #693: the split path's reduce stores four bf16 (uint2)
    at 4-aligned columns, so a ``[M, rows]`` view whose row stride is 2 mod 4
    is refused by the extension for ``k_split > 1`` and ``dense_forward``
    routes it through ``S = 1`` -- same answer as a contiguous output."""
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    m = 1
    sms = rf._sm_count(torch.cuda.current_device())
    assert rf.dense_k_split(m, ROWS, COLS, sms, tile_words=role.tile_words) > 1
    _x, xq, a = _inputs(family, m, COLS, 7500)
    wide = torch.zeros(m, ROWS + 2, dtype=torch.bfloat16, device="cuda")
    view = wide[:, :ROWS]
    assert view.stride(0) % 4 == 2
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    rf.dense_forward(role, xq, a, view, counter)
    assert torch.equal(view, _fused(role, xq, a))            # S forced to 1: the one-pass answer
    assert torch.equal(wide[:, ROWS:], torch.zeros_like(wide[:, ROWS:]))
    lib = rf._ext(family)
    s = 2
    partial = torch.empty((s, m, ROWS), dtype=torch.float32, device="cuda")
    empty = xq.new_empty(0, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="multiple of 4"):
        lib.dense_forward(bool(role.fp8), xq, a if a is not None else empty,
                          role.words, role.table16, role.init, role.has_init, role.wscale,
                          role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), counter, s,
                          partial, view, sms)


def test_the_run_pair_and_block_descriptor_are_the_packers_layout():
    """Pure host arithmetic: the run pair restates a one- or two-run table and
    refuses three runs, a wrong offset or a run that does not tile K; the
    block descriptor lists each block's low-rate columns then its high-rate
    columns in ascending position with the running low-rate count -- checked
    against a direct restatement over the packer's permutation."""
    cols = 128
    rates = _sched(cols, 928)                                        # 3/4 mixed, 80 at rate 4
    n_lo = rates.count(3)
    perm = torch.tensor(sorted(range(cols), key=lambda c: (rates[c], c)), dtype=torch.int32)
    runs = torch.tensor([[3, 0, n_lo, 0], [4, n_lo, cols - n_lo, 16 * 3 * n_lo]], dtype=torch.int32)
    pair, why = rf.run_pair(runs, cols)
    assert why is None and pair.tolist() == [3, 0, n_lo, 0, 4, n_lo, cols - n_lo, 48 * n_lo]
    assert rf.pair_tile_words(pair) == 16 * sum(rates)
    one, why = rf.run_pair(torch.tensor([[5, 0, cols, 0]], dtype=torch.int32), cols)
    assert why is None and one.tolist() == [5, 0, cols, 0, 0, cols, 0, 80 * cols]
    assert rf.pair_tile_words(one) == 16 * 5 * cols
    for bad, word in ((torch.tensor([[2, 0, 64, 0], [3, 64, 32, 2048], [4, 96, 32, 3584]]), "3 runs"),
                      (torch.tensor([[3, 0, n_lo, 0], [4, n_lo, cols - n_lo, 0]]), "do not tile"),
                      (torch.tensor([[4, 0, n_lo, 0], [3, n_lo, cols - n_lo, 64 * n_lo]]), "not above"),
                      (torch.tensor([[9, 0, cols, 0]]), "rate in 1..8"),
                      (torch.tensor([[4, 0, cols - 32, 0]]), "covers")):
        got, why = rf.run_pair(bad.to(torch.int32), cols)
        assert got is None and word in why, (bad.tolist(), why)
    assert rf.perm_reason(perm, n_lo, cols) is None
    swapped = perm.clone()
    swapped[0], swapped[1] = perm[1], perm[0]                        # a descent inside the low run
    assert "ascending" in rf.perm_reason(swapped, n_lo, cols)
    assert "not a permutation" in rf.perm_reason(torch.zeros(cols, dtype=torch.int32), n_lo, cols)
    desc = rf.block_desc(perm, n_lo, cols)
    assert tuple(desc.shape) == (1, cols // 32, rf.BDESC_INTS) and desc.dtype == torch.int32
    is_hi = [rates[c] == 4 for c in range(cols)]
    before = 0
    for kc in range(cols // 32):
        block = list(range(32 * kc, 32 * kc + 32))
        order = [c - 32 * kc for c in block if not is_hi[c]] + [c - 32 * kc for c in block if is_hi[c]]
        words = desc[0, kc].tolist()
        got = [(words[i // 4] >> (8 * (i % 4))) & 0xFF for i in range(32)]
        assert got == order, kc
        cnt_lo = sum(1 for c in block if not is_hi[c])
        assert words[8] == before and words[9] == cnt_lo and words[10] == words[11] == 0
        before += cnt_lo
    assert before == n_lo
