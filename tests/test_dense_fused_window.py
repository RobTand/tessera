"""The fused window kernel's DENSE identity (``tessera.routed_fused.dense_forward``;
contract v43, the dense follow-up to tessera#640).

A dense Linear is the E = 1, top-1, unweighted case of the routed lane, so the
same kernel serves it with identity routing and, for small M, a split over K
into an fp32 partial workspace that a fixed-order reduce sums before the one
bf16 rounding.  What this pins, against the same per-role definition oracle
the grouped and fused routed tests use (``test_window_gemm_grouped.Expert``)
and against the Triton window GEMM the module keeps as its other lane:

* parity for both window families at M = 1, 3, 64, 65, 200 and 1536, and at
  the first M the split model runs in one pass, so both the split-K regime
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

#: The three libraries every kernel test runs on (see
#: ``test_routed_fused_window.LIBRARY_IDS``): the value family and the E4M3
#: family on each tensor-core instruction.
LIBRARY_IDS = ["value", "e4m3", "e4m3mma"]


@pytest.fixture
def family(request, monkeypatch):
    lib = request.param
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3" if lib == "e4m3mma" else "f16")
    return "e4m3" if lib == "e4m3mma" else lib

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
#: The value family's dense rungs above rate 8 (tessera#750 item 4): every
#: one-run rate 9..14 (q256 2304..3584) and every adjacent pair from 8/9 to
#: 13/14 at its midpoint -- the rate-8 low run of 2176 is the only pair whose
#: low rate the routed launches also read.  The E4M3 grids' codes are 8 bits,
#: so these are value-only.
VALUE_DENSE_Q256 = [2304, 2560, 2816, 3072, 3328, 3584, 2176, 2432, 2688, 2944, 3200, 3456]


def _sched(cols, q256, cap=8):
    """The grammar's Bresenham schedule for ``q256`` over ``cols`` columns
    (cap 8 by default: the packer's range on both families, so the value
    family's rungs are read too; 14 for the value family's rungs above 8)."""
    from fractions import Fraction

    return bresenham_rate_schedule(Fraction(q256, 256), cols, cap=cap)


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
    """The K split the kernel runs at ``m`` rows (``dense_k_split``: the model
    ``TESSERA_DENSE_MODULE_LAUNCH`` selects, the bandwidth model by default)."""
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("module_launch", ["0", "1"])
def test_the_k_split_model_picks_both_regimes_and_both_are_exact(family, module_launch, monkeypatch):
    """``dense_k_split`` splits K when the items leave SMs idle (decode) and
    runs one pass once they fill whole waves (prefill); parity holds in both,
    at the first one-pass M this device reaches rather than a fixed M, under
    either model (``TESSERA_DENSE_MODULE_LAUNCH``): the bandwidth model's range
    runs to ``dense_split_max`` (tessera#805), the makespan model's to ``dense_fixup_split_max``."""
    monkeypatch.setenv(rf.ENV_DENSE_MODULE, module_launch)
    sms = rf._sm_count(torch.cuda.current_device())
    n_blocks = ROWS // rf.BN
    split = rf.dense_k_split(1, ROWS, COLS, sms)
    most = rf.dense_fixup_split_max(COLS) if module_launch == "1" else rf.dense_split_max(COLS)
    assert 1 < split <= most, (split, sms, module_launch)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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


#: The rungs the dense CUDA-graph capture test replays (tessera#694): every
#: one-rate rung the dense lane reads, 1..8 (q256 256..2048), and the GLM
#: two-run tables 832 (3/4) and 1088 (4/5); 256 columns realise each exactly.
DENSE_CAPTURE_Q256 = [1024, 256, 512, 768, 1280, 1536, 1792, 2048, 832, 1088]


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("q256", DENSE_CAPTURE_Q256)
def test_dense_forward_captures_and_replays_against_eager(family, q256):
    """The work counter is zeroed INSIDE the captured region and the partial
    workspace is a graph-pool allocation, so a replay starts a fresh work
    list; two replays equal the eager forward bitwise, and new inputs copied
    into the static buffers replay to the new answer.  At every rung of
    ``DENSE_CAPTURE_Q256``: the run pair, block descriptors, tile stride and
    slot are launch arguments and device tensors the graph holds, so a
    mixed-rate role replays exactly like the rate-4 one."""
    _capture_replays(family, q256)


@cuda
@pytest.mark.parametrize("family", ["value"], indirect=True)
@pytest.mark.parametrize("q256", VALUE_DENSE_Q256)
def test_dense_forward_captures_and_replays_above_rate_8(family, q256):
    """The value family's dense rungs above rate 8 (tessera#750 item 4)
    capture and replay like the rungs below it: the 20-, 24- and 28-word
    slots are launch arguments too."""
    _capture_replays(family, q256, cap=14)


def _capture_replays(family, q256, cap=8):
    rates = _sched(COLS, q256, cap)
    assert set(rates) <= set(rf.dense_rates(family)) and len(set(rates)) in (1, 2), \
        (q256, sorted(set(rates)))
    _expert, bundle = _role(family, rates=rates, seed=7500 + q256)
    assert rf.fused_dense_window_supported(bundle) is None, (family, q256)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    role's rows run with a K split of at most ``dense_split_max(cols)`` (the
    default bandwidth model's range; the makespan model's,
    ``dense_fixup_split_max(cols)``, is inside it), and the bound grows with
    ``S``, so that split bounds every role on the fused lane under either
    setting, and the Triton lane's one pass (``S = 1``) with it."""
    cols = int(ref_w.shape[1])
    a64 = _a64(family, xq, a) if family == "e4m3" else x.double()
    return fb.dense_bound(family, a64, ref_w, cols, rf.dense_split_max(cols))


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    library = rf.library_for(family)
    decoder = {"e4m3": telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE,
               "e4m3mma": telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA,
               "value": telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED}[library]
    assert module.launch_pair == (FUSED_WINDOW_DENSE_SYMBOL, decoder)
    table_dtype, table_bytes = (torch.uint8, 1) if library == "e4m3mma" else (torch.int16, 2)
    assert (module.symbol, module.decoder) == module.launch_pair
    assert module.role_names == ("gate_proj", "up_proj")
    named = dict(module.named_tensors())
    for index, facts in enumerate(module.layout_facts()):
        assert named[f"roles.{index}.fused_table16"].shape == (1, rf.TABLE_ENTRIES)
        assert named[f"roles.{index}.fused_table16"].dtype == table_dtype
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
    # table (16-bit, or the E4M3 bytes on the E4M3 instruction), one int32
    # flag, the run pair (8 int32) and the block descriptor (BDESC_INTS int32
    # per 32 columns) per role (tessera#694).
    assert module.packed_bytes() - twin.packed_bytes() == len(roles) * (
        rf.TABLE_ENTRIES * table_bytes + 4 + 8 * 4 + (256 // 32) * rf.BDESC_INTS * 4)
    assert not any(name.endswith(("fused_table16", "fused_has_init"))
                   for name, _ in twin.named_tensors())


@cuda
@pytest.mark.parametrize("family", ["value"], indirect=True)
@pytest.mark.parametrize("q256", [2304, 2432, 3456, 3584])
def test_an_encoded_bf16_module_above_rate_8_takes_the_fused_lane(family, q256, monkeypatch):
    """tessera#750 item 4, end to end: a BF16 module encoded at a rung above
    rate 8 loads through ``prepare_window_compact`` at the dense bound, takes
    the fused dense launch, and serves within the derived bound of its
    materialised weight on both lanes.  The same wire at the routed stacks'
    bound (``WINDOW_GEMM_RATE_MAX``) is refused by name."""
    from tessera.compact_prep import (DENSE_WINDOW_RATE_MAX, WINDOW_GEMM_RATE_MAX,
                                      prepare_window_compact)
    from tessera.errors import GrammarError as Refusal
    from tessera.serving.native_window import LANE_FUSED, LANE_TRITON
    from tessera.serving.scheme import parse_compact_blob_for_scheme, validate_tessera_scheme

    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE, raising=False)
    roles = [("gate_proj", 256), ("up_proj", 128)]
    blob, scheme, ref_w = _encode_module(family, roles, cols=256, q256=q256, seed=11)
    module = _module(blob, scheme)
    assert module.lane == LANE_FUSED and module.lane_reason is None, module.lane_reason
    assert all(max(f.rates) > 8 for f in module.layout_facts()), q256
    monkeypatch.setenv(rf.ENV_TOGGLE_DENSE, "0")
    twin = _module(blob, scheme)
    assert twin.lane == LANE_TRITON
    for m in (1, 64, 129):
        x, xq, a = _inputs(family, m, 256, 800 + m)
        _within(_served(module, family, xq, x, a), _module_bound(family, ref_w, xq, x, a),
                f"q256={q256} module M={m}", triton=_served(twin, family, xq, x, a))
    declared = validate_tessera_scheme(scheme, "test")
    compact = parse_compact_blob_for_scheme(blob, scheme, "test", device="cuda")
    _name, wire = compact[0]
    assert DENSE_WINDOW_RATE_MAX == 14 and WINDOW_GEMM_RATE_MAX == 8, declared
    unit = prepare_window_compact(wire, device="cuda", family="value", rate_max=DENSE_WINDOW_RATE_MAX)
    assert max(unit.rep.rates) == -(-q256 // 256)
    with pytest.raises(Refusal, match=r"outside this lane's window GEMM rates 1\.\.8"):
        prepare_window_compact(wire, device="cuda", family="value")


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_a_module_the_predicate_refuses_keeps_the_triton_lane_and_says_why(family, monkeypatch):
    from tessera.serving.native_window import LANE_TRITON
    from tessera.serving.scheme import WINDOW_GEMM_SYMBOL

    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE, raising=False)
    blob, scheme, ref_w = _encode_module(family, [("weight", 190)], cols=256, seed=9)
    module = _module(blob, scheme)
    assert module.lane == LANE_TRITON
    assert module.lane_reason == (
        f"role 'weight': 190 rows; the dense identity needs a multiple of {rf.DENSE_ROW_QUANTUM}")
    assert module.launch_pair[0] == WINDOW_GEMM_SYMBOL
    x, xq, a = _inputs(family, 7, 256, 61)
    _within(_served(module, family, xq, x, a), _module_bound(family, ref_w, xq, x, a),
            f"{family} 190-row module on the Triton lane")


# --- the predicate ---------------------------------------------------------------

@cuda
def test_the_predicate_refuses_by_name():
    _e, ok = _role("value")
    assert rf.fused_dense_window_supported(ok) is None
    _e, ok8 = _role("e4m3")
    assert rf.fused_dense_window_supported(ok8) is None
    _e, short = _role("value", rows=190)
    assert rf.fused_dense_window_supported(short) == \
        f"190 rows; the dense identity needs a multiple of {rf.DENSE_ROW_QUANTUM}"
    # an N-tail (the last 128-row block partial) is admitted since tessera#750 WP2
    _e, tail = _role("value", rows=160)
    assert rf.fused_dense_window_supported(tail) is None
    _e, narrow = _role("value", cols=64)
    assert rf.fused_dense_window_supported(narrow) == \
        f"64 columns; the kernel needs a multiple of {rf.BK} and at least {rf.MIN_COLS}"
    # two rates (the two bracketing a root) are read since contract v45 (#694)
    _e, mixed = _role("value", rates=tuple(3 if c % 2 else 4 for c in range(COLS)))
    assert rf.fused_dense_window_supported(mixed) is None
    # the value family's dense launch reads every rate its 14-bit window holds
    # (tessera#750 item 4); the E4M3 family's stops at 8, its grid's code width
    assert rf.dense_rates("value") == tuple(range(1, 15)) and rf.dense_rates("e4m3") == rf.RATES
    _e, r14 = _role("value", rates=(14,) * COLS)
    assert rf.fused_dense_window_supported(r14) is None
    _e, r9 = _role("e4m3", rates=(9,) * COLS)
    assert rf.fused_dense_window_supported(r9) == \
        f"first run (9, 0, {COLS}, 0) is not (rate in 1..8, 0, n, 0)"
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


def test_the_default_k_split_is_the_bandwidth_model(monkeypatch):
    """Pure arithmetic: by default (``TESSERA_DENSE_MODULE_LAUNCH`` unset or
    ``0``) ``dense_k_split`` is the bandwidth model the dense identity ran
    before tessera#778 -- one pass once every SM has an item; otherwise the
    integer minimiser of ``wire * sms / min(S * items0, sms) + 2 S M N 4``
    over ``1 .. min(K / 64, ceil(sms / items0))``, restated here (the
    ``dense_split_max`` cut of tessera#805)."""
    monkeypatch.delenv(rf.ENV_DENSE_MODULE, raising=False)
    sms = 48

    def restated(m, rows, cols, words=None, blocks=None):
        items0 = -(-m // rf.BM) * (-(-rows // rf.BN) if blocks is None else blocks)
        if m <= 0 or items0 >= sms:
            return 1
        wire = rows * cols // 2 if words is None else rows * words * 4 // 512
        return min(range(1, min(cols // rf.BK // 2, -(-sms // items0)) + 1),
                   key=lambda s: (wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4, s))

    for model in (rf.dense_k_split, rf.dense_k_split_bandwidth):
        assert model(0, 256, 4096, sms) == 1
        assert model(8192, 256, 4096, sms) == 1
        assert model(1, 6144, 4096, sms) == 1          # 48 blocks: every SM busy
    # The GLM-5.3 dense roles per TP2 rank (the four MLP shapes, the KDA
    # input module's roles, f_b) and the shapes the earlier test restated.
    shapes = ((256, 4096), (4096, 2048), (4096, 4096), (2048, 4096), (4096, 6144), (12288, 4096),
              (128, 128), (6144, 4096), (1024, 4096), (4096, 1024), (32, 4096), (64, 4096), (4096, 128))
    for rows, cols in shapes:
        for m in (*range(1, 257), 512, 2048, 2049, 8192):
            got = rf.dense_k_split(m, rows, cols, sms)
            assert got == rf.dense_k_split_bandwidth(m, rows, cols, sms) == restated(m, rows, cols), (m, rows, cols)
            assert 1 <= got <= rf.dense_split_max(cols)
            # the wire is the role's own words per tile: rate 4 restates the default
            assert rf.dense_k_split(m, rows, cols, sms, tile_words=64 * cols) == got
    # The default model is cut at dense_split_max (tessera#805), not at the
    # in-kernel fixup's tighter dense_fixup_split_max.
    assert rf.dense_k_split(1, 32, 4096, sms) == restated(1, 32, 4096) <= rf.dense_split_max(4096)
    assert rf.dense_k_split(1, 4096, 128, sms) == restated(1, 4096, 128) <= rf.dense_split_max(128)
    assert rf.dense_k_split(1, 256, 4096, sms) > 1
    light = rf.dense_k_split(32, 256, 4096, sms, tile_words=16 * 4096)
    heavy = rf.dense_k_split(32, 256, 4096, sms, tile_words=128 * 4096)
    assert light < rf.dense_k_split(32, 256, 4096, sms) < heavy, (light, heavy)
    assert rf.dense_k_split(1, 2048, 4096, sms) <= rf.dense_k_split(1, 256, 4096, sms)
    # Several roles priced as one item list (``blocks=``), for a caller of
    # ``dense_forward_roles`` under the default setting.
    kda_rows, kda_blocks = 3 * 4096 + 32 + 64 + 64, 3 * 32 + 3
    for m in (1, 4, 16, 64, 512):
        got = rf.dense_k_split(m, kda_rows, 4096, sms, blocks=kda_blocks)
        assert got == restated(m, kda_rows, 4096, blocks=kda_blocks), (m, got)


def test_the_module_launch_setting_selects_the_model_and_is_read_per_call(monkeypatch):
    """``TESSERA_DENSE_MODULE_LAUNCH``: unset and ``0`` are the bandwidth
    model, ``1`` the makespan model, read on every call; any other value is
    refused by name."""
    sms = 48
    cases = [(m, rows, cols) for m in (1, 2, 6, 31, 64, 128, 192, 512)
             for rows, cols in ((4096, 6144), (1024, 4096), (4096, 1024), (4096, 4096))]
    assert any(rf.dense_k_split_bandwidth(*c, sms) != rf.dense_k_split_makespan(*c, sms) for c in cases)
    for value, model in ((None, rf.dense_k_split_bandwidth), ("0", rf.dense_k_split_bandwidth),
                         ("1", rf.dense_k_split_makespan)):
        if value is None:
            monkeypatch.delenv(rf.ENV_DENSE_MODULE, raising=False)
        else:
            monkeypatch.setenv(rf.ENV_DENSE_MODULE, value)
        assert rf.dense_module_launch_enabled() is (value == "1")
        for m, rows, cols in cases:
            assert rf.dense_k_split(m, rows, cols, sms) == model(m, rows, cols, sms), (value, m, rows, cols)
    for bad in ("2", "yes", "true", ""):
        monkeypatch.setenv(rf.ENV_DENSE_MODULE, bad)
        with pytest.raises(GrammarError, match="TESSERA_DENSE_MODULE_LAUNCH"):
            rf.dense_k_split(1, 4096, 4096, sms)


def test_the_module_op_launches_per_role_unless_the_module_launch_is_set(monkeypatch):
    """``tessera::fused_window_dense`` on the E4M3 family: one
    ``dense_forward`` per role by default, as before tessera#778, and one
    ``dense_forward_roles`` for the module under
    ``TESSERA_DENSE_MODULE_LAUNCH=1``.  The value family is per role under
    either.  The launches are replaced by recorders, so this runs on CPU."""
    from tessera.serving import native_window as nw

    calls = []
    monkeypatch.setattr(rf, "dense_forward", lambda role, x, a, out, counter, **kw: calls.append(("role", role.rows)))
    monkeypatch.setattr(rf, "dense_forward_roles",
                        lambda roles, x, a, out, **kw: calls.append(("module", tuple(r.rows for r in roles))))
    rows, cols = [256, 32, 64], 256
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32)
    lists = dict(words=[i32(1, 8) for _ in rows], tables=[torch.zeros(1, 4, dtype=torch.int16) for _ in rows],
                 inits=[i32(1, cols) for _ in rows], has_inits=[i32(1) for _ in rows],
                 wscales=[torch.ones(1, r) for r in rows], runs=[i32(1, 8) for _ in rows],
                 bdescs=[i32(1, cols // rf.BK, rf.BDESC_INTS) for _ in rows])
    x = torch.zeros(3, cols, dtype=torch.float8_e4m3fn)
    a = torch.ones(3)
    for family_e4m3, value, want in ((True, None, [("role", 256), ("role", 32), ("role", 64)]),
                                     (True, "0", [("role", 256), ("role", 32), ("role", 64)]),
                                     (True, "1", [("module", (256, 32, 64))]),
                                     (False, "1", [("role", 256), ("role", 32), ("role", 64)])):
        if value is None:
            monkeypatch.delenv(rf.ENV_DENSE_MODULE, raising=False)
        else:
            monkeypatch.setenv(rf.ENV_DENSE_MODULE, value)
        calls.clear()
        out = nw._fused_window_dense(x if family_e4m3 else x.to(torch.bfloat16), a if family_e4m3 else None,
                                     role_rows=rows, tile_words=[64] * 3, slot_words=[8] * 3, cols=cols,
                                     family_e4m3=family_e4m3, folded=not family_e4m3, **lists)
        assert out.shape == (3, sum(rows))
        assert calls == want, (family_e4m3, value, calls)


def test_the_k_split_model_is_the_makespan_model():
    """Pure arithmetic: the integer minimiser, smaller ``S`` on a tie, of
    ``ceil(S items0 / sms) sms (item ceil(nk / S) / nk + c) + [S > 1] 2 S M N 4``
    over ``1 .. min(nk / (STAGES + 1), sms)``, restated here (``item`` the wire
    bytes of one 128-row block over all of K, ``c`` the measured per-item
    cost, ``items0`` over the launch's blocks: one role's, or the sum of a
    module's roles' when they share a launch)."""
    sms = 48
    assert rf.STAGES == 2
    for cols, most in ((4096, 42), (1536, 16), (512, 5), (256, 2), (128, 1), (6144, 64)):
        assert rf.dense_fixup_split_max(cols) == most

    def restated(m, rows, cols, words=None, blocks=None):
        if m <= 0:
            return 1
        items0 = -(-m // rf.BM) * (-(-rows // rf.BN) if blocks is None else blocks)
        nk = cols // rf.BK
        item = rf.BN * (64 * cols if words is None else words) * 4 / 512

        def t(s):
            waves = -(-(s * items0) // sms)
            return (waves * sms * (item * -(-nk // s) / nk + rf.DENSE_ITEM_FIXED_BYTES)
                    + (2.0 * s * m * rows * 4 if s > 1 else 0.0))

        return min(range(1, max(1, min(nk // (rf.STAGES + 1), sms)) + 1), key=lambda s: (t(s), s))

    assert rf.dense_k_split_makespan(0, 256, 4096, sms) == 1
    assert rf.dense_k_split_makespan(8192, 256, 4096, sms) == 1
    assert rf.dense_k_split_makespan(1, 6144, 4096, sms) == 1          # 48 items: one full wave
    for m, rows, cols in ((1, 256, 4096), (1, 4096, 2048), (8, 4096, 4096), (64, 2048, 4096),
                          (200, 4096, 6144), (1, 12288, 4096), (3, 128, 128), (1, 8192, 1536),
                          (1, 512, 4096), (1, 32, 4096), (1, 4096, 128)):
        got = rf.dense_k_split_makespan(m, rows, cols, sms)
        assert got == restated(m, rows, cols), (m, rows, cols, got)
        assert 1 <= got <= rf.dense_fixup_split_max(cols)
        # the wire is the role's own words per tile: rate 4 restates the default
        assert rf.dense_k_split_makespan(m, rows, cols, sms, tile_words=64 * cols) == got
    for words in (16 * 4096, 128 * 4096):
        assert rf.dense_k_split_makespan(32, 256, 4096, sms, tile_words=words) == restated(32, 256, 4096, words)
    assert rf.dense_k_split_makespan(1, 256, 4096, sms) > 1
    # The wave count decides: 32 items (a 4096-row role at decode) split
    # three ways fill two waves exactly, where two ways leave the second wave
    # a third full and cost as much as one pass (tessera#750).
    assert rf.dense_k_split_makespan(1, 4096, 4096, sms) == 3
    # 64 items (q_b, 8192 rows) are two waves at S = 1 for 1.33 waves of work.
    assert rf.dense_k_split_makespan(1, 8192, 1536, sms) == 3
    # the wire's bytes move the optimum: at M = 32 x 256 x 4096 (2 items)
    # rate 1 (16 words per column per tile), rate 4 and rate 8 pick three
    # splits, heavier wire for more
    light = rf.dense_k_split_makespan(32, 256, 4096, sms, tile_words=16 * 4096)
    heavy = rf.dense_k_split_makespan(32, 256, 4096, sms, tile_words=128 * 4096)
    assert light < rf.dense_k_split_makespan(32, 256, 4096, sms) < heavy, (light, heavy)
    # More rows means more items and never a larger split at the same M.
    assert rf.dense_k_split_makespan(1, 2048, 4096, sms) <= rf.dense_k_split_makespan(1, 256, 4096, sms)
    # A module's roles in one launch: their blocks are one item list.  The
    # GLM KDA input module at TP2 (q, k, v 4096 rows each, b 32, f_a 64,
    # g_a 64: 99 blocks) splits four ways at decode.
    kda_rows, kda_blocks = 3 * 4096 + 32 + 64 + 64, 3 * 32 + 3
    for m in (1, 4, 16, 64, 512):
        got = rf.dense_k_split_makespan(m, kda_rows, 4096, sms, blocks=kda_blocks)
        assert got == restated(m, kda_rows, 4096, blocks=kda_blocks), (m, got)
    assert rf.dense_k_split_makespan(1, kda_rows, 4096, sms, blocks=kda_blocks) == 4


#: GLM-5.3's dense roles as one rank's dense launch sees them (rows x K): the
#: KDA, MLA and indexer projections, the LM head's TP2 vocab cut, the dense MLP
#: and shared expert at TP2, and the whole and TP2 shapes above.
#: ``g_b_proj`` (K = 64) is absent: the launch refuses K < 128.
GLM_DENSE_ROLES = {
    "kda_qkv_o": (4096, 4096), "b_proj": (32, 4096), "f_a_g_a_proj": (64, 4096),
    "f_b_proj": (4096, 128), "q_a_proj": (1536, 4096), "kv_a_proj_with_mqa": (512, 4096),
    "q_b_proj": (8192, 1536), "indexer_wq_b": (4096, 1536), "lm_head_tp2": (77440, 4096),
    "mlp_gate_up_tp2": (6144, 4096), "mlp_down_tp2": (4096, 6144),
    "shared_gate_up_tp2": (1024, 4096), "shared_down_tp2": (4096, 1024),
    "dense_down": (4096, 6144), "dense_gate_up": (12288, 4096), "dense_down_tp2_cut": (4096, 3072),
}


def test_every_split_the_model_picks_keeps_two_chunks_per_item():
    """tessera#805: the producers rewrite an item's descriptor slot two items
    after claiming it, so the split launch is legal only while every item keeps
    two K chunks (``floor(nk / S) >= 2``), and ``dense_k_split`` searches no
    further.  Swept over few-row and small-K roles, where ``ceil(sms /
    items0)`` would pass ``nk / 2``, and over three SM counts."""
    for sms in (48, 132, 188):
        for rows in (4, 32, 64, 128, 256, 512, 4096):
            for cols in (128, 160, 256, 512, 1024, 4096):
                cap = rf.dense_split_max(cols)
                assert cap == cols // rf.BK // 2 >= 2
                for m in (1, 2, 3, 6, 16, 64, 65, 200):
                    for tile_words in (None, 16 * cols, 128 * cols):
                        s = rf.dense_k_split(m, rows, cols, sms, tile_words=tile_words)
                        assert 1 <= s <= cap, (sms, rows, cols, m, tile_words, s)
                        assert (cols // rf.BK) // s >= 2
    # where the bound binds: a 128 x 256 role at M = 1 asked for S = 8 (= nk,
    # one chunk per item) before the bound, and takes 4 now
    assert rf.dense_k_split(1, 128, 256, 48) == 4


def test_the_split_bound_moves_no_glm_role_on_gb10():
    """The bound changes no launch on GLM-5.3 on GB10 (48 SMs).  The model's
    search ceiling before the bound was ``min(nk, ceil(sms / ceil(rows /
    128)))`` for every M and every rate (``items0 >= ceil(rows / 128)``), and
    no GLM role's ceiling passes ``nk / 2``, so the bound never binds there.
    Cross-checked by the unbounded model restated, at every rate 1..16 up to
    M = 128 and at rates 4 and 16 up to M = 8192."""
    sms = 48

    def unbounded(m, rows, cols, tile_words):
        items0 = -(-m // rf.BM) * -(-rows // rf.BN)
        if m <= 0 or items0 >= sms:
            return 1
        wire = rows * tile_words * 4 // 512
        best_s, best_t = 1, None
        for s in range(1, min(cols // rf.BK, -(-sms // items0)) + 1):
            t = wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4
            if best_t is None or t < best_t:
                best_s, best_t = s, t
        return best_s

    for name, (rows, cols) in GLM_DENSE_ROLES.items():
        nk = cols // rf.BK
        ceiling = min(nk, -(-sms // -(-rows // rf.BN)))
        assert ceiling <= rf.dense_split_max(cols), (name, rows, cols, ceiling)
        for rate in range(1, 17):
            tile_words = 16 * cols * rate
            for m in range(1, 8193 if rate in (4, 16) else 129):
                want = unbounded(m, rows, cols, tile_words)
                assert rf.dense_k_split(m, rows, cols, sms, tile_words=tile_words) == want, \
                    (name, rate, m, want)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_a_split_that_leaves_an_item_one_chunk_is_refused_by_name(family):
    """tessera#805: the library refuses ``k_split > dense_split_max(K)`` by
    name, as the E2M1 launch does, and takes the bound itself."""
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    lib = rf._ext(role.library)
    sms = rf._sm_count(torch.cuda.current_device())
    m = 3
    _x, xq, a = _inputs(family, m, COLS, 8051)
    empty = xq.new_empty(0, dtype=torch.float32)
    cap = rf.dense_split_max(COLS)

    def native(s):
        out = torch.empty(m, ROWS, dtype=torch.bfloat16, device="cuda")
        partial = torch.empty((s, m, ROWS), dtype=torch.float32, device="cuda")
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        lib.dense_forward(bool(role.fp8), xq, a if a is not None else empty,
                          role.words, role.table16, role.init, role.has_init, role.wscale,
                          role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), counter, s,
                          partial, out, sms, rf.BM)
        return out

    with pytest.raises(RuntimeError, match="two K chunks"):
        native(cap + 1)
    with pytest.raises(RuntimeError, match="two K chunks"):
        native(COLS // rf.BK)
    _within(native(cap), _bound(expert, family, xq, a, cap), f"{family} M={m} S={cap} (the bound)")


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
        extra = ((module.DENSE_FUSED_MMA_E4M3_LAUNCH, module.DENSE_DECODE_ONCE_LAUNCH)
                 if module is fp8_route else ())
        assert module.DENSE_LAUNCHES == (module.DENSE_LAUNCH, module.DENSE_FUSED_LAUNCH, *extra)
        # contract v47 (tessera#747) earned the E4M3 instruction's pair its
        # cells; the decode-once pair (v56, tessera#931) is resident-only and
        # experimental, so the attested view is every launch the route makes
        # but that one, and the experimental view adds it at resident only
        decode_once = {getattr(module, "DENSE_DECODE_ONCE_LAUNCH", None)} - {None}
        for mode in ("resident", "streamed"):
            for regime in ("decode", "batch"):
                assert set(module.DENSE_LAUNCHES) - decode_once == launch_pairs(
                    route, structure=STRUCTURE_DENSE, regime=regime, mode=mode), (route, regime, mode)
                assert set(module.DENSE_LAUNCHES) - (decode_once if mode == "streamed" else set()) \
                    == launch_pairs(route, structure=STRUCTURE_DENSE, regime=regime, mode=mode,
                                    include_experimental=True), (route, regime, mode)


# --- every rate, and the two-rate schedules (tessera#694) -------------------------

@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("q256", Q256_CASES)
def test_dense_forward_decodes_every_rate_exactly(family, q256):
    """One-hot rows through the real kernel: each output element is one
    decoded weight through the epilogue, so the fused output must equal the
    torch restatement of that arithmetic BITWISE at every (row, column) --
    which catches a wrong column map, run offset, or field position at any
    rate, in either run, with and without a start state.  ``S`` is whatever
    the makespan model picks for ``M = cols`` rows, so the split path's
    partials and reduce are read too (a split adds exact zeros only)."""
    _one_hot_exact(family, q256)


@cuda
@pytest.mark.parametrize("family", ["value"], indirect=True)
@pytest.mark.parametrize("q256", VALUE_DENSE_Q256)
def test_dense_forward_decodes_every_value_rate_above_8_exactly(family, q256):
    """The value family's dense launch at every rate 9..14 and every adjacent
    pair 8/9..13/14 (tessera#750 item 4): the lane's decode window spans up to
    five words per 8-row lane group at rates 13 and 14, and the one-hot
    products hold it to the definition bitwise, with and without a start
    state."""
    _one_hot_exact(family, q256, cap=14)


def _one_hot_exact(family, q256, cap=8):
    rates = _sched(COLS, q256, cap)
    assert set(rates) <= set(rf.dense_rates(family)) and len(set(rates)) in (1, 2)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("q256", [768, 832, 928, 1088, 1152, 2048])
@pytest.mark.parametrize("cols", [128, COLS])
def test_dense_forward_at_every_rate_is_within_the_derived_bound(family, q256, cols):
    """Random inputs at the GLM rungs and the extremes, K = 128 (the smallest
    the kernel takes) and 256, M in the split and one-pass regimes: the fused
    output is within ``fused_bound.dense_bound`` of the fp64 reference per
    element, and within twice it of the Triton lane (follow-up 3 of #693:
    the fused-vs-Triton statistic is a pass criterion with a derived limit)."""
    _within_bound(family, q256, cols)


@cuda
@pytest.mark.parametrize("family", ["value"], indirect=True)
@pytest.mark.parametrize("q256", [2176, 2304, 2944, 3456, 3584])
@pytest.mark.parametrize("cols", [128, COLS])
def test_dense_forward_above_rate_8_is_within_the_derived_bound(family, q256, cols):
    """The value family's dense rungs above rate 8 (tessera#750 item 4), from
    the 8/9 pair to rate 14, against the fp64 reference and the Triton lane
    within the same derived bounds as the rungs below 8."""
    _within_bound(family, q256, cols, cap=14)


def _within_bound(family, q256, cols, cap=8):
    rates = _sched(cols, q256, cap)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    lib = rf._ext(role.library)
    s = 2
    partial = torch.empty((s, m, ROWS), dtype=torch.float32, device="cuda")
    empty = xq.new_empty(0, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="multiple of 4"):
        lib.dense_forward(bool(role.fp8), xq, a if a is not None else empty,
                          role.words, role.table16, role.init, role.has_init, role.wscale,
                          role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), counter, s,
                          partial, view, sms, rf.BM)


# --- the N-tail: a role whose last 128-row block is partial (tessera#750 WP2) ------

#: Role heights with a partial last block: the quantum itself (4), the GLM KDA
#: input module's small roles at TP2 (``b_proj`` 32, ``f_a_proj``/``g_a_proj``
#: 64), a tail after whole blocks (160 = 128 + 32, 300 = 2 * 128 + 44) and a
#: tail past a 512-row wire tile (548 = 512 + 36).
TAIL_ROWS = [4, 32, 64, 160, 300, 548]


def _sentinel_out(m, rows, *, even_only):
    """A ``[m, rows]`` view into a wider bf16 buffer filled with a sentinel, 8
    columns of it on the left and 8 (or 10) on the right: a row stride that is
    a multiple of 4 lets the split path run, one that is 2 mod 4 forces the
    one-pass path (``dense_forward``)."""
    right = 10 if even_only else 8
    wide = torch.full((m, 8 + rows + right), 7.0, dtype=torch.bfloat16, device="cuda")
    view = wide[:, 8:8 + rows]
    assert (view.stride(0) % 4 == 2) == even_only
    return wide, view


def _untouched(wide, rows):
    return bool((wide[:, :8] == 7).all()) and bool((wide[:, 8 + rows:] == 7).all())


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("rows", TAIL_ROWS)
def test_an_n_tail_role_decodes_exactly_and_stores_only_its_rows(family, rows):
    """One-hot rows through the real kernel at a role height 128 does not
    divide, one run (q256 1024) and two runs (1088), with and without a start
    state, on the split path and the one-pass path: every stored element is
    the definition BITWISE, and not one sentinel column beside the role's
    slice moves -- the partial block's pad rows are decoded and never
    stored."""
    for q256 in (1024, 1088):
        rates = _sched(COLS, q256)
        for init in (None, _init(COLS, 7600 + rows)):
            expert, bundle = _role(family, rows=rows, rates=rates, seed=7700 + rows + q256, init=init)
            assert rf.fused_dense_window_supported(bundle) is None, (family, rows, q256)
            role = rf.prepare_dense_role(bundle)
            _x, xq, a, hot = fb.one_hot_inputs(family, COLS, _quant)
            want = fb.one_hot_expected(expert, family, hot, a)
            m = int(xq.shape[0])
            for even_only in (False, True):
                wide, view = _sentinel_out(m, rows, even_only=even_only)
                rf.dense_forward(role, xq, a, view, torch.zeros(1, dtype=torch.int32, device="cuda"))
                what = (f"{family} rows={rows} q256={q256} init={init is not None} "
                        f"{'one-pass' if even_only else 'split'}")
                bad = view != want
                assert not bool(bad.any()), (
                    f"{what}: {int(bad.sum())} of {bad.numel()} one-hot products differ; first at "
                    f"{bad.nonzero()[0].tolist()}")
                assert _untouched(wide, rows), f"{what}: a store left the role's column slice"


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("rows", [32, 300])
def test_an_n_tail_role_is_within_the_bound_deterministic_and_replays(family, rows):
    """Random inputs at a tail height: within the derived bound of the fp64
    reference and of the Triton lane in the split and one-pass regimes,
    bitwise equal across runs, and a captured forward replays to the eager
    answer."""
    expert, bundle = _role(family, rows=rows, rates=_sched(COLS, 1088), seed=7800 + rows)
    role = rf.prepare_dense_role(bundle)
    sms = rf._sm_count(torch.cuda.current_device())
    one_pass_m = rf.BM * -(-sms // -(-rows // rf.BN))
    assert rf.dense_k_split(one_pass_m, rows, COLS, sms, tile_words=role.tile_words) == 1
    assert rf.dense_k_split(1, rows, COLS, sms, tile_words=role.tile_words) > 1
    for m in (1, 64, 200, one_pass_m):
        _x, xq, a = _inputs(family, m, COLS, 7900 + m + rows)
        fused = _fused(role, xq, a)
        s = _split(role, m)
        _within(fused, _bound(expert, family, xq, a, s), f"{family} rows={rows} M={m} S={s}",
                triton=_triton(bundle, xq, a))
        for _ in range(2):
            assert torch.equal(_fused(role, xq, a), fused), (family, rows, m)
    m = 40
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
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_the_kda_input_module_takes_the_fused_lane(family, monkeypatch):
    """The GLM KDA input module at TP2 is q, k, v, b, f_a and g_a (12448 rows
    per rank: 3 x 4096 + 32 + 64 + 64).  Its small roles used to keep the
    whole module on the Triton lane; with the N-tail every role is served by
    the fused identity into its column slice.  Scaled here to K = 256 and
    256-row q/k/v, with the tail roles between and after the whole ones, so a
    stray store from any role lands in a neighbour the bound check reads."""
    from tessera.serving.native_window import LANE_FUSED

    monkeypatch.delenv(rf.ENV_TOGGLE_DENSE, raising=False)
    roles = [("q_proj", 256), ("b_proj", 32), ("k_proj", 256), ("f_a_proj", 64),
             ("v_proj", 256), ("g_a_proj", 64)]
    blob, scheme, ref_w = _encode_module(family, roles, cols=256, seed=13)
    module = _module(blob, scheme)
    assert module.lane == LANE_FUSED and module.lane_reason is None
    rows = sum(r for _, r in roles)
    for m in (1, 5, 64, 129):
        x, xq, a = _inputs(family, m, 256, 800 + m)
        got = _served(module, family, xq, x, a)
        assert got.shape == (m, rows)
        _within(got, _module_bound(family, ref_w, xq, x, a),
                f"{family} KDA-shaped module M={m} vs the materialised reference")


def test_the_run_pair_and_block_descriptor_are_the_packers_layout():
    """Pure host arithmetic: the run pair restates a one- or two-run table and
    refuses three runs, two rates that are not adjacent, a wrong offset or a
    run that does not tile K; the
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
                      # a pair no grammar schedule emits: the lane reads adjacent rates only
                      (torch.tensor([[2, 0, 64, 0], [4, 64, 64, 16 * 2 * 64]]), "not adjacent"),
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


# --- the wide superblock (tessera#741) ------------------------------------------

@cuda
@pytest.mark.parametrize("family,q256", [(lib, q) for lib in ("e4m3", "e4m3mma") for q in (1024, 832, 1088, 2048)],
                         indirect=["family"])
def test_wide_dense_superblocks_are_bitwise_the_64_row_launch(family, q256, monkeypatch):
    """The E4M3 dense launch at 128-row superblocks (unsplit) gives the
    64-row launch's bits on both E4M3 libraries at every fill of the last
    superblock -- M below, at and past 64 and 128 -- at the one-run and
    two-run rungs, with and without a row cut's start state.  K is held
    unsplit so both widths run at every M; the oracle parity tests hold the
    64-row launch."""
    monkeypatch.setattr(rf, "dense_k_split", lambda *a, **k: 1)
    for init in (None, _init(COLS, 43)):
        _expert, bundle = _role(family, rates=_sched(COLS, q256), seed=720 + q256, init=init)
        role = rf.prepare_dense_role(bundle)
        for m in (1, 63, 64, 65, 127, 128, 129, 200, 300):
            _x, xq, a = _inputs(family, m, COLS, 90 + m)
            monkeypatch.setenv(rf.ENV_WIDE, "0")
            narrow = _fused(role, xq, a)
            monkeypatch.setenv(rf.ENV_WIDE, "1")
            wide = _fused(role, xq, a)
            assert torch.equal(wide, narrow), (role.library, q256, m, init is not None)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_the_dense_width_follows_the_split_and_the_family(family, monkeypatch):
    """A split launch keeps 64-row superblocks (the split model is the 64-row
    one, and a split has SMs to fill, not rows); an unsplit E4M3 launch takes
    the width ``superblock_rows`` picks, on either instruction; the value
    family has 64 only."""
    _expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    lib = rf._ext(role.library)
    seen = []

    class Spy:
        def __getattr__(self, name):
            return getattr(lib, name)

        def dense_forward(self, *args):
            seen.append((int(args[-5]), int(args[-1])))          # (k_split, bm)
            return lib.dense_forward(*args)

    monkeypatch.setattr(rf, "_ext", lambda _library: Spy())
    sms = rf._sm_count(torch.cuda.current_device())
    one_pass_m = rf.BM * -(-sms // (ROWS // rf.BN))
    monkeypatch.setenv(rf.ENV_WIDE, "1")
    for m in (1, one_pass_m):
        _x, xq, a = _inputs(family, m, COLS, 60 + m)
        seen.clear()
        _fused(role, xq, a)
        (s, bm), = seen
        want = rf.BM_WIDE if (family == "e4m3" and s == 1) else rf.BM
        assert bm == want and (s > 1) == (m == 1), (role.library, m, s, bm)
    monkeypatch.setenv(rf.ENV_WIDE, "0")
    seen.clear()
    _x, xq, a = _inputs(family, one_pass_m, COLS, 61)
    _fused(role, xq, a)
    assert seen == [(1, rf.BM)]


# --- random fractional mixes inside every run table (tessera#750) ----------------

#: The dense twin of ``test_routed_fused_window.test_random_mixes_inside_every_pair_decode_exactly``:
#: every adjacent pair the family's dense identity reads (1/2..7/8 on E4M3,
#: 1/2..13/14 on the value family since contract v51) is attested at random
#: rungs inside it, with the upper-rate columns placed at random.
DENSE_MIX_CASES = [(lib, r) for lib in LIBRARY_IDS
                   for r in rf.dense_rates("e4m3" if lib == "e4m3mma" else lib)
                   if r + 1 in rf.dense_rates("e4m3" if lib == "e4m3mma" else lib)]


def _mix_rungs(r, seed, draws=4):
    """Rungs strictly inside ``(256 r, 256 (r + 1))`` over 256 columns: one
    upper-rate column, all but one, and random interior counts (fixed seed)."""
    import random

    rng = random.Random(seed)
    return [256 * r + 1, 256 * r + 255] + sorted(rng.sample(range(256 * r + 2, 256 * r + 255), draws - 2))


@cuda
@pytest.mark.parametrize("family,r", DENSE_MIX_CASES, indirect=["family"])
def test_dense_random_mixes_inside_every_pair_decode_exactly(family, r):
    """At random rungs of the pair ``(r, r + 1)``, placed at random: one-hot
    rows decode bitwise with and without a start state, random inputs sit
    within the derived bound in the split and one-pass regimes, and the
    forward replays in a CUDA graph.  A pair above rate 8 is scheduled at the
    value family's dense cap, 14; the pairs up to 7/8 keep cap 8, so their
    rungs and placements are the ones v50 attested."""
    from fractions import Fraction

    from tessera.grammar import rate_set

    cap = 8 if r + 1 <= 8 else max(rf.dense_rates(family))
    sms = rf._sm_count(torch.cuda.current_device())
    for q256 in _mix_rungs(r, 7700 + r):
        assert rate_set(Fraction(q256, 256), cap=cap) == (r, r + 1), q256
        g = torch.Generator().manual_seed(7800 + q256)
        sched = _sched(COLS, q256, cap)
        rates = tuple(sched[i] for i in torch.randperm(COLS, generator=g).tolist())
        for seed, init in ((7900 + q256, None), (8000 + q256, _init(COLS, 8100 + q256))):
            expert, bundle = _role(family, rates=rates, seed=seed, init=init)
            assert rf.fused_dense_window_supported(bundle) is None, (family, q256)
            role = rf.prepare_dense_role(bundle)
            assert role.tile_words == 16 * sum(rates)
            _x, xq, a, hot = fb.one_hot_inputs(family, COLS, _quant)
            got = _fused(role, xq, a)
            want = fb.one_hot_expected(expert, family, hot, a)
            bad = got != want
            assert not bool(bad.any()), (
                f"{family} q256={q256} init={init is not None}: {int(bad.sum())} of "
                f"{bad.numel()} one-hot products differ; first at {bad.nonzero()[0].tolist()}")
        w64 = fb.fp64_weight(expert, family)
        one_pass_m = rf.BM * -(-sms // (ROWS // rf.BN))
        for m in (1, one_pass_m):
            s = rf.dense_k_split(m, ROWS, COLS, sms, tile_words=role.tile_words)
            _x, xq, a = _inputs(family, m, COLS, 8200 + m + q256)
            r64, bound = fb.dense_bound(family, _a64(family, xq, a), w64, COLS, s)
            fb.check_within(_fused(role, xq, a), r64, bound,
                            f"{family} q256={q256} M={m} S={s}: fused vs the fp64 reference")
        _x, xq, a = _inputs(family, 40, COLS, 8300 + q256)
        eager = _fused(role, xq, a)
        out = torch.empty_like(eager)
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _fused(role, xq, a, out, counter)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fused(role, xq, a, out, counter)
        for _ in range(2):
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eager), (family, q256)


# --- one launch per module, the split reduced in-kernel (tessera#750 WP2) ----------

#: The E4M3 libraries: their dense launch takes a module's roles and reduces a
#: K split in-kernel (``dense_forward_roles``).  The value library keeps one
#: launch per role and the reduce launch.
E4M3_LIBRARY_IDS = ["e4m3", "e4m3mma"]
#: 512 columns: 16 K chunks, so a fixup split may be 1 .. 5 (``dense_fixup_split_max``),
#: and S = 5 leaves items of three chunks, the fewest the pipeline allows.
WIDE_COLS = 512


def _force_split(monkeypatch, s):
    """Every launch after this runs at ``s`` (``dense_k_split`` is read per call)."""
    monkeypatch.setattr(rf, "dense_k_split", lambda *args, **kw: s)


def _roles_out(roles, xq, a, *, fixup=True):
    out = torch.empty(int(xq.shape[0]), sum(r.rows for r in roles), dtype=torch.bfloat16, device="cuda")
    rf.dense_forward_roles(roles, xq, a, out, fixup=fixup)
    return out


@cuda
@pytest.mark.parametrize("family", E4M3_LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("rows", [256, 160, 32, 12, 4])
def test_the_in_kernel_fixup_is_the_reduce_kernel_bitwise(family, rows, monkeypatch):
    """The same partials reduced two ways -- in the kernel by the last split
    of each tile to arrive, and by ``dense_reduce_kernel`` after the launch --
    agree BITWISE at every split the launch takes (2 .. 5, the last at three K
    chunks per item), at one superblock (M = 1, 40, 64) and several (65, 200),
    for whole blocks and N-tails down to the row quantum (4).  Both sum in
    split order from 0.f, so the arrival order the fixup sees does not reach
    the output."""
    _expert, bundle = _role(family, rows=rows, cols=WIDE_COLS, rates=_sched(WIDE_COLS, 1088),
                            seed=9100 + rows)
    role = rf.prepare_dense_role(bundle)
    assert rf.dense_fixup_split_max(WIDE_COLS) == 5
    for m in (1, 40, 64, 65, 200):
        _x, xq, a = _inputs(family, m, WIDE_COLS, 9200 + m)
        for s in range(2, rf.dense_fixup_split_max(WIDE_COLS) + 1):
            _force_split(monkeypatch, s)
            reduced = _fused(role, xq, a)                     # one role, the reduce launch
            fixed = _roles_out([role], xq, a)                 # the in-kernel fixup
            assert torch.equal(fixed, reduced), (family, rows, m, s)
            assert torch.equal(_roles_out([role], xq, a, fixup=False), reduced)


#: Modules in miniature with role boundaries everywhere: whole blocks, N-tails
#: (32, 64, 160 rows: the KDA input module's small roles and a tail after a
#: whole block), two roles of the same height apart, and roles of 4 and 12
#: rows ahead of others, so a role's first column is 4-aligned but not
#: 32-aligned (the row quantum, ``DENSE_ROW_QUANTUM``).
MODULE_ROWS = [256, 32, 64, 160, 256]
MODULE_ROW_SETS = [MODULE_ROWS, [4, 256, 12, 32, 160]]


@cuda
@pytest.mark.parametrize("family", E4M3_LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("module_rows", MODULE_ROW_SETS, ids=["blocks-and-tails", "quantum-offsets"])
def test_one_launch_of_a_modules_roles_is_each_role_alone_bitwise(family, module_rows, monkeypatch):
    """One launch of five roles against each role launched alone at the same
    split: every role's columns BITWISE equal at its column offset, at S = 1
    (including the wide superblock at M = 200) and in the split regime.  With
    ``test_the_in_kernel_fixup_is_the_reduce_kernel_bitwise`` this ties the
    module launch to the one-role launch the definition-bound tests read."""
    rates = _sched(WIDE_COLS, 832)
    roles = [rf.prepare_dense_role(_role(family, rows=r, cols=WIDE_COLS, rates=rates, seed=9300 + i)[1])
             for i, r in enumerate(module_rows)]
    for m in (1, 40, 200):
        _x, xq, a = _inputs(family, m, WIDE_COLS, 9400 + m)
        for s in (1, 2, rf.dense_fixup_split_max(WIDE_COLS)):
            _force_split(monkeypatch, s)
            together = _roles_out(roles, xq, a)
            offset = 0
            for role in roles:
                alone = _roles_out([role], xq, a)
                assert torch.equal(together[:, offset:offset + role.rows], alone), (family, m, s, role.rows)
                offset += role.rows
    # Past MAX_ROLES the module is launched in groups; the answer does not move.
    many = [rf.prepare_dense_role(_role(family, rows=32, cols=WIDE_COLS, rates=rates, seed=9500 + i)[1])
            for i in range(rf.MAX_ROLES + 1)]
    _x, xq, a = _inputs(family, 1, WIDE_COLS, 9600)
    _force_split(monkeypatch, 3)
    together = _roles_out(many, xq, a)
    for i, role in enumerate(many):
        assert torch.equal(together[:, 32 * i:32 * (i + 1)], _roles_out([role], xq, a))


@cuda
@pytest.mark.parametrize("family", E4M3_LIBRARY_IDS, indirect=True)
def test_the_fixup_serves_a_row_stride_that_is_only_even(family, monkeypatch):
    """The fixup stores through the epilogue's four-byte ``store_seg``, which
    needs only an even row stride, and reads the fp32 workspace, which is
    contiguous.  So unlike the reduce launch (uint2 stores) it keeps its split
    for a ``[M, rows]`` view whose row stride is 2 mod 4, gives the answer of
    a contiguous output, and leaves the columns past the view untouched."""
    rows = 160
    role = rf.prepare_dense_role(_role(family, rows=rows, cols=WIDE_COLS, rates=_sched(WIDE_COLS, 1088),
                                       seed=9650)[1])
    for m in (1, 40):
        _x, xq, a = _inputs(family, m, WIDE_COLS, 9660 + m)
        _force_split(monkeypatch, 3)
        wide = torch.zeros(m, rows + 2, dtype=torch.bfloat16, device="cuda")
        view = wide[:, :rows]
        assert view.stride(0) % 4 == 2
        rf.dense_forward_roles([role], xq, a, view)
        assert torch.equal(view, _roles_out([role], xq, a)), (family, m)
        assert torch.equal(wide[:, rows:], torch.zeros_like(wide[:, rows:]))


@cuda
@pytest.mark.parametrize("family", E4M3_LIBRARY_IDS, indirect=True)
def test_a_module_launch_captures_and_replays_against_eager(family, monkeypatch):
    """The work counter and the arrival counts are zeroed inside the captured
    region and the workspace is a graph-pool allocation, so two replays equal
    the eager module launch bitwise, and new inputs replay to the new answer."""
    rates = _sched(WIDE_COLS, 1088)
    roles = [rf.prepare_dense_role(_role(family, rows=r, cols=WIDE_COLS, rates=rates, seed=9700 + i)[1])
             for i, r in enumerate(MODULE_ROWS)]
    _force_split(monkeypatch, 3)
    m = 40
    _x, xq, a = _inputs(family, m, WIDE_COLS, 9800)
    eager = _roles_out(roles, xq, a)
    out = torch.empty_like(eager)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            rf.dense_forward_roles(roles, xq, a, out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rf.dense_forward_roles(roles, xq, a, out)
    for _ in range(2):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    _x2, xq2, a2 = _inputs(family, m, WIDE_COLS, 9801)
    xq.copy_(xq2)
    a.copy_(a2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, _roles_out(roles, xq2, a2))


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_dense_forward_roles_takes_one_rung_of_one_e4m3_library(family):
    """A launch's roles share the rung (cols, tile_words, slot_words) and the
    library; the value family's roles take ``dense_forward``."""
    _expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    _x, xq, a = _inputs(family, 1, COLS, 9950)
    if family == "value":
        with pytest.raises(ValueError, match="one E4M3 library"):
            _roles_out([role], xq, a)
        return
    other = rf.prepare_dense_role(_role(family, cols=WIDE_COLS, seed=9951)[1])
    with pytest.raises(ValueError, match="share cols"):
        rf.dense_forward_roles([role, other], xq, a,
                               torch.empty(1, 2 * ROWS, dtype=torch.bfloat16, device="cuda"))


# --- a split the kernel reduces keeps every item at least STAGES + 1 chunks ----------------

@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_a_split_that_leaves_an_item_fewer_than_three_chunks_is_refused_by_the_fixup_only(family, monkeypatch):
    """The producers run at most ``STAGES`` K chunks ahead of the consumers,
    so an item of fewer than ``STAGES + 1`` chunks could let them rewrite the
    descriptor and row-scale slot of the item two back while it is still
    read.  The in-kernel fixup (``dense_forward_roles``, the E4M3 libraries)
    refuses such a split by name, and ``dense_forward_roles`` keeps a forced
    one inside its range: past it, the launch is the one at the range's end,
    bitwise.  The one-role entry keeps the tessera#805 bound,
    ``dense_split_max`` (two chunks per item), on every library."""
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    lib = rf._ext(role.library)
    assert lib.STAGES == rf.STAGES and rf.dense_fixup_split_max(COLS) == 2
    m = 1
    sms = rf._sm_count(torch.cuda.current_device())
    _x, xq, a = _inputs(family, m, COLS, 9900)
    s = rf.dense_fixup_split_max(COLS) + 1
    assert s <= rf.dense_split_max(COLS)
    empty = xq.new_empty(0, dtype=torch.float32)
    a_or_empty = a if a is not None else empty
    # The one-role entry: every split up to the tessera#805 bound, within the numeric bound.
    for k in (s, rf.dense_split_max(COLS)):
        _force_split(monkeypatch, k)
        _within(_fused(role, xq, a), _bound(expert, family, xq, a, k), f"{family} one-role S={k}")
    if family == "value":
        return
    blocks = -(-ROWS // rf.BN)
    nsb = -(-m // rf.BM)
    partial = torch.empty((s, m, ROWS), dtype=torch.float32, device="cuda")
    out = torch.empty(m, ROWS, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="fewer than 3"):
        lib.dense_forward_roles(True, xq, a_or_empty, [role.words], [role.table16], [role.init],
                                [role.has_init], [role.wscale], [role.runs], [role.bdesc],
                                int(role.tile_words), int(role.slot_words),
                                torch.zeros(1 + blocks * nsb, dtype=torch.int32, device="cuda"), s, partial,
                                out, sms, rf.BM, True)
    _force_split(monkeypatch, s)
    past = _roles_out([role], xq, a)
    _force_split(monkeypatch, s - 1)
    assert torch.equal(past, _roles_out([role], xq, a)), family
