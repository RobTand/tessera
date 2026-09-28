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

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import routed_fused as rf                    # noqa: E402
from tessera import window_gemm as wg                     # noqa: E402
from tessera.errors import GrammarError                   # noqa: E402

from test_window_gemm_grouped import Expert, _quant, _tol  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

L = 14
# One role: two 128-row N blocks, eight 32-column K steps (split-K up to 8).
ROWS, COLS = 256, 256
M_CASES = [1, 3, 64, 65, 200, 1536]
FAMILIES = ["value", "e4m3"]


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


def _reference(expert, family, xq, a):
    """The definition: exact fp32 products of the decoded weights, the row
    scale (folded before the dot for the value family, on the accumulator for
    E4M3) and the per-token activation scale, rounded once to bf16."""
    if family == "e4m3":
        return (expert.reference(xq.float(), "e4m3") * a.reshape(-1, 1)).bfloat16()
    return expert.reference(xq, "value", folded=True).bfloat16()


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


def _close(out, ref, what):
    err = float((out.float() - ref.float()).abs().max())
    assert err < _tol(ref), f"{what}: max abs err {err} vs tol {_tol(ref)}"
    return err


def _within_row_ulps(a, b, what, ulps=2):
    """Two bf16 renderings of one fp32 sum taken in two orders differ by at
    most a straddle of one bf16 rounding; with cancellation an element near
    zero carries the row's absolute error, so the unit is the bf16 ulp at the
    row's max magnitude (``2^(e-7)`` for a row max in ``[2^e, 2^(e+1))``)."""
    ra, rb = a.double(), b.double()
    rowmax = torch.maximum(ra.abs().amax(dim=-1, keepdim=True),
                           rb.abs().amax(dim=-1, keepdim=True)).clamp(min=2.0 ** -126)
    unit = torch.exp2(torch.floor(torch.log2(rowmax)) - 7)
    worst = float(((ra - rb).abs() / unit).max())
    assert worst <= ulps, f"{what}: {worst} row-max bf16 ulps apart (limit {ulps})"
    return worst


# --- the kernel against the definition and the Triton lane ------------------------

@cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("m", M_CASES)
def test_dense_forward_matches_the_definition_and_the_triton_lane(family, m):
    expert, bundle = _role(family)
    role = rf.prepare_dense_role(bundle)
    assert (role.rows, role.cols, role.fp8) == (ROWS, COLS, family == "e4m3")
    _x, xq, a = _inputs(family, m, COLS, 900 + m)
    ref = _reference(expert, family, xq, a)
    fused = _fused(role, xq, a)
    assert fused.shape == (m, ROWS) and fused.dtype == torch.bfloat16
    _close(fused, ref, f"{family} M={m}: fused vs the definition")
    # The Triton lane computes the same function of the same wire in another
    # accumulation order: parity is tolerance-bound, not bitwise.
    _within_row_ulps(fused, _triton(bundle, xq, a), f"{family} M={m}: fused vs Triton")


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
        _close(_fused(role, xq, a), _reference(expert, family, xq, a),
               f"{family} M={m} (S={rf.dense_k_split(m, ROWS, COLS, sms)})")


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
        _close(_fused(role, xq, a), _reference(expert, family, xq, a),
               f"{family} M={m} with a start state")
        _within_row_ulps(_fused(role, xq, a), _triton(bundle, xq, a),
                         f"{family} M={m} with a start state: fused vs Triton")


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
U_ACC = 2.0 ** -23          # one fp32 ulp per accumulation step (a truncating adder)
U32 = 2.0 ** -24            # fp32 unit roundoff (round-to-nearest multiply)
U64 = 2.0 ** -53            # fp64 unit roundoff (the reference's own dot)


def _gamma(n, u):
    return n * u / (1.0 - n * u)


def _bf16_ulp(v):
    """The bf16 ulp at magnitude ``v`` (fp64, >= 0): ``2^(e - 7)`` for ``v`` in
    ``[2^e, 2^(e+1))``, floored at the smallest normal's ulp."""
    return torch.exp2(torch.floor(torch.log2(v.clamp(min=2.0 ** -126))) - 7)


def _fp64_weight(expert, family):
    """The role's decoded weight in fp64 ``[rows, cols]``, row scale applied:
    the E4M3 bytes times the fp32 row scale (both exact in fp64), or the value
    table folded once to bf16 with the row scale, as the folded contract does."""
    states = expert.states.cuda()
    if family == "e4m3":
        byte = expert.unit.native[expert.unit.codes_of_state[states].long()]
        return byte.view(torch.float8_e4m3fn).double() * expert.scale.double()[:, None]
    values = expert.values.float().cuda()[states]
    return (values * expert.scale[:, None]).bfloat16().double()


def _glm_bound(family, a64, w64, k, s):
    """``(r, bound)``: the fp64 reference and the per-element bound on
    ``|fused - r|`` stated in the test's docstring."""
    r = a64 @ w64.t()
    sigma = a64.abs() @ w64.abs().t()
    e_acc = (_gamma(k + s + 2, U_ACC) + _gamma(k, U64)) * sigma
    e_pre = e_acc
    if family == "e4m3":
        e_pre = e_acc + _gamma(2, U32) * (r.abs() + e_acc)
    return r, e_pre + 0.5 * _bf16_ulp(r.abs() + e_pre)


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
    weight (fp32 [rows, cols]: ``stock_dequant`` of the materialised E4M3
    tiles, or the folded BF16 tile)."""
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
            refs.append(stock.stock_dequant(tiles).to("cuda").float())
        else:
            refs.append(decode.materialize_bf16_folded(unit, forests, export.DEFAULT_CODE)
                        .to("cuda").float())
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


def _module_reference(family, ref_w, xq, x, a):
    if family == "e4m3":
        return ((xq.float() * a.reshape(-1, 1)) @ ref_w.t()).bfloat16()
    return (x.float() @ ref_w.t()).bfloat16()


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
        _close(got, _module_reference(family, ref_w, xq, x, a),
               f"{family} module M={m}: fused vs the materialised reference")
    monkeypatch.setenv(rf.ENV_TOGGLE_DENSE, "0")
    twin = _module(blob, scheme)
    assert twin.lane == LANE_TRITON
    assert twin.launch_pair == (WINDOW_GEMM_SYMBOL, (
        telemetry.DECODER_NATIVE_WINDOW_GEMM if family == "e4m3"
        else telemetry.DECODER_NATIVE_WINDOW_GEMM_FOLDED))
    assert twin.lane_reason == f"role 'gate_proj': disabled by {rf.ENV_TOGGLE_DENSE}=0"
    for m in (1, 129):
        x, xq, a = _inputs(family, m, 256, 600 + m)
        _within_row_ulps(_served(module, family, xq, x, a), _served(twin, family, xq, x, a),
                         f"{family} module M={m}: fused vs Triton over the same bytes")
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
    _close(_served(module, family, xq, x, a), _module_reference(family, ref_w, xq, x, a),
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
    _e, mixed = _role("value", rates=tuple(3 if c % 2 else 4 for c in range(COLS)))
    assert "run table is not" in rf.fused_dense_window_supported(mixed)
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
    assert rf.dense_k_split(1, 256, 4096, sms) > 1
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
