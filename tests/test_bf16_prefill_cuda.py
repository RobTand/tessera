"""Real-wire current-arithmetic tests, not performance or serving qualification.

GPU execution requires the assignment's exact-tuple authorization and PB
allocation. CPU collection/skips are not evidence for any CUDA assertion here.
"""
import pytest
import torch

import fused_bound as bounds
from test_dense_fused_window import GLM_DENSE_ROLES, _module, _scheme, _tessera
from tessera.compact_prep import DENSE_WINDOW_RATE_MAX
from tessera.serving.bf16_prefill import decode_bf16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="raw-BF16 prefill requires CUDA")


@pytest.fixture(autouse=True)
def packed_triton_lane(monkeypatch):
    from tessera.serving import flags

    flags.reset_for_tests()
    monkeypatch.setenv("TESSERA_DENSE_FUSED", "0")
    yield
    flags.reset_for_tests()


def encode(roles, columns, q256):
    fused, export, _stock, decode, alphabet = _tessera()
    generator = torch.Generator(device="cuda").manual_seed(q256)
    blobs, values, scales = [], [], []
    for name, rows in roles:
        source = torch.randn(rows, columns, device="cuda", generator=generator)
        exported, unit, forests = export.encode_linear_planes(
            source.contiguous(), grid=alphabet.BF16_GRID, q256=q256,
            name=name, verify=False)
        raw, scale = decode.materialize_bf16(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
        values.append(raw.to("cuda"))
        scales.append(scale.to("cuda", torch.float32).reshape(-1))
    blob = fused.pack_fused(blobs)
    scheme = _scheme("value", sum(rows for _, rows in roles), columns,
                     roles, len(blob), q256)
    return _module(blob, scheme), torch.cat(values), torch.cat(scales)


def compare(owner, raw, scale, token_counts):
    copy = decode_bf16(owner, min_m=256, chunk=192)
    assert torch.equal(copy.weight, raw)
    assert torch.equal(copy.scale, scale)
    packed_bytes = owner.packed_bytes()
    fingerprint = owner.fingerprints()
    generator = torch.Generator(device="cuda").manual_seed(owner.columns)
    effective = raw.double() * scale.double()[:, None]
    for tokens in token_counts:
        activations = torch.randn(tokens, owner.columns, device="cuda",
                                  generator=generator).bfloat16()
        packed_result = owner.apply(activations)
        result = copy.apply(activations)
        reference, bound = bounds.dense_bound(
            "value", activations.double(), effective, owner.columns, s=owner.columns)
        bounds.check_within(result, reference, bound, f"decoded M={tokens}")
        bounds.check_within(packed_result, reference, bound, f"packed M={tokens}")
        bounds.check_within(result, packed_result.double(), bound,
                            f"decoded vs packed M={tokens}", scale=2.0)
    assert owner.fingerprints() == fingerprint
    owner.attach_decoded(copy)
    assert owner.packed_bytes() == packed_bytes + copy.nbytes
    assert owner.launch_pair_for(token_counts[-1]) == copy.launch_pair
    assert torch.equal(owner.apply(activations), result)
    return owner


RATE_RUNGS = tuple(256 * rate for rate in range(1, DENSE_WINDOW_RATE_MAX + 1)) + tuple(
    256 * rate + 128 for rate in range(1, DENSE_WINDOW_RATE_MAX))


@pytest.mark.parametrize("q256", RATE_RUNGS)
def test_real_wire_rates_and_adjacent_mixtures_through_m8192(q256):
    owner, raw, scale = encode([("gate_proj", 128), ("up_proj", 64)], 256, q256)
    compare(owner, raw, scale, (1, 256, 8192))


PREFILL_GEOMETRIES = tuple(
    pytest.param([(name, GLM_DENSE_ROLES[name][0])], GLM_DENSE_ROLES[name][1], id=name)
    for name in ("dense_gate_up", "dense_down", "mlp_gate_up_tp2", "mlp_down_tp2",
                 "shared_gate_up_tp2", "shared_down_tp2", "q_b_proj")
) + (pytest.param([("o_proj", 4096)], 8192, id="mla_o_proj-4096x8192"),)


@pytest.mark.parametrize("roles,columns", PREFILL_GEOMETRIES)
def test_real_wire_mlp_shared_and_corrected_attention_through_m8192(roles, columns):
    """MLA output uses full K=8192, not the historical partial K=4096."""
    owner, raw, scale = encode(roles, columns, 1024)
    compare(owner, raw, scale, (256, 2048, 8192))


def test_graph_replays_current_inputs_with_the_same_resident_copy():
    owner, raw, scale = encode([("q_proj", 128)], 256, 1024)
    copy = decode_bf16(owner, min_m=256, chunk=192)
    owner.attach_decoded(copy)
    activations = torch.randn(256, owner.columns, device="cuda").bfloat16()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        owner.apply(activations)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = owner.apply(activations)
    before = owner.fingerprints()
    for factor in (1.0, -2.0):
        activations.mul_(factor)
        expected = owner.apply(activations)
        graph.replay()
        assert torch.equal(captured, expected)
        reference, bound = bounds.dense_bound(
            "value", activations.double(), raw.double() * scale.double()[:, None],
            owner.columns, s=owner.columns)
        bounds.check_within(captured, reference, bound, "graph replay")
    assert owner.fingerprints() == before
