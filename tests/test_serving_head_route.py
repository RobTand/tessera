"""The LM head through the Tessera dense route (tessera#750 WP3).

``head_route.build_tessera_head_method`` hands a declared ``ParallelLMHead``
the family's dense method.  vLLM calls that method on the head exactly as it
calls it on a Linear, but positionally from ``VocabParallelEmbedding``:
``create_weights(layer, embedding_dim, [num_embeddings_per_partition],
embedding_dim, num_embeddings_padded, params_dtype=..., weight_loader=...)``.
These tests drive that call on a head-shaped layer with a real E4M3 wire, at
one rank and at each rank of two, and compare every rank's logits against the
stock pair's product over that rank's vocabulary rows.

The layer is a stand-in carrying what ``ParallelLMHead`` has set when it asks
for a method: its TP coordinates and its vocabulary geometry, and no
``prefix``.  The vLLM classes are stubbed as in ``test_serving_fp8_route``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_serving_fp8_route import (_LAST_A, _encode_module, _install_vllm_stubs,  # noqa: E402
                                    _reference_fp8_quant, requires_cuda)
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import native_ops                               # noqa: E402
from tessera.serving.head_route import build_tessera_head_method     # noqa: E402
from tessera.serving.lane import MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV  # noqa: E402
from tessera.serving.scheme import TESSERA_FP8                        # noqa: E402

PREFIX = "language_model.lm_head"


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


class _Head(torch.nn.Module):
    """``ParallelLMHead`` at ``create_weights``: TP coordinates and an
    unpadded vocabulary, no ``prefix`` (vLLM's embedding classes store none)."""

    def __init__(self, vocab: int, tp_rank: int, tp_size: int):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.org_vocab_size = self.num_embeddings = self.num_embeddings_padded = vocab
        self.num_embeddings_per_partition = vocab // tp_size


def _drive_head(monkeypatch, mode, *, vocab=512, cols=1024, tp=(0, 1), m=8, seed=0):
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_ops, "native_fp8_quant", _reference_fp8_quant)
    monkeypatch.setattr(native_ops, "require_native_fp8_quant", lambda context: None)
    blob, scheme, _weight, _scale, ref_w = _encode_module([("lm_head", vocab)], cols=cols, seed=seed)
    rank, size = tp
    head = _Head(vocab, rank, size)
    method = build_tessera_head_method(scheme, PREFIX, mode, head)
    assert type(method).__name__ == "TesseraFp8LinearMethod"
    per = vocab // size
    # VocabParallelEmbedding's call, positionally as vLLM makes it.
    method.create_weights(head, cols, [per], cols, vocab, params_dtype=torch.bfloat16,
                          weight_loader=None)
    head.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    head.to(torch.device("cuda"))
    method.process_weights_after_loading(head)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed + 1))
    got = method.apply(head, x)
    want = (_LAST_A["value"] @ ref_w[rank * per:(rank + 1) * per].t()).to(torch.bfloat16)
    return got, want, head


def _rel(got, want):
    return (got.float() - want.float()).abs().max().item() / max(want.float().abs().max().item(), 1e-9)


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
@pytest.mark.parametrize("tp", [(0, 1), (0, 2), (1, 2)], ids=["tp1", "tp2-rank0", "tp2-rank1"])
def test_each_rank_serves_its_vocabulary_rows(monkeypatch, mode, tp):
    got, want, head = _drive_head(monkeypatch, mode, tp=tp)
    per = 512 // tp[1]
    assert tuple(got.shape) == (8, per) and got.dtype == torch.bfloat16
    assert head.tessera_rows == per and head.tessera_family == TESSERA_FP8
    assert _rel(got, want) < 8e-3


@requires_cuda
def test_the_two_ranks_are_the_one_rank_head_split(monkeypatch):
    """What ``LogitsProcessor`` gathers is the one-rank head's logits."""
    whole, _, _ = _drive_head(monkeypatch, MODE_RESIDENT, tp=(0, 1))
    halves = [_drive_head(monkeypatch, MODE_RESIDENT, tp=(r, 2))[0] for r in (0, 1)]
    assert _rel(torch.cat(halves, dim=1), whole) < 8e-3


@requires_cuda
def test_the_route_trace_names_the_head(monkeypatch):
    from tessera.serving.telemetry import read_route

    _got, _want, head = _drive_head(monkeypatch, MODE_RESIDENT)
    assert head.prefix == PREFIX
    record = read_route(head)
    assert record is not None and record["state"] == "served" and record["kind"] == "dense"
    assert record["policy"] == f"{TESSERA_FP8}:{MODE_RESIDENT}"
