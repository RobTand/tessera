"""The seal's per-unit digest is taken ahead of consumption, off the encode thread.

``ActivationSource._require_sealed_unit`` digests every unit before the
encoder sees a byte (tessera#302, kept through the resident binding of
tessera#440).  On the GLM census that digest -- stage 64 MiB to host, sha256
it -- ran on the encode thread at the head of every batch with the GPU idle
behind it (PrismaQuant boundary-feed, 2026-09-11).  The seal now keeps a memo
of each unit's digest beside the tensor's signature and an exact fingerprint
of the bytes digested: a plain mapping fills it while sealing, a bound
``ReferenceHessians`` fills it on a helper thread in roster order, and the
consumer accepts the memo only for the same tensor whose bytes, re-read on
the device, fingerprint to what was hashed.  Everything else digests inline
as before.

What these pin: the memo agrees with ``tensor_identity`` byte for byte; a
consumption served from it makes no digest; an edit after the digest -- in
place, through ``.data``, from a rewrite of equal bytes -- is either refused
or re-digested, never served stale; the prefetch observes no unit; the lever
``TESSERA_SEAL_PREFETCH=0`` is the inline path; and the helper's staging
(a pinned buffer on its own stream) runs beside the encoder's graph captures
without faulting them, which is the contract ``window_viterbi.capture``
states for other threads.
"""
import threading
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tessera import cached_unit as cached_unit_module
from tessera.cached_unit import digest_host_tensor, host_fingerprint, tensor_identity
from tessera.errors import GrammarError
from tessera.export import (ActivationSource, _SEAL_PREFETCH_ENV, _prefetch_seal_digests,
                            _tensor_signature, device_fingerprint)
from tessera.manifest import ScalePlaneKind

from test_hessian_reference_capture import reference  # noqa: F401  (fixture)
from test_hessian_resident_binding import bound_source, counted, encode_kwargs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PROVENANCE = {"text_sha256": "0" * 64, "fit_tokens": 16, "fit_ids_sha256": "1" * 64}


def _plain(H, **settings):
    settings.setdefault("ldlq_sigma", None)
    return ActivationSource(hessians=H, provenance=PROVENANCE, **settings)


def _hessians(device="cpu", n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {name: (torch.randn(n, n, generator=g)).to(device)
            for name in ("a", "b", "c")}


def _consume(source, name="a", n=4, device="cpu"):
    return source.for_unit(f"{name}.weight", n, device, scale_plane=ScalePlaneKind.CHANNEL)


# --- the digest construction is unchanged ------------------------------------

def test_digest_host_tensor_is_tensor_identitys_digest():
    value = torch.randn(6, 5)
    assert digest_host_tensor(value) == tensor_identity(value)["sha256"]
    assert digest_host_tensor(value.to(torch.float16)) == \
        tensor_identity(value.to(torch.float16))["sha256"]
    assert digest_host_tensor(value.t().contiguous()) == \
        tensor_identity(value.t())["sha256"]
    with pytest.raises(ValueError):
        digest_host_tensor(value.t())


def test_fingerprints_agree_between_host_and_device_paths():
    value = torch.randn(7, 9)
    assert host_fingerprint(value) == device_fingerprint(value)
    odd = torch.randint(0, 255, (3, 5), dtype=torch.uint8)      # not a whole word count
    assert host_fingerprint(odd) == device_fingerprint(odd)
    assert device_fingerprint(value.t()) == host_fingerprint(value.t().contiguous())


# --- a plain mapping -----------------------------------------------------------

def test_a_plain_mapping_is_consumed_from_its_own_seal(monkeypatch):
    H = _hessians()
    source = _plain(H)
    source.capture_sha256()
    memo = source._seal_memo
    assert {name: entry.sha256 for name, entry in memo.items()} == \
        {name: tensor_identity(v)["sha256"] for name, v in H.items()}
    digests = counted(monkeypatch, cached_unit_module, "tensor_identity")
    kwargs = _consume(source)
    assert kwargs["refit_metric"].equal(H["a"])
    assert digests == [], "a unit the seal digested must not be digested again at consumption"


def test_an_in_place_edit_after_the_seal_is_refused():
    H = _hessians()
    source = _plain(H)
    source.capture_sha256()
    H["a"][0, 0] += 1.0
    with pytest.raises(GrammarError, match="edited in place or replaced"):
        _consume(source)


def test_a_data_swap_after_the_seal_is_refused():
    H = _hessians()
    source = _plain(H)
    source.capture_sha256()
    H["a"].data = torch.zeros(4, 4)
    with pytest.raises(GrammarError, match="edited in place or replaced"):
        _consume(source)


def test_a_rewrite_with_equal_bytes_is_redigested_and_served(monkeypatch):
    """``copy_`` bumps the version counter: the memo no longer matches, the
    unit is digested inline, it agrees with the seal, and the memo is
    refreshed so the next consumption is free again."""
    H = _hessians()
    source = _plain(H)
    source.capture_sha256()
    H["a"].copy_(H["a"].clone())
    digests = counted(monkeypatch, cached_unit_module, "tensor_identity")
    _consume(source)
    assert len(digests) == 1
    _consume(source)
    assert len(digests) == 1


def test_prefetch_off_is_the_inline_digest(monkeypatch):
    monkeypatch.setenv(_SEAL_PREFETCH_ENV, "0")
    H = _hessians()
    source = _plain(H)
    source.capture_sha256()
    assert getattr(source, "_seal_memo", None) is None
    digests = counted(monkeypatch, cached_unit_module, "tensor_identity")
    _consume(source)
    assert len(digests) == 1


def test_the_prefetch_lever_is_zero_or_one(monkeypatch):
    monkeypatch.setenv(_SEAL_PREFETCH_ENV, "2")
    with pytest.raises(GrammarError, match="TESSERA_SEAL_PREFETCH"):
        _plain(_hessians()).capture_sha256()


# --- a bound ReferenceHessians ---------------------------------------------

@pytest.fixture
def resident(reference):  # noqa: F811
    handoff, payload, H, _, _ = reference
    return handoff, payload, H


def test_a_bound_roster_is_digested_ahead_and_consumed_without_a_digest(resident, monkeypatch):
    handoff, payload, H = resident
    source = bound_source(handoff, H)
    assert source.capture_sha256() == payload["capture_sha256"]
    assert source._seal_prefetch_thread is not None
    source.wait_seal_prefetch()
    assert {n: e.sha256 for n, e in source._seal_memo.items()} == \
        {n: tensor_identity(v)["sha256"] for n, v in H.items()}
    digests = counted(monkeypatch, cached_unit_module, "tensor_identity")
    kwargs = encode_kwargs(source)
    assert kwargs["refit_metric"] is H["a"]
    assert digests == []
    # The prefetch read both units; only 'a' was served.
    assert source.hessians.receipt()["resident_units_observed"] == ["a"]
    source.hessians.close()


def test_a_bound_unit_edited_after_the_prefetch_is_refused(resident):
    handoff, _, H = resident
    source = bound_source(handoff, H)
    source.capture_sha256()
    source.wait_seal_prefetch()
    H["a"][0, 0] = 99.0
    with pytest.raises(GrammarError, match="edited in place or replaced"):
        encode_kwargs(source)
    source.hessians.close()


def test_the_helper_stops_when_the_owner_closes(resident):
    handoff, _, H = resident
    source = bound_source(handoff, H)
    source.capture_sha256()
    source.hessians.close()
    source.stop_seal_prefetch()
    assert source.hessians.resident_mapping() is None


def test_an_unbound_reference_prefetches_nothing(reference):  # noqa: F811
    handoff, _, _, _, _ = reference
    source = ActivationSource.from_capture(handoff, ldlq_sigma=None)
    source.capture_sha256()
    assert source._seal_prefetch_thread is None
    assert source._seal_memo == {}
    source.hessians.close()


# --- CUDA: staging through pinned memory, beside the encoder's captures -----

@cuda
def test_a_resident_cuda_roster_is_digested_through_pinned_memory(resident, monkeypatch):
    handoff, _, H = resident
    device = {name: value.cuda() for name, value in H.items()}
    source = bound_source(handoff, device)
    source.capture_sha256()
    source.wait_seal_prefetch()
    assert {n: e.sha256 for n, e in source._seal_memo.items()} == \
        {n: tensor_identity(v)["sha256"] for n, v in H.items()}
    digests = counted(monkeypatch, cached_unit_module, "tensor_identity")
    kwargs = encode_kwargs(source, device="cuda")
    assert kwargs["refit_metric"] is device["a"]
    assert digests == []
    device["a"][1, 1] = -5.0
    with pytest.raises(GrammarError, match="edited in place or replaced"):
        encode_kwargs(source, device="cuda")
    source.hessians.close()


@cuda
def test_the_helper_runs_beside_the_encoders_graph_captures():
    """The helper's stage-and-hash on its own stream, while the encode thread
    captures and replays window Viterbi graphs, faults nothing and digests
    what ``tensor_identity`` digests."""
    from tessera.encode import viterbi_window
    from tessera.window_viterbi import fused_available, window_plan_cache_clear

    if not fused_available():
        pytest.skip("needs the fused window Viterbi")
    g = torch.Generator().manual_seed(11)
    units = {f"u{i}": torch.randn(1024, 1024, generator=g).cuda() for i in range(12)}
    expected = {n: tensor_identity(v)["sha256"] for n, v in units.items()}
    memo, lock, stop = {}, threading.Lock(), threading.Event()
    helper = threading.Thread(
        target=_prefetch_seal_digests,
        args=(lambda: units, sorted(units), memo, lock, stop))
    targets = torch.randn(128, 96, generator=g).cuda()
    vectors = torch.randn(1 << 14, 1, generator=g).cuda()
    ref, sse_ref = viterbi_window(targets, vectors, 14, 4, impl="reference")
    window_plan_cache_clear()
    helper.start()
    answers = []
    for L, R in ((14, 4), (12, 3), (10, 2), (14, 5)):        # new shapes: captures
        vecs = torch.randn(1 << L, 1, generator=g).cuda()
        for _ in range(3):
            answers.append(viterbi_window(targets, vecs, L, R, impl="fused"))
    got, sse = viterbi_window(targets, vectors, 14, 4, impl="fused")
    helper.join(timeout=120)
    assert not helper.is_alive()
    assert torch.equal(got, ref) and sse == sse_ref
    assert {n: e.sha256 for n, e in memo.items()} == expected
    assert all(e.signature == _tensor_signature(units[n]) for n, e in memo.items())
    assert all(e.fingerprint == device_fingerprint(units[n]) for n, e in memo.items())
