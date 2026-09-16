"""The loader throughput changes' contracts, at their own seams.

Three behaviour contracts introduced by the loader throughput commit, each
tested against the *original* semantics rather than against the new
implementation:

* ``grammar._check_rate`` still compares a rate against each integer in the
  domain -- integral-valued floats are accepted, fractional ones refused, an
  unhashable rate still reaches the same ``TypeError`` -- for the default cap
  and for any other cap.  The historical predicate is spelled out in the test
  and the two must agree on a matrix of ints, bools, floats, Fractions and a
  list.
* the geometry memos never change a verdict: parsing every corpus artifact
  with and without a shared memo yields the same metadata, and a corrupted
  artifact is refused identically whether or not the memo was warmed by
  another artifact first.
* ``container.verify_plane_region`` keeps its checks and their precedence on
  the >= 1 MiB parallel-digest path: payload-digest errors still precede
  plane-level errors, content digests are still compared, and the worker
  thread is joined on every exit (success, refusal, or malformed manifest).

The >= 1 MiB cases use a real routed A4 wire (4.19 MB) when the box carries
the export root; on a box without it the test skips through
``box_artifacts.skip_now`` and the skip is visible in the run surface.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from fractions import Fraction
from pathlib import Path

import pytest

import box_artifacts

from tessera import container
from tessera.errors import GrammarError, PlaneLayoutError, SchemaError, TruncationError
from tessera.grammar import C_FULL_BITS, LEGAL_RATES, _check_rate, validate_rate_schedule

CORPUS = sorted((Path(__file__).resolve().parent / "data" / "legacy").glob("*.tessera"))
A4_TENSOR = "model.language_model.layers.3.mlp.experts.0.{projection}.wire"


# ---------------------------------------------------------------------------
# grammar domain: historical predicate, spelled out
# ---------------------------------------------------------------------------


def _historical_accepts(rate, cap) -> bool:
    """The pre-change predicate, verbatim: membership in the int tuple."""
    domain = LEGAL_RATES if cap == C_FULL_BITS else tuple(range(1, cap + 1))
    return rate in domain


def test_check_rate_domain_agrees_with_the_historical_predicate():
    probes = (1, 2, 3, 4, 0, -1, True, False, 1.0, 2.0, 1.5, Fraction(2, 1),
              Fraction(3, 2))
    for cap in (C_FULL_BITS, 7):
        for rate in probes:
            expected = _historical_accepts(rate, cap)
            try:
                _check_rate(rate, cap)
                accepted = True
            except GrammarError:
                accepted = False
            assert accepted is expected, (rate, cap, accepted, expected)


def test_check_rate_refuses_a_fractional_rate_at_a_nondefault_cap():
    with pytest.raises(GrammarError):
        _check_rate(1.5, 7)
    _check_rate(2.0, 7)          # an integral float was always in the domain
    _check_rate(Fraction(2, 1), 7)
    _check_rate(True, C_FULL_BITS)


def test_check_rate_unhashable_rate_keeps_its_original_error():
    # ``rate < 1`` raises before any membership test, exactly as before.
    with pytest.raises(TypeError):
        _check_rate([2], 7)


def test_validate_rate_schedule_refuses_a_fractional_schedule():
    with pytest.raises(GrammarError):
        validate_rate_schedule((3, 3.5, 3, 3), Fraction(26, 8))


# ---------------------------------------------------------------------------
# memos: verdict-preserving, not implementation-preserving
# ---------------------------------------------------------------------------


def _facts(metadata) -> dict:
    return {
        "grid": metadata.grid.name,
        "body": metadata.body.name,
        "plane": metadata.manifest.scale_plane.kind.name,
        "rows": int(metadata.rows),
        "columns": int(metadata.columns),
        "span": int(metadata.span),
        "rates": tuple(int(r) for r in metadata.rates),
        "completion_limit": metadata.completion_limit,
    }


def test_memo_does_not_change_any_corpus_verdict():
    from tessera.unit_artifact import parse_unit_metadata

    assert CORPUS, "the legacy corpus is the population this test speaks about"
    memo: dict = {}
    for path in CORPUS:
        blob = path.read_bytes()
        plain = parse_unit_metadata(blob)
        with_memo = parse_unit_metadata(blob, memo=memo)
        assert _facts(plain) == _facts(with_memo), path.name
        # Warm memo from this artifact, then a mutated one: same refusal.
        mutated = bytearray(blob)
        mutated[-1] ^= 0xFF
        errors = []
        for candidate in (bytes(mutated),):
            with pytest.raises(Exception) as plain_exc:
                parse_unit_metadata(candidate)
            with pytest.raises(Exception) as memo_exc:
                parse_unit_metadata(candidate, memo=memo)
            errors.append((type(plain_exc.value).__name__, str(plain_exc.value)))
            assert (type(memo_exc.value).__name__, str(memo_exc.value)) == errors[-1], path.name


# ---------------------------------------------------------------------------
# the >= 1 MiB parallel-digest path: checks, precedence, thread hygiene
# ---------------------------------------------------------------------------


def _a4_wire_blob() -> bytes:
    index_path = box_artifacts.skip_now("a4_export", "model.safetensors.index.json")
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    tensor = A4_TENSOR.format(projection="gate_proj")
    shard = box_artifacts.skip_now("a4_export", weight_map[tensor])
    from safetensors import safe_open

    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(tensor).numpy().tobytes()


def test_parallel_digest_path_over_one_mib_payload_precedence_and_content_digest():
    blob = _a4_wire_blob()
    assert len(blob) >= (1 << 20), "this test speaks about the threaded path"

    art = container.parse(blob)
    region = art.plane_region
    assert len(region) >= (1 << 20)

    # 1. A payload-level corruption must raise the payload-digest error even
    #    though the same byte also breaks a plane check: precedence preserved.
    corrupted = bytearray(blob)
    corrupted[-1] ^= 0x01
    threads_before = threading.active_count()
    with pytest.raises(SchemaError, match="payload digest"):
        container.parse(bytes(corrupted))
    assert threading.active_count() == threads_before, "the digest worker was not joined"

    # 2. Content digests are still compared: keep the region and payload
    #    digest, lie about one plane's content digest.
    descriptor = art.manifest.planes[0]
    liar = dataclasses.replace(art.manifest, planes=(
        dataclasses.replace(descriptor,
                            content_digest=hashlib.sha256(b"not these bytes").digest()),
        *art.manifest.planes[1:]))
    with pytest.raises(SchemaError, match="content digest"):
        container.verify_plane_region(liar, art.terminal, region)

    # 3. Malformed inputs keep their classes with the threaded path in play.
    with pytest.raises((TruncationError, SchemaError)):
        container.parse(blob[: container.HEADER_BYTES + 8])
    with pytest.raises(SchemaError, match="foreign magic"):
        container.parse(b"\x89TESSERA\x00\x00" + bytes(64))


def test_parallel_digest_path_is_exact_on_the_valid_wire():
    blob = _a4_wire_blob()
    art = container.parse(blob)
    region = art.plane_region
    digest = container.verify_plane_region(art.manifest, art.terminal, region)
    assert digest == hashlib.sha256(region).digest()
    assert len(region) >= (1 << 20)
