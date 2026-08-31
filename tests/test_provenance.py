"""S14 ledger invariant: content-addressed ancestry, not a text denylist."""

import hashlib

import pytest

from tessera.errors import ProvenanceError
from tessera.provenance import (
    Denylist,
    InputNode,
    assert_denylist_populated,
    check_closure,
    content_digest,
)


def d(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def test_empty_denylist_refuses_to_certify():
    """A closure checked against nothing is not a passing check."""
    with pytest.raises(ProvenanceError, match="empty"):
        assert_denylist_populated(Denylist())


def test_clean_closure_passes():
    nodes = [
        InputNode(d("bf16"), "checkpoint"),
        InputNode(d("calib"), "calibration", parents=(d("bf16"),)),
        InputNode(d("probe"), "probe", parents=(d("calib"),)),
    ]
    check_closure(nodes, Denylist({d("prohibited")}, "test"))


def test_direct_prohibited_input_is_rejected():
    nodes = [InputNode(d("bad"), "calibration", label="corpus")]
    with pytest.raises(ProvenanceError, match="prohibited source"):
        check_closure(nodes, Denylist({d("bad")}, "test"))


def test_prohibited_ancestor_is_rejected_transitively():
    """Relabelling or copying does not change identity.

    A closed closure necessarily contains its own ancestors, so the direct
    scan reports first; the ancestry walk is the second line of defence for
    the case where a node cites a parent the scan has not reached. Either way
    the derived, clean-looking leaf does not survive.
    """
    nodes = [
        InputNode(d("bad"), "checkpoint", label="prohibited"),
        InputNode(d("mid"), "activation_cache", parents=(d("bad"),)),
        InputNode(d("leaf"), "cost_table", label="clean-looking",
                  parents=(d("mid"),)),
    ]
    with pytest.raises(ProvenanceError, match="prohibited"):
        check_closure(nodes, Denylist({d("bad")}, "test"))


def test_stripping_the_prohibited_parent_still_fails():
    """Omitting the prohibited ancestor makes the closure unclosed, not clean."""
    nodes = [
        InputNode(d("mid"), "activation_cache", parents=(d("bad"),)),
        InputNode(d("leaf"), "cost_table", parents=(d("mid"),)),
    ]
    with pytest.raises(ProvenanceError, match="not closed"):
        check_closure(nodes, Denylist({d("bad")}, "test"))


def test_unclosed_closure_fails_rather_than_passes():
    nodes = [InputNode(d("leaf"), "cost_table", parents=(d("absent"),))]
    with pytest.raises(ProvenanceError, match="not closed"):
        check_closure(nodes, Denylist({d("x")}, "test"))


def test_duplicate_digests_are_rejected():
    nodes = [InputNode(d("a"), "x"), InputNode(d("a"), "y")]
    with pytest.raises(ProvenanceError, match="duplicate"):
        check_closure(nodes, Denylist(set(), "test"))


def test_malformed_digest_is_rejected():
    with pytest.raises(ProvenanceError, match="64 lowercase hex"):
        InputNode("not-a-digest", "x")


def test_content_digest_is_sha256():
    assert content_digest(b"") == hashlib.sha256(b"").hexdigest()
