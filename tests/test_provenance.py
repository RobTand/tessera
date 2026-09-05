"""S14 ledger invariant: content-addressed ancestry, not a text denylist."""

import hashlib
import json

import pytest

from tessera.errors import ProvenanceError
from tessera.provenance import (
    DIGEST_LENGTH,
    Denylist,
    InputNode,
    assert_denylist_populated,
    check_closure,
    content_digest,
    load_denylist,
)


def d(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def write_denylist(tmp_path, entries):
    path = tmp_path / "owner-denylist.json"
    path.write_text(json.dumps({"prohibited_digests": entries}))
    return path


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


def test_the_identity_length_is_the_hash_s_and_not_a_number_anyone_chose():
    """The one shape, derived from the digest this module actually takes."""
    assert DIGEST_LENGTH == len(content_digest(b""))


# ------------------------------- what the owner's list may contain (tessera#226)
#
# `load_denylist` lowercased its entries and checked only their *length*, while
# `InputNode` requires 64 lowercase hex characters. So a configured entry of
# sixty-four `z`s certified a denylist as populated -- `assert_denylist_populated`
# passed, `check_closure` ran, "certified 1" printed -- against an identity no
# possible input can carry. The gate that exists to refuse a check against an
# ineffective list was passable by a malformed list.


@pytest.mark.parametrize("entry", [
    "z" * 64,                       # the issue's own entry
    "Z" * 64,                       # and its upper case, which normalises to it
    "0" * 63,
    "0" * 65,
    "",
    "0x" + "0" * 62,
    " " + "0" * 63,
    None,
    12345,
])
def test_a_digest_no_input_can_ever_carry_is_refused_at_load(tmp_path, entry):
    with pytest.raises(ProvenanceError, match="denylist digest"):
        load_denylist(write_denylist(tmp_path, [entry]))


def test_a_valid_entry_still_loads_normalized_and_still_catches_its_input(tmp_path):
    """Normalisation is retained: an owner may write the digest in either case."""
    denylist = load_denylist(write_denylist(tmp_path, [d("prohibited").upper()]))
    assert denylist.digests == {d("prohibited")}
    assert_denylist_populated(denylist)
    with pytest.raises(ProvenanceError, match="prohibited source"):
        check_closure([InputNode(d("prohibited"), "calibration")], denylist)


def test_one_malformed_entry_refuses_the_whole_list(tmp_path):
    """Not "drop it and certify the rest": the owner's record is not this
    module's to edit, and a list that lost a row silently is a list nobody
    checked against."""
    with pytest.raises(ProvenanceError, match="denylist digest"):
        load_denylist(write_denylist(tmp_path, [d("bad"), "z" * 64]))


def test_a_parent_digest_is_the_same_identity_as_a_node_digest():
    """A parent that cannot be a digest cannot be the node it names."""
    with pytest.raises(ProvenanceError, match="64 lowercase hex"):
        InputNode(d("leaf"), "cost_table", parents=("not-a-digest",))
