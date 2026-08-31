"""S9 identity: disjoint parser, collision, and name novelty."""

import hashlib

import pytest

from tessera.container import MAGIC
from tessera.errors import IdentityError
from tessera.identity import (
    AssignmentRecord,
    FamilyDescriptor,
    check_name_novelty,
    legacy_tcq_parse,
    parse_family_descriptor,
    require_terminal_record,
)
from tessera.manifest import BranchIdentity, ContainerClass, RotationState

TESSERA_NAMES = [
    "TESSERA_E2M1_R256",
    "TESSERA_E2M1_R512",
    "TESSERA_E4M3_R768",
]
LEGACY_NAMES = ["TCQ_E2M1_R512", "TCQ_E4M3_R1152", "TCQ_ABC_R0"]


def branch():
    return BranchIdentity(
        unit_id="u", root_q256=512, rotation=RotationState.NONE,
        container=ContainerClass.GRIDBOOK,
    )


@pytest.mark.parametrize("name", TESSERA_NAMES)
def test_tessera_names_parse(name):
    assert str(parse_family_descriptor(name)) == name


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_legacy_names_parse_to_the_legacy_two_tuple(name):
    parsed = legacy_tcq_parse(name)
    assert isinstance(parsed, tuple) and len(parsed) == 2


@pytest.mark.parametrize("name", TESSERA_NAMES)
def test_legacy_parser_never_accepts_a_tessera_name(name):
    """The legacy two-tuple parser never silently accepts a Tessera artifact."""
    with pytest.raises(IdentityError):
        legacy_tcq_parse(name)


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_tessera_parser_never_accepts_a_legacy_name(name):
    with pytest.raises(IdentityError):
        parse_family_descriptor(name)


def test_the_two_languages_are_disjoint():
    for name in TESSERA_NAMES + LEGACY_NAMES:
        accepted_by_tessera = True
        accepted_by_legacy = True
        try:
            parse_family_descriptor(name)
        except IdentityError:
            accepted_by_tessera = False
        try:
            legacy_tcq_parse(name)
        except IdentityError:
            accepted_by_legacy = False
        assert not (accepted_by_tessera and accepted_by_legacy)


def test_binary_artifact_is_not_parseable_as_either_name():
    with pytest.raises(Exception):
        parse_family_descriptor(MAGIC.decode("latin-1"))
    with pytest.raises(Exception):
        legacy_tcq_parse(MAGIC.decode("latin-1"))


def test_descriptor_is_never_normative():
    descriptor = parse_family_descriptor("TESSERA_E2M1_R512")
    assert descriptor.is_normative is False
    with pytest.raises(IdentityError, match="not a terminal record"):
        require_terminal_record(descriptor)


def test_bare_string_is_refused_where_a_record_is_required():
    with pytest.raises(IdentityError, match="names never carry"):
        require_terminal_record("TESSERA_E2M1_R512")


def test_assignment_record_is_accepted_and_digests_stably():
    record = AssignmentRecord(
        encoder_profile_id=hashlib.sha256(b"p").digest(),
        terminal_id=hashlib.sha256(b"t").digest(),
        branch=branch(),
        payload_digest=hashlib.sha256(b"d").digest(),
    )
    assert require_terminal_record(record) is record
    assert record.record_digest() == record.record_digest()
    assert len(record.record_digest()) == 32


def test_assignment_digest_changes_with_the_branch():
    common = dict(
        encoder_profile_id=hashlib.sha256(b"p").digest(),
        terminal_id=hashlib.sha256(b"t").digest(),
        payload_digest=hashlib.sha256(b"d").digest(),
    )
    a = AssignmentRecord(branch=branch(), **common)
    b = AssignmentRecord(
        branch=BranchIdentity(
            unit_id="u", root_q256=512, rotation=RotationState.R_IN_ONLY,
            container=ContainerClass.GRIDBOOK,
        ),
        **common,
    )
    assert a.record_digest() != b.record_digest()


def test_malformed_digests_are_rejected():
    with pytest.raises(IdentityError):
        AssignmentRecord(
            encoder_profile_id=b"short",
            terminal_id=hashlib.sha256(b"t").digest(),
            branch=branch(),
            payload_digest=hashlib.sha256(b"d").digest(),
        )


def test_name_novelty_passes_on_a_clean_registry():
    check_name_novelty(["NVFP4", "MXFP8_E4M3", "FP8_DYNAMIC", "TCQ_E2M1_R512"])


def test_name_novelty_fails_on_a_collision():
    with pytest.raises(IdentityError, match="name-novelty"):
        check_name_novelty(["NVFP4", "TESSERA_E2M1_R512"])
