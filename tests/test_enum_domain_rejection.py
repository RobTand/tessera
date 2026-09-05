"""An enum ordinal no member names is a Tessera rejection, not a bare ValueError.

``errors.py`` states that every rejection path in the package raises a
``TesseraError`` and that the wire contract is fail-closed.  Ten enum-decoding
sites across ``manifest.py`` and ``planes.py`` broke that: they handed the raw
ordinal straight to the ``IntEnum`` constructor, so one corrupt byte reached a
loader that was catching ``TesseraError`` around ``container.parse`` as an
unnamed exception from a different taxonomy.

The rule lives on the reader (``canonical.Reader.enum``), which is the one
module every decoder already depends on and the one that already makes this
argument for ``bool`` (a flag byte of 2) and ``blob`` (an impossible length):
an ordinal outside the declared domain is a byte string no conforming encoder
produces, so it is refused by name where the bytes are read.
"""

import ast
import pathlib

import pytest

from conftest import make_artifact
from tessera.canonical import Reader, Writer
from tessera.container import HEADER_BYTES, parse
from tessera.errors import CanonicalEncodingError, TesseraError
from tessera.manifest import SCHEMA_ID, ArrangementMode, RotationState


def _rotation_offset(manifest) -> int:
    """Where the branch's rotation ordinal sits in the serialized artifact.

    Replayed through the writer rather than hand-counted: the prefix is
    ``schema id | profile digest | unit id | root``, and if that order ever
    moves this walks with it instead of silently patching another field.
    """
    prefix = (
        Writer()
        .text(SCHEMA_ID)
        .digest32(manifest.encoder_profile_id)
        .text(manifest.branch.unit_id)
        .uint(manifest.branch.root_q256)
        .bytes
    )
    return HEADER_BYTES + len(prefix)


def test_a_corrupt_enum_byte_is_refused_inside_the_taxonomy():
    """One flipped byte on the rotation ordinal.  Pre-fix this escaped
    ``container.parse`` as ``ValueError: 127 is not a valid RotationState``,
    which no ``except TesseraError`` loader catches."""
    manifest, _region, data = make_artifact()
    offset = _rotation_offset(manifest)
    assert data[offset] == int(RotationState.NONE)
    # 0x7f, not 0xff: the high bit is LEB128's continuation flag, and 0xff
    # would be rejected as a truncated varint before the enum is reached.
    patched = data[:offset] + b"\x7f" + data[offset + 1 :]
    with pytest.raises(TesseraError, match="RotationState"):
        parse(patched)


def test_the_rejection_names_the_enum_and_its_domain():
    reader = Reader(Writer().uint(9).bytes)
    with pytest.raises(CanonicalEncodingError) as excinfo:
        reader.enum(ArrangementMode)
    message = str(excinfo.value)
    assert "ArrangementMode" in message
    assert "9" in message
    assert str(sorted(int(m) for m in ArrangementMode)) in message


def test_a_legal_ordinal_decodes_to_its_member():
    for member in RotationState:
        reader = Reader(Writer().uint(int(member)).bytes)
        assert reader.enum(RotationState) is member


def _bare_enum_decodes() -> list[str]:
    """Every ``SomeEnum(reader.uint())`` left in the package.

    The rule, not the roster (AGENTS.md 3): the enum names are derived from
    the ``IntEnum`` subclasses the package declares, so a new enum decoded the
    old way is caught without anyone editing a list here.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tessera"
    sources = {path: ast.parse(path.read_text()) for path in sorted(root.rglob("*.py"))}
    enums = {
        node.name
        for tree in sources.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id in ("IntEnum", "Enum", "IntFlag")
            for base in node.bases
        )
    }
    assert enums, "no enum declarations found; the walk is looking in the wrong place"
    offenders = []
    for path, tree in sources.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in enums or len(node.args) != 1:
                continue
            argument = node.args[0]
            if (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Attribute)
                and argument.func.attr in ("uint", "sint")
                and isinstance(argument.func.value, ast.Name)
                and argument.func.value.id == "reader"
            ):
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno} "
                    f"{node.func.id}(reader.{argument.func.attr}())"
                )
    return offenders


def test_no_decoder_builds_an_enum_from_a_raw_ordinal():
    """Pre-fix this listed ten sites: BranchIdentity's rotation/container,
    Manifest's arrangement/body, and all six of PlaneDescriptor's."""
    assert _bare_enum_decodes() == []
