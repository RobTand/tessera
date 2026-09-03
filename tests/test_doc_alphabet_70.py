"""Schema section 3 must price the ALPHABET element at the normative 8 (#70).

Section 3 is the normative element table, but it priced ALPHABET/DESCENDANT
at "8, or 16 on a grid wider than a byte", while section 1e and the code keep
``NORMATIVE_ELEMENT_BITS[ALPHABET]`` at 8: a two-byte grid code is two
elements, and a producer following section 3 gets ``PlaneLayoutError``.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_schema_section3_alphabet_row_matches_normative_width():
    """Section 3 is the normative element table: ALPHABET is 8 bits per element."""
    from tessera.planes import NORMATIVE_ELEMENT_BITS, PlaneKind

    assert NORMATIVE_ELEMENT_BITS[PlaneKind.ALPHABET] == 8
    assert NORMATIVE_ELEMENT_BITS[PlaneKind.DESCENDANT] == 8
    doc = (ROOT / "docs/schema/prismaquant.tessera.v1.md").read_text(encoding="utf-8")
    section3 = doc.split("## 3. Plane element units", 1)[1].split("## 3b.", 1)[0]
    row = next(
        line for line in section3.splitlines()
        if "ALPHABET" in line and "DESCENDANT" in line
    )
    assert "or 16" not in row, f"row prices a 16-bit element the code refuses: {row!r}"
    assert re.search(r"\b8\b", row), f"row states no 8-bit width: {row!r}"
