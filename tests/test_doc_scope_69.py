"""README and schema section 7 must not sell the 1a/1b scope as current (#69).

The tree grew an encoder, a decoder, checkpoint export and a self-housed
serving plugin; the prose still denied all four in the present tense. Each
check derives the shipped fact from the code that owns it, then reads the
prose that denied it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tessera"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _shipped_capabilities() -> dict[str, bool]:
    contract = json.loads((SRC / "serving" / "runtime_contract.json").read_text())
    return {
        "encoder": (SRC / "encode.py").exists(),
        "decoder": (SRC / "decode.py").exists(),
        "export": (SRC / "export.py").exists(),
        "serving": (SRC / "serving" / "__init__.py").exists()
        and int(contract["contract_version"]) >= 7
        and "TESSERA_BF16" in (SRC / "serving" / "scheme.py").read_text(),
    }


def test_readme_scope_block_does_not_deny_shipped_capabilities():
    """README's deliberately-absent block is the historical 1a/1b scope."""
    caps = _shipped_capabilities()
    assert all(caps.values()), f"test premise moved: {caps}"
    readme = _read("README.md")
    # The stale block denied, in the present tense, exactly what caps says ships.
    assert "No encoder." not in readme
    assert "Gridbook owns the decoder" not in readme
    assert "No menu, DP, export, or serving wiring" not in readme


def test_readme_layout_names_only_files_the_tree_has():
    """Every ``*.py`` the README layout block names must exist under src/tessera."""
    readme = _read("README.md")
    block = readme.split("## Layout", 1)[1].split("```")[1]
    named = re.findall(r"(\w+\.py)", block)
    assert named, "README layout block names no files"
    missing = [n for n in named if not (SRC / n).exists()]
    assert missing == [], f"README layout names files absent from src/tessera/: {missing}"


def test_schema_section7_is_historical_scope_not_present_denial():
    """Schema section 7 must not deny the encoder/decoder/export/serving the tree has."""
    caps = _shipped_capabilities()
    assert all(caps.values()), f"test premise moved: {caps}"
    doc = _read("docs/schema/prismaquant.tessera.v1.md")
    section7 = doc.split("## 7. Deliberately absent", 1)[1]
    # Cut at the next top-level heading if one ever follows section 7.
    nxt = re.search(r"^## \S", section7, re.M)
    section7 = section7[: nxt.start()] if nxt else section7
    assert "No encoder" not in section7
    assert "Gridbook" not in section7
    assert "no menu, DP, export, or serving wiring" not in section7


def test_export_docstring_does_not_claim_no_runtime_decodes():
    """export.py's library path declares unbacked; it must not claim no runtime exists."""
    text = (SRC / "export.py").read_text(encoding="utf-8")
    assert "No serving runtime\ndecodes this container today" not in text
    assert "No serving runtime decodes this container today" not in text


def test_gridbook_container_comment_does_not_claim_sole_consumer():
    """ContainerClass.GRIDBOOK is a legacy lane name, not the only consumer."""
    text = (SRC / "manifest.py").read_text(encoding="utf-8")
    assert "the only consumer of Tessera bytes" not in text
