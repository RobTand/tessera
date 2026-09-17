"""The lane_eligibility cells contract v31 withdrew, for the tests built on them.

Contract v31 (tessera#538) removed eight dense cells whose ``executes`` named
the window-GEMV dispatch that ``1b767a207`` retired.  Two mechanisms in this
suite -- the census join (``census.cell_launch_agreement``) and lane
reachability -- were written against the served records those cells were minted
from, and the records are real: replaying them against an invented cell would
weaken the checks the withdrawal has no quarrel with.

So the cells are quoted here, from outside the published document.  Nothing
resolves this as a contract; the shipped file is what a consumer reads, and
each module that uses these also asserts what the shipped file now says about
the same records.

The two serve-image digests are placeholders in the fixture and are resolved
here, because ``tests/test_runtime_image_pin.py`` refuses a second copy of the
pin in any acting file under ``tests/``.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "lane_eligibility_cells_withdrawn_v31.json"
GFX1201_RECEIPT = ROOT / "docs/measurements/tessera-gfx1201-bf16-k1-served-2026-09-13.md"

PIN_PLACEHOLDER = "<versions.default_serve_image>"
GFX1201_PLACEHOLDER = "<gfx1201 receipt image>"


def _gfx1201_image() -> str:
    found = sorted(set(re.findall(
        r"192\.168\.1\.107/prismaquant/vllm-rocm@sha256:[0-9a-f]{64}",
        GFX1201_RECEIPT.read_text(encoding="utf-8"))))
    assert len(found) == 1, found
    return found[0]


def withdrawn_cells(ids=None) -> list[dict]:
    """The withdrawn cells, images resolved, as v30 published them."""
    from tessera.serving.runtime_image import pinned_reference

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    images = {PIN_PLACEHOLDER: pinned_reference(), GFX1201_PLACEHOLDER: _gfx1201_image()}
    out = []
    for cell in payload["cells"]:
        if ids is not None and cell["id"] not in ids:
            continue
        cell = json.loads(json.dumps(cell))
        cell["runtime"]["image"] = images[cell["runtime"]["image"]]
        out.append(cell)
    if ids is not None:
        assert len(out) == len(ids), sorted(c["id"] for c in out)
    return out


WITHDRAWN_IDS = frozenset(
    cell["id"] for cell in json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"])
