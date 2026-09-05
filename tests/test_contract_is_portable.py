"""The packaged contract may not name a path only this fleet can open (#181).

``runtime_contract.json`` ships inside the wheel.  Its changelog prose carried
three paths to one person's boxes -- ``/mnt/shared/tessera-runs/...``,
``/home/rob/tessera-runs/...`` -- which is a pointer no reader who installed
the distribution can follow, and which goes public with the tag.

THE RULE IS DERIVED, NOT A ROSTER of the three strings that were removed.  A
distribution does not know its own install prefix at authoring time: a wheel
unpacks under whatever ``site-packages`` the environment has, an editable
install under the checkout, a container under neither.  So an absolute
filesystem path written into a distributed data file *cannot* name anything
inside the distribution -- it always names the author's box.  The evidence
grammar already reached the same conclusion for the field a gate reads
(``evidence.kl[].receipt`` must be a repository path under
``docs/measurements/``, ``test_cell_evidence.py``); this file holds the prose
to it, because a changelog entry is what a person reads when a gate has
nothing to say.

Two claims, because "no absolute path" alone is satisfiable by deleting the
evidence pointer, which would be worse than an unfollowable one:

* nothing in the packaged document is an absolute path;
* every repository path it does name resolves in this checkout.

A URL is not refused: ``https://host/path`` is exactly what a reader outside
these boxes *can* open, and the lookbehind lets it through on purpose.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tessera.serving.contract import contract_path, load_serving_contract

ROOT = Path(__file__).resolve().parents[1]

#: A POSIX absolute path: a ``/`` that opens a token rather than separating
#: one.  The lookbehind is what keeps a URL (``https://host/x`` -- the ``/``s
#: follow ``:`` and ``/``) and a bare separator (``gate/up``, ``112/112``)
#: out of the match, so the rule refuses locations and not punctuation.
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.:/+~-])/[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]*)+")

#: The top-level directories of the repository, read from the tree rather than
#: listed here: a roster would go stale the day one is added or renamed.
_REPO_DIRS = sorted(p.name for p in ROOT.iterdir()
                    if p.is_dir() and not p.name.startswith("."))
_REPO_PATH = re.compile(
    r"(?<![A-Za-z0-9_.:/+~-])(?:%s)/[A-Za-z0-9_./+-]+" % "|".join(_REPO_DIRS))


def _strings(value, where="runtime_contract"):
    """Every string in the document, with the field that carries it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{where}[{i}]")
    elif isinstance(value, str):
        yield where, value


def _hits(document, pattern):
    return [(where, match.group(0))
            for where, text in _strings(document)
            for match in pattern.finditer(text)]


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def test_the_packaged_contract_names_no_absolute_path(contract):
    """No field, prose included, may point at a filesystem outside the wheel."""
    found = _hits(contract, _ABSOLUTE_PATH)
    assert found == [], (
        "the packaged contract names absolute path(s) no installed reader can open; "
        "point at a tracked receipt, the experiment that produced it, or say in words "
        f"that the artifact is a build-box run log: {found}")


def test_the_rule_catches_a_path_a_reader_cannot_open():
    """The green test above is not vacuous: it detects what it forbids.

    On a synthetic document, not the real one, so the control says what the
    rule does rather than what today's contract happens to contain.  One
    planted path per shape that was actually removed -- a shared mount, a home
    directory -- in prose rather than in a receipt field, because prose is
    where the validator does not look.  The mount and the user are invented:
    ``test_box_artifacts.py`` forbids a test file from naming a real box
    address space, and that rule and this one are the same rule read at two
    altitudes.
    """
    for planted in ("/example-mount/tessera-runs/some-checkpoint",
                    "/home/someone/tessera-runs/census.json"):
        doctored = {"changelog": [{"change": f"measured on {planted}, streamed, eager"}]}
        assert _hits(doctored, _ABSOLUTE_PATH) == [
            ("runtime_contract.changelog[0].change", planted)]


def test_a_url_and_a_separator_are_not_absolute_paths():
    """The rule refuses locations, not every ``/`` in the document."""
    doctored = {"changelog": [{"change": (
        "see https://github.com/RobTand/tessera/issues/181; gate/up; 112/112; "
        "docs/measurements/tessera-gemv-lane-reachable-2026-09-03.md")}]}
    assert _hits(doctored, _ABSOLUTE_PATH) == []


def test_every_repository_path_the_contract_names_resolves(contract):
    """A relative pointer is only better than an absolute one if it exists."""
    missing = [(where, token) for where, token in _hits(contract, _REPO_PATH)
               if not (ROOT / token.split("::")[0].rstrip(".,;)")).exists()]
    assert missing == [], f"the contract names repository path(s) that are not here: {missing}"


def test_the_raw_file_carries_nothing_the_parsed_document_hides():
    """The same rule over the bytes that ship, so a key or a comment-shaped
    string cannot smuggle a host path past the walker."""
    raw = contract_path().read_text(encoding="utf-8")
    # ``json.loads`` then re-dump would normalise escapes; scan the text the
    # wheel actually carries, minus the JSON structure the walker covers.
    found = sorted({m.group(0) for m in _ABSOLUTE_PATH.finditer(raw)})
    assert found == [], f"the shipped bytes name absolute path(s): {found}"


def test_the_document_is_still_the_json_it_claims_to_be():
    """Guard for the edit that removes a path: prose is inside a JSON string."""
    raw = contract_path().read_text(encoding="utf-8")
    assert json.loads(raw)["schema"] == "tessera.runtime-contract.v1"
