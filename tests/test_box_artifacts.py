"""One home for the trees this repository reads but does not own (tessera#146).

A gate whose evidence is a directory on one machine reads green everywhere
else while covering nothing.  The fix is not "stop using the evidence" -- a
built checkpoint is exactly the right thing to hold a bit-exactness gate to --
it is that the *root* stops being a per-file decision.

So the rule below is about placement, not about absence: a documented root's
default path may be written in exactly one module, and that module is
``tests/box_artifacts.py``.  Everything else asks it for a path and gets a
skip reason it did not have to write.  Derived from ``ROOTS`` rather than from
a list of filenames, so a new root is covered the day it is added and a
retired one stops being pinned the day it goes (AGENTS.md rule 3).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import box_artifacts

TESTS = Path(__file__).resolve().parent
#: The module that owns the defaults is the one place allowed to spell them.
OWNER = "box_artifacts.py"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """The ``ast.Constant`` nodes that are docstrings, by identity.

    Prose is evidence, not a gate: a docstring that records which serve log a
    number came from is a receipt and must stay readable.  Comments never
    reach the AST at all, so they need no exclusion.
    """

    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        first = node.body[0] if node.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            ids.add(id(first.value))
    return ids


def _code_strings(path: Path):
    """Every string literal this module uses as a value, with its line."""

    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.lineno, node.value


def test_a_root_default_is_written_in_exactly_one_module():
    """The pre-fix failure this test was written for::

        AssertionError: a box-artifact root is spelled outside box_artifacts.py

    A path literal in a test file is a claim about which machine the suite is
    running on, and no gate can read it.  Route it through
    ``box_artifacts.require`` / ``.path`` instead, which turns an absent
    artifact into a skip whose reason names the root and the variable that
    moves it.
    """

    defaults = {spec.default: spec for spec in box_artifacts.ROOTS.values()}
    offenders = []
    for module in sorted(TESTS.glob("*.py")):
        if module.name == OWNER:
            continue
        for lineno, text in _code_strings(module):
            for default, spec in defaults.items():
                if default in text:
                    offenders.append(
                        f"{module.name}:{lineno} spells the {spec.key!r} root "
                        f"({spec.env}, default {spec.default}) inside {text!r}"
                    )
    assert not offenders, (
        "a box-artifact root is spelled outside box_artifacts.py:\n  "
        + "\n  ".join(offenders)
    )


def test_every_root_says_what_it_is_and_how_to_move_it():
    """A root with no sentence produces a skip reason nobody can act on."""

    for key, spec in box_artifacts.ROOTS.items():
        assert spec.key == key
        assert spec.env and spec.env == spec.env.strip()
        assert spec.default.startswith("/"), spec
        assert spec.what and not spec.what.endswith("."), spec


def test_the_skip_reason_names_the_artifact_the_variable_and_the_default():
    """``conftest`` reads the prefix; a human reads the rest of the sentence."""

    text = box_artifacts.reason("runs", Path("/nowhere/at/all"))
    assert text.startswith(box_artifacts.ABSENT + ":")
    assert "/nowhere/at/all" in text
    assert "TESSERA_RUNS_DIR" in text
    assert box_artifacts.ROOTS["runs"].default in text


def test_a_non_opt_in_root_resolves_its_documented_default(monkeypatch):
    monkeypatch.delenv("TESSERA_RUNS_DIR", raising=False)
    assert box_artifacts.root("runs") == Path(box_artifacts.ROOTS["runs"].default)
    monkeypatch.setenv("TESSERA_RUNS_DIR", "/elsewhere")
    assert box_artifacts.path("runs", "a", "b") == Path("/elsewhere/a/b")


def test_an_unknown_root_is_refused_by_name():
    with pytest.raises(KeyError, match="no such box-artifact root"):
        box_artifacts.root("the-root-that-never-was")
