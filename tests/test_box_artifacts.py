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
from pathlib import Path, PurePosixPath

import pytest

import box_artifacts

TESTS = Path(__file__).resolve().parent
#: The module that owns the defaults is the one place allowed to spell them.
OWNER = "box_artifacts.py"


def box_address_space() -> list:
    """The directories the documented roots live in, minimised.

    A root's PARENT is the machine's address space: ``/home/rob`` is one
    box's home, ``/mnt/shared`` is the mount both boxes carry.  The rule is
    written over that space rather than over the roster of root defaults
    because the root a roster cannot catch is exactly the one nobody
    declared -- ``/mnt/shared/models/Qwen3.8-Flash-Next`` was a literal in a
    test file while the scan looked only for the seven strings it already
    knew (AGENTS.md rule 3).  Derived from ``ROOTS``, so a new root widens
    the rule the day it is added.
    """

    parents = {str(PurePosixPath(spec.default).parent)
               for spec in box_artifacts.ROOTS.values()}
    return sorted(space for space in parents
                  if not any(other != space and space.startswith(other + "/")
                             for other in parents))


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


def test_no_test_file_names_a_path_in_the_box_address_space():
    """The pre-fix failure this test was written for::

        AssertionError: a test names a path on somebody's box:
          test_export_moe_layouts.py:762 names '/mnt/shared/models/Qwen3.8-Flash-Next'
          in the '/mnt/shared' address space

    A path literal in a test file is a claim about which machine the suite is
    running on, and no gate can read it.  Route it through
    ``box_artifacts.require`` / ``.path`` instead, which turns an absent
    artifact into a skip whose reason names the root and the variable that
    moves it.

    This replaces a scan for the ``ROOTS`` defaults themselves, which by
    construction could not see the one case that matters: a root nobody
    declared.  ``tests/**`` and not ``tests/*``, for the same reason.
    """

    spaces = box_address_space()
    assert spaces, "ROOTS declares no defaults, so this rule covers nothing"
    offenders = []
    for module in sorted(TESTS.rglob("*.py")):
        if module.name == OWNER:
            continue
        for lineno, text in _code_strings(module):
            for space in spaces:
                if space + "/" in text:
                    offenders.append(
                        f"{module.relative_to(TESTS)}:{lineno} names {text!r} in the "
                        f"{space!r} address space"
                    )
    assert not offenders, (
        "a test names a path on somebody's box:\n  " + "\n  ".join(offenders)
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


def test_an_opt_in_root_does_not_resolve_its_default(monkeypatch):
    """The kl instrument is an untracked file shared with PrismaQuant: its
    default is documented so a reader can find it, and NOT resolved, so the
    released suite never reads one machine's filesystem unasked (#153)."""
    monkeypatch.delenv("KL_TOOL_DIR", raising=False)
    assert box_artifacts.ROOTS["kl_instrument"].opt_in is True
    assert box_artifacts.root("kl_instrument") is None
    with pytest.raises(pytest.skip.Exception) as refused:
        box_artifacts.require_module("kl_instrument", "kl_tool.py")
    assert "KL_TOOL_DIR" in str(refused.value)
    monkeypatch.setenv("KL_TOOL_DIR", "/elsewhere")
    assert box_artifacts.root("kl_instrument") == Path("/elsewhere")
