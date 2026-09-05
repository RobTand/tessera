"""The evidence a test needs that no checkout carries, named in one place.

Some gates in this suite are held by bytes too large to commit: a built
checkpoint, a served census, a serve log, a BF16 source model, the PrismaQuant
tree this repository prices against.  Those live under roots this repository
does not own, and until now every test spelled its own root -- 17 absolute
literals under ``/home/rob/tessera-runs`` across 8 files at ``b83fd17``, plus
three more files gating on other absolute roots (tessera#146).

A literal is not the problem.  A literal *per test file* is: it is a per-file
decision about which machine the suite runs on, it is invisible to any gate,
and on a box without the directory the file skips and the run reads green
while covering nothing.  That is the same sentence as tessera#112, one level
down -- a pass count quoted without the population it was measured on.

So: this module owns the roots.  Each is an environment variable with a
documented default, a test asks for a path rather than spelling one, and a
missing artifact becomes a skip whose reason is **issued from here** and names
the root, the variable and the path.  That last part is what gives the skip
teeth: because one function writes the sentence, ``conftest`` can tell "this
box has no copy of the evidence" from "this test chose not to run" by reading
a declared prefix rather than by pattern-matching forty authors' prose -- and
``--strict-cuda`` refuses a run that skipped for it (tessera#152).

``tests/test_box_artifacts.py`` holds the rule that keeps this the only home:
a documented default may appear in exactly one module, which is this one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

#: Every skip reason this module issues starts with this sentence.  It is a
#: declared marker, not a classification: ``conftest`` matches the prefix it
#: knows was written here, so a new gate cannot silently fall outside the set.
ABSENT = "box artifact absent"


@dataclass(frozen=True)
class Root:
    """One tree this repository reads but does not own."""

    key: str
    env: str
    default: str
    what: str
    #: An opt-in root has a documented default that is NOT used unless the
    #: variable is set.  The kl instrument is the one such case: it is an
    #: untracked file shared with PrismaQuant, so a default that resolved
    #: itself would put one machine's filesystem back in every run of the
    #: released suite (tessera#153).
    opt_in: bool = False


ROOTS: dict[str, Root] = {
    root.key: root
    for root in (
        Root(
            "runs",
            "TESSERA_RUNS_DIR",
            "/home/rob/tessera-runs",
            "checkpoints, served censuses and serve logs this box produced",
        ),
        Root(
            "shared_runs",
            "TESSERA_SHARED_RUNS_DIR",
            "/mnt/shared/tessera-runs",
            "the run tree both boxes mount",
        ),
        Root(
            "models",
            "TESSERA_MODELS_DIR",
            "/home/rob/models",
            "BF16 source checkpoints the encoder reads",
        ),
        Root(
            "prismaquant",
            "TESSERA_PRISMAQUANT_DIR",
            "/home/rob/prismaquant",
            "the PrismaQuant checkout whose pricing this suite is pinned against",
        ),
        Root(
            "prismaquant_worktree",
            "TESSERA_PRISMAQUANT_WORKTREE",
            "/home/rob/pq-wt/tessera-continuous",
            "the PrismaQuant worktree carrying the continuous-rate branch",
        ),
        Root(
            "scratch",
            "TESSERA_SCRATCH_DIR",
            "/home/rob/tmp",
            "the scratch root child processes get as TMPDIR (/tmp is forbidden here)",
        ),
        Root(
            "kl_instrument",
            "KL_TOOL_DIR",
            "/home/rob/dq-runs",
            "kl_tool.py and kl_estimator.py, the untracked served-KL instrument",
            opt_in=True,
        ),
    )
}


def _root(key: str) -> Root:
    try:
        return ROOTS[key]
    except KeyError:
        raise KeyError(
            f"no such box-artifact root: {key!r}. Known roots: "
            + ", ".join(sorted(ROOTS))
        ) from None


def root(key: str) -> Path | None:
    """Where this box keeps ``key``, or ``None`` when nobody said.

    ``None`` is only ever returned for an opt-in root whose variable is unset:
    the default is documented but deliberately not resolved, so the released
    suite does not read one machine's filesystem unless somebody asked it to.
    """

    spec = _root(key)
    override = os.environ.get(spec.env, "")
    if override:
        return Path(override)
    return None if spec.opt_in else Path(spec.default)


def path(key: str, *parts: str) -> Path | None:
    """The artifact at ``parts`` under root ``key``, or ``None`` (see ``root``)."""

    base = root(key)
    return None if base is None else base.joinpath(*parts)


def reason(key: str, target: Path | None = None) -> str:
    """The one sentence a missing box artifact skips with.

    It names what is wanted, where this run looked, and the variable that
    moves it, because a skip reason is the only thing a reader of a population
    histogram gets.
    """

    spec = _root(key)
    where = (
        f"nothing set {spec.env} and its default is not resolved for this root"
        if target is None
        else f"{target} is not on this box"
    )
    return (
        f"{ABSENT}: {spec.what} -- {where} "
        f"(set {spec.env}; documented default {spec.default})"
    )


def present(key: str, *parts: str) -> bool:
    target = path(key, *parts)
    return target is not None and target.exists()


def require(key: str, *parts: str):
    """A ``skipif`` marker for one box artifact, with this module's reason."""

    return pytest.mark.skipif(not present(key, *parts), reason=reason(key, path(key, *parts)))


def require_module(key: str, *parts: str) -> Path:
    """The artifact, or a module-level skip carrying this module's reason."""

    target = path(key, *parts)
    if target is None or not target.exists():
        pytest.skip(reason(key, target), allow_module_level=True)
    return target


def skip_now(key: str, *parts: str) -> Path:
    """The artifact, or an in-test skip carrying this module's reason."""

    target = path(key, *parts)
    if target is None or not target.exists():
        pytest.skip(reason(key, target))
    return target


def first_present(key: str, *keys: str) -> Path | None:
    """The first of several roots that is actually on this box."""

    for candidate in (key, *keys):
        base = root(candidate)
        if base is not None and base.exists():
            return base
    return None


def scratch_tmpdir() -> str:
    """``TMPDIR`` for a child process this suite starts.

    Not a gate -- a child that inherits ``/tmp`` writes where this fleet
    forbids writing, so the value is required rather than skipped on.
    """

    base = root("scratch")
    assert base is not None  # not an opt-in root
    return str(base)
