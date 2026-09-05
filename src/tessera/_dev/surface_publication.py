"""The one home for what a run says when it publishes its population.

``tests/conftest.py`` writes one line per surface file it publishes, and
``tools/merge_suite.py`` reads that line back out of a pool attempt's captured
stdout to decide which attempt wrote the population sitting at a path.  That
line is therefore a **contract between two modules**, and until #331 each of
them spelled the sentence itself: the producer at ``conftest.py`` and the
consumer's own ``f"tessera surface: population written to {path}"``.  Nothing
told either that the other existed, so a reword of the producer would have
silently stopped every resumed receipt from binding -- measured: one word
changed, and ``tests/test_merge_suite.py`` stayed green because its fixture
wrote the parser's own string back to it.  AGENTS.md rule 4: the module that
owns the grammar owns the message, and everyone else calls it.

**A digest, not a sentence.**  The line carries the SHA-256 of the bytes the
run just wrote, so the join is evidence rather than prose: an attempt is bound
to the file at a path only when that file still hashes to what the attempt
announced.  A later attempt that overwrote the population with one of
identical counts -- which the summary-counts leg cannot see -- is refused
here.  ``docs/ARCHITECTURE.md`` named that gap when #294 landed ("the attempt
is bound by path and counts, not by a digest of the population's bytes,
because the conftest does not yet print one"); this is the conftest printing
one.

The sentence survives only as the way a reader finds the digest on the line,
and a line without a digest is still understood: the captured stdout of every
run under ``/mnt/shared/tessera-suite-receipts`` predates this module, and a
resume of one of those must keep binding on what it does carry.  ``None`` in
the returned list is that case, stated rather than guessed at.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: What the run calls the two kinds of file it publishes.  Only the
#: controller (or a serial run, which is its own controller) writes a
#: ``population``; under ``-n N`` each worker writes a ``worker share`` to its
#: own path.  A consumer that wants the run's answer wants ``POPULATION``.
POPULATION = "population"
WORKER_SHARE = "worker share"

_MARK = "sha256:"


def digest_bytes(data: bytes) -> str:
    """The digest a publication line announces for ``data``."""

    return hashlib.sha256(data).hexdigest()


def publication_line(what: str, path, digest: str | None = None) -> str:
    """The one spelling of "this run wrote ``what`` to ``path``".

    ``digest`` is the SHA-256 of the bytes at ``path``; omit it only to build
    the shape a pre-#331 tree printed, which is what ``published_digests``
    matches historical stdout against.
    """

    line = f"tessera surface: {what} written to {Path(path)}"
    return f"{line} {_MARK}{digest}" if digest else line


def published_digests(stdout: str, path, what: str = POPULATION) -> list:
    """The digests ``stdout`` announced for ``what`` at ``path``.

    Empty where the run never said it published there -- an attempt that died
    before writing, or one that wrote somewhere else.  An entry is ``None``
    where the line names the path but carries no digest, which is every run
    from before this module: such an attempt said what it wrote and not what
    was in it, and a consumer must decide on the weaker evidence rather than
    pretend to the stronger.

    Every announcement is returned, not the first: a caller that finds two
    disagreeing digests for one path is looking at a stdout that cannot be
    read as one publication, and that is its decision to make, not this
    function's to hide by picking one.
    """

    head = publication_line(what, path)
    prefix = f"{head} {_MARK}"
    found: list = []
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if line == head:
            found.append(None)
        elif line.startswith(prefix):
            found.append(line[len(prefix):].strip() or None)
    return found
