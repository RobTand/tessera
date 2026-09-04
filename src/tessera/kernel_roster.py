"""The window GEMV's supported set, read from the kernel that instantiates it.

The lane's eligibility is two numbers -- which column rates it can read and
which window width it was built for -- and they are consumed in four places:
``kernel_window_gemv.repack_window_body`` refuses a unit at load,
``serving.bf16_route.gemv_eligible_for_unit`` refuses one at prepare,
``serving.ext.WINDOW_GEMV_LANE`` publishes them as the lane predicate a
PRODUCER refuses a plan against (#104), and ``runtime_contract.json`` ships
that block to a consumer.  Each of those was a literal, tied to the others by
tests and to ``csrc/window_gemv.cu`` -- the file that decides what actually
exists -- by nothing.

**Why the kernel is the source and not Python.**  The alternative was to make
the Python constants authoritative and generate the ``.cu``'s dispatch from
them at build time (``-D`` flags).  Both remove the drift; they differ in
where the disagreement surfaces.  A ``-D``-parameterised build fails at
``nvcc`` time, on a GPU host, at first use -- and the whole point of
publishing the predicate is that a producer with no torch and no device
refuses a plan BEFORE it encodes a checkpoint that could never take the lane.
Parsing the kernel puts the single source where the instantiation is, keeps
the ``.cu`` compilable on its own, and makes the answer available to the
torch-free reader.  So: the kernel declares, Python reads.

**What the kernel declares.**  Two preprocessor definitions, near the top of
``window_gemv.cu``::

    #define TESSERA_GEMV_RATES(X) X(1) X(2) X(4)
    #define TESSERA_GEMV_WINDOW_BITS 14

They are not a comment for this parser to read: every rate dispatch in the
file is *generated* by expanding ``TESSERA_GEMV_RATES``, and every window
width in a template argument, a ``TORCH_CHECK`` and its message is
``TESSERA_GEMV_WINDOW_BITS``.  So the declaration is the kernel, and
``tests/test_kernel_roster.py`` pins that no second spelling has crept back
in.  Adding a rate to the list is a real change with a real consequence:
``consume_chunk``'s ``static_assert(chunk_width_supported(RPL * R))`` refuses
a rate whose lane is not a whole number of words, which is why R = 3 is
absent rather than merely unlisted.

**Fail closed.**  A source this module cannot read raises
:class:`~tessera.errors.KernelSourceError` naming the file.  It never falls
back to a default set: a wrong roster serves wrong weights, and an empty one
is indistinguishable from "no artifact can reach this lane".

Torch-free, and it must stay that way -- ``serving.ext`` and
``serving.contract`` are read by producers with no torch.
"""
from __future__ import annotations

import dataclasses
import os
import re

from .errors import KernelSourceError

__all__ = [
    "WINDOW_GEMV_SOURCE",
    "KernelRoster",
    "parse_window_gemv_roster",
    "read_window_gemv_roster",
    "SUPPORTED_RATES",
    "WINDOW_BITS_SUPPORTED",
]

#: The kernel source this package builds and reads its roster from.  ONE
#: constant: ``kernel_window_gemv._ext`` hands this exact path to
#: ``cpp_extension.load``, so the file parsed is the file compiled.
WINDOW_GEMV_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "csrc", "window_gemv.cu")

_RATES_DECL = re.compile(r"^[ \t]*#define[ \t]+TESSERA_GEMV_RATES\(X\)(.*)$", re.M)
_RATE_ITEM = re.compile(r"X\(\s*(\d+)\s*\)")
_WINDOW_DECL = re.compile(r"^[ \t]*#define[ \t]+TESSERA_GEMV_WINDOW_BITS[ \t]+(\d+)[ \t]*$", re.M)


@dataclasses.dataclass(frozen=True)
class KernelRoster:
    """What a kernel source says it instantiates."""

    rates: "tuple[int, ...]"
    window_bits: "tuple[int, ...]"


def _one(matches: list, what: str, where: str) -> object:
    if not matches:
        raise KernelSourceError(
            f"{where} declares no {what}: the window GEMV's supported set is read from the "
            "kernel that instantiates it, and a source without the declaration is a "
            "packaging or edit defect, not an empty lane")
    if len(matches) > 1:
        raise KernelSourceError(
            f"{where} declares {what} {len(matches)} times; the roster has one home, and a "
            "source with two declarations does not say which one the dispatch expanded")
    return matches[0]


def parse_window_gemv_roster(text: str, *, where: str) -> KernelRoster:
    """The rates and window ``text`` declares, or :class:`KernelSourceError`.

    ``where`` names the source in every refusal -- a parse failure is read by
    someone holding a file, not a string.
    """
    body = _one(_RATES_DECL.findall(text), "TESSERA_GEMV_RATES(X)", where)
    rates = tuple(int(v) for v in _RATE_ITEM.findall(body))
    leftover = _RATE_ITEM.sub("", body).strip()
    if leftover:
        raise KernelSourceError(
            f"{where}: TESSERA_GEMV_RATES(X) holds {leftover!r} beside its X(<rate>) items; "
            "the declaration is a plain list because a parser and the preprocessor must "
            "agree on it without either evaluating the other")
    if not rates:
        raise KernelSourceError(
            f"{where}: TESSERA_GEMV_RATES(X) lists no rate. A lane that reads nothing is "
            "refused here rather than published as a predicate no checkpoint can satisfy")
    if any(r <= 0 for r in rates):
        raise KernelSourceError(
            f"{where}: TESSERA_GEMV_RATES(X) lists {rates}, and a rate is bits per code -- "
            "a positive integer")
    if sorted(set(rates)) != list(rates):
        raise KernelSourceError(
            f"{where}: TESSERA_GEMV_RATES(X) lists {rates}; it must be ascending and without "
            "repeats, because it is published verbatim as the contract's "
            "lane.requires.column_rates, which is validated for exactly that")
    bits = int(_one(_WINDOW_DECL.findall(text), "TESSERA_GEMV_WINDOW_BITS", where))
    if bits <= 0:
        raise KernelSourceError(
            f"{where}: TESSERA_GEMV_WINDOW_BITS is {bits}; a window is a positive width")
    return KernelRoster(rates=rates, window_bits=(bits,))


def read_window_gemv_roster(path: str = WINDOW_GEMV_SOURCE) -> KernelRoster:
    """:func:`parse_window_gemv_roster` over a packaged source file."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
    except OSError as exc:
        raise KernelSourceError(
            f"tessera is installed without {path}: the window GEMV's supported set is read "
            "from its own kernel source, so the package cannot say what the lane reads. "
            "This is a packaging defect -- reinstall tessera or install from a checkout."
        ) from exc
    return parse_window_gemv_roster(text, where=path)


_ROSTER = read_window_gemv_roster()

#: Column rates the kernel instantiates a lane for.  A unit is readable by
#: this lane iff EVERY column rate is in here (``repack_window_body``).
SUPPORTED_RATES = _ROSTER.rates

#: Window widths this build instantiates.
WINDOW_BITS_SUPPORTED = _ROSTER.window_bits
