"""Issue #109: the box-side instrument's window, and the campaign it lost.

``experiments/box_power_window.py`` is principle 15's second instrument -- the
one no in-process profiler can supply -- and ``window_gemv_latency_ab.sh``
calls it once per timed arm with that arm's own marks::

    --window=2026-09-04T09:16:12Z:2026-09-04T09:17:26Z

An ISO-8601 UTC stamp carries two colons of its own, and the first spelling of
this parser split the argument on the FIRST colon.  So the tool refused the
usage line in its own docstring, exited 1, and the driver's
``subprocess.run(..., check=False)`` dropped the refusal on the floor: the
2026-09-04 two-rep campaign produced four latency receipts, four chrome traces
and **zero** ``power-arm*.json``.  Every timed window in that run is unreadable
from the box side, and nothing in the log said so.

What is pinned here is therefore not "a parser handles a format".  It is:

* the exact argument the A/B driver constructs, from a real receipt's marks;
* that an offset pair (``-1800:0``) still reads the same way it always did, so
  the idle-window callers are unmoved;
* that an unreadable or ambiguous window is REFUSED rather than guessed at --
  a box-side window silently off by a minute would describe a different box
  state than the one the arm was timed in, which is worse than no reading.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "box_power_window", ROOT / "experiments" / "box_power_window.py")
BPW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BPW)

#: A fixed "now" so the offset forms are arithmetic rather than a clock read.
NOW = 1788000000.0

#: Verbatim from ``latency-armA-streamed-eager-rep2.json``'s ``marks_utc``:
#: the decode window that campaign timed, and the window whose box-side
#: reading it never took.
ARM_DECODE_START = "2026-09-04T09:16:12Z"
ARM_PREFILL_END = "2026-09-04T09:17:26Z"


def test_the_driver_s_own_window_is_readable():
    """The argument ``window_gemv_latency_ab.sh`` builds, character for character."""
    after, before = BPW.split_window(
        f"{ARM_DECODE_START}:{ARM_PREFILL_END}", NOW)
    assert BPW._parse_instant(ARM_DECODE_START, NOW) == after
    assert BPW._parse_instant(ARM_PREFILL_END, NOW) == before
    # 09:16:12 -> 09:17:26 is 74 s, which is the length of a real decode window
    # on these arms: the box-side reading covers the arm, not the hour round it.
    assert before - after == 74


def test_offsets_are_unmoved():
    """``-1800:0``, the idle-window callers' form, reads as it always did."""
    after, before = BPW.split_window("-1800:0", NOW)
    assert (after, before) == (NOW - 1800, NOW)


def test_a_stamp_against_an_offset_reads_both_ways_round():
    """Mixed forms are legal: an absolute start against "now"."""
    after, before = BPW.split_window(f"{ARM_DECODE_START}:0", NOW)
    assert after == BPW._parse_instant(ARM_DECODE_START, NOW)
    assert before == NOW


def test_an_unreadable_window_is_refused_not_guessed():
    with pytest.raises(SystemExit) as caught:
        BPW.split_window("yesterday:today", NOW)
    assert "readable instants" in str(caught.value)


def test_a_window_without_a_separator_is_refused():
    with pytest.raises(SystemExit) as caught:
        BPW.split_window("-1800", NOW)
    assert "AFTER:BEFORE" in str(caught.value)


def test_the_first_colon_split_would_have_failed_here():
    """The regression itself, stated as the thing that must not come back.

    ``partition(':')`` on the driver's argument yields ``2026-09-04T09`` and
    ``16:12Z:2026-09-04T09:17:26Z``.  Neither is an instant, which is exactly
    why the old code raised -- so a future edit that reintroduces a fixed split
    position fails this rather than failing a campaign.
    """
    window = f"{ARM_DECODE_START}:{ARM_PREFILL_END}"
    head, _, tail = window.partition(":")
    for half in (head, tail):
        with pytest.raises(ValueError):
            BPW._parse_instant(half, NOW)
    head, _, tail = window.rpartition(":")
    with pytest.raises(ValueError):
        BPW._parse_instant(head, NOW)
