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


def test_recorded_aligned_power_response_cannot_escape_arm(monkeypatch):
    import json
    fixture=json.loads((ROOT/'tests/fixtures/netdata-power-aligned-arm-819.json').read_text())
    after,before=fixture['requested_window_unix']
    monkeypatch.setattr(BPW,'SERIES',[("nvidia_smi.gpu_power_draw",("power_draw",))])
    monkeypatch.setattr(BPW,'_fetch',lambda *args:{"url":"recorded://issue819","doc":fixture['document']})
    series=BPW.collect('recorded',after,before,4)['nvidia_smi.gpu_power_draw']
    assert all(after < stamp <= before for stamp,_ in series['samples']), "aligned power sample escaped requested steady arm"


def grouped_document(width, stamps):
    return {"view":{"update_every":width,"after":90,"before":130},
            "result":{"labels":["time","power_draw"],"data":[[stamp,[value,0,0]] for stamp,value in stamps]}}


def test_straddling_groups_are_not_interpolated_into_window():
    raw=grouped_document(10,[(105,100),(110,90),(120,80),(125,1)])
    bounded,coverage=BPW.bounded_groups(raw,100,120)
    assert [r[0] for r in bounded['result']['data']]==[110,120]
    assert len(raw['result']['data'])==4
    assert coverage['accepted_group_intervals_unix']==[[100,110],[110,120]]
    assert coverage['accepted_coverage_s']==20
    assert coverage['unobserved_requested_s']==0
    assert len(coverage['rejected_groups'])==2
    assert BPW._stats(bounded,('power_draw',))['power_draw']['mean']==85


def test_no_whole_group_is_missing_measurement_not_zero():
    raw=grouped_document(10,[(105,1),(115,1)])
    bounded,coverage=BPW.bounded_groups(raw,104,106)
    stats=BPW._stats(bounded,('power_draw',))['power_draw']
    assert stats['points']==0 and 'mean' not in stats
    assert coverage['accepted_coverage_window_unix'] is None
    assert coverage['unobserved_requested_s']==2


@pytest.mark.parametrize('width',[None,0,-1,float('nan'),True])
def test_unknown_group_duration_is_refused(width):
    with pytest.raises(ValueError,match='group duration'):
        BPW.bounded_groups(grouped_document(width,[(110,90)]),100,120)


def test_duplicate_groups_are_refused():
    with pytest.raises(ValueError,match='repeats'):
        BPW.bounded_groups(grouped_document(10,[(110,90),(110,90)]),100,120)


def test_recorded_response_keeps_raw_request_and_actual_coverage(monkeypatch):
    import json
    fixture=json.loads((ROOT/'tests/fixtures/netdata-power-aligned-arm-819.json').read_text())
    after,before=fixture['requested_window_unix'];raw=fixture['document']
    monkeypatch.setattr(BPW,'SERIES',[("nvidia_smi.gpu_power_draw",("power_draw",))])
    monkeypatch.setattr(BPW,'_fetch',lambda *args:{"url":"recorded://issue819","doc":raw})
    result=BPW.collect('recorded',after,before,4)['nvidia_smi.gpu_power_draw']
    assert result['raw_response'] is raw
    assert result['coverage']['requested_window_unix']==[after,before]
    assert result['coverage']['returned_window_unix']==[raw['view']['after'],raw['view']['before']]
    assert result['coverage']['accepted_groups']==3
    assert result['coverage']['accepted_coverage_window_unix']==[1790920180,1790920201]
    assert result['coverage']['unobserved_requested_s']==6


def test_fetch_requests_unaligned_but_still_bounds_server_response(monkeypatch):
    import json,urllib.parse
    raw=grouped_document(10,[(105,1),(110,90),(120,80),(125,1)])
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(raw).encode()
    urls=[]
    monkeypatch.setattr(BPW.urllib.request,'urlopen',lambda url,**kwargs:urls.append(url) or Response())
    fetched=BPW._fetch('recorded','nvidia_smi.gpu_power_draw',('power_draw',),100,120,4)
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query)['options']==['unaligned']
    assert [r[0] for r in fetched['doc']['result']['data']]==[110,120]
    assert len(fetched['raw_doc']['result']['data'])==4


def test_refused_fetch_preserves_unreadable_raw_window(monkeypatch):
    import json
    raw=grouped_document(None,[(110,90)])
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(raw).encode()
    monkeypatch.setattr(BPW.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(BPW,'SERIES',[("nvidia_smi.gpu_power_draw",("power_draw",))])
    result=BPW.collect('recorded',100,120,4)['nvidia_smi.gpu_power_draw']
    assert 'error' in result and 'stats' not in result
    assert result['raw_response']==raw
    assert result['requested_window_unix']==[100,120]
