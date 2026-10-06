"""The nightly arm summarizer judges an arm only against an eager pool it has.

``summarize-arms.py`` applies the tessera#508 membership criterion: a choice is
a member when it equals the same choice of some eager run of the same batch,
where the pool brings the eager serves other than the arm itself.  An arm with
no other eager run in its pool -- a lone eager serve (the planned smE1 on the
newer nightly, tessera#695), or one listed against only itself -- has no
criterion.  Before this fix the summarizer crashed on it
(``ValueError: min() iterable argument is empty``, reproduced on the preserved
2026-09-30 mtp receipts with ``summarize-arms.py RECEIPTS mtE1 mtE1``) and the
crash, arriving mid-loop, discarded the summary rows of every arm after it.
Now the arm is named unjudged in the table and on stdout, and carries no
membership numbers; the other arms of the invocation are still judged.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "summarize_arms_under_test",
    HERE.parent / "experiments" / "graph_attest_nightly" / "summarize-arms.py")
summarize_arms = importlib.util.module_from_spec(_spec)
_argv_at_import, sys.argv = sys.argv, sys.argv[:1]
_spec.loader.exec_module(summarize_arms)
sys.argv = _argv_at_import


def _choice(prompt, tokens, tops):
    """One equality-suite choice as equal-508.py writes it."""
    return {"prompt_token_ids": prompt, "token_ids": tokens,
            "logprobs": {"top_logprobs": tops}}


def _choices(prompt=(1, 2, 3)):
    return [
        _choice(list(prompt), [10, 11],
                [{"10": -0.1, "11": -2.0}, {"11": -0.2, "12": -3.0}]),
        _choice(list(prompt), [20, 21],
                [{"20": -0.3, "21": -1.5}, {"21": -0.4, "22": -2.5}]),
    ]


def _write_arm(recs, name, cases):
    (recs / f"{name}.eq.summary.json").write_text(json.dumps(
        {"_admission": "paused", **{case: {} for case in cases}}))
    for case, choices in cases.items():
        (recs / f"{name}.eq.{case}.json").write_text(json.dumps({"choices": choices}))


def _run(recs, pool, arms, capsys):
    argv, sys.argv = sys.argv, ["summarize-arms.py", str(recs), pool, *arms]
    try:
        summarize_arms.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    summary = recs / f"summary-{pool.split(',')[0]}.json"
    return json.loads(summary.read_text()), out


def test_a_lone_eager_serve_is_named_not_judged(tmp_path, capsys):
    recs = tmp_path
    _write_arm(recs, "e1", {"b1": _choices()})
    table, out = _run(recs, "e1", ["e1"], capsys)
    row = table["e1"]["equality"]
    assert row["judged"] is False
    assert "no other eager run" in row["why"]
    assert "members" not in row
    assert "NOT JUDGED" in out


def test_one_unjudgeable_arm_does_not_sink_the_table(tmp_path, capsys):
    recs = tmp_path
    _write_arm(recs, "e1", {"b1": _choices()})
    _write_arm(recs, "g1", {"b1": _choices()})
    table, out = _run(recs, "e1", ["e1", "g1"], capsys)
    assert table["e1"]["equality"]["judged"] is False
    judged = table["g1"]["equality"]
    assert judged["judged"] is True
    assert judged["members"] == 2 and judged["choices"] == 2
    assert judged["worst_same_prefix_delta"] == 0.0
    assert "NOT JUDGED" in out and "members 2/2" in out


def test_a_judged_row_keeps_the_membership_columns(tmp_path, capsys):
    recs = tmp_path
    _write_arm(recs, "e1", {"b1": _choices()})
    departed = _choices()[0]
    departed["token_ids"] = [99, *departed["token_ids"][1:]]
    _write_arm(recs, "g1", {"b1": [departed, _choices()[1]]})
    table, out = _run(recs, "e1", ["g1"], capsys)
    judged = table["g1"]["equality"]
    assert judged["members"] == 1 and judged["non_members"] == 1
    assert judged["non_member_choices"] == ["b1[0]"]
    assert judged["first_difference_step_min"] == 0
    assert judged["first_difference_context_min"] == 3
    assert judged["changed_a_token"] == 1
    assert judged["prefill_token_equal"] is False
    assert "members 1/2" in out
