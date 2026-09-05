"""The degeneration rule behind a routed-MoE cell's ``evidence.smoke.status``.

``experiments/moe_greedy_smoke.py`` decides ``repetitive`` versus ``recorded``
for a greedy completion, and the contract copies that word.  A rule nobody
tests is a rule that can drift from the status it feeds, so the cases below
pin it: the campaign's own observation must read ``repetitive`` (the positive
control), a single coincidental repeat is not a cycle, a cycle the completion
ends in is degeneration however late it started, and a cycle the completion
left again is not.
"""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "moe_greedy_smoke.py"
_SPEC = importlib.util.spec_from_file_location("moe_greedy_smoke", _PATH)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)

# The v17 smoke: ' France is' x 8 as the two tokens it decodes to, repeated.
FRANCE_IS = [3450, 374] * 8


def test_the_campaign_observation_is_the_positive_control():
    """``' France is France is ...'`` for 16 tokens is a whole-completion cycle."""
    verdict = smoke.classify(FRANCE_IS)
    assert verdict["status"] == "repetitive"
    assert (verdict["period"], verdict["periodic_suffix"], verdict["tokens"]) == (2, 16, 16)
    assert verdict["whole_completion_periodic"] is True
    assert verdict["coverage"] == 1.0


def test_a_completion_with_no_cycle_is_recorded():
    verdict = smoke.classify(list(range(40)))
    assert verdict["status"] == "recorded"
    assert (verdict["period"], verdict["periodic_suffix"]) == (0, 0)


def test_one_repeat_is_a_coincidence_not_a_cycle():
    """``s >= 2p``: a period must be observed twice in full before it counts."""
    tokens = list(range(10)) + [7, 8]  # "... 7 8 9 7 8": under two full periods of 3
    assert smoke.periodic_tail(tokens) == (0, 0)
    assert smoke.classify(tokens)["status"] == "recorded"
    # "... 7 8 9 7 8 9" IS the period observed twice, and the completion ends
    # in it: that is the cycle greedy decoding will not leave.
    twice = list(range(10)) + [7, 8, 9]
    assert smoke.periodic_tail(twice) == (3, 6)
    assert smoke.classify(twice)["status"] == "repetitive"


def test_the_shortest_period_names_the_cycle():
    """A p=2 cycle is also p=4-periodic; the report names p=2."""
    assert smoke.periodic_tail([1, 2] * 6) == (2, 12)


def test_a_late_cycle_is_still_a_cycle():
    """The first run's student P3: 33 novel tokens, then ``' trans'`` to the
    64-token cap.  A majority rule read that ``recorded``; where ``max_tokens``
    cut the cycle is not evidence about the model, so the share does not
    matter.  Below, at and above a majority all read ``repetitive``."""
    trans = [33]  # one token id, repeated
    late = list(range(100, 133)) + trans * 31            # 33 novel + 31 repeated = 64
    verdict = smoke.classify(late)
    assert verdict["status"] == "repetitive"
    assert (verdict["period"], verdict["periodic_suffix"], verdict["tokens"]) == (1, 31, 64)
    head = list(range(100, 110))                          # 10 novel tokens
    assert smoke.classify(head + [1, 2] * 5)["status"] == "repetitive"   # 10 vs 10
    assert smoke.classify(head + [1, 2] * 6)["status"] == "repetitive"   # 12 vs 10
    assert smoke.classify(head + [1, 2] * 1)["status"] == "recorded"     # one period: not observed twice


def test_a_cycle_the_completion_leaves_again_does_not_count():
    """The rule is about the TAIL: greedy decoding never leaves a cycle, so a
    repeat in the middle that the completion moved past is not degeneration."""
    tokens = [1, 2] * 6 + list(range(50, 60))
    assert smoke.classify(tokens)["status"] == "recorded"


@pytest.mark.parametrize("tokens", [[], [5], [5, 5]])
def test_short_completions(tokens):
    verdict = smoke.classify(tokens)
    # [] is no completion at all and reads ``not_recorded`` (below).  [5] is a
    # one-token completion with no cycle.  [5, 5] is p=1 observed twice: a cycle
    # covering the whole (2-token) completion.
    expected = {(): "not_recorded", (5,): "recorded", (5, 5): "repetitive"}[tuple(tokens)]
    assert verdict["status"] == expected


def test_an_empty_completion_is_not_a_completion():
    """``recorded`` is a verdict ABOUT a completion, so nothing is not one (#327).

    The rule detects cycles, and an empty token list trivially has none -- which
    made ``classify([])`` return ``recorded``, the same word a coherent 372-token
    answer gets.  That is a hole in a machine condition and not a hypothetical:
    the aggregation a cell's ``evidence.smoke.status`` is derived from asks
    whether some prompt read ``recorded`` on BOTH arms, so two arms that each
    returned ``""`` for one prompt would together manufacture a positive record
    out of two non-answers.  ``not_recorded`` is the vocabulary's own word for
    "no completion came back" and is what the empty list means.
    """
    verdict = smoke.classify([])
    assert verdict["status"] == "not_recorded"
    assert (verdict["tokens"], verdict["period"], verdict["periodic_suffix"]) == (0, 0, 0)
    assert verdict["coverage"] == 0.0
    assert verdict["whole_completion_periodic"] is False


def test_two_empty_arms_cannot_manufacture_a_positive_record(tmp_path):
    """The hole above, closed where the pair is joined: ``both_recorded`` and
    ``positive_record`` are what an aggregation reads, and neither may count a
    (prompt, form) pair on which neither arm said anything."""
    import json

    def receipt(arm, ids):
        return {"schema": "tessera.moe-greedy-smoke/1", "arm": arm, "rule": smoke.RULE,
                "tokenizer": {"tokenizer_json_sha256": "t"}, "prompts_sha256": "p",
                "results": [{"id": "P9", "form": "campaign", "interface": "raw_completion",
                             "request": {"max_tokens": 64}, "completion": "",
                             "finish_reason": "stop", "token_ids": ids,
                             "status": "recorded", "tokens": 0, "period": 0,
                             "periodic_suffix": 0, "coverage": 0.0}]}

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(receipt("bf16", [])))
    b.write_text(json.dumps(receipt("tessera", [])))
    out = tmp_path / "pair.json"
    smoke.main(["compare", str(a), str(b), "--out", str(out)])
    pair = json.loads(out.read_text())
    (row,) = pair["rows"]
    assert row["bf16"]["status"] == row["tessera"]["status"] == "not_recorded"
    assert row["both_recorded"] is False
    assert pair["positive_record"] == []


def test_compare_scores_the_stored_token_ids_under_the_current_rule(tmp_path):
    """A pair joined after the rule changed must read under the rule the
    module states, from the token ids the receipts keep, and must say what
    rule the arms were run under."""
    import json

    def receipt(arm, ids):
        return {"schema": "tessera.moe-greedy-smoke/1", "arm": arm, "rule": "an older rule",
                "tokenizer": {"tokenizer_json_sha256": "t"}, "prompts_sha256": "p",
                "results": [{"id": "P3", "form": "campaign", "request": {"max_tokens": 64},
                             "completion": arm, "finish_reason": "length", "token_ids": ids,
                             "status": "recorded", "tokens": len(ids), "period": 0,
                             "periodic_suffix": 0, "coverage": 0.0}]}

    late = list(range(100, 133)) + [33] * 31
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(receipt("bf16", late)))
    b.write_text(json.dumps(receipt("tessera", list(range(64)))))
    out = tmp_path / "pair.json"
    smoke.main(["compare", str(a), str(b), "--out", str(out)])
    pair = json.loads(out.read_text())
    assert pair["rule"] == smoke.RULE
    assert pair["rule_at_run"] == ["an older rule"]
    (row,) = pair["rows"]
    assert row["bf16"]["status"] == "repetitive"       # re-scored, not copied
    assert (row["bf16"]["period"], row["bf16"]["periodic_suffix"]) == (1, 31)
    assert row["tessera"]["status"] == "recorded"
    assert pair["positive_record"] == []
