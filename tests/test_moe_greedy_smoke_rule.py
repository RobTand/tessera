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


# --- the launch boundary the in-process tests cannot see (#341) --------------
#
# Everything above imports the instrument into a process that ALREADY holds a
# matching ``tessera.serving.contract``, because the conftest put this
# checkout's ``src`` on ``sys.path`` before pytest collected anything.  The
# wrapper does not: it runs ``$PY $TS/experiments/moe_greedy_smoke.py`` in a
# fresh interpreter, and until #341 nothing bound that interpreter to the
# checkout ``$TS`` selected, so the aggregation import resolved against
# whatever Tessera the host happened to carry.  These tests cross that
# boundary in a subprocess, which is the only place the defect is visible.

CHECKOUT = Path(__file__).resolve().parents[1]
WRAPPER = CHECKOUT / "experiments" / "moe_greedy_smoke_pair.sh"


def _ambient_package(tmp_path, *, body):
    """A stand-in Tessera on the host, importable and NOT this checkout's.

    ``body`` is the whole of ``tessera/serving/contract.py``.  The pre-v22
    package is spelled by what it LACKS -- it has the v7 control vocabulary the
    argument parser reads and no ``derive_smoke_status`` -- which is exactly the
    shape of a host that installed Tessera before contract v22.
    """
    root = tmp_path / "ambient"
    (root / "tessera" / "serving").mkdir(parents=True)
    (root / "tessera" / "__init__.py").write_text("")
    (root / "tessera" / "serving" / "__init__.py").write_text("")
    (root / "tessera" / "serving" / "contract.py").write_text(body)
    return root


#: A host still carrying the package that shipped before contract v22.
_PRE_V22 = "EVIDENCE_CONTROL_REFERENCES = ('bf16_source',)\n"


def _receipts(tmp_path):
    """Two ordinary non-empty matching per-arm receipts, one per arm."""
    import json

    ids = {"recorded": list(range(40)), "repetitive": [1, 2] * 8}
    plan = [("P0", "campaign", "prompt", "repetitive", "repetitive"),
            ("P4", "campaign", "messages", "recorded", "recorded")]
    paths = {}
    for arm, column in (("bf16", 4), ("tessera", 3)):
        results = []
        for prompt, form, key, student, source in plan:
            verdict = student if arm == "tessera" else source
            results.append({"id": prompt, "form": form,
                            "interface": ("chat_template" if key == "messages"
                                          else "raw_completion"),
                            "request": {"max_tokens": 64, key: "x"},
                            "completion": f"{arm}-{prompt}", "finish_reason": "stop",
                            "token_ids": ids[verdict]})
        path = tmp_path / f"smoke_{arm}.json"
        path.write_text(json.dumps(
            {"schema": "tessera.moe-greedy-smoke/1", "arm": arm, "rule": smoke.RULE,
             "tokenizer": {"tokenizer_json_sha256": "t"}, "prompts_sha256": "p",
             "results": results}))
        paths[arm] = str(path)
    return paths


def _run_compare(tmp_path, ambient, out):
    """The wrapper-owned join, launched the way the wrapper launches it.

    ``cwd`` is outside the checkout and ``PYTHONPATH`` is the ambient package
    alone, so nothing but the instrument's own resolution can put this
    checkout's ``tessera`` in reach.  ``-P`` keeps the script's directory off
    ``sys.path`` so ``experiments/`` cannot stand in for the package either.
    """
    import os
    import subprocess
    import sys

    paths = _receipts(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ambient)}
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-P", str(_PATH), "compare", paths["bf16"], paths["tessera"],
         "--out", str(out), "--subject", "tessera", "--reference", "bf16_source"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True)


def test_the_join_reads_the_aggregation_from_its_own_checkout(tmp_path):
    """#341: selecting the instrument must select the aggregation it calls.

    The wrapper runs ``$PY $TS/experiments/moe_greedy_smoke.py compare
    --subject ...`` and the pre-fix invocation bound no import path, so
    ``_contract_record``'s ``from tessera.serving.contract import
    derive_smoke_status`` resolved against the HOST's Tessera.  A host still
    carrying the pre-v22 package has no such name, so both expensive serves ran
    and the run then died at the final join with an ImportError, writing no
    ``pair.json``.  Installing the plugin inside the student container does not
    move the host interpreter.

    Here the only importable Tessera on ``PYTHONPATH`` is a pre-v22 stand-in,
    and the join must still succeed -- by resolving the aggregation from the
    checkout the instrument itself lives in -- and must emit the derived record.
    """
    import json

    out = tmp_path / "pair.json"
    done = _run_compare(tmp_path, _ambient_package(tmp_path, body=_PRE_V22), out)
    assert done.returncode == 0, f"stdout={done.stdout}\nstderr={done.stderr}"
    assert out.is_file(), "the join wrote no pair.json"
    emitted = json.loads(out.read_text())["contract_record"]
    assert emitted["derived"] == {"status": "recorded",
                                  "attribution": "shared_with_reference"}
    assert emitted["record"]["instrument"] == "experiments/moe_greedy_smoke.py"
    assert emitted["record"]["rule"] == smoke.RULE


def test_the_join_also_runs_with_no_ambient_tessera_at_all(tmp_path):
    """The absent-package half of the same boundary: a host that never
    installed Tessera is the commoner case, and it failed the same way."""
    import json

    out = tmp_path / "pair.json"
    done = _run_compare(tmp_path, tmp_path / "empty", out)
    assert done.returncode == 0, f"stdout={done.stdout}\nstderr={done.stderr}"
    assert json.loads(out.read_text())["contract_record"]["derived"]["status"] == "recorded"


def test_the_wrapper_refuses_a_missing_aggregation_before_it_serves_anything(tmp_path):
    """The cost half of #341 (`experiments/runtime_image.sh`'s own rule).

    Binding the instrument to its own checkout fixes the wrong-code half; it
    cannot help a checkout that genuinely has no aggregation to supply -- an
    `experiments/` tree deployed without its `src/`, say -- and discovering
    that AFTER two serves is what this finding is expensive about.  So the
    wrapper proves the join possible before it touches docker, the box's one
    serve lock, or either arm.

    `$TS` here is a checkout carrying the instrument and its shell helpers and
    NO `src/`, run by a real interpreter with nothing ambient to fall back on.
    The wrapper must refuse by name, and the `docker` recorder on PATH must
    stay empty.
    """
    import os
    import shutil
    import subprocess
    import sys

    checkout = tmp_path / "checkout"          # experiments/, deliberately no src/
    shutil.copytree(CHECKOUT / "experiments", checkout / "experiments")
    assert not (checkout / "src").exists()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "docker-was-called"
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\necho "$@" >> {called}\nexit 0\n')
    docker.chmod(0o755)
    # A real interpreter, with the script's own directory off sys.path so
    # `experiments/` cannot stand in for the package: the checkout's `src` is
    # the only thing that could supply the aggregation, and it is not there.
    python = bin_dir / "python-bare"
    python.write_text(f'#!/bin/sh\nexec {sys.executable} -P "$@"\n')
    python.chmod(0o755)

    out = tmp_path / "out"
    env = {**os.environ,
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "TS": str(checkout), "PY": str(python),
           "IMAGE": "example/image@sha256:" + "0" * 64,
           "EXT": str(tmp_path / "ext"), "VLLM_CACHE": str(tmp_path / "cache")}
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        ["bash", str(WRAPPER), str(tmp_path / "source"), str(tmp_path / "artifact"),
         str(tmp_path / "seal.json"), str(out)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True)
    combined = done.stdout + done.stderr
    assert done.returncode != 0, f"the wrapper did not refuse: {combined}"
    assert "REFUSED" in combined and "aggregation" in combined, combined
    assert not called.exists(), (
        "the wrapper reached docker before checking the join it ends with; the whole "
        "point of the preflight is that a mismatch cannot burn a serve")
    assert not (out / "pair.json").exists()
    # The refusal is in the run's own log, not only on a terminal nobody kept.
    assert "REFUSED" in (out / "driver.log").read_text()


def test_the_wrapper_preflights_with_the_instrument_it_will_join_with(tmp_path):
    """The preflight must be the join's OWN import, not a restatement of it.

    A preflight that lists the symbols it expects is a second copy of the rule,
    and the day the two disagree it waves through a run that fails after both
    serves.  So `preflight` and the join take the same names from the same
    `aggregation()`, and a real checkout answers it with where the contract
    actually resolved from.
    """
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    done = subprocess.run([sys.executable, "-P", str(_PATH), "preflight"],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert done.returncode == 0, f"stdout={done.stdout}\nstderr={done.stderr}"
    said = json.loads(done.stdout)
    assert said["ok"] is True
    assert said["checkout"] == str(CHECKOUT)
    assert said["contract_module"] == str(CHECKOUT / "src/tessera/serving/contract.py"), (
        "the preflight resolved the contract from somewhere other than the checkout "
        "holding the instrument, which is the defect #341 is about")
    assert said["lane_eligibility_schema"].startswith("tessera.lane-eligibility.v")
