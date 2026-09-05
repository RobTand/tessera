"""The promotion gate must believe the screen's own proofs (tessera#250).

``experiments/refit_trailing_pair.py`` records, per unit and per arm, the
evidence that can invalidate its own experiment: the first/last drift control's
reconstruction identity, the sink-versus-wire agreement that licenses reading
any arm off the landing sink at all, and -- for the trailing arms -- the
matched-pair legs that say the two arms differ in the last scale plane and in
nothing else.  ``experiments/refit_trailing_pair_gate.py`` read none of them: a
document whose control DIFFERS, or whose trailing arm changed its codes, still
reached ``assert_plane_promotion`` on its numbers alone and could print
PROMOTED.

These tests run the real gate over the committed receipts, then over copies
with one recorded proof falsified or removed.  The valid receipts must keep the
verdicts they have; a falsified or missing proof must refuse **by name** and
name the field, whatever the ratios say.  ``plane_moved`` is deliberately not
in that set: an arm whose lever reached nothing is an ineffective arm, not a
broken comparison.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from refit_trailing_screen import (  # noqa: E402
    assert_screen_receipt, classify_trailing_pair, control_arm, wire_arms)

GATE = REPO / "experiments" / "refit_trailing_pair_gate.py"
QWEN = REPO / "experiments" / "results" / "refit_trailing_pair_qwen.json"
GLM = REPO / "experiments" / "results" / "refit_trailing_pair_glm.json"
QWEN_CL = REPO / "experiments" / "results" / "refit_trailing_pair_qwen_cl.json"

CONTROL = "A drift control FIRST [refit h^1.0 x4]"
B_JAC = "B-Jac  T R_h T R_h T R_h T R_H          (trailing full-H, Jacobi)"
C_GS = "C-GS   T R_H T R_H T R_H T R_H(GS)      (full-H every pass, sweep)"
B_GS_CL = ("B-GS+CL T R_h T R_h T R_h T R_H(GS,CL)   "
           "(trailing full-H, sweep, coupled landing)")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def screens(tmp_path):
    """The two committed receipts, copied so a test may falsify one proof."""
    qwen, glm = _load(QWEN), _load(GLM)
    assert qwen["units"] and glm["units"], "the receipts carry the screen"

    def write(q=None, g=None):
        qp, gp = tmp_path / "qwen.json", tmp_path / "glm.json"
        qp.write_text(json.dumps(q if q is not None else qwen))
        gp.write_text(json.dumps(g if g is not None else glm))
        return qp, gp

    return qwen, glm, write


def run_gate(tmp_path, qwen_path, glm_path, *, served=B_JAC, kl="0.5"):
    out = tmp_path / "gate.json"
    argv = [sys.executable, str(GATE),
            "--qwen", str(qwen_path), "--glm", str(glm_path),
            "--out", str(out)]
    if served is not None:
        argv += ["--served-arm", served.split()[0], "--served-kl", kl]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    record = json.loads(out.read_text()) if out.exists() else None
    return proc, record


def arm_line(stdout: str, arm: str) -> str:
    """The gate's verdict line for one arm."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("== ") and line[3:] == arm:
            return "\n".join(lines[i:i + 4])
    raise AssertionError(f"no block for {arm!r} in:\n{stdout}")


def test_the_committed_screen_still_promotes(screens, tmp_path):
    """A valid receipt keeps the verdict it has: nothing here is a new bar."""
    _, _, write = screens
    proc, record = run_gate(tmp_path, *write())
    assert proc.returncode == 0, proc.stderr
    assert "PROMOTED" in arm_line(proc.stdout, B_JAC)
    assert record["arms"][B_JAC]["verdict"]["promoted"] is True


def test_a_failed_drift_control_refuses_every_arm(screens, tmp_path):
    """First-versus-last reconstructions that DIFFER mean the two arms were
    encoded by two different processes, so no ratio in the document is a
    controlled comparison."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][CONTROL]["drift_control_identical"] = False
    proc, record = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode != 0, proc.stdout
    assert "drift_control_identical" in proc.stderr
    assert unit in proc.stderr
    assert "PROMOTED" not in proc.stdout


def test_a_removed_drift_control_proof_refuses(screens, tmp_path):
    """A proof that is absent is not a proof that passed."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    del qwen["units"][unit][CONTROL]["drift_control_identical"]
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode != 0, proc.stdout
    assert "drift_control_identical" in proc.stderr


def test_the_glm_population_is_checked_too(screens, tmp_path):
    """The cross-check population's control is evidence the gate reads."""
    _, glm, write = screens
    unit = next(iter(glm["units"]))
    glm["units"][unit][CONTROL]["drift_control_identical"] = False
    proc, _ = run_gate(tmp_path, *write(g=glm))
    assert proc.returncode != 0, proc.stdout
    assert "drift_control_identical" in proc.stderr
    assert str(GLM.name) in proc.stderr or "glm.json" in proc.stderr


def test_a_sink_versus_wire_disagreement_refuses(screens, tmp_path):
    """The sink the arms are scored off must be the wire that ships."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][C_GS]["sink_vs_wire_bit_identical"] = False
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode != 0, proc.stdout
    assert "sink_vs_wire_bit_identical" in proc.stderr


def test_changed_codes_refuse_the_trailing_arm(screens, tmp_path):
    """A trailing arm whose packed codes moved is comparing two encodings,
    not two scale planes -- and it is the arm the served KL is quoted for."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["matched_pair"]["codes_identical"] = False
    proc, record = run_gate(tmp_path, *write(q=qwen))
    block = arm_line(proc.stdout, B_JAC)
    assert "PROMOTED" not in block, block
    assert "codes_identical" in block
    assert "refused" in record["arms"][B_JAC]


def test_a_moved_blob_length_refuses_the_trailing_arm(screens, tmp_path):
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["matched_pair"]["bytes_equal"] = False
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    block = arm_line(proc.stdout, B_JAC)
    assert "PROMOTED" not in block, block
    assert "bytes_equal" in block


def test_a_removed_matched_pair_block_refuses_the_trailing_arm(screens, tmp_path):
    """Which arms owe the proof is derived from the recorded schedule -- the
    inner objectives equal to the control's and the trailing one swapped --
    so deleting the block is a missing proof, not an exempt arm."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    del qwen["units"][unit][B_JAC]["matched_pair"]
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    block = arm_line(proc.stdout, B_JAC)
    assert "PROMOTED" not in block, block
    assert "matched_pair" in block


def test_an_ineffective_lever_is_not_a_broken_pair(screens, tmp_path):
    """``plane_moved=false`` says the arm changed nothing, which is a result
    and not a failed control: it must not be refused as one."""
    qwen, _, write = screens
    for unit in qwen["units"]:
        qwen["units"][unit][B_JAC]["matched_pair"]["plane_moved"] = False
    proc, record = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode == 0, proc.stderr
    assert "PROMOTED" in arm_line(proc.stdout, B_JAC)
    assert record["arms"][B_JAC]["verdict"]["promoted"] is True


#: Every committed screen document: the producer's output, which is the pair
#: receipt and never the gate's verdict file.
SCREENS = sorted(
    path for path in (REPO / "experiments" / "results").glob("refit_trailing_pair_*.json")
    if "gate" not in path.name)


@pytest.mark.parametrize("receipt", SCREENS, ids=lambda p: p.stem)
def test_every_committed_screen_receipt_carries_its_proofs(receipt):
    """Including the coupled-landing pair.  #50's coupled landing re-assigns
    blocks and is *expected* to move the codes the next trellis pass sees, so
    ``B-GS+CL`` carries no matched-pair record and must not be required to --
    which is why the requirement is derived from the recorded schedule and the
    refit's own ``coupled`` diagnostics, not from an arm-name roster."""
    document = _load(receipt)
    failures = assert_screen_receipt(
        document, name=receipt.name, where="tessera#250")
    assert failures == {}, failures
    unit = next(iter(document["units"]))
    record = document["units"][unit]
    control = control_arm(record, where="tessera#250", unit=unit)
    classified = {arm: classify_trailing_pair(record[arm], record[control])
                  for arm in wire_arms(record) if arm != control}
    unknown = {arm: why for arm, (_pair, why) in classified.items() if why}
    assert unknown == {}, unknown
    pairs = [arm for arm, (pair, _why) in classified.items() if pair]
    assert [arm.split()[0] for arm in pairs] == ["B-Jac", "B-GS"], pairs


# --- the classification evidence itself (tessera#299) ------------------------
#
# Which arms owe the matched-pair proof is DERIVED from the recorded schedule
# and the refit's own coupled-landing diagnostics, so those recordings are
# evidence exactly as the proofs are.  Reading them with ``.get()`` and
# answering "not a trailing pair" made a receipt able to exempt an arm from
# every pair check by deleting, emptying or corrupting one field -- over an
# explicitly recorded ``codes_identical: false``, the very failure the #250
# validator exists to catch.  Unknown is not exempt.


def test_a_missing_candidate_schedule_cannot_clear_a_recorded_pair_failure(
        screens, tmp_path):
    """The #299 repro: a receipt that records ``codes_identical: false`` and
    then loses the field that classifies the comparison must not come back
    clean.  A proof that is present and FAILED does not disappear because a
    field needed to decide whether it was owed is missing."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["matched_pair"]["codes_identical"] = False
    del qwen["units"][unit][B_JAC]["schedule"]
    proc, record = run_gate(tmp_path, *write(q=qwen))
    block = arm_line(proc.stdout, B_JAC)
    assert "PROMOTED" not in block, block
    assert "schedule" in block
    assert "codes_identical" in block
    assert "refused" in record["arms"][B_JAC]


@pytest.mark.parametrize("schedule", [
    pytest.param([], id="empty"),
    pytest.param("1,1,1,2", id="not-a-list"),
    pytest.param([[1, False], ["trailing", False]], id="unreadable-objective"),
    pytest.param([[], []], id="stepless"),
])
def test_an_unreadable_candidate_schedule_does_not_exempt_the_arm(
        screens, tmp_path, schedule):
    """Empty, wrong-typed and unparseable schedules are all *unknown*, and an
    arm that cannot be told from the control's trailing pair is refused, not
    silently excused from the proof."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["schedule"] = schedule
    proc, record = run_gate(tmp_path, *write(q=qwen))
    block = arm_line(proc.stdout, B_JAC)
    assert "PROMOTED" not in block, block
    assert "schedule" in block
    assert "refused" in record["arms"][B_JAC]


def test_a_missing_control_schedule_refuses_the_whole_document(screens, tmp_path):
    """The control's schedule is the baseline every other arm is classified
    against, so losing it exempts *every* arm in the unit at once.  It refuses
    the document, exactly as a failed drift control does, rather than being
    reported one arm at a time."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["matched_pair"]["codes_identical"] = False
    del qwen["units"][unit][CONTROL]["schedule"]
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode != 0, proc.stdout
    assert "schedule" in proc.stderr
    assert unit in proc.stderr
    assert "PROMOTED" not in proc.stdout


def test_a_coupled_arm_that_records_no_diagnostics_is_not_exempt():
    """The other half of the classification: an arm is exempt from the pair
    proof because its refit diagnostics say a coupled landing re-assigned
    blocks (#50).  Absent diagnostics are not that statement, so a
    trailing-shaped arm that carries none is refused rather than read as
    coupled."""
    document = _load(QWEN_CL)
    unit = next(iter(document["units"]))
    del document["units"][unit][B_GS_CL]["refit"]
    failures = assert_screen_receipt(
        document, name=QWEN_CL.name, where="tessera#299")
    assert B_GS_CL in failures, failures
    assert any("coupled" in reason for reason in failures[B_GS_CL]), \
        failures[B_GS_CL]


def test_the_coupled_arm_keeps_its_exemption_when_it_proves_it():
    """The control for the case above: the committed coupled-landing receipt
    records the diagnostics, so ``B-GS+CL`` stays exempt and the document is
    clean."""
    assert assert_screen_receipt(
        _load(QWEN_CL), name=QWEN_CL.name, where="tessera#299") == {}


def test_a_non_wire_landing_in_the_document_refuses(screens, tmp_path):
    """Only the wire promotes (tessera#85); a wire arm that records another
    landing is a mislabelled row, not a ceiling read the gate may skip."""
    qwen, _, write = screens
    unit = next(iter(qwen["units"]))
    qwen["units"][unit][B_JAC]["landing"] = "none"
    qwen["units"][unit][B_JAC]["serialisable"] = False
    proc, _ = run_gate(tmp_path, *write(q=qwen))
    assert proc.returncode != 0, proc.stdout
    assert "landing" in proc.stderr
