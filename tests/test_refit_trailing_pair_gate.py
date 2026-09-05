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
    assert_screen_receipt, control_arm, is_trailing_pair, wire_arms)

GATE = REPO / "experiments" / "refit_trailing_pair_gate.py"
QWEN = REPO / "experiments" / "results" / "refit_trailing_pair_qwen.json"
GLM = REPO / "experiments" / "results" / "refit_trailing_pair_glm.json"

CONTROL = "A drift control FIRST [refit h^1.0 x4]"
B_JAC = "B-Jac  T R_h T R_h T R_h T R_H          (trailing full-H, Jacobi)"
C_GS = "C-GS   T R_H T R_H T R_H T R_H(GS)      (full-H every pass, sweep)"


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
    pairs = [arm for arm in wire_arms(record)
             if arm != control and is_trailing_pair(record[arm], record[control])]
    assert [arm.split()[0] for arm in pairs] == ["B-Jac", "B-GS"], pairs


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
