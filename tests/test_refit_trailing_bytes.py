"""The matched-pair check must refuse a pair that moved more than the plane.

``experiments/refit_trailing_bytes.py`` reads two exported checkpoints and says
whether they are tessera#75's pair: passes 1-3 are identical calls, so the
packed codes must be identical on every unit and only the trailing scale plane
may differ.  It loaded ``.weight`` and ``.input_global_scale`` and recorded
their differences -- and then decided on the packed codes, the scale plane, the
tensor-name sets and the wire byte totals alone, so a B side whose BF16
passthrough weight or activation quantizer had also moved still returned 0 with
``verdict: "the matched pair"`` (tessera#248).

The pairs here are three-tensor toys: what is exercised is the deciding
predicate, not an encoder.

It also owns the tool's **exit-code contract**, because a second process reads
it: ``refit_trailing_serve.sh compare-drift`` exists to report "NOT the matched
pair", so that verdict cannot share a status with the tool failing to reach any
verdict at all (tessera#269).  Zero is the matched pair, three is the computed
NOT-matched verdict, one is the tool failing.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "experiments" / "refit_trailing_bytes.py"
WRAPPER = REPO / "experiments" / "refit_trailing_serve.sh"

UNIT = "model.layers.0.mlp.down_proj"


def tool_module():
    """The module itself, for the constants it owns."""
    sys.path.insert(0, str(REPO / "experiments"))
    import refit_trailing_bytes as tool

    return tool


def base_tensors() -> dict:
    """One quantized unit plus the passthrough weights around it."""
    return {
        f"{UNIT}.weight_packed": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        f"{UNIT}.weight_scale": torch.ones(4, 1, dtype=torch.float32),
        f"{UNIT}.weight_global_scale": torch.tensor([2.0]),
        f"{UNIT}.input_global_scale": torch.tensor([0.125]),
        "model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16),
    }


def write_side(root: Path, tensors: dict, *, wire_bytes: int | None = None) -> Path:
    twin = root
    twin.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(twin / "model.safetensors"))
    if wire_bytes is not None:
        (twin / "tessera_serving_manifest.json").write_text(json.dumps(
            {"totals": {"wire_bytes": wire_bytes, "on_disk_bytes": wire_bytes,
                        "units": 1, "modules": 1}}))
    return twin


def make_pair(tmp_path, *, mutate=None, wire_b_bytes=4096):
    """A genuine scale-only pair, with one optional extra B-side change."""
    a = base_tensors()
    b = base_tensors()
    # The trailing refit's whole intervention: the scale plane moves.
    b[f"{UNIT}.weight_scale"] = torch.full((4, 1), 1.5, dtype=torch.float32)
    if mutate is not None:
        mutate(b)
    write_side(tmp_path / "a", a, wire_bytes=4096)
    write_side(tmp_path / "b", b, wire_bytes=wire_b_bytes)
    return tmp_path / "a", tmp_path / "b"


def run_tool(tmp_path, a: Path, b: Path, *, wire: bool = True):
    out = tmp_path / "bytes.json"
    argv = [sys.executable, str(TOOL), str(a), str(b), "--out", str(out)]
    if wire:
        argv += ["--wire-a", str(a), "--wire-b", str(b)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    record = json.loads(out.read_text()) if out.exists() else None
    return proc, record


def test_a_scale_only_pair_is_the_matched_pair(tmp_path):
    proc, record = run_tool(tmp_path, *make_pair(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert record["verdict"] == "the matched pair"


def test_a_moved_passthrough_weight_is_not_the_matched_pair(tmp_path):
    """A BF16 weight the intervention does not touch has moved, so the served
    KL of this pair cannot isolate the trailing refit's objective."""
    def touch_weight(b):
        b["model.embed_tokens.weight"] = torch.full(
            (4, 4), 2.0, dtype=torch.bfloat16)

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=touch_weight))
    assert proc.returncode != 0, proc.stdout
    assert record["verdict"] == "NOT the matched pair"
    assert record["immutable_changed"][".weight"] == ["model.embed_tokens.weight"]
    assert "model.embed_tokens.weight" in proc.stdout


def test_a_moved_activation_scale_is_not_the_matched_pair(tmp_path):
    """The activation quantizer is held fixed by this experiment's design;
    a pair whose A4 input scales moved is measuring two treatments."""
    def touch_input_scale(b):
        b[f"{UNIT}.input_global_scale"] = torch.tensor([0.25])

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=touch_input_scale))
    assert proc.returncode != 0, proc.stdout
    assert record["verdict"] == "NOT the matched pair"
    assert record["immutable_changed"][".input_global_scale"] == [
        f"{UNIT}.input_global_scale"]


def test_the_weight_global_scale_may_move_with_the_plane(tmp_path):
    """The weight plane's own global scale is part of the intervention, not a
    third treatment: a pair that moves it with the block scales still passes."""
    def touch_weight_global(b):
        b[f"{UNIT}.weight_global_scale"] = torch.tensor([2.5])

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=touch_weight_global))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert record["verdict"] == "the matched pair"


def test_moved_codes_are_still_refused(tmp_path):
    def touch_codes(b):
        b[f"{UNIT}.weight_packed"] = torch.arange(
            1, 17, dtype=torch.int32).reshape(4, 4)

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=touch_codes))
    assert proc.returncode != 0, proc.stdout
    assert record["codes_identical_on_every_unit"] is False


def test_a_plane_that_did_not_move_is_still_refused(tmp_path):
    def revert_plane(b):
        b[f"{UNIT}.weight_scale"] = torch.ones(4, 1, dtype=torch.float32)

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=revert_plane))
    assert proc.returncode != 0, proc.stdout
    assert record["the_plane_moved"] is False


def test_a_moved_wire_length_is_still_refused(tmp_path):
    proc, record = run_tool(tmp_path, *make_pair(tmp_path, wire_b_bytes=4097))
    assert proc.returncode != 0, proc.stdout
    assert record["wire"]["wire_bytes_equal"] is False


def test_the_not_matched_verdict_is_not_the_tool_failing(tmp_path):
    """The verdict this stage exists to report gets its own status, and the
    receipt is written before it is returned: ``compare-drift`` reads a 3 as a
    reading and anything else as the tool failing (tessera#269)."""
    def touch_codes(b):
        b[f"{UNIT}.weight_packed"] = torch.arange(
            1, 17, dtype=torch.int32).reshape(4, 4)

    proc, record = run_tool(tmp_path, *make_pair(tmp_path, mutate=touch_codes))
    assert record["verdict"] == "NOT the matched pair"
    assert proc.returncode != 1, \
        "the computed verdict shares its status with the tool failing"
    assert proc.returncode == tool_module().EXIT_NOT_MATCHED, (
        proc.stdout + proc.stderr)


def test_a_missing_manifest_is_the_tool_failing_not_a_verdict(tmp_path):
    """``--wire-a`` at a directory with no ``tessera_serving_manifest.json``
    reaches no verdict at all, so it must not be reported as one."""
    a, b = make_pair(tmp_path)
    (a / "tessera_serving_manifest.json").unlink()
    proc, record = run_tool(tmp_path, a, b)
    assert proc.returncode == tool_module().EXIT_TOOL_FAILED, (
        proc.stdout + proc.stderr)
    assert record is None, "a failed run wrote a receipt"


def test_an_export_with_no_tensors_is_the_tool_failing_not_a_verdict(tmp_path):
    """An absent or unreadable export dir loads zero tensors and compares
    nothing.  Before tessera#269 that fell out as "NOT the matched pair" --
    which is exactly the reading ``compare-drift`` accepts, so an absent
    ``$INCUMBENT`` would have certified a drift receipt built from nothing."""
    _, b = make_pair(tmp_path)
    proc, record = run_tool(tmp_path, tmp_path / "not-exported", b, wire=False)
    assert "no *.safetensors" in proc.stderr, \
        f"nothing was compared and the tool returned {proc.returncode} anyway"
    assert record is None, "a run that compared nothing wrote a receipt"
    assert proc.returncode == tool_module().EXIT_TOOL_FAILED, (
        proc.stdout + proc.stderr)


def test_the_exit_codes_are_named_once_and_the_shell_reads_the_same_numbers():
    """Three outcomes, three codes, named in the module that owns them.  The
    numbers themselves are the contract, because a second process reads them,
    and ``refit_trailing_serve.sh`` cannot import a Python constant -- so its
    own list of accepted codes is pinned here against the module's."""
    tool = tool_module()
    assert (tool.EXIT_MATCHED, tool.EXIT_NOT_MATCHED, tool.EXIT_TOOL_FAILED) \
        == (0, 3, 1)
    declared = re.search(r'drift_reading_codes="([0-9 ]+)"',
                         WRAPPER.read_text())
    assert declared, "compare-drift declares no accepted exit codes"
    assert {int(c) for c in declared.group(1).split()} == {
        tool.EXIT_MATCHED, tool.EXIT_NOT_MATCHED}


def test_every_loaded_suffix_has_a_declared_policy():
    """The policy is a total function of what the loader reads: adding a
    suffix to ``SUFFIXES`` without saying whether it may move is a refusal,
    not a silently-ignored tensor."""
    tool = tool_module()

    assert set(tool.SUFFIXES) == set(tool.CHANGE_POLICY)
    assert set(tool.CHANGE_POLICY.values()) <= {
        tool.IDENTICAL, tool.MUST_MOVE, tool.MAY_MOVE}
