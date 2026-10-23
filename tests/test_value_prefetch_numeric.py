"""CPU controls only: synthetic two-arm orchestration is not CUDA proof."""
import hashlib
from pathlib import Path
import subprocess
import pytest
import torch
from experiments.t8r_speed import value_prefetch_numeric as numeric
from _whole_phase_sdk import whole_phase_sdk, manifest_value, write_manifest


def _numeric_entries(tmp_path):
    # Deliberately absent artifacts: validation may open only the manifest.
    return [dict(path=str(tmp_path / "bank.pt"), offset=0, bytes=32, sha256="0" * 64),
            dict(path=str(tmp_path / "source.so"), offset=0, bytes=16, sha256="1" * 64)]


@pytest.mark.parametrize("schema,variant", [
    ("v1", "whole"), ("v1", "trailing-zero"), ("v1", "v1-advisory-bytes"),
    ("v2", "whole"), ("v2", "trailing-zero"), ("v2", "read-order"), ("v2", "resident"),
])
def test_whole_phase_public_schema_and_read_order(tmp_path, whole_phase_sdk, schema, variant):
    entries = _numeric_entries(tmp_path)
    path = write_manifest(tmp_path / "declared.json",
                          manifest_value(entries, "whole-synthetic-bank", schema, variant))
    validated, encoding = whole_phase_sdk.read_data_manifest(path)
    assert encoding == "identity"
    assert whole_phase_sdk.manifest_read_entries(validated) == entries
    manifest, reads = numeric.validate_readset(path)
    assert manifest == validated
    assert reads == entries


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize("variant", ["missing", "wrong", "split", "multiple", "incomplete", "leading-zero"])
def test_whole_phase_public_phase_refusals(tmp_path, whole_phase_sdk, schema, variant):
    entries = _numeric_entries(tmp_path)
    path = write_manifest(tmp_path / "declared.json",
                          manifest_value(entries, "whole-synthetic-bank", schema))
    numeric.validate_readset(path)
    write_manifest(path, manifest_value(entries, "whole-synthetic-bank", schema, variant))
    with pytest.raises(ValueError):
        numeric.validate_readset(path)


def test_whole_phase_requires_identity_encoding(tmp_path, whole_phase_sdk):
    import gzip
    path = write_manifest(tmp_path / "declared.json",
                          manifest_value(_numeric_entries(tmp_path), "whole-synthetic-bank", "v1"))
    numeric.validate_readset(path)
    path.write_bytes(gzip.compress(path.read_bytes()))
    with pytest.raises(ValueError):
        numeric.validate_readset(path)


def test_repair_reports_ranges_while_validator_returns_reads(tmp_path, whole_phase_sdk, monkeypatch, capsys):
    """Synthetic metadata probe only; do not bypass the real frozen-bank gate."""
    import json

    entries = _numeric_entries(tmp_path)
    original_value = manifest_value(entries, "whole-synthetic-bank", "v1")
    original_value["annotations"]["phases"][0]["cumulative_bytes"] -= 1
    original = write_manifest(tmp_path / "original.json", original_value)
    output = tmp_path / "repaired.json"
    real_digest = numeric.digest
    # Exercise repair with synthetic metadata, not the unavailable frozen bank.
    # Only its original-file admission digest is substituted; output hashing,
    # parsing, validation, exclusive creation and reporting are real.
    monkeypatch.setattr(numeric, "digest", lambda path:
                        "f8f880774c06a4bb8fa0f99d73469645f5521827b34928c8983d0dd33fe20dc5"
                        if Path(path) == original else real_digest(path))
    numeric.repair_readset(original, output)
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    checked, encoding = whole_phase_sdk.read_data_manifest(output)
    assert encoding == "identity"
    assert checked["entries"] == entries == original_value["entries"]
    assert report == {"scope": numeric.SCOPE, "readset": str(output),
                      "sha256": real_digest(output), "entry_count": 2,
                      "total_bytes": 48,
                      "phases": [{"name": "whole-synthetic-bank", "start_bytes": 0,
                                  "end_bytes": 48}],
                      "retained_entries_unchanged": True}
    # These are deliberately different shapes: returning phases from the
    # validator or serializing read entries under "phases" must both fail.
    manifest, reads = numeric.validate_readset(output)
    assert manifest == checked
    assert reads == whole_phase_sdk.manifest_read_entries(checked) == entries


@pytest.mark.parametrize("bad_source", [False, True], ids=["short-phase", "foreign-source"])
def test_native_preflight_rejects_input_before_native_load_and_releases_reader(tmp_path, monkeypatch, bad_source):
    """CPU gate model only; actual DSO FD mapping is a separate admitted action."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class Reader:
        entries = {("source", 0): {}}
        manifest = {"total_bytes": 4 if bad_source else 5}
        closed = False
        def read(self, path, offset=0):
            return b"bad!"
        def close(self):
            self.closed = True
    reader = Reader()
    monkeypatch.setattr(numeric, "StagedInputs", lambda manifest: reader)
    with pytest.raises(ValueError, match="retained source" if bad_source else "byte total"):
        numeric.native_preflight(tmp_path / "bank", tmp_path / "readset", tmp_path / "out")
    assert reader.closed


def test_native_preflight_rejects_visible_cuda_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="CPU-only"):
        numeric.native_preflight(tmp_path / "bank", tmp_path / "readset", tmp_path / "out")
    assert not (tmp_path / "out").exists()



def test_exact_comparator_rejects_signed_zero_and_metadata():
    numeric.compare_bits(torch.tensor([1.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match='bits'):
        numeric.compare_bits(torch.tensor([0.0]), torch.tensor([-0.0]))
    with pytest.raises(ValueError, match='metadata'):
        numeric.compare_bits(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize("failure", ["existing-directory", "file-parent"])
def test_sanitizer_output_initialization_failure_releases_reader(tmp_path, monkeypatch, failure):
    """Causal CPU lease model with real filesystem refusal, not GPU proof."""
    class Reader:
        closed = False
        def close(self):
            self.closed = True
    reader = Reader()
    monkeypatch.setattr(numeric, "StagedInputs", lambda manifest: reader)
    out = tmp_path / "out"
    if failure == "existing-directory":
        (out / "sanitizer").mkdir(parents=True)
    else:
        out.write_text("not a directory")
    with pytest.raises(OSError):
        numeric.sanitize(tmp_path / "bank", tmp_path / "manifest.json", out)
    assert reader.closed



def test_sanitizer_uses_only_pinned_bytes_and_propagates_failure(tmp_path, monkeypatch):
    bank = Path('/forbidden/synthetic-bank')
    tool_bytes = b'sealed sanitizer control'
    monkeypatch.setattr(numeric, 'SANITIZER_SHA', hashlib.sha256(tool_bytes).hexdigest())
    class Reader:
        entries = {(str(bank / 'sanitizer/compute-sanitizer'), 0): {},
                   (str(bank / 'sanitizer/libsanitizer-public.so'), 0): {}}
        closed = False
        def read(self, path, offset=0):
            assert (path, offset) in self.entries
            return tool_bytes
        def close(self):
            self.closed = True
    reader = Reader()
    monkeypatch.setattr(numeric, 'StagedInputs', lambda manifest: reader)
    calls = []
    def run(command, check):
        calls.append(command)
        assert check
        raise subprocess.CalledProcessError(99, command)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        numeric.sanitize(bank, tmp_path / 'manifest.json', tmp_path / 'out')
    assert reader.closed
    assert calls[0][1:7] == ['--tool', 'memcheck', '--error-exitcode', '99', '--target-processes', 'all']
    assert 'consume' in calls[0]
    assert calls[0][0] == str(tmp_path / 'out/sanitizer/compute-sanitizer')


def test_synthetic_matrix_covers_every_routed_run_pair():
    assert numeric.MS == [1, 7, 71]
    assert {q // 256 for q in numeric.Q256_CASES if q % 256 == 0} == set(range(1, 9))
    assert {q // 256 for q in numeric.Q256_CASES if q % 256} == set(range(1, 8))
    assert numeric.SCOPE == "syntheticgeometry/nonshipping"


@pytest.mark.parametrize("complete_phase", [False, True])
def test_public_pb_readset_and_whole_bank_phase_contract(tmp_path, complete_phase, whole_phase_sdk):
    import json
    manifest = {"schema": "prismaquant.prismabuild.data_manifest.v1",
                "produced_by": {"tool": "test"}, "mount_prefix": str(tmp_path),
                "annotations": {"phases": [{"name": "whole-synthetic-bank", "bytes": 32, "cumulative_bytes": 32 if complete_phase else 31}]},
                "entries": [{"path": str(tmp_path / "synthetic.pt"), "offset": 0, "bytes": 32, "sha256": "0" * 64}],
                "entry_count": 1, "total_bytes": 32}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if complete_phase:
        checked, reads = numeric.validate_readset(path)
        assert checked == manifest
        # Numeric consumers need the public SDK's ordered read entries. The
        # repair CLI's separate "phases" field remains normalized ranges.
        assert reads == manifest["entries"]
    else:
        with pytest.raises(ValueError, match="staging phase"):
            numeric.validate_readset(path)


@pytest.mark.parametrize("history", ["none", "zero", "fallback", "declared-wins"])
def test_fixture_transfer_preserves_native_history_precedence_and_prefix(history):
    import dataclasses
    from test_window_gemm_grouped import Expert, _states
    from tessera.window_gemm import _resolve_initial_state
    initial = None if history == "none" else torch.zeros(32, dtype=torch.int32)
    if history in ("fallback", "declared-wins"):
        initial = torch.arange(32, dtype=torch.int32) + 101
    expert = Expert(64, 32, (1,) * 32, 874, family="value", init=initial, device="cpu")
    unit = expert.unit
    if history == "declared-wins":
        unit = dataclasses.replace(unit, initial_state=initial)
        unit.rep.initial_state = torch.full((32,), 999, dtype=torch.int32)
    rep_history = getattr(unit.rep, "initial_state", None)
    declared_history = unit.initial_state
    # The old recursive replacement drops the dynamic rep history.
    old_transfer = dataclasses.replace(unit, rep=dataclasses.replace(unit.rep))
    old_states = _states(expert.body, expert.rates, 14, _resolve_initial_state(old_transfer, None))
    transferred = numeric.move(unit, "cpu")
    resolved = _resolve_initial_state(transferred, None)
    decoded = _states(expert.body, expert.rates, 14, resolved)
    assert torch.equal(decoded, expert.states)
    assert torch.equal(old_states[13:], expert.states[13:])
    if history == "fallback":
        assert not torch.equal(old_states[:13], expert.states[:13])
    else:
        assert torch.equal(old_states, expert.states)
    assert unit.initial_state is declared_history
    assert getattr(unit.rep, "initial_state", None) is rep_history
    if initial is None:
        assert resolved is None
    else:
        assert torch.equal(resolved, initial)
    for items, _ in transferred.items_by_mt.values():
        assert items.device == transferred.rep.words.device


def test_actual_retained_fixture_histories_through_public_pinned_reader():
    import io, json, os
    from test_window_gemm_grouped import _states
    from tessera.window_gemm import _resolve_initial_state
    manifest = os.environ.get("T16_SYNTHETIC_READSET")
    if manifest is None:
        pytest.skip("explicit admitted synthetic bank proof only")
    reader = numeric.StagedInputs(manifest)
    units, histories = 0, 0
    try:
        packet_path = next(path for path, offset in reader.entries if Path(path).name == "packet.json")
        packet = json.loads(reader.read(packet_path))
        assert packet["scope"] == numeric.SCOPE
        assert [case["q256"] for case in packet["cases"]] == numeric.Q256_CASES
        for case in packet["cases"]:
            payload = torch.load(io.BytesIO(reader.read(case["path"])), map_location="cpu", weights_only=False)
            for stack in payload["stacks"]:
                for expert in stack:
                    original = _resolve_initial_state(expert.unit, None)
                    transferred = numeric.move(expert.unit, "cpu")
                    resolved = _resolve_initial_state(transferred, None)
                    if original is None:
                        assert resolved is None
                    else:
                        assert torch.equal(resolved, original)
                        histories += 1
                    assert torch.equal(_states(expert.body, expert.rates, 14, resolved), expert.states)
                    assert _resolve_initial_state(expert.unit, None) is original
                    units += 1
        assert (units, histories) == (270, 108)
        print(f"RETAINED_SYNTHETIC_CPU_STATE_PROOF units={units} histories={histories} readset={reader.manifest_sha256}")
    finally:
        reader.close()
