"""CPU controls only: synthetic two-arm orchestration is not CUDA proof."""
import hashlib
from pathlib import Path
import subprocess
import pytest
import torch
from experiments.t8r_speed import value_prefetch_numeric as numeric


def test_exact_comparator_rejects_signed_zero_and_metadata():
    numeric.compare_bits(torch.tensor([1.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match='bits'):
        numeric.compare_bits(torch.tensor([0.0]), torch.tensor([-0.0]))
    with pytest.raises(ValueError, match='metadata'):
        numeric.compare_bits(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))


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
    assert numeric.SCOPE == 'syntheticgeometry/nonshipping'
