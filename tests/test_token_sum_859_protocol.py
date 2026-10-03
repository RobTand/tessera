"""CPU checks of the exact original-binding GPU proposal, not CUDA evidence."""
import hashlib
import json
from pathlib import Path
import runpy

from box_artifacts import ROOTS

ROOT = Path(__file__).resolve().parents[1]
DRIVER = runpy.run_path(str(ROOT / 'experiments/t8r_speed/token_sum_859.py'))
PROTOCOL = json.loads((ROOT / 'experiments/configs/token_sum_859_protocol.json').read_bytes())


def test_protocol_binds_unchanged_compiled_original_binding_source():
    source = ROOT / 'src/tessera/serving/csrc/routed_fused_window.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == DRIVER['SOURCE']
    assert PROTOCOL['implementation_owner']['kernel_sha256'] == DRIVER['SOURCE']
    assert PROTOCOL['libraries'] == list(DRIVER['LIBS'])
    assert set(DRIVER['BASELINE_HASHES']) == set(DRIVER['LIBS'])
    body = source.read_text().split('void token_sum(', 1)[1].split('void token_sum_shared(', 1)[0]
    assert body.index('out.is_cuda() && out.device() == routed.device()') < body.index('if (vecs == 0) return;') < body.index('token_sum_kernel<<<')


def test_root_gpu_proposal_has_two_concrete_exclusive_single_attempt_commands():
    assert PROTOCOL['gpu_authorized'] is False
    assert PROTOCOL['campaign_max_inflight'] == 1
    resources = PROTOCOL['resources_each']
    assert resources['exclusive'] and resources['max_attempts'] == 1
    assert resources['cpu'] == resources['native_threads'] == 1
    assert resources['deadline_seconds'] == resources['queue_wait_seconds'] == 180
    assert resources['output_max_bytes'] == 32 << 20
    assert len(PROTOCOL['submission_argv']) == len(PROTOCOL['gpu_rows']) == 2
    for row, argv in zip(PROTOCOL['gpu_rows'], PROTOCOL['submission_argv']):
        # Frozen proposal uses the published client, not a per-test host path.
        client = Path(ROOTS['prismabuild_tools'].default) / 'pbrun.py'
        assert argv[0:2] == ['python3', str(client)]
        assert '--exclusive' in argv
        for option, value in [('--max-attempts', '1'), ('--cpus', '1'), ('--timeout-s', '180'), ('--wait-s', '180'), ('--container-image', PROTOCOL['image']), ('--data-manifest', PROTOCOL['readsets'][row['arm']]['path'])]:
            assert argv[argv.index(option) + 1] == value
        assert argv[-len(row['args']):] == row['args']
        assert ('--before' in argv) == (row['arm'] == 'before')


def test_single_gpu_corner_and_separate_native_arms_remain_unqualified():
    assert PROTOCOL['native_cpu_gate']['not_qualified_yet']
    assert 'different CUDA devices unobservable' in ' '.join(PROTOCOL['limitations'])
    for arm in ['t4_prefetch_877', 't16_prefetch_876', 'piece_major']:
        assert 'excluded' in PROTOCOL['source_reconciliation'][arm]
