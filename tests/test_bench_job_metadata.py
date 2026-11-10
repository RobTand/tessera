"""The regular bench writer marks its Flux identity with its schema."""
import ast
from pathlib import Path


def test_regular_bench_record_declares_the_job_schema():
    from tessera.job_metadata import JOB_META_SCHEMA, job_identity
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.py').read_text())
    fields = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                  and any(isinstance(k, ast.Constant) and k.value == 'tessera_file' for k in node.keys))
    schema = next(ast.literal_eval(v) for k, v in zip(fields.keys, fields.values)
                  if isinstance(k, ast.Constant) and k.value == 'meta_schema')
    assert schema == JOB_META_SCHEMA
    assert job_identity({'meta_schema': schema, 'flux_job_id': 'job'}) == 'job'


def test_reader_selection_has_no_dead_transport_switch():
    root = Path(__file__).resolve().parents[1]
    for name in ('bench_t8r.py', 'bench_t8r.sh', 'paired_k32_action.py'):
        text = (root / 'experiments/t8r_speed' / name).read_text()
        assert '--direct-vllm-inputs' not in text
        assert 'direct_vllm_inputs' not in text
