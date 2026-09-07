"""Complete-answer and route-roster proof for the all-original research reference."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT: Path
BASELINE = Path('/mnt/shared/tessera-clean-runtime-20260907/full-model-original-r1024')
CONTROL = Path('/mnt/shared/tessera-clean-runtime-20260907/original-wire-layer2')
CONFIG = Path('/mnt/shared/tessera-native376-resource/configs/lfm25_first_model_fixed_kv_20260907.json')
CONFIG_SHA = 'f5064609d62a3e61ef1d9bb87b2b62ea10b31b71759db3dce666543d7585233e'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'artifact': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}), flush=True)


def observe_all(model):
    from original_wire_generation import observed
    from tessera.serving.telemetry import read_route
    result = observed(model)
    methods, routes = {}, {}
    for path, layer in model.named_modules():
        if getattr(layer, 'tessera_family', None) is None:
            continue
        method = getattr(layer, 'quant_method', None)
        if method is None:
            continue
        name = path.removesuffix('.routed_experts')
        assert name not in methods, 'Duplicate normalized runtime owner'
        methods[name] = {'actual_module_path': path,
                         'method': type(method).__module__ + '.' + type(method).__qualname__,
                         'family': layer.tessera_family, 'mode': layer.tessera_mode,
                         'kind': 'moe' if getattr(layer, 'tessera_structure', None) == 'routed_moe' else 'dense'}
        routes[name] = read_route(layer)
    result.update(methods=methods, routes=routes)
    return result


def main():
    global ROOT
    ROOT = Path(sys.argv[1])
    started = time.time()
    assert digest(CONFIG) == CONFIG_SHA
    config = json.loads(CONFIG.read_text())
    (ROOT / 'serving-config.json').write_bytes(CONFIG.read_bytes())
    checkpoint = BASELINE / 'checkpoint'
    proof_path = BASELINE / 'export-proof.json'
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'passed' and len(proof['wires']) == 2142
    for name, row in proof['checkpoint_files'].items():
        assert digest(checkpoint / name) == row['sha256'], name
    manifest = json.loads((checkpoint / 'tessera_serving_manifest.json').read_text())
    expected = set(manifest['modules'])
    assert len(expected) == 38
    assert sum(len(row['roles']) for row in manifest['modules'].values()) == 2142
    package_path = CONTROL / 'controls/package-identity-proof.json'
    assert digest(package_path) == '26ab4829797e81b96ec8f6e4bf4257757c25905324df5d1f6baba687e692a4f7'
    package = json.loads(package_path.read_text())
    helper = CONTROL / 'controls/full_engine_kv.py'
    assert digest(helper) == '3320b77dd392e71bd07c967a3e2aa2e85707f64b4992ec185256936ec0986651'
    assert os.environ['TESSERA_SERVE_MODE'] == 'resident'
    from tessera.cached_unit import encoder_source_sha256
    assert encoder_source_sha256() == package['installed_package_source_sha256']
    from full_engine_kv import inspect_worker_kv
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(checkpoint), seed=0, **config['engine_args'])
    try:
        before = llm.apply_model(observe_all)[0]
        write('loaded-runtime.json', before)
        kv = llm.collective_rpc(inspect_worker_kv, timeout=60, args=(config['capacity_assertions'],))
        write('actual-kv-capacity.json', {'helper_sha256': digest(helper), 'workers': kv})
        assert len(kv) == 1 and kv[0]['capacity_assertions']['passed']
        assert set(before['methods']) == expected, 'Loaded owners differ from exported dense/MoE roster'
        messages = [{'role': 'user', 'content': 'Return exactly the word blue.'}]
        outputs = llm.chat(messages, SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=512), use_tqdm=False)
        after = llm.apply_model(observe_all)[0]
        write('served-runtime.json', after)
        def route_key(row):
            return json.dumps({k: v for k, v in row.items() if k not in ('launches', 'modules')}, sort_keys=True)
        old = {route_key(row): row['launches'] for row in before['trace']['entries']}
        delta = [{**row, 'launches': row['launches'] - old.get(route_key(row), 0),
                  'new_shape_since_request': route_key(row) not in old}
                 for row in after['trace']['entries'] if row['launches'] > old.get(route_key(row), 0)]
        assert len(outputs) == 1 and len(outputs[0].outputs) == 1
        prompt_rows = len(outputs[0].prompt_token_ids)
        new_prefill = [row for row in delta if row['new_shape_since_request']
                       and row['shape'].startswith(f'M{prompt_rows}:')]
        # Telemetry publishes distinct module counts, not module names. For
        # this one prefill, each owner's fixed N/K geometry occupies one key.
        observed_counts = {kind: sum(row['modules'] for row in new_prefill if row['kind'] == kind)
                           for kind in ('dense', 'moe')}
        expected_counts = {kind: sum(row['kind'] == kind for row in after['methods'].values())
                           for kind in ('dense', 'moe')}
        answer = outputs[0].outputs[0]
        final = answer.text.strip()
        closed = '<think>' not in final or (final.count('<think>') == 1 and final.count('</think>') == 1)
        if '<think>' in final:
            final = final.rsplit('</think>', 1)[-1].strip() if closed else ''
        checks = {
            'complete_stop': answer.finish_reason == 'stop', 'exact_final_blue': final == 'blue',
            'closed_reasoning': closed, 'actual_kv_capacity': kv[0]['capacity_assertions']['passed'],
            'exact_exported_owner_roster': set(after['methods']) == expected,
            'post_request_prefill_owner_counts_match_roster': observed_counts == expected_counts,
            'native_plugin_methods': all(row['method'].startswith('tessera.serving.') for row in after['methods'].values()),
            'served_routes': all(row and row['state'] == 'served' for row in after['routes'].values()),
            'both_dense_and_moe_launches': {row['kind'] for row in delta} == {'dense', 'moe'},
            'loaded_module_origins': all(row['package_identity']['loaded_module_origins_verified'] for row in (before, after)),
            'exact_installed_sources': all(row['package_identity']['fresh_encoder_source_sha256'] ==
                row['package_identity']['independently_recomputed_source_sha256'] == package['installed_package_source_sha256']
                and row['package_identity']['package_source_files'] == package['expected_installed_source_files'] for row in (before, after)),
        }
        sys.path.insert(0, '/mnt/shared/tessera-clean-runtime-20260907/control-854e672')
        from per_job_install import files
        inventory = Path('/mnt/shared/tessera-clean-runtime-20260907/official-primary/runtime-inventory.json')
        actual = files(Path(importlib.util.find_spec('vllm').origin).parent)
        checks['stock_core_unchanged'] = actual == json.loads(inventory.read_text())['files']
        write('core-after-generation.json', {'files': actual, 'baseline_sha256': digest(inventory)})
        write('generation-proof.json', {'schema': 'tessera.full_model_original_wire_generation.v1',
            'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'request': {'messages': messages, 'max_tokens': 512, 'temperature': 0.0, 'seed': 0},
            'response': {'text': answer.text, 'finish_reason': answer.finish_reason,
                         'stop_reason': answer.stop_reason, 'token_ids': answer.token_ids,
                         'prompt_token_ids': outputs[0].prompt_token_ids},
            'final_answer': final, 'post_request_route_delta': delta, 'export_proof_sha256': digest(proof_path),
            'new_prefill_owner_counts': observed_counts, 'expected_owner_counts': expected_counts,
            'actual_kv_capacity_sha256': digest(ROOT / 'actual-kv-capacity.json'),
            'package_identity_proof_sha256': digest(package_path),
            'serving_config_sha256': CONFIG_SHA, 'plugin_receipt_sha256': digest(ROOT / 'plugin-install/per-job-runtime.json'),
            'started_epoch': started, 'finished_epoch': time.time(),
            'scope': 'One complete answer from the uniform original 2142-unit E4M3/R1024 research reference; not allocator-derived, statistical quality, performance, concurrency or release promotion.'})
        assert all(checks.values()), checks
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == '__main__':
    main()
