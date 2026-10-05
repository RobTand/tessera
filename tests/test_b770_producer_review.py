"""Consumer-visible PR950 review regressions; CPU control flow, not GPU proof."""
from __future__ import annotations

import builtins
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import test_export_serving as selected
import test_t8_partition_launcher as launcher
from tessera import serving_parts as parts

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = Path('/mnt/shared/tessera-measurements/b770-partition-producer-20261005/producer-source-d735aa23b')


def tensor_file(path, names):
    header = {n: {'dtype': 'BF16', 'shape': [1], 'data_offsets': [2*i, 2*i+2]}
              for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0\0' * len(names))


def bundle(tmp_path, producer, source_ref):
    source = tmp_path / 'checkpoint'
    source.mkdir()
    names = ['model.layers.0.norm.weight', 'model.layers.1.norm.weight', 'lm_head.weight']
    tensor_file(source / 'model.safetensors', names)
    (source / 'config.json').write_text(json.dumps({'architectures': ['Example']}))
    authority = tmp_path / 'producer_authority.py'
    authority.write_text('authority-placeholder\n')
    capture = tmp_path / 'capture.json'
    capture.write_text('{}\n')
    launcher.HESSIAN = capture
    stamp = launcher._full_stamp(tmp_path, producer, source_ref)
    out = tmp_path / 'census' / 'stubs' / f'parts-{launcher.STUB}' / 'part-0'
    out.mkdir(parents=True)
    owned = [n for n in names if parts.partition_owner(n, 8) == 0]
    tensor_file(out / 'model.safetensors', owned)
    (out / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {n: 'model.safetensors' for n in owned}}))
    (out / 'tessera_part_config.json').write_text(json.dumps({'architectures': ['Example'], 'quantization_config': {
        'quant_method': 'tessera', 'format': 'mixed-precision', 'config_groups': {}, 'ignore': []}}))
    receipt = stamp['producer_receipt']
    opts = {'plan': json.loads(launcher.PLAN.read_text()), 'hessian_sha256': parts.sha256_file(capture),
            'producer_authority_sha256': parts.sha256_file(authority), 'input_scales_sha256': None, 'encode_batch': 1}
    opts["plan_sha256"] = parts.sha256_file(launcher.PLAN)
    manifest = {'schema': parts.SCHEMA, 'producer': receipt, 'encode_batch': 1, 'modules': {},
                'plan_sha256': parts.sha256_file(launcher.PLAN),
                'totals': {'passthrough_bytes': 2 * len(owned)},
                'export_partition': {'schema': parts.SCHEMA, 'index': 0, 'count': 8, 'source_tensors': owned,
                    'identity': {'source': parts.source_part_identity(source), 'producer': receipt, 'options': opts},
                    'output_sha256': {'model.safetensors': parts.sha256_file(out / 'model.safetensors')}}}
    return source, out, stamp, manifest


def stub_producer(tmp_path):
    path = tmp_path / 'producer'
    # Stub only initial authentication and exporter; execute the actual owner code.
    path.write_text(f'#!{sys.executable}\n' + launcher.STUB_SCRIPT)
    path.chmod(0o755)
    return path


def run_part(tmp_path, producer, source_ref, source, **env):
    return launcher._run_t8(tmp_path, launcher._launch_env(tmp_path,
        TESSERA_PRODUCER_PYTHON=producer, TESSERA_PRODUCER_SOURCE=source_ref,
        SOURCE_CHECKPOINT=source, PYTHONPATH=ROOT / 'src', **env))


def test_real_installed_python_m_entrypoint(tmp_path):
    """Real -m invocation, genuine ancestry, no canonical-module driver/alias."""
    repo = tmp_path / 'genuine'
    subprocess.run(['git', 'clone', '--quiet', '--shared', str(FROZEN_SOURCE), str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'fixture'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'fixture@tessera'], check=True)
    shutil.rmtree(repo / 'src' / 'tessera')  # Only this test's newly created clone.
    shutil.copytree(ROOT / 'src' / 'tessera', repo / 'src' / 'tessera', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'pyproject.toml', repo / 'pyproject.toml')
    subprocess.run(['git', '-C', str(repo), 'add', 'src/tessera', 'pyproject.toml'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '--quiet', '--allow-empty', '-m', 'exact fixture payload'], check=True)
    site = selected._fixture_install(tmp_path / 'site')
    source = selected._write(tmp_path, selected._checkpoint(), selected._config())
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({selected.STACK: {'grid': 'E4M3', 'q256': 896}}))
    out = tmp_path / 'out'
    done = subprocess.run([sys.executable, '-m', 'tessera.export_serving',
        *selected._selected_export_argv(source, out, plan)], cwd=tmp_path,
        env=selected._selected_export_env(site, repo), capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    manifest = parts.read_serving_manifest(out / 'tessera_serving_manifest.json')
    assert manifest['producer']['git_head'] == selected._git('rev-parse', 'HEAD', cwd=repo).stdout.strip()
    assert manifest['plan_sha256'] == parts.sha256_file(plan)


@pytest.mark.parametrize('defect', ['empty_seals', 'missing_source', 'extra_source', 'missing_index_tensor',
    'extra_index_tensor', 'missing_actual_tensor', 'extra_actual_tensor', 'extra_shard',
    'producer_identity', 'batch_identity', 'index_float', 'count_bool', 'batch_bool'])
def test_real_skip_refuses_incomplete_owner_proof(tmp_path, defect):
    producer = stub_producer(tmp_path)
    source_ref = tmp_path / 'qualified' / 'src' / 'tessera'
    source_ref.mkdir(parents=True)
    source, out, stamp, manifest = bundle(tmp_path, producer, source_ref)
    part = manifest['export_partition']
    index_path = out / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    if defect == 'empty_seals': part['output_sha256'] = {}
    elif defect == 'missing_source': part['source_tensors'].pop()
    elif defect == 'extra_source': part['source_tensors'].append('model.layers.1.norm.weight')
    elif defect == 'missing_index_tensor': index['weight_map'].pop('lm_head.weight')
    elif defect == 'extra_index_tensor': index['weight_map']['extra'] = 'model.safetensors'
    elif defect in ('missing_actual_tensor', 'extra_actual_tensor'):
        held = part['source_tensors'][:-1] if defect == 'missing_actual_tensor' else [*part['source_tensors'], 'extra']
        tensor_file(out / 'model.safetensors', held)
        part['output_sha256']['model.safetensors'] = parts.sha256_file(out / 'model.safetensors')
    elif defect == 'extra_shard': tensor_file(out / 'unindexed.safetensors', ['extra'])
    elif defect == 'producer_identity': part['identity']['producer'] = {'foreign': True}
    elif defect == 'batch_identity': part['identity']['options']['encode_batch'] = 2
    elif defect == 'index_float': part['index'] = 0.0
    elif defect == 'count_bool': part['count'] = 8.0
    elif defect == 'batch_bool': manifest['encode_batch'] = True
    index_path.write_text(json.dumps(index))
    launcher._write_marker(tmp_path, stamp, json.dumps(manifest).encode())
    done = run_part(tmp_path, producer, source_ref, source)
    assert done.returncode != 0, f'{defect}: invalid completed part was skipped: {done.stdout} {done.stderr}'
    assert 'already done and verified' not in done.stdout


def image_cli(tmp_path):
    path = tmp_path / "image-cli"
    path.write_text(f"#!{sys.executable}\n" +
        "import json, subprocess, sys\n"
        "a=sys.argv[1:]\n"
        "if a[:2] == ['-m', 'tessera.serving.runtime_image']:\n"
        "    if a[2] == 'resolve': print(json.dumps({'resolved_digest': 'sha256:'+'ab'*32, 'resolved_reference': a[-1], 'local_id': 'fixture'}))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(subprocess.call([sys.executable, *a]))\n")
    path.chmod(0o755)
    return path


def test_completion_without_manifest_returns_failure(tmp_path):
    producer = stub_producer(tmp_path)
    source_ref = tmp_path / 'qualified' / 'src' / 'tessera'
    source_ref.mkdir(parents=True)
    capture = tmp_path / 'capture.json'
    capture.write_text('{}')
    launcher.HESSIAN = capture
    done = run_part(tmp_path, producer, source_ref, tmp_path / 'unused', RUNTIME_IMAGE_PY=image_cli(tmp_path))
    assert 'rc=0' in done.stdout, 'fixture must reach the actual exporter completion path'
    assert done.returncode != 0, 'exporter rc0 without its sealed manifest must not report completion'
    assert not (tmp_path / 'census' / 'stubs' / f'parts-{launcher.STUB}' / 'part-0.done.json').exists()


@pytest.mark.parametrize('script', [launcher.T8_LAUNCHER, launcher.T16_LAUNCHER])
def test_zero_bound_refuses_before_authentication(tmp_path, script):
    done = subprocess.run(['bash', str(script), launcher.STUB, '0', '8'], cwd=ROOT,
        env=launcher._launch_env(tmp_path, PART_BOUND_S=0, TESSERA_PRODUCER_PYTHON='/absent',
                                TESSERA_PRODUCER_SOURCE='/qualified/src/tessera'),
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 2
    assert 'PART_BOUND_S' in done.stdout + done.stderr


def test_default_profile_authenticates_before_cuda_or_source_access(tmp_path):
    out = tmp_path / 'profile'
    done = subprocess.run(['bash', str(ROOT / 'experiments/t8_census/profile_unit_encode.sh'), str(out),
        '--q256', '768', '--hessian', str(tmp_path / 'missing-capture')], cwd=ROOT,
        env=launcher._launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable,
            TESSERA_PRODUCER_SOURCE='/missing/genuine/src/tessera', PYTHONPATH=ROOT / 'src'),
        capture_output=True, text=True, timeout=90)
    assert done.returncode != 0
    assert 'TESSERA_PRODUCER_SOURCE' in done.stdout + done.stderr
    assert 'a GPU measurement with no GPU' not in done.stdout + done.stderr


def test_provisioner_never_deletes_preexisting_staging():
    """Safety pre-fix proof is static: forbidden to execute the unsafe provisioner."""
    import ast
    tree = ast.parse((ROOT / 'tools/provision_producer_env.py').read_text())
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.If)
           and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute)
           and n.test.func.attr == 'exists' and any(
               isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'rmtree'
               for statement in n.body for c in ast.walk(statement))]
    assert not bad, f'pre-existing path recursively deleted at lines {bad}'


def test_projection_uses_toml_backport_without_tomllib(monkeypatch):
    from pip._vendor import tomli
    from tessera.source_profiles import packaged_projection_config
    monkeypatch.setitem(sys.modules, 'tomli', tomli)
    real_import = builtins.__import__
    def without_tomllib(name, *args, **kwargs):
        if name == 'tomllib': raise ModuleNotFoundError('tomllib unavailable on Python 3.10')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_tomllib)
    assert packaged_projection_config(b'[project]\nname="tessera-quant"\n')['project']['name'] == 'tessera-quant'


@pytest.mark.parametrize("mode", ["skipped_anchor", "tail", "shared_split", "timed_mismatch"])
def test_unanchored_shape_cannot_enter_timed_population(tmp_path, monkeypatch, mode):
    sys.path.insert(0, str(ROOT / 'experiments' / 't8_census'))
    import ab_batched_best_form as ab
    from tessera import export_serving, export
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    stack = 'model.language_model.layers.3.mlp.experts'
    population = 2 if mode == 'skipped_anchor' else 5 if mode == 'tail' else 8
    units = [{'stack': stack, 'rows': 2, 'cols': 2, 'projection': 'gate_proj',
              'tensor': f'{stack}.{i}.gate_proj.weight'} for i in range(population)]
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {u['tensor']: 'source.safetensors' for u in units}}))
    monkeypatch.setattr(export_serving, 'authenticate_producer_python', lambda: {'qualified': True})
    monkeypatch.setattr(ab.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ab.torch.cuda, 'get_device_name', lambda _: 'CPU control-flow fixture')
    monkeypatch.setattr(ab, 'load_producer_authority', lambda _: (None, None))
    monkeypatch.setattr(export.ActivationSource, 'from_capture', lambda *a, **k: None)
    monkeypatch.setattr(ab, 'quantizable', lambda _: (None, None, None, {}))
    monkeypatch.setattr(ab, 'expert_stacks', lambda _: {stack: [0, 1]})
    monkeypatch.setattr(ab, 'plan_expert_stack', lambda *a, **k: {'units': units})
    monkeypatch.setattr(ab, 'grid_for', lambda _: 'grid')
    monkeypatch.setattr(ab, 'served_recipe', lambda *a: None)
    monkeypatch.setattr(ab, 'bind_source', lambda *a: {'shards': [], 'digest_s': 0, 'cache': {},
        'cached_shards': [], 'hashed_shards': [], 'receipt': {}, 'identity': {'files': {'source.safetensors': 'a'*64}}})
    monkeypatch.setattr(ab.time, 'perf_counter', lambda: 0)
    anchors = []
    def anchor(positions, *a, **k):
        anchors.extend(positions)
        return {'digests': {units[i]['tensor']: 'a'*64 for i in positions}, 'units': len(positions),
                'wall_s': 1, 's_per_unit': 0.5, 'start_utc': 'fixture', 'end_utc': 'fixture', 'power': {}}
    monkeypatch.setattr(ab, 'anchor_units', anchor)
    calls = []
    def run(positions, *a, **k):
        calls.append(positions)
        digests = {units[i]['tensor']: 'a'*64 for i in positions}
        if mode == 'timed_mismatch' and a[-1].endswith('-b000'):
            digests[units[positions[0]]['tensor']] = 'b'*64
        widths = ([2, 2] if positions[0] == 0 else [1, 3]) if mode == 'shared_split' else [len(positions)]
        return {'positions': positions, 'units': len(positions), 'key': '2x2', 'widths_observed': widths,
                'digests': digests, 'wall_s': 1, 's_per_unit': 0.5, 'power': {},
                'start_utc': 'fixture', 'end_utc': 'fixture', 'workload': 'encode+frame+verify'}
    monkeypatch.setattr(ab, 'run_owner_batch', run)
    batch, budget = ('2', '35') if mode == 'skipped_anchor' else ('4', '1000')
    monkeypatch.setattr(sys, 'argv', ['probe', str(tmp_path / 'out'), '--src', str(source), '--rungs', 'E4M3:768',
        '--hessian', 'h', '--producer-authority', 'a', '--batch', batch, '--budget-s', budget, '--no-profile'])
    result = ab.main()
    packet = json.loads((tmp_path / 'out' / 'ab_batched_best_form.json').read_text())
    row = packet['rungs']['E4M3:768']
    if mode == 'skipped_anchor':
        assert not calls, 'skipped warm+anchor cohort still executed a timed batch'
        assert not row['batches']
        assert result != 0, 'unqualified population must not be reported as accepted'
    elif mode == 'timed_mismatch':
        assert result == 4, 'timed digests were discarded instead of compared'
        assert not row['batches']
    else:
        assert set(anchors) == set(range(population)), 'tail/shared schedule members were not independently anchored'
        assert row['batches'] and all('digests' in b for b in row['batches']), 'timed digests must be retained'


def test_completion_binds_consumed_input_identity(tmp_path):
    producer = stub_producer(tmp_path)
    source_ref = tmp_path / 'qualified' / 'src' / 'tessera'
    source_ref.mkdir(parents=True)
    source, out, stamp, manifest = bundle(tmp_path, producer, source_ref)
    scales = tmp_path / 'input_scales.safetensors'
    scales.write_bytes(b'consumed scales')
    manifest['export_partition']['identity']['options']['input_scales_sha256'] = parts.sha256_file(scales)
    (out / 'tessera_serving_manifest.json').write_text(json.dumps(manifest))
    prepared = tmp_path / 'prepared'
    shutil.copytree(out, prepared)
    code = producer.read_text().replace(
        "    raise SystemExit(0)\nif args[:1] == ['-c']:",
        "    import shutil\n    shutil.copytree(os.environ['PREPARED_PART'], args[3])\n"
        "    open(os.environ['INPUT_SCALES'], 'wb').write(b'mutated after export')\n"
        "    raise SystemExit(0)\nif args[:1] == ['-c']:")
    producer.write_text(code)
    done = run_part(tmp_path, producer, source_ref, source, INPUT_SCALES=scales,
                    PREPARED_PART=prepared, RUNTIME_IMAGE_PY=image_cli(tmp_path))
    assert 'rc=0' in done.stdout, 'fixture must reach exporter completion'
    assert done.returncode != 0, 'same-path input mutation must not publish fresh false completion'
    assert 'consumed.input_scales_sha256' in done.stdout + done.stderr
    assert not (out.parent / 'part-0.done.json').exists()



@pytest.mark.parametrize("leg", ["interpreter", "payload"])
def test_default_profile_refuses_wrong_actual_producer(tmp_path, leg):
    site = selected._fixture_install(tmp_path / "site")
    python = Path(sys.executable)
    if leg == "interpreter":
        python = tmp_path / "requested-python"
        python.write_text(f"#!{sys.executable}\nimport os, sys\nos.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n")
        python.chmod(0o755)
    done = subprocess.run(["bash", str(ROOT / "experiments/t8_census/profile_unit_encode.sh"),
        str(tmp_path / "profile"), "--q256", "768", "--hessian", str(tmp_path / "missing-capture")],
        cwd=ROOT, env=launcher._launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=python,
            TESSERA_PRODUCER_SOURCE=FROZEN_SOURCE / "src" / "tessera", PYTHONPATH=site),
        capture_output=True, text=True, timeout=90)
    text = done.stdout + done.stderr
    assert done.returncode != 0
    field = "TESSERA_PRODUCER_PYTHON" if leg == "interpreter" else "TESSERA_PRODUCER_SOURCE"
    assert field in text, f"{leg}: actual process never authenticated: {text}"
    assert "a GPU measurement with no GPU" not in text


def test_completion_refuses_formatting_only_plan_mutation(tmp_path):
    producer = stub_producer(tmp_path)
    source_ref = tmp_path / 'qualified' / 'src' / 'tessera'
    source_ref.mkdir(parents=True)
    source, out, _stamp, manifest = bundle(tmp_path, producer, source_ref)
    plan = tmp_path / 'experiments' / 't8_census' / f'plan-{launcher.STUB}.json'
    plan.parent.mkdir(parents=True)
    shutil.copy2(launcher.PLAN, plan)
    (tmp_path / 'experiments' / 'runtime_image.sh').write_bytes((ROOT / 'experiments' / 'runtime_image.sh').read_bytes())
    (out / 'tessera_serving_manifest.json').write_text(json.dumps(manifest))
    prepared = tmp_path / 'prepared'
    shutil.copytree(out, prepared)
    code = producer.read_text().replace(
        "    raise SystemExit(0)\nif args[:1] == ['-c']:",
        "    import shutil\n    shutil.copytree(os.environ['PREPARED_PART'], args[3])\n"
        "    with open(os.environ['PLAN_TO_MUTATE'], 'ab') as f: f.write(b'\\n')\n"
        "    raise SystemExit(0)\nif args[:1] == ['-c']:")
    producer.write_text(code)
    done = subprocess.run(['bash', str(launcher.T8_LAUNCHER), launcher.STUB, '0', '8'], cwd=tmp_path,
        env=launcher._launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=producer,
            TESSERA_PRODUCER_SOURCE=source_ref, SOURCE_CHECKPOINT=source, PYTHONPATH=ROOT / 'src',
            PREPARED_PART=prepared, PLAN_TO_MUTATE=plan, RUNTIME_IMAGE_PY=image_cli(tmp_path)),
        capture_output=True, text=True, timeout=90)
    assert 'rc=0' in done.stdout, 'fixture must reach exporter completion'
    assert done.returncode != 0, 'raw plan mutation with equal parsed allocation published false completion'
    assert 'consumed.plan_sha256' in done.stdout + done.stderr
    assert not (out.parent / 'part-0.done.json').exists()
