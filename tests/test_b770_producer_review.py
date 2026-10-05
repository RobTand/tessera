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
import box_artifacts

@pytest.fixture
def selected():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import test_export_serving
    return test_export_serving
import test_t8_partition_launcher as launcher
from tessera import serving_parts as parts

ROOT = Path(__file__).resolve().parents[1]
@pytest.fixture
def frozen_source():
    return box_artifacts.skip_now("measurements", "b770-partition-producer-20261005", "producer-source-d735aa23b")


tensor_file = launcher.tensor_file
bundle = launcher.bundle


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


def test_real_installed_python_m_entrypoint(tmp_path, selected, frozen_source):
    """Real -m invocation, genuine ancestry, no canonical-module driver/alias."""
    repo = tmp_path / 'genuine'
    subprocess.run(['git', 'clone', '--quiet', '--shared', str(frozen_source), str(repo)], check=True)
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


@pytest.mark.parametrize('script', [launcher.T8_LAUNCHER, launcher.T16_LAUNCHER])
def test_empty_bound_refuses_before_authentication(tmp_path, script):
    """An explicitly empty PART_BOUND_S is a malformed bound, not an unset one."""
    done = subprocess.run(['bash', str(script), launcher.STUB, '0', '8'], cwd=ROOT,
        env=launcher._launch_env(tmp_path, PART_BOUND_S='', TESSERA_PRODUCER_PYTHON='/absent',
                                TESSERA_PRODUCER_SOURCE='/qualified/src/tessera'),
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 2
    assert 'PART_BOUND_S' in done.stdout + done.stderr


def test_empty_bound_refuses_in_the_profile_wrapper_before_any_work(tmp_path):
    out = tmp_path / 'profile'
    done = subprocess.run(['bash', str(ROOT / 'experiments/t8_census/profile_unit_encode.sh'), str(out),
        '--q256', '768', '--hessian', str(tmp_path / 'missing-capture')], cwd=ROOT,
        env=launcher._launch_env(tmp_path, PART_BOUND_S='', TESSERA_PRODUCER_PYTHON=sys.executable,
                                TESSERA_PRODUCER_SOURCE='/qualified/src/tessera'),
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 2
    assert 'PART_BOUND_S' in done.stdout + done.stderr
    assert not out.exists(), 'the refusal must come before the output directory is made'


def test_default_profile_authenticates_before_cuda_or_source_access(tmp_path, selected):
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
    tomli = pytest.importorskip("tomli", reason="the declared Python3.10 TOMLI backport is not installed in this population")
    from tessera.source_profiles import packaged_projection_config
    monkeypatch.setitem(sys.modules, 'tomli', tomli)
    real_import = builtins.__import__
    def without_tomllib(name, *args, **kwargs):
        if name == 'tomllib': raise ModuleNotFoundError('tomllib unavailable on Python 3.10')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_tomllib)
    assert packaged_projection_config(b'[project]\nname="tessera-quant"\n')['project']['name'] == 'tessera-quant'





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



def _refuses_the_installed_payload(text: str) -> bool:
    """Whether ``text`` is the exporter refusing an installed payload that differs
    from its source: the roster refusal or the byte-digest refusal."""
    return "the installed payload" in text and (
        "projected wheel roster" in text or "TESSERA_PRODUCER_SOURCE" in text)


@pytest.mark.parametrize("leg", ["interpreter", "payload"])
def test_default_profile_refuses_wrong_actual_producer(tmp_path, leg, selected, frozen_source):
    site = selected._fixture_install(tmp_path / "site")
    python = Path(sys.executable)
    if leg == "interpreter":
        python = tmp_path / "requested-python"
        python.write_text(f"#!{sys.executable}\nimport os, sys\nos.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n")
        python.chmod(0o755)
    done = subprocess.run(["bash", str(ROOT / "experiments/t8_census/profile_unit_encode.sh"),
        str(tmp_path / "profile"), "--q256", "768", "--hessian", str(tmp_path / "missing-capture")],
        cwd=ROOT, env=launcher._launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=python,
            TESSERA_PRODUCER_SOURCE=frozen_source / "src" / "tessera", PYTHONPATH=site),
        capture_output=True, text=True, timeout=90)
    text = done.stdout + done.stderr
    assert done.returncode != 0
    if leg == "interpreter":
        assert "TESSERA_PRODUCER_PYTHON" in text, f"{leg}: actual process never authenticated: {text}"
    else:
        # The installed package here is this repository's, the source is the frozen
        # genuine checkout. The roster check runs first and names the interpreter
        # variable when their file lists differ; the byte digest names the source
        # variable when the lists agree. Either refuses the installed payload, and
        # which one fires depends on the two file lists, not on this case. Each is
        # pinned on its own in test_export_serving.
        assert _refuses_the_installed_payload(text), \
            f"{leg}: actual process never authenticated: {text}"
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
