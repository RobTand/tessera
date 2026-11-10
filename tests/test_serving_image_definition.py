"""The image job keeps all patches as files and requests no GPU."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_import_search_path():
    before = sys.path[:]
    try:
        yield
    finally:
        sys.path[:] = before


def _builder():
    spec = importlib.util.spec_from_file_location("image_builder", ROOT / "tools/build_serving_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_uses_the_repository_dockerfile_and_no_gpu(tmp_path):
    manifest = json.loads((ROOT / "images/serving/glm53-sm121.json").read_text())
    job = _builder().build_job(manifest, "example/serving:test", tmp_path / "image-output")
    assert job["gpu"] is False and job["submitted"] is False
    assert "--gpu" not in job["payload"] and "gpu" not in job["request"]
    assert job["request"] == {"architecture": "aarch64", "cores": 4, "memory_gib": 8, "timeout_s": 1790}
    assert "filename=images/serving/Dockerfile" in job["builder_command"]
    dockerfile = (ROOT / "images/serving/Dockerfile").read_text()
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "<<" not in dockerfile
    assert "TESSERA_KERNEL_BUILD" in dockerfile
    for name in manifest["patches"]:
        source = (ROOT / "images/serving/patches" / name).read_text()
        compile(source, name, "exec")
    patch = ast.parse((ROOT / "images/serving/patches/glm53_sm121.py").read_text())
    assert len([node for node in ast.walk(patch) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "replace_once"]) == 8


def test_dry_run_does_not_start_a_build(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/build_serving_image.py"),
        "--manifest", str(ROOT / "images/serving/glm53-sm121.json"),
        "--image-tag", "example/serving:test", "--output-directory", str(tmp_path / "image-output"),
        "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job = json.loads(result.stdout)
    assert job["submitted"] is False and job["gpu"] is False


def _manifest():
    return json.loads((ROOT / "images/serving/glm53-sm121.json").read_text())


def test_payload_uses_an_owned_builder_without_a_host_socket(tmp_path):
    manifest = _manifest()
    job = _builder().build_job(manifest, "example/serving:test", tmp_path)
    assert job["payload"][:2] == ["tools/build_serving_image.py", "--run"]
    assert "--inside-action" not in job["payload"]
    assert job["builder_command"][:2] == ["docker", "run"]
    assert not any("docker.sock" in value for value in job["builder_command"])
    assert manifest["builder_image"] in job["builder_command"]
    assert "type=docker,name=example/serving:test,dest=/out/serving-image.tar" in job["builder_command"]


def test_the_job_names_no_prismabuild_runner(tmp_path):
    builder = _builder()
    job = builder.build_job(_manifest(), "example/serving:test", tmp_path)
    text = json.dumps(job).lower()
    assert "pbrun" not in text and "prismabuild" not in text and "--priority" not in text
    assert not hasattr(builder, "PB_RUN")


@pytest.mark.parametrize("architecture,expected", [("x86", "x86_64"), ("aarch64", "aarch64")])
def test_the_request_is_the_suite_transports_stock_demand_and_refuses_more(tmp_path, architecture, expected):
    builder = _builder()
    job = builder.build_job({**_manifest(), "architecture": architecture}, "t:1", tmp_path, cpus=2, memory_gb=4)
    assert job["request"] == {"architecture": expected, "cores": 2, "memory_gib": 4, "timeout_s": 1790}
    for demand in ({"cpus": 5}, {"memory_gb": 17}, {"cpus": 0}):
        with pytest.raises(ValueError, match="native suite profile"):
            builder.build_job(_manifest(), "t:1", tmp_path, **demand)


def test_submit_refuses_before_the_suite_transport(tmp_path, monkeypatch):
    builder = _builder()
    job = builder.build_job(_manifest(), "t:1", tmp_path)
    seen = {}
    import _suite_flux
    monkeypatch.setattr(_suite_flux, "submit", lambda *a, **k: seen.update(args=a, kwargs=k) or {"state": "running"})
    with pytest.raises(ValueError, match="root Docker store.*lead"):
        builder.submit(job, tmp_path / "launch", tmp_path / "checkout", "/named/python")
    assert seen == {}
    assert not (tmp_path / "launch").exists()


def test_submit_cli_refuses_before_staging_or_transport(tmp_path, monkeypatch, capsys):
    builder = _builder()
    calls = []
    monkeypatch.setattr(builder._suite_flux, 'submit', lambda *a, **k: calls.append(a))
    code = builder.main(['--submit', '--manifest-json', json.dumps(_manifest()),
                         '--image-tag', 't:1', '--output-directory', str(tmp_path / 'out'),
                         '--launch-directory', str(tmp_path / 'launch'), '--checkout', str(ROOT)])
    assert code == 2
    assert 'root Docker store' in capsys.readouterr().err
    assert calls == [] and not (tmp_path / 'launch').exists() and not (tmp_path / 'out').exists()


@pytest.mark.parametrize('identity,expected', [('fAbc-123', 'fAbc123'), ('', '')])
def test_container_name_derivation_needs_no_flux_process(identity, expected):
    assert _builder().container_name(identity) == 'tessera-image-build-' + expected


@pytest.mark.parametrize('job_value', [None, '', 'f123'])
def test_run_refuses_without_a_reviewed_build_path(tmp_path, job_value):
    docker = tmp_path / 'docker'
    docker.write_text('#!/bin/sh\ntouch "$CONTROL/docker-reached"\nexit 0\n')
    docker.chmod(0o755)
    env = {**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}',
           'TMPDIR': str(tmp_path), 'CONTROL': str(tmp_path)}
    env.pop('FLUX_JOB_ID', None)
    if job_value is not None:
        env['FLUX_JOB_ID'] = job_value
    result = subprocess.run([sys.executable, str(ROOT / 'tools/build_serving_image.py'),
                             '--run', '--manifest-json', json.dumps(_manifest()),
                             '--image-tag', 't:1', '--output-directory', str(tmp_path / 'out')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'root Docker store' in result.stderr and 'reviewed' in result.stderr
    assert not (tmp_path / 'docker-reached').exists()
    assert not (tmp_path / 'out').exists()
    assert 'Traceback' not in result.stderr


def test_imported_patch_licenses_reach_the_image():
    license_root = ROOT / "images/serving/patches"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (license_root / "LICENSE").read_text()
    assert "GNU Affero General Public License v3.0" in (license_root / "LICENSE.MIT").read_text()
    dockerfile = (ROOT / "images/serving/Dockerfile").read_text()
    assert "COPY images/serving/patches/LICENSE images/serving/patches/LICENSE.MIT /opt/tessera-image/licenses/" in dockerfile


def _patch_case(tmp_path):
    source = (ROOT / "images/serving/patches/glm53_sm121.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in tree.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "replace_once"]
    edits = [(ast.literal_eval(node.value.args[0]), ast.literal_eval(node.value.args[1])) for node in calls]
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id.startswith("old_")}
    pairs = next(ast.literal_eval(node.iter) for node in tree.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Tuple))
    files = {
        "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "\n".join(old for old, new in edits),
        "models/glm5next/nvidia/model.py": constants["old_width"],
        "models/glm5next/nvidia/mtp.py": constants["old_width"],
        "v1/attention/backends/mla/flashinfer_mla_sparse.py": constants["old_block_sizes"],
        "platforms/cuda.py": constants["old_align"] + constants["old_pdl"],
        "model_executor/layers/sparse_attn_indexer_kpool.py": "\n".join(old for old, new in pairs),
        "model_executor/warmup/kernel_warmup.py": constants["old_sparse_warmup"] + constants["old_autotune"],
    }
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    source = source.replace("/usr/local/lib/python3.12/dist-packages/vllm", str(tmp_path))
    return source, edits


def test_extracted_patch_applies_every_edit(tmp_path):
    source, edits = _patch_case(tmp_path)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    target = tmp_path / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    assert all(new in target.read_text() for old, new in edits)
    assert "return major in (9, 10)" in (tmp_path / "platforms/cuda.py").read_text()
    assert "pool_ids[:, : select_k - 1]" in (tmp_path / "model_executor/layers/sparse_attn_indexer_kpool.py").read_text()


def test_extracted_patch_refuses_a_missing_target(tmp_path):
    source, edits = _patch_case(tmp_path)
    target = tmp_path / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    original = target.read_text().replace(edits[0][0], "", 1)
    target.write_text(original)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode != 0 and "expected exactly one patch target" in result.stderr
    assert target.read_text() == original


@pytest.mark.parametrize('record,expected', [
    ({'state': 'running'}, 1), ({'state': 'unknown'}, 1),
    ({'state': 'terminal', 'returncode': 7}, 1),
    ({'state': 'terminal', 'returncode': 0, 'evidence_errors': ['missing evidence']}, 1),
    ({'state': 'terminal', 'returncode': 0, 'evidence_errors': []}, 0),
])
def test_collect_requires_terminal_success_with_complete_evidence(tmp_path, monkeypatch, record, expected):
    builder = _builder()
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    import _suite_flux
    seen = []
    monkeypatch.setattr(_suite_flux, 'collect', lambda path: seen.append(path) or record)
    assert builder.main(['--collect', str(tmp_path)]) == expected
    assert seen == [tmp_path]


def test_repeated_job_construction_does_not_grow_import_search_path(tmp_path):
    builder = _builder()
    before = list(sys.path)
    builder.build_job(_manifest(), 't:1', tmp_path)
    builder.build_job(_manifest(), 't:2', tmp_path)
    assert sys.path == before
