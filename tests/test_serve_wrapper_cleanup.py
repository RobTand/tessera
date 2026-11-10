"""Exercise the real teacher wrapper's exit paths without launching a serve.

The speculative-decoding gate is here too (tessera#247).  It used to be
``curl -s .../metrics | grep -q 'vllm:spec_decode'``, which cannot detect the
condition it owns: ``grep -q`` exits at the first match and closes the pipe, so
on a real metrics response -- the marker followed by thousands of further lines
-- curl fails its write, ``set -o pipefail`` makes the pipeline non-zero, the
``if`` reads false and the wrapper dumps anyway.  A failed fetch or an HTTP
error took the same accepting branch.  The stub below is therefore a curl that
honours ``-o`` and ``-w '%{http_code}'`` and can answer with a body larger than
a pipe buffer, because a stub that emits one marker line and exits cannot
exercise either failure.
"""
from pathlib import Path
import os
import shutil
import subprocess
from tools import _flux_serve as policy

WRAPPER = Path(__file__).resolve().parents[1] / "experiments" / "serve_and_dump_kl.sh"

#: A curl faithful enough to fail the way the real one does: it writes the
#: whole body where ``-o`` says, prints the code ``-w`` asks for, and can be
#: told to fail at the transport or answer an HTTP error.
CURL_STUB = r'''#!/bin/bash
out=""; wfmt=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -w) wfmt="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
emit() { if [ -n "$out" ]; then cat > "$out"; else cat; fi; }
bulk() { seq 1 20000 | sed 's/^/vllm:num_requests_running /'; }
case "$url" in
  */v1/models)
    [ "$READY" = 1 ] || exit 7
    echo '{"data":[{"id":"kl-target"}]}' | emit
    [ -z "$wfmt" ] || printf '200'
    exit 0 ;;
  */metrics)
    case "$METRICS" in
      unreachable) exit 7 ;;
      http_500)    echo 'internal error' | emit
                   [ -z "$wfmt" ] || printf '500'
                   exit 0 ;;
      empty)       : | emit
                   [ -z "$wfmt" ] || printf '200'
                   exit 0 ;;
      clean)       bulk | emit ;;
      spec_bulk)   { echo 'vllm:spec_decode_num_accepted_tokens_total 3'; bulk; } | emit ;;
      spec_one)    echo 'vllm:spec_decode' | emit ;;
      *)           echo "unknown METRICS=$METRICS" >&2; exit 90 ;;
    esac
    # curl reports a failed write, and dies on SIGPIPE, exactly as it does
    # when a downstream `grep -q` exits at the first match: that status is the
    # whole of tessera#247, so the stub must carry it rather than swallow it.
    st=$?
    [ "$st" -eq 0 ] || exit "$st"
    [ -z "$wfmt" ] || printf '200'
    exit 0 ;;
esac
exit 1
'''


def _run_wrapper(tmp_path, *, fail_remove=False, ready=False, metrics="clean"):
    import json
    import sys

    root = WRAPPER.parents[1]
    module = root / 'tools/_flux_serve.py'
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    wrapper = scripts / WRAPPER.name
    shutil.copyfile(WRAPPER, wrapper)
    shutil.copyfile(WRAPPER.parent / 'serve_metrics.sh', scripts / 'serve_metrics.sh')
    (scripts / 'flux_serve.sh').write_text(f'source "{root}/experiments/flux_serve.sh"\n')
    (scripts / 'runtime_image.sh').write_text('runtime_image_require() { :; }\n')
    (scripts / 'build_identity.sh').write_text('build_identity_docker_env() { :; }\n')
    image = 'test/serve@sha256:' + 'a' * 64
    commands = {
        'docker': r'''#!/usr/bin/env python3
import json,os,pathlib,sys
p=pathlib.Path(os.environ['TEST_ROOT']); a=sys.argv[1:]
if a[:2]==['image','inspect']:
 print(json.dumps([{'Id':'sha256:cccc','RepoDigests':[os.environ['TESSERA_KL_IMAGE']]}]))
elif a[0]=='run': (p/'live').touch(); print('b'*64)
elif a[:2]==['container','inspect']:
 if not (p/'live').exists(): sys.exit(1)
 print(json.dumps([{'Id':'b'*64,'Image':'sha256:cccc','State':{'Running':True,'Pid':123,'StartedAt':'now'}}]))
elif a[0]=='ps': print('b'*64 if (p/'live').exists() else '')
elif a[0]=='logs':
 with (p/'events').open('a') as f: f.write('logs\n')
 print('preserved serve log')
elif a[0]=='rm':
 with (p/'events').open('a') as f: f.write('remove\n')
 if os.environ['FAIL_REMOVE']=='1': sys.exit(1)
 (p/'live').unlink(missing_ok=True)
''',
        'curl': CURL_STUB,
        'score-python': '#!/bin/bash\nexit 91\n',
    }
    for name, body in commands.items():
        path = scripts / name
        path.write_text(body)
        path.chmod(0o755)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'weights').write_text('independent fixture')
    request = {'model': str(model), 'plugin': None, 'port': 8000, 'util': '0.85',
               'options': ['-v', '/mnt/shared:/mnt/shared:ro', '-v', f'{tmp_path}:{tmp_path}:ro'],
               'image': image, 'command': [str(model), '--served-model-name', 'kl-target',
                   '--host', '0.0.0.0', '--port', '8000', '--max-model-len', '4096',
                   '--max-num-seqs', '8', '--gpu-memory-utilization', '0.85',
                   '--max-logprobs', '1024', '--enforce-eager', '--trust-remote-code']}
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(policy.make_spec(request)))
    (tmp_path / 'peak').write_text('1234')
    driver = tmp_path / 'driver.py'
    driver.write_text('''import importlib.util,json,os,pathlib,sys
s=importlib.util.spec_from_file_location('serve_policy',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.container_peak_path=lambda info: pathlib.Path(os.environ['TEST_ROOT'])/'peak'
m.decimal_job_id=lambda value: '12345'
def models(url,timeout):
 if os.environ['READY']!='1': raise OSError('not ready')
 return {'data':[{'id':'kl-target'}]}
m.fetch_models=models
original=m.Serve.start
def start(self,request,**kwargs): return original(self,request,health_seconds=0)
m.Serve.start=start
sys.exit(m.supervise(json.loads(pathlib.Path(sys.argv[2]).read_text()),sys.argv[3],sys.argv[4:]))
''')
    env = os.environ | {
        'PATH': f'{scripts}:{os.environ["PATH"]}', 'TEST_ROOT': str(tmp_path),
        'FAIL_REMOVE': str(int(fail_remove)), 'READY': str(int(ready)), 'METRICS': metrics,
        'TESSERA_KL_LOGDIR': str(tmp_path), 'TESSERA_KL_IMAGE': image,
        'TESSERA_KL_PYTHON': str(scripts / 'score-python'),
        'TESSERA_KL_CORPUS': str(tmp_path / 'no-such-corpus.json'),
        'FLUX_JOB_ID': 'fABC', 'CUDA_VISIBLE_DEVICES': '0', 'FLUX_MEMORY_GIB': '108',
        'FLUX_SCRATCH': str(tmp_path / 'scratch'),
    }
    result = subprocess.run([sys.executable, str(driver), str(module), str(spec),
                             str(tmp_path / 'receipt.json'), 'bash', str(wrapper),
                             str(model), str(tmp_path / 'dump.json'), 'teacher'],
                            env=env, text=True, capture_output=True)
    events = tmp_path / 'events'
    return result, events.read_text().splitlines() if events.exists() else []


def test_teacher_wrapper_reaps_after_unexpected_exit(tmp_path):
    result, events = _run_wrapper(tmp_path)
    assert result.returncode == 2, result.stderr
    assert 'HEALTH_DEADLINE' in result.stderr
    assert not (tmp_path / 'live').exists(), 'health refusal left the serve live'
    assert not (tmp_path / 'serve.lock').exists()
    assert events == ['logs', 'remove']
    assert (tmp_path / 'serve_dump.log').read_text() == 'preserved serve log\n'


def test_teacher_wrapper_retains_ownership_when_reap_fails(tmp_path):
    import json
    result, events = _run_wrapper(tmp_path, fail_remove=True)
    assert result.returncode == 2, result.stderr
    assert (tmp_path / 'live').exists()
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['removed'] is False
    assert receipt['name'] == 'flux-12345'
    assert events == ['logs', 'remove']
    assert 'CLEANUP_UNVERIFIED' in receipt['cleanup_refusal']


def test_teacher_wrapper_refusal_cleanup_preserves_logs_once(tmp_path):
    result, events = _run_wrapper(tmp_path, ready=True, metrics='spec_one')
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / 'live').exists()
    assert events == ['logs', 'remove']
    assert (tmp_path / 'serve_dump.log').read_text() == 'preserved serve log\n'


def test_a_marker_followed_by_a_large_body_still_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='spec_bulk')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'spec-decode active' in result.stderr
    assert not (tmp_path / 'live').exists(), 'refusal left the serve live'


def test_an_unreachable_metrics_endpoint_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='unreachable')
    assert result.returncode == 2, result.stdout + result.stderr
    assert "could not read the serve's /metrics" in result.stderr


def test_an_http_error_from_metrics_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='http_500')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'answered HTTP 500' in result.stderr


def test_an_empty_metrics_body_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='empty')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'empty body' in result.stderr


def test_a_clean_metrics_response_passes_the_gate(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='clean')
    assert result.returncode == 3, result.stdout + result.stderr
    assert 'dump FAILED' in result.stdout
    metrics = tmp_path / 'serve_dump.metrics.txt'
    assert metrics.exists() and 'vllm:spec_decode' not in metrics.read_text()
