"""Real paired-arm scoring and lifecycle, driven through the serving supervisor.

The extracted producer function runs against a fake Docker store and HTTP
boundary. Real child signals exercise startup cleanup; scoring failure retains
its own status. The pre-existing-name, failed-start, thread-bound and cleanup
assertions remain. Legacy cidfile and CPU-mask units retain their old helper.
No container, port or GPU is used.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from box_artifacts import FLUX_RENDER_ROOT
from tools import _flux_serve as serve_policy

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(os.environ.get("TESSERA_PAIR_WRAPPER_UNDER_TEST",
                              ROOT / "experiments" / "moe_greedy_smoke_pair.sh"))
HELPER = ROOT / "experiments" / "owned_container.sh"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# A fake Docker: a container store under $FAKE_DOCKER_STATE, one JSON file per
# container id, plus an append-only log of every argv.  It answers only the
# subcommands the wrapper uses, and refuses anything else loudly, so a wrapper
# that starts calling something new cannot pass by silence.
FAKE_DOCKER = r'''#!/usr/bin/env python3
import hashlib, json, os, sys, time
from pathlib import Path
state = Path(os.environ["FAKE_DOCKER_STATE"])
state.mkdir(parents=True, exist_ok=True)
argv = sys.argv[1:]
with (state / "argv.log").open("a") as fh:
    fh.write(json.dumps(argv) + "\n")

def containers():
    return {p.stem: json.loads(p.read_text()) for p in state.glob("*.json")}

def resolve(ref):
    cs = containers()
    if ref in cs:
        return ref
    for cid, c in cs.items():
        if c["name"] == ref:
            return cid
    return None

def write(cid, rec):
    (state / f"{cid}.json").write_text(json.dumps(rec))

def event(line):
    with (state / "events.log").open("a") as fh:
        fh.write(line + "\n")

cmd = argv[0] if argv else ""
if cmd == "image" and argv[1] == "inspect":
    print(json.dumps([{"Id": "sha256:cccc", "RepoDigests": [os.environ["IMAGE"]]}]))
elif cmd == "create" or (cmd == "run" and "-d" in argv):
    name = next((a.split("=", 1)[1] for a in argv if a.startswith("--name=")), None)
    if name is None: name = argv[argv.index("--name") + 1]
    if resolve(name) is not None:
        print(f"Conflict. The container name /{name} is already in use", file=sys.stderr)
        sys.exit(125)
    cid = hashlib.sha256(f"{name}-{time.time_ns()}".encode()).hexdigest()
    write(cid, {"name": name, "running": False, "argv": argv})
    event(f"created {name} {cid}")
    if "--cidfile" in argv:
        # Docker writes the cidfile between create and start, so it is written
        # here -- before the start-failure branch below -- and not after it.
        Path(argv[argv.index("--cidfile") + 1]).write_text(cid)
    if cmd == "run":
        if os.environ.get("FAKE_DOCKER_START_FAILS") == "1":
            print("Error response from daemon: start failed", file=sys.stderr)
            sys.exit(125)
        rec = containers()[cid]; rec["running"] = True; write(cid, rec)
        event(f"started {name} {cid}")
    print(cid)
elif cmd == "start":
    cid = resolve(argv[1])
    if cid is None:
        sys.exit(1)
    if os.environ.get("FAKE_DOCKER_START_FAILS") == "1":
        print("Error response from daemon: start failed", file=sys.stderr)
        sys.exit(125)
    rec = containers()[cid]; rec["running"] = True; write(cid, rec)
    event(f"started {rec['name']} {cid}")
elif cmd == "stop":
    cid = resolve(argv[-1])
    if cid:
        rec = containers()[cid]; rec["running"] = False; write(cid, rec)
elif cmd == "rm":
    ref = argv[-1]
    cid = resolve(ref)
    if cid is None:
        sys.exit(1)
    event(f"removed {containers()[cid]['name']} {cid} by={ref}")
    (state / f"{cid}.json").unlink()
elif cmd == "container" and argv[1:2] == ["inspect"]:
    cid = resolve(argv[-1])
    if cid is None: sys.exit(1)
    print(json.dumps([{"Id": cid, "Image": "sha256:cccc", "State": {
        "Running": containers()[cid]["running"], "Pid": 123, "StartedAt": "fixture-start"}}]))
elif cmd == "inspect":
    cid = resolve(argv[-1])
    if cid is None:
        sys.exit(1)
    print("true" if containers()[cid]["running"] else "false")
elif cmd == "ps":
    want = None
    for i, a in enumerate(argv):
        if a in ("-f", "--filter"):
            want = argv[i + 1]
    for cid, c in containers().items():
        if not c["running"] and not any("a" in a for a in argv[1:] if a.startswith("-")):
            continue
        if want is None or want in (f"name=^/{c['name']}$", f"id={cid}"):
            print(cid)
elif cmd == "logs":
    sys.exit(0 if resolve(argv[-1]) is not None else 1)
else:
    print(f"fake docker: unexpected command {argv}", file=sys.stderr)
    sys.exit(64)
'''

# curl never succeeds: the serve under test never comes up, which is the state
# the wrapper is in when a signal or a start failure arrives.
FAKE_CURL = "#!/usr/bin/env bash\nexit 7\n"
# The wrapper polls on a 10 s cadence; the tests do not need to pay for it.
FAKE_SLEEP = "#!/usr/bin/env bash\nexec /usr/bin/sleep 0.05\n"
FAKE_NVIDIA_SMI = "#!/usr/bin/env bash\nexit 0\n"


def _serve_arm_source(text: str) -> str:
    """The wrapper's own serve_arm (and the cleanup it calls), verbatim."""
    lines = text.splitlines()
    arm = next(i for i, line in enumerate(lines) if line.startswith("serve_arm() {"))
    start = arm
    for i in range(arm, 0, -1):
        if lines[i].startswith("SERVE_ARM_TELE="):
            start = i
            break
    end = next(i for i in range(arm + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


def _driver(tmp_path: Path) -> Path:
    """A script that runs the real serve_arm with every other stage stubbed."""
    body = _serve_arm_source(WRAPPER.read_text())
    helper = f'source "{ROOT / "experiments/flux_serve.sh"}"\n'
    if 'source "$TS/experiments/owned_container.sh"' in WRAPPER.read_text():
        helper += f'source "{HELPER}"\n'
    script = tmp_path / "driver.sh"
    script.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
TS="{ROOT}"
PY=${{SCORE_PY:-python3}}
PORT=${{PORT:?the test must pin a port that binds nothing}}
NAME_PREFIX=ts198-smoke
IMAGE=${{IMAGE:?}}
UTIL=0.35
EXT="$OUT/ext"; VLLM_CACHE="$OUT/cache"; ARTIFACT="$OUT"; PROMPTS="$OUT/prompts.json"
mkdir -p "$OUT" "$EXT" "$VLLM_CACHE"
{helper}
require_memory() {{ :; }}
mem_avail_gib() {{ echo 99; }}
identity() {{ echo '{{}}'; }}
# The wrapper gates its image with runtime_image_require before it serves;
# these tests stub the gate, and `test_an_ungated_image_is_refused` is what
# proves the stub is standing in for a check that really does bite.
RUNTIME_IMAGE_JSON=${{RUNTIME_IMAGE_JSON-'{{"stub": true}}'}}
runtime_image_require() {{
  [ -n "$RUNTIME_IMAGE_JSON" ] || {{ echo runtime_image_require-required >&2; return 2; }}
}}
serve_lock_acquire() {{ SERVE_LOCK_TOKEN=test:1:2; export SERVE_LOCK_TOKEN; echo lock_acquired >> "$OUT/events"; }}
serve_lock_release() {{ [ -n "${{SERVE_LOCK_TOKEN:-}}" ] || return 0; unset SERVE_LOCK_TOKEN; echo lock_released >> "$OUT/events"; }}
serve_require_no_spec_decode() {{ return 0; }}
build_identity_docker_env() {{ :; }}
build_identity_stamp() {{ :; }}
telemetry() {{ while :; do sleep 1; done; }}
{body}
serve_arm bf16 "$OUT/model"
''')
    script.chmod(0o755)
    driver = tmp_path / 'supervise.py'
    driver.write_text('''#!''' + sys.executable + '''
import os,pathlib,sys
sys.path.insert(0, ''' + repr(str(ROOT)) + ''')
from tools import _flux_serve as policy
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
model=out/'model'; model.mkdir(); (model/'weights').write_text('independent weights')
(out/'peak').write_text('1234')
policy.decimal_job_id=lambda value: '12345'
policy.cpu_list=lambda: '2-5'
policy.container_peak_path=lambda info: out/'peak'
def models(url,timeout):
 if os.environ.get('FAKE_READY')!='1': raise OSError('startup not ready')
 return {'data':[]}
policy.fetch_models=models
req={'model':str(model),'plugin':None,'port':0,'util':'.35',
 'image':os.environ['IMAGE'],
 'options':['-v','/mnt/shared:/mnt/shared:ro','-v',f'{model}:{model}:ro','--entrypoint','vllm'],
 'command':['serve',str(model),'--served-model-name','kl-target','--host','0.0.0.0','--port','8000',
 '--max-model-len','4096','--max-num-seqs','8','--gpu-memory-utilization','0.35',
 '--max-logprobs','1024','--enforce-eager','--trust-remote-code']}
os.environ.update(FLUX_JOB_ID='fABC',CUDA_VISIBLE_DEVICES='0',FLUX_MEMORY_GIB='47',FLUX_SCRATCH=str(out/'scratch'))
sys.exit(policy.supervise(policy.make_spec(req),out/'receipt.json',['bash',
 ''' + repr(str(script)) + ''',str(out)]))
''')
    driver.chmod(0o755)
    return driver


def _env(tmp_path: Path) -> dict[str, str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name, text in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL),
                       ("sleep", FAKE_SLEEP), ("nvidia-smi", FAKE_NVIDIA_SMI)):
        path = bindir / name
        path.write_text(text)
        path.chmod(0o755)
    state = tmp_path / "docker-state"
    return dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        FAKE_DOCKER_STATE=str(state),
        IMAGE='test/serve@sha256:' + 'a' * 64,
        # Nothing binds here; the value only keeps the wrapper's curl URL and
        # -p flag off any port a real serve could be holding.
        PORT="0",
    )


def _store(env: dict[str, str]) -> dict[str, dict]:
    state = Path(env["FAKE_DOCKER_STATE"])
    return {p.stem: json.loads(p.read_text()) for p in state.glob("*.json")}


def _argvs(env: dict[str, str]) -> list[list[str]]:
    log = Path(env["FAKE_DOCKER_STATE"]) / "argv.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _events(env: dict[str, str]) -> list[str]:
    log = Path(env["FAKE_DOCKER_STATE"]) / "events.log"
    return log.read_text().splitlines() if log.exists() else []


def test_each_pair_arm_is_a_reviewed_job_and_refuses_without_flux(tmp_path):
    argv = serve_policy.render_serving(
        [str(WRAPPER), '--arm', 'bf16', 'source', 'artifact', 'seal', 'out'],
        {'gpu_memory_utilization': '.35'}, out=str(FLUX_RENDER_ROOT / 'moe-arm'),
        spec_path=tmp_path / 'spec.json')
    assert '--resources-shape=slot/[core=4;memory=47;gpu=1]' in argv
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('FLUX_', 'TESSERA_SERVE_'))}
    result = subprocess.run(['bash', str(WRAPPER), '--arm', 'bf16', 'source', 'artifact', 'seal', 'out'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'OUTSIDE_FLUX_JOB' in result.stderr


def test_a_scoring_failure_reaps_the_real_arm(tmp_path):
    env = _env(tmp_path)
    scorer = tmp_path / 'score'
    scorer.write_text('#!/bin/sh\ntouch "$SCORE_CALLED"\nexit 17\n')
    scorer.chmod(0o755)
    env.update(FAKE_READY='1', SCORE_PY=str(scorer), SCORE_CALLED=str(tmp_path / 'scored'))
    out = tmp_path / 'out'
    result = subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert (tmp_path / 'scored').exists(), result.stderr
    assert result.returncode == 17, result.stderr
    assert _store(env) == {}
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['removed'] and receipt['container_memory_peak_bytes'] == 1234


def test_parent_collects_both_arms_before_comparing(tmp_path):
    text = WRAPPER.read_text()
    start = text.index('# Submit each scored arm')
    calls = tmp_path / 'calls'
    driver = tmp_path / 'parent.sh'
    driver.write_text('''#!/bin/bash
set -euo pipefail
ONLY_ARM=
SOURCE=source; ARTIFACT=artifact; SEAL=seal; OUT=$PARENT_OUT
mkdir -p "$OUT"
flux_serve_score() { echo "${3}:collected" >> "$CALLS"; }
score() { echo compared >> "$CALLS"; }
PY=score; TS=tree
''' + text[start:])
    result = subprocess.run(['bash', str(driver)],
                            env=os.environ | {'CALLS': str(calls), 'PARENT_OUT': str(tmp_path / 'out')},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ['bf16:collected', 'tessera:collected', 'compared']


def test_a_failed_start_leaves_no_container(tmp_path):
    """A container whose start fails is an abort with a container on disk.

    The container exists the moment it is created, so the failure that stops
    the wrapper between creation and a healthy serve is precisely the one that
    used to strand it.
    """
    env = _env(tmp_path)
    env["FAKE_DOCKER_START_FAILS"] = "1"
    out = tmp_path / "out"
    proc = subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, "a start failure must not be reported as success"
    assert any(e.startswith("created ") for e in _events(env)), "no container was created"
    assert _store(env) == {}, f"leaked: {_store(env)}"


def test_a_signal_during_startup_reaps_the_container(tmp_path):
    """SIGTERM while waiting for the serve: EXIT alone never sees this."""
    env = _env(tmp_path)
    out = tmp_path / "out"
    proc = subprocess.Popen([str(_driver(tmp_path)), str(out)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 60
    while time.time() < deadline:  # wait on the event, never on a clock
        if any(e.startswith("started ") for e in _events(env)):
            break
        assert proc.poll() is None, "the driver exited before starting a container"
        time.sleep(0.05)
    else:
        pytest.fail("the container never started")
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    assert _store(env) == {}, f"leaked after SIGTERM: {_store(env)}"
    assert json.loads((out / 'receipt.json').read_text())['removed']


def test_cleanup_in_the_window_before_the_id_is_recorded_still_reaps(tmp_path):
    """The window `test_a_signal_during_startup_reaps_the_container` races for.

    `docker run -d --cidfile` creates the container and writes its id to the
    cidfile before it returns, and the id reaches `OWNED_CONTAINER_IDS` on the
    next line.  A trap runs at exactly that boundary, so cleanup can be called
    with a container that exists and an empty id list.  The signal test can
    only reach this state by winning a race -- it passed on an 8-shard fleet
    run and failed in CI -- so this one assembles the state instead: the
    cidfile registered as `owned_container_start` registers it, the container
    written as the fake docker writes it, and nothing in `OWNED_CONTAINER_IDS`.

    Without the cidfile scan, `owned_container_cleanup` returns on the empty
    array and the container survives.
    """
    env = _env(tmp_path)
    state = Path(env["FAKE_DOCKER_STATE"])
    state.mkdir(parents=True, exist_ok=True)
    cid = "b" * 64
    (state / f"{cid}.json").write_text(
        json.dumps({"name": "ts198-smoke-bf16", "running": True, "argv": ["run", "-d"]}))
    cidfile = tmp_path / "owned-container-1-0.cid"
    cidfile.write_text(cid)

    script = tmp_path / "window.sh"
    script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
source "{HELPER}"
OWNED_CONTAINER_CIDFILES=("{cidfile}")
OWNED_CONTAINER_IDS=()
owned_container_cleanup
""")
    script.chmod(0o755)
    proc = subprocess.run([str(script)], env=env, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert _store(env) == {}, f"leaked from the pre-record window: {_store(env)}"
    assert not cidfile.exists(), "cleanup left the cidfile behind"


def test_a_preexisting_name_is_refused_not_removed(tmp_path):
    """Someone else's container under our name is refused, never reaped."""
    env = _env(tmp_path)
    state = Path(env["FAKE_DOCKER_STATE"])
    state.mkdir(parents=True, exist_ok=True)
    squatter = "f" * 64
    (state / f"{squatter}.json").write_text(
        json.dumps({"name": "flux-12345", "running": True, "argv": ["someone-else"]}))
    out = tmp_path / "out"
    proc = subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, "a taken name must stop the run"
    assert squatter in _store(env), "the wrapper removed a container it did not create"
    assert not any(e.startswith("removed ") for e in _events(env))


def test_every_removal_addresses_an_id(tmp_path):
    """No `docker rm`/`logs`/liveness call may be made by name."""
    env = _env(tmp_path)
    env["FAKE_DOCKER_START_FAILS"] = "1"
    out = tmp_path / "out"
    subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                   capture_output=True, text=True, timeout=120)
    for argv in _argvs(env):
        if argv[:1] in (["rm"], ["logs"]) or argv[:1] == ["inspect"]:
            # A failed Docker run can register its unique job name before it
            # returns an id. The supervisor owns exactly this one name.
            assert HEX64.match(argv[-1]) or argv[-1] == 'flux-12345', argv


def test_the_container_is_given_explicit_thread_bounds(tmp_path):
    """Host exports do not cross into a container; these flags do."""
    env = _env(tmp_path)
    env["OWNED_CONTAINER_CPUSET"] = "2-5"
    env["FAKE_DOCKER_START_FAILS"] = "1"  # the create argv is already logged
    out = tmp_path / "out"
    subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                   capture_output=True, text=True, timeout=120)
    create = next(a for a in _argvs(env) if a[:1] in (["create"], ["run"]))
    assert '--cpuset-cpus=2-5' in create
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS"):
        assert f"{var}=1" in create, f"{var} is not explicitly bounded: {create}"


def test_the_wrapper_never_calls_docker_start(tmp_path):
    """PrismaBuild's docker shim refuses `docker start` (exit 125).

    Its `_starts_unowned_container` rejects the lifecycle forms that cannot be
    given PB's owner label, so a `docker create` + `docker start` spelling of
    ownership -- which is otherwise the obvious one -- cannot run under an
    admitted action at all.  The id has to come from `--cidfile` instead, and
    this test is what keeps it there.
    """
    env = _env(tmp_path)
    out = tmp_path / "out"
    proc = subprocess.Popen([str(_driver(tmp_path)), str(out)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        if any(e.startswith("started ") for e in _events(env)):
            break
        assert proc.poll() is None, "the driver exited before starting a container"
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    calls = _argvs(env)
    assert calls, "the wrapper made no docker calls"
    assert not any(a[:1] == ["start"] or a[:2] == ["container", "start"]
                   for a in calls), f"docker start is refused under PB: {calls}"


def test_an_ungated_image_is_refused(tmp_path):
    """No container starts before `runtime_image_require` has run (#100)."""
    if not HELPER.exists():
        pytest.skip("wrapper under test predates experiments/owned_container.sh")
    env = _env(tmp_path)
    env["RUNTIME_IMAGE_JSON"] = ""          # the gate did not run
    out = tmp_path / "out"
    proc = subprocess.run([str(_driver(tmp_path)), str(out)], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "runtime_image_require" in proc.stderr, proc.stderr
    assert _store(env) == {}, f"a container started ungated: {_store(env)}"


@pytest.mark.parametrize("mask,count", [
    ("0-19", 20), ("3", 1), ("0-3,8,10-11", 7), ("0,2,4", 3),
])
def test_the_cpu_count_is_read_off_the_mask(tmp_path, mask, count):
    """The thread count is the size of the granted mask, not a constant."""
    if not HELPER.exists():
        pytest.skip("wrapper under test predates experiments/owned_container.sh")
    proc = subprocess.run(
        ["bash", "-c", f'source "{HELPER}"; owned_container_cpu_count "{mask}"'],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(count)
