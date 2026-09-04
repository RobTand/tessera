"""One-shot PB-only LFM census/student stages against a sealed merged artifact.

Run `census` and `student` as separate exclusive-GPU PrismaBuild actions, each
under an outer timeout with cleanup grace. Fixed paths identify this campaign;
an existing stage directory is a refusal, never permission to overwrite it.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd())]
from tessera.serving_parts import source_identity, sha256_file
from experiments.ts5_stage_cleanup import cleanup_stage


def campaign_stage_paths(stage, attempt):
    """An explicit retry keeps all prior outputs and container identities."""
    if attempt < 1:
        raise ValueError("campaign attempt must be positive")
    return (CAMPAIGN / f"{stage}-bound-r{attempt}",
            f"ts5-lfm-r2-{stage}-bound-r{attempt}",
            Path(f"/home/rob/tmp/ts5-lfm-r2-final-{stage}-output-r{attempt}"))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("census", "student"))
parser.add_argument("--attempt", type=int, default=1)
args = parser.parse_args()
stage = args.stage
CAMPAIGN = Path("/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2")
MODEL = CAMPAIGN / "full-model"
OUT, NAME, LOCAL_OUTPUT = campaign_stage_paths(stage, args.attempt)
TEACHER = CAMPAIGN / "teacher-bound-r1"
CORPUS = Path("/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json")
IMAGE = "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
PY = "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python"
EXT = "/home/rob/tmp/ts5-lfm-r2-final-ext"

def interrupted(signum, _frame):
    raise TimeoutError(f"{stage} action interrupted by signal {signum}")

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
OUT.mkdir()
stop = threading.Event()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def capture(command):
    return subprocess.check_output(command, text=True, timeout=30).strip()

def gpu_processes():
    return capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])

def telemetry():
    with (OUT / "telemetry.log").open("w") as log:
        while not stop.is_set():
            log.write(datetime.datetime.now().astimezone().isoformat() + "\n")
            try:
                log.write(capture(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"]) + "\n")
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:")):
                        log.write(line + "\n")
            except Exception as exc:
                log.write("telemetry_error=" + repr(exc) + "\n")
            log.flush()
            stop.wait(10)

monitor = threading.Thread(target=telemetry, daemon=True)
monitor.start()
completed = False
launched = False
try:
    assert not capture(["docker", "ps", "-aq", "--filter", f"name=^/{NAME}$"]), "unique container exists"
    assert not gpu_processes(), "previous GPU stage has not finished cleanup"
    seal_path = CAMPAIGN / "merge-action-r1/artifact-seal.json"
    seal = json.loads(seal_path.read_text())
    pre = source_identity(MODEL)
    assert pre == seal["checkpoint_identity"], "merged checkpoint differs from checked assembly"
    assert sha256_file(CORPUS) == "2fdd48eeab69109c6222ef2f857815d2b35d5422747815c495e0467712751d44"
    commit = capture(["git", "rev-parse", "HEAD"])
    evidence = {"schema": "tessera.lfm-served-artifact-binding/1", "stage": stage,
                "checkpoint_pre": pre, "assembly_seal_sha256": sha256_file(seal_path),
                "runtime_image": IMAGE, "execution_mode": "eager", "residency": "resident",
                "source_snapshot": commit, "corpus_sha256": sha256_file(CORPUS),
                "model_mount_read_only": True}
    write("artifact-before.json", evidence)
    env = os.environ.copy()
    env.update({"TS": str(Path.cwd()), "RUNS": str(OUT), "EXT": EXT})
    if stage == "census":
        local = LOCAL_OUTPUT
        local.mkdir()
        env["IMG"] = IMAGE
        command = ["experiments/tessera_plugin_run.sh", "--name", NAME,
                   "--memory=64g", "--memory-swap=64g", "--ipc=host",
                   "-e", "TESSERA_SERVE_MODE=resident", "-v", "/mnt/shared:/mnt/shared:ro",
                   "-v", f"{MODEL}:{MODEL}:ro", "-v", f"{local}:/census", "--",
                   f"python3 tools/tessera_route_census.py {MODEL} /census/census.json "
                   f"--runtime-image {IMAGE} --tessera-commit {commit} "
                   "--prompt-tokens 64 --max-model-len 512 --gpu-memory-utilization 0.35"]
    else:
        binding_path = TEACHER / "source-bound-result.json"
        binding = json.loads(binding_path.read_text())
        assert binding["source_pre"] == binding["source_post"] == seal["export_identity"]["source"]
        assert binding["runtime_image"] == IMAGE and binding["execution_mode"] == "eager"
        for name, sha in binding["outputs"].items():
            assert sha256_file(TEACHER / name) == sha, f"teacher output changed: {name}"
        evidence["teacher_binding_sha256"] = sha256_file(binding_path)
        env.update({"IMAGE": IMAGE, "VLLM_CACHE": "/home/rob/tmp/ts5-lfm-r2-final-vllm-cache",
                    "TESSERA_KL_PORT": "8151", "TESSERA_KL_NAME": NAME,
                    "TESSERA_KL_CORPUS": str(CORPUS),
                    "TESSERA_KL_TEACHER": str(TEACHER / "teacher_bf16.json"),
                    "TESSERA_KL_DUMP": str(OUT / "student_tessera.json"),
                    "TESSERA_KL_LOG": str(OUT / "serve_student.log"),
                    "TESSERA_GPU_MEM_UTIL": "0.35", "TESSERA_KL_TOPK": "1024",
                    "TESSERA_LANE_EAGER": "1",
                    "TESSERA_LANE_DOCKER_EXTRA": f"--memory=64g --memory-swap=64g -v {MODEL}:{MODEL}:ro"})
        command = ["experiments/tessera_plugin_served.sh", str(MODEL), "ts5lfm", "resident"]
    with (OUT / "action.log").open("w") as log:
        launched = True  # Ownership starts only when the launch can begin.
        subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    post = source_identity(MODEL)
    assert post == pre == seal["checkpoint_identity"], "assembled checkpoint changed across serve"
    assert sha256_file(CORPUS) == evidence["corpus_sha256"], "corpus changed across serve"
    outputs = []
    if stage == "census":
        shutil.copy2(local / "census.json", OUT / "census.json")
        with (OUT / "check.log").open("w") as log:
            subprocess.run([PY, "experiments/ts5_census_check.py", "--plan", str(CAMPAIGN / "plan.json"),
                            "--checkpoint", str(MODEL), "--census", str(OUT / "census.json"),
                            "--runtime-image", IMAGE, "--out", str(OUT / "check-before-promotion.json")],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        outputs = ["census.json", "check-before-promotion.json"]
    else:
        build = json.loads((OUT / "student_tessera.build.json").read_text())
        meta = json.loads((OUT / "student_tessera.meta.json").read_text())
        teacher_meta = json.loads((TEACHER / "teacher_bf16.meta.json").read_text())
        assert build["complete"] is True and build["identity"]["eager"] is True
        assert build["identity"]["compiled_forward"] is False
        assert build["identity"]["image"] == IMAGE and build["identity"]["image_digest"] == IMAGE.split("@", 1)[1]
        assert meta["role"] == "student" and meta["regime"]["name"] == teacher_meta["regime"]["name"] == "prefill"
        assert meta["payload"]["positions"] == teacher_meta["payload"]["positions"] == 4096
        assert meta["metric"]["requested_top_k"] == teacher_meta["metric"]["requested_top_k"] == 1024
        assert meta["corpus"] == teacher_meta["corpus"]
        assert meta["tokenizer"]["identity_sha256"] == teacher_meta["tokenizer"]["identity_sha256"]
        for name, sha in binding["outputs"].items():
            assert sha256_file(TEACHER / name) == sha, f"teacher output changed across student serve: {name}"
        outputs = ["student_tessera.json.npz", "student_tessera.meta.json", "student_tessera.build.json",
                   "serve_student.log", "kl_tessera_ts5lfm.json"]
    evidence.update({"checkpoint_post": post, "checkpoint_unchanged": True,
                     "outputs": {name: sha256_file(OUT / name) for name in outputs}})
    write("artifact-bound-result.json", evidence)
    completed = True
finally:
    cleanup = cleanup_stage(NAME, launched=launched, completed=completed,
                            stop=stop, monitor=monitor)
    write("cleanup.json", cleanup)
    if launched:
        assert cleanup["safe_to_release"], "GPU/container cleanup not verified"
print(json.dumps({"result": str(OUT / "artifact-bound-result.json"), "cleanup": str(OUT / "cleanup.json")}), flush=True)
