"""One-shot PB-only LFM teacher measurement with source-bound receipts.

The fixed campaign paths are deliberate measurement identities, not defaults.
Dispatch through pbrun under an outer 1500s timeout with 120s cleanup grace.
"""
import datetime
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading

sys.path.insert(0, str(Path.cwd() / "src"))
from tessera.serving_parts import source_identity, sha256_file

CAMPAIGN = Path("/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2")
SOURCE = Path("/mnt/shared/models/LFM2.5-8B-A1B-BF16")
OUT = CAMPAIGN / "teacher-bound-r1"
CORPUS = Path("/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json")
IMAGE = "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
NAME = "ts5-lfm-r2-teacher-bound-r1"
PY = "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python"
def interrupted(signum, _frame):
    raise TimeoutError(f"teacher action interrupted by signal {signum}")

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
OUT.mkdir()  # Exclusive one-shot guard: retries never start a second serve.
stop = threading.Event()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def capture(command):
    return subprocess.check_output(command, text=True, timeout=30).strip()

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
try:
    assert not capture(["docker", "ps", "-aq", "--filter", f"name=^/{NAME}$"]), "unique container name already exists"
    manifest_path = CAMPAIGN / "part0/tessera_serving_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["export_partition"]["identity"]
    assert identity["runtime_image"] == IMAGE
    expected = identity["source"]
    pre = source_identity(SOURCE)
    assert pre == expected, "teacher source differs from the encoded source"
    assert sha256_file(CORPUS) == "2fdd48eeab69109c6222ef2f857815d2b35d5422747815c495e0467712751d44"
    evidence = {
        "schema": "tessera.lfm-teacher-source-binding/1",
        "source_path": str(SOURCE), "source_pre": pre,
        "part_manifest_sha256": sha256_file(manifest_path),
        "runtime_image": IMAGE, "execution_mode": "eager",
        "source_snapshot": capture(["git", "rev-parse", "HEAD"]),
        "corpus_sha256": sha256_file(CORPUS),
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_mount_read_only": True,
    }
    write("source-before.json", evidence)
    env = os.environ.copy()
    env.update({
        "TESSERA_KL_IMAGE": IMAGE,
        "TESSERA_KL_DOCKER_EXTRA": f"--entrypoint=vllm --memory=64g --memory-swap=64g -v {SOURCE}:{SOURCE}:ro",
        "TESSERA_KL_IMAGE_COMMAND": "serve",
        "TESSERA_KL_CORPUS": str(CORPUS), "TESSERA_KL_LOGDIR": str(OUT),
        "TESSERA_KL_NAME": NAME, "TESSERA_KL_PORT": "8152",
        "TESSERA_GPU_MEM_UTIL": "0.35", "TESSERA_KL_TOPK": "1024",
        "TESSERA_KL_EAGER": "1", "TESSERA_KL_REGIME": "prefill",
    })
    with (OUT / "action.log").open("w") as log:
        subprocess.run([
            "timeout", "--signal=TERM", "--kill-after=30s", "900s",
            "experiments/serve_and_dump_kl.sh", str(SOURCE), str(OUT / "teacher_bf16.json"),
            "teacher", "BF16-LFM2.5-8B-A1B-5dd22602c2e9f6a097b1de4c4efe0658b605015c",
        ], env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    post = source_identity(SOURCE)
    assert post == pre == expected, "teacher source changed across serve"
    assert sha256_file(CORPUS) == evidence["corpus_sha256"], "corpus changed across serve"
    build = json.loads((OUT / "teacher_bf16.build.json").read_text())
    meta = json.loads((OUT / "teacher_bf16.meta.json").read_text())
    assert build["complete"] is True and build["identity"]["eager"] is True
    assert build["identity"]["compiled_forward"] is False
    assert build["identity"]["image"] == IMAGE
    assert build["identity"]["image_digest"] == IMAGE.split("@", 1)[1]
    assert meta["role"] == "teacher" and meta["regime"]["name"] == "prefill"
    assert meta["payload"]["positions"] == meta["corpus"]["scored_positions"] == 4096
    assert meta["metric"]["requested_top_k"] == 1024
    assert meta["corpus"]["contract_sha256"] == json.loads(CORPUS.read_text())["contract_sha256"]
    assert all(expected["auxiliary_sha256"][name] == sha for name, sha in meta["tokenizer"]["files"].items())
    with (OUT / "usability.log").open("w") as log:
        subprocess.run([PY, "experiments/kl_reference_usable.py", str(OUT / "teacher_bf16.json.npz"), str(CORPUS), "--json", str(OUT / "teacher_reference_gate.json")], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    evidence.update({
        "source_post": post,
        "source_unchanged": True,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "outputs": {name: sha256_file(OUT / name) for name in (
            "teacher_bf16.json.npz", "teacher_bf16.meta.json", "teacher_bf16.build.json",
            "serve_teacher_bf16.log", "teacher_reference_gate.json")},
    })
    write("source-bound-result.json", evidence)
    completed = True
finally:
    cleanup = {"container_name": NAME, "measurement_completed": completed}
    try:
        existing = capture(["docker", "ps", "-aq", "--filter", f"name=^/{NAME}$"])
        cleanup["container_before_cleanup"] = existing
        if existing:
            subprocess.run(["docker", "rm", "-f", NAME], check=True, timeout=45)
        cleanup["container_after_cleanup"] = capture(["docker", "ps", "-aq", "--filter", f"name=^/{NAME}$"])
        cleanup["gpu_compute_processes"] = capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        cleanup["safe_to_release"] = not cleanup["container_after_cleanup"] and not cleanup["gpu_compute_processes"]
        write("cleanup.json", cleanup)
        assert cleanup["safe_to_release"], "GPU/container cleanup not verified"
    finally:
        stop.set()
        monitor.join(timeout=35)
print(json.dumps({"result": str(OUT / "source-bound-result.json"), "cleanup": str(OUT / "cleanup.json")}), flush=True)

