#!/usr/bin/env python3
"""Generate one tessera#1204 fold arm run dir and its pbgang manifest.

File generation only: no GPU, no serve, no hashing. The caller submits
the printed gang manifest with pbgang.py. The engine.json (derived
scorer/peer argv) is written by a separate CPU job before submission.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "graph_attest_702"))
from managed_window import MEMORY_POLICY

ARTIFACT = "/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported"
RUNDIR = Path("/mnt/shared/tessera-measurements/fold-tp2-1204")
ROSTER = RUNDIR / "roster.json"
ROSTER_SHA = "bb07140cb5c23ad574f32e138333942e95df5be31b2510afe0971aa32539e1d8"
DIGEST_CACHE = RUNDIR / "candidate-digest-cache.sparklina.json"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
TS_COMMIT = "4c0fd6aca04f1c3e95b094ae006c2479043c3372"
PANEL = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/final_panel_handoff.json"
ARRAYS = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/arrays"
TEACHER = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04/artifact/teacher.json"
TEACHER_SHA = "1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae"
TEACHER2 = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-exl3ref-01/artifact/teacher.json"
TEACHER2_SHA = "da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b"
PROMPTS = "/mnt/shared/tessera-runs/exl3-preflight/rdv-validated-9ccccd9945214c309b28bb0ddb0c860a/inputs/prompts.json"
PROMPTS_SHA = "8cd21019b03ffe1875704441878acf3bbd8546177036e2734ab5cfa85f1a4cb8"
PQ = "/mnt/shared/tessera-measurements/glm-pact-u4-20260927/src/prismaquant"
RANK_PYTHON = "/home/rob/venvs/pb-cpu/bin/python"


def scorer_argv(out_tr3: Path) -> list[str]:
    return ["--model", ARTIFACT, "--candidate-digest-cache", str(DIGEST_CACHE),
            "--panel", PANEL, "--arrays-root", ARRAYS,
            "--teacher", TEACHER, "--teacher-sha256", TEACHER_SHA,
            "--serve-image", IMAGE,
            "--output", str(out_tr3 / "full-vocabulary-kl.json"),
            "--kv-cache-dtype", "fp8_ds_mla", "--expected-kv-cache-dtype", "fp8_ds_mla",
            "--kernel-config", '{"enable_flashinfer_autotune":false}',
            "--gpu-memory-utilization", "0.5", "--logits-layout", "vllm_v2_chunk1024",
            "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "0",
            "--master-addr", "10.100.96.2", "--master-port", "29541",
            "--distributed-executor-backend", "mp", "--data-parallel-backend", "mp",
            "--moe-backend", "triton", "--kv-cache-memory-bytes", "1073741824",
            "--teacher2", TEACHER2, "--teacher2-sha256", TEACHER2_SHA,
            "--qualify-then-score", str(out_tr3 / "hook-qualification.json")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "gate"), required=True)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"fold1204-{args.arm}-{stamp}"
    run = RUNDIR / run_id
    if run.exists():
        raise SystemExit(f"run dir exists: {run}")
    for path in (ARTIFACT, str(ROSTER), PANEL, TEACHER, TEACHER2, PROMPTS, PQ):
        if not Path(path).exists():
            raise SystemExit(f"missing input: {path}")
    pq_commit = subprocess.check_output(["git", "-C", PQ, "rev-parse", "HEAD"],
                                        text=True).strip()
    (run / "barrier").mkdir(parents=True)
    (run / "generation").mkdir(parents=True)
    arm_name = f"fold-{args.arm}"
    fold = args.arm == "gate"
    setup = {"run": str(run), "run_id": run_id, "arm": arm_name, "fold": int(fold),
             "artifact": ARTIFACT, "roster": str(ROSTER), "roster_sha256": ROSTER_SHA,
             "digest_cache": str(DIGEST_CACHE), "image": IMAGE, "ts_commit": TS_COMMIT,
             "prompts": PROMPTS, "prompts_sha256": PROMPTS_SHA,
             "pq": PQ, "pq_commit": pq_commit,
             "panel": PANEL, "arrays_root": ARRAYS,
             "teacher": TEACHER, "teacher_sha256": TEACHER_SHA,
             "teacher2": TEACHER2, "teacher2_sha256": TEACHER2_SHA,
             "window_seconds": 5400, "master_port": 29541, "api_port": 8142}
    (run / "inputs.json").write_text(json.dumps(setup, indent=1, sort_keys=True) + "\n")
    (run / "memory-policy.json").write_text(json.dumps(MEMORY_POLICY, indent=1) + "\n")
    (run / "scorer-argv.json").write_text(
        json.dumps(scorer_argv(run / arm_name / "tr3"), indent=1) + "\n")
    env_lines = [f"ARM_ARTIFACT={ARTIFACT}", f"ARM_AUDIT_ROSTER={ROSTER}",
                 f"ARM_AUDIT_ROSTER_SHA256={ROSTER_SHA}",
                 f"TS_PIN_{arm_name}={TS_COMMIT[:8]}",
                 f"TESSERA_GLM53_FOLD_SHARED_ADD={int(fold)}"]
    (run / f"{arm_name}.env").write_text("\n".join(env_lines) + "\n")
    manifest = {"prompt_file": "prompts.json", "prompt_sha256": PROMPTS_SHA,
                "concurrency": [1], "warmup": 1, "temperature": 0, "ignore_eos": True,
                "lens": [512, 2048, 8192], "trials": 10, "output_tokens": 128,
                "identities": {"tessera_artifact": "glm53-artifact",
                               "artifact_path": ARTIFACT,
                               "artifact_roster_sha256": ROSTER_SHA,
                               "run_id": run_id, "arm": arm_name}}
    (run / "generation" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    link = run / "generation" / "prompts.json"
    if not link.exists():
        link.symlink_to(PROMPTS)
    digest = hashlib.sha256((run / "inputs.json").read_bytes()).hexdigest()
    gang = {"priority": 0,
            "priority_reason": ("tessera#1204 served fold arm; CEO dec-1010-040208-6b0b "
                                "(120 GPU-min); main campaign GPU work first at 10"),
            "timeout_s": 7200, "skew_s": 300,
            "members": [
                {"tag": "sparklina", "demand": {"cpu": 8, "mem_gb": 104, "gpu": 1},
                 "gpu_memory_gb": 102, "exclusive": True, "measurement": True,
                 "host_class": "gb10", "container_images": [IMAGE], "max_attempts": 1,
                 "env": {"FOLD_INPUT_SHA256": digest},
                 "argv": [RANK_PYTHON, "experiments/graph_attest_702/fold_tp2_arm.py",
                          "--rank", "0", "--run", str(run)]},
                {"tag": "sparky", "demand": {"cpu": 6, "mem_gb": 104, "gpu": 1},
                 "gpu_memory_gb": 102, "exclusive": True, "measurement": True,
                 "host_class": "gb10", "container_images": [IMAGE], "max_attempts": 1,
                 "env": {"FOLD_INPUT_SHA256": digest},
                 "argv": [RANK_PYTHON, "experiments/graph_attest_702/fold_tp2_arm.py",
                          "--rank", "1", "--run", str(run)]}]}
    (run / "gang.json").write_text(json.dumps(gang, indent=1) + "\n")
    print(f"run={run}\nrun_id={run_id}\ninputs_sha256={digest}\npq_commit={pq_commit}")
    print(f"gang_manifest={run / 'gang.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
