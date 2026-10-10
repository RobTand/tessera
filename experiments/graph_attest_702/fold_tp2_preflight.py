#!/usr/bin/env python3
"""D38 CPU dry run for the tessera#1204 served TP2 shared-add fold arm.

Checks the pieces the GPU campaign drivers use, on a small slice:
imports and argument parsing, arm .env schema, lever_chain wiring with
stub collaborators, the fold flag default-off latch, small reads of the
artifact, and the exact serve argv the recipe builds at this head.
No GPU, no serve, no model launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "graph_attest_702"))

IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
HEAD = "4c0fd6aca04f1c3e95b094ae006c2479043c3372"
ARTIFACT = Path("/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported")


def check_imports() -> None:
    import comparison_input_intake  # noqa: F401
    import comparison_arm_identity  # noqa: F401
    import served_generation_client  # noqa: F401
    import tp2_recipe  # noqa: F401
    print("imports ok: intake, identity, generation client, tp2_recipe")


def check_fold_latch() -> None:
    os.environ.pop("TESSERA_GLM53_FOLD_SHARED_ADD", None)
    from tessera.serving import glm53_shared_fold as fold
    from tessera.serving.flags import latched_bool
    assert fold.FLAG == "TESSERA_GLM53_FOLD_SHARED_ADD"
    assert latched_bool(fold.FLAG) is False
    assert fold.install_for_current_config() is False
    assert IMAGE.rpartition("@sha256:")[2][:12] == "5be13705acae"[:12]
    assert "5be13705" in Path(fold.__file__).read_text()
    print("fold latch ok: default off, install declines without the flag")
def check_artifact() -> None:
    config = json.loads((ARTIFACT / "config.json").read_bytes())
    text = config.get("text_config", config)
    manifest = json.loads((ARTIFACT / "tessera_serving_manifest.json").read_bytes())
    index = json.loads((ARTIFACT / "model.safetensors.index.json").read_bytes())
    print(f"artifact ok: layers={text.get('num_hidden_layers')} "
          f"export_commit={manifest.get('git')} "
          f"contract={manifest.get('serving_gate', {}).get('contract_version')} "
          f"shards={len(index.get('weight_map', {}))}")


def check_arm_schema(tmp: Path) -> None:
    import comparison_arm_identity as owner
    roster = {"artifact": str(ARTIFACT), "files": [], "manifest_sha256": "x",
              "all_files_bytes": 0}
    (tmp / "roster.json").write_text(json.dumps(roster))
    for name in ("base", "gate"):
        env = (tmp / f"{name}.env")
        env.write_text(
            f"ARM_ARTIFACT={ARTIFACT}\n"
            f"ARM_AUDIT_ROSTER={tmp / 'roster.json'}\n"
            "ARM_AUDIT_ROSTER_SHA256=" + hashlib.sha256(
                (tmp / "roster.json").read_bytes()).hexdigest() + "\n"
            + ("TESSERA_GLM53_FOLD_SHARED_ADD=1\n" if name == "gate" else ""))
    print("arm schema ok: base and gate .env files written")


def check_lever_chain() -> None:
    gates = (ROOT / "tools" / "serve_comparison_gates.sh").read_text()
    fn = re.search(r"^lever_chain\(\) \{.*?^\}", gates, re.M | re.S).group()
    for rc, ship in (("0", "G"), ("1", "")):
        script = ("say() { :; }\n"
                  "run_val() { RV_STATUS=ran; RV_LV=\"levers as declared\"; RV_KV=0; }\n"
                  "tr3_gate() { echo \"BITWISE exact\"; }\n"
                  f"generation_gate() {{ return {rc}; }}\n"
                  "BASE_ARM=B; GATE_ARM=G; FALLBACK_ARMS=\"\"; SHIP=\"\"; SHIP_RUN=\"\"; "
                  "CHAIN_ROOT=/unused; U=/unused; WID=w; N=/dev/null; GATE_TRAIL=\"\"\n"
                  + fn + "\nlever_chain; printf \"%s\" \"$SHIP\"\n")
        out = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                             check=True, timeout=60)
        assert out.stdout == ship, (rc, out.stdout, out.stderr)
    print("lever_chain ok: ships on green gates, holds on generation failure")


def check_serve_argv() -> None:
    import tp2_recipe as recipe
    assert recipe.IMAGE == IMAGE
    config = {"artifact": str(ARTIFACT), "window_mode": "window4-eager-2048-4096",
              "profile_dir": "/tmp/fold-d38/profiles"}
    arm = {"arm": "eager2048", "eager": "1", "compilation": "",
           "spec": '{"method":"mtp","num_speculative_tokens":1,'
                   '"draft_tensor_parallel_size":2,"moe_backend":"triton"}',
           "max_batched": 2048, "fabric": "socket"}
    argv = recipe.serve(config, arm, 0)
    blob = " ".join(argv)
    assert "--tensor-parallel-size 2" in blob and "--nnodes 2" in blob
    assert "--enforce-eager" in argv
    assert "--profiler-config" in blob
    print(f"serve argv ok: sha256={hashlib.sha256(blob.encode()).hexdigest()[:16]} "
          f"eager tp2 profiler-armed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=HEAD)
    ap.add_argument("--tmp", default="/tmp/fold-d38")
    args = ap.parse_args()
    subprocess.check_call(["git", "-C", str(ROOT), "cat-file", "-e", args.commit + "^{commit}"])
    base = subprocess.check_output(["git", "-C", str(ROOT), "merge-base", args.commit, "HEAD"], text=True).strip()
    assert base == args.commit, f"snapshot without {args.commit} as ancestor"
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    check_imports()
    check_fold_latch()
    check_artifact()
    check_arm_schema(tmp)
    check_lever_chain()
    check_serve_argv()
    print("D38 PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
