"""CPU artifacts exercise the real f16 A/B acceptance entry point."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.t8r_speed.finalize_native_build import finalize
from tessera._dev.native_identity import loaded_native_identity

ARMS = ("base", "new", "newb", "baseb")


def cpu_artifacts(root):
    sources = {}
    for arm in ("base", "new"):
        source = root / ("master-src" if arm == "base" else "new-src") / "tessera/serving/csrc/routed_fused_window.cu"
        source.parent.mkdir(parents=True)
        source.write_text(f"/* CPU acceptance fixture: {arm} */\nint witness(int x) {{ return x * 3 + 7; }}\n")
        sources[arm] = source
    libraries = []
    for arm in ARMS:
        source = sources["base" if arm.startswith("base") else "new"]
        directory = root / arm / "home/torch_extensions/cpu_tessera_guarded_v1"
        directory.mkdir(parents=True)
        library = directory / "tessera_routed_fused_e4m3.so"
        (directory / "build.ninja").write_text(
            "rule compile\n  command = cc -x c -fPIC -c $in -o $out\n"
            "rule link\n  command = cc -shared $in -o $out\n"
            f"build witness.o: compile {source}\n"
            f"build {library.name}: link witness.o\n")
        subprocess.run(["ninja", "-C", str(directory)], check=True, capture_output=True)
        finalized = finalize(directory, source)
        module = ctypes.CDLL(str(library))
        libraries.append(module)
        identity = loaded_native_identity(SimpleNamespace(__file__=str(library)), source)
        finalization = directory / "native-finalization.json"
        meta = {"kernel_sha": identity["source_sha256"], "library": "e4m3",
                "cases": ["r4", "r3+q"], "modes": [0, 2], "ms": [1, 7],
                "native": identity, "native_after": identity,
                "finalization_path": str(finalization),
                "finalization_sha256": hashlib.sha256(finalization.read_bytes()).hexdigest(),
                "fixture_scope": "CPU scalar witness; no CUDA, timing, or serving qualification"}
        results = []
        for case in meta["cases"]:
            for mode in meta["modes"]:
                cells = {}
                for m in meta["ms"]:
                    output = struct.pack("i", module.witness(m + mode))
                    cells[str(m)] = {"out_sha256": hashlib.sha256(output).hexdigest(),
                                     "wall": {"median_ms": 1.0}}
                results.append({"case": case, "mode": mode, "cells": cells})
        (root / arm / "bench_pairs.json").write_text(json.dumps({"meta": meta, "results": results}))
        assert finalized["no_pending_work"]
    return sources, libraries


def invoke(root, sources):
    env = dict(os.environ, AB_STEPS="analyze", AB_CASES="r4,r3+q", AB_MODES="0,2", AB_MS="1,7",
               MASTER_SRC=str(sources["base"].parents[3]), NEW_SRC=str(sources["new"].parents[3]),
               ORACLE_IMAGE="CPU-offline-unused", AB_PYTHON=sys.executable)
    return subprocess.run(["bash", str(ROOT / "experiments/t8r_speed/f16_prefetch_ab.sh"), str(root)],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("defect", ["output_mismatch", "missing_every_arm", "analyzer_json_failure", "wrong_source_arm"])
def test_real_harness_refuses_consumer_failure(tmp_path, defect):
    sources, held = cpu_artifacts(tmp_path)
    if defect == "analyzer_json_failure":
        (tmp_path / "new/bench_pairs.json").write_text("{invalid json")
    else:
        for arm in (ARMS if defect == "missing_every_arm" else ("new",)):
            path = tmp_path / arm / "bench_pairs.json"
            doc = json.loads(path.read_text())
            if defect == "output_mismatch":
                doc["results"][0]["cells"]["1"]["out_sha256"] = "0" * 64
            elif defect == "missing_every_arm":
                del doc["results"][0]["cells"]["1"]
            else:
                baseline = json.loads((tmp_path / "base/bench_pairs.json").read_text())
                doc["meta"] = baseline["meta"]
            path.write_text(json.dumps(doc))
    result = invoke(tmp_path, sources)
    assert result.returncode != 0, f"{defect} escaped acceptance:\n{result.stdout}\n{result.stderr}"
    assert "ALL_DONE" not in result.stdout
    assert held


def test_partial_cli_reports_requested_scope_only(tmp_path):
    sources, held = cpu_artifacts(tmp_path)
    result = invoke(tmp_path, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = tmp_path / "ab_acceptance.json"
    assert receipt.exists(), "the actual CLI supplied no explicit partial acceptance receipt"
    doc = json.loads(receipt.read_text())
    assert doc["cells"] == 8
    assert doc["issue_acceptance"] == "not_established"
    assert doc["stages"] == ["analyze"]
    assert held
