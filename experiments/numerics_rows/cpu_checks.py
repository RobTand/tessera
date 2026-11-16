"""Run the scoped CPU checks inside an admitted Flux job."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.numerics_rows.common import check_source, write_json

TESTS = ("tests/test_numerics_rows.py", "tests/test_numerics_rows_sass.py",
         "tests/test_dense_fused_window.py", "tests/test_routed_fused_e2m1.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image")
    args = parser.parse_args()
    check_source(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bazel-targets.txt").write_text(Path(__file__).with_name("bazel-targets.txt").read_text())
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src") + ":" + str(root), "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    records = []
    def run(name, command):
        with (args.out / f"{name}.stdout").open("w") as stdout, (args.out / f"{name}.stderr").open("w") as stderr:
            result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        records.append({"name": name, "command": command, "exit_status": result.returncode})
        write_json(args.out / "checks.json", records)
        return result.returncode
    paths = sorted(str(p) for p in (root / "experiments/numerics_rows").glob("*.py"))
    run("syntax", [sys.executable, "-m", "py_compile", *paths, *TESTS])
    run("pytest", [sys.executable, "-m", "pytest", *TESTS, "-q", "-n", "4", "--dist", "worksteal", "--durations=25",
                   "--surface-json", str(args.out / "surface.json"), "--junitxml", str(args.out / "junit.xml")])
    run("build-files", [sys.executable, "tools/bazel/build_files.py"])
    run("full-shards", [sys.executable, "tools/ci/full_shards.py"])
    with (args.out / "metadata.patch").open("w") as stream:
        subprocess.run(["git", "diff", "--", "*BUILD.bazel", "tools/bazel/full_targets"], cwd=root, stdout=stream, check=True)
    raise SystemExit(0 if all(r["exit_status"] == 0 for r in records) else 3)


if __name__ == "__main__":
    main()
