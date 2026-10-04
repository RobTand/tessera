"""tessera#696: inside one serving image, answer for the image itself.

Prints the runtime identity and the two decisive source hashes, and writes a
sha256 manifest of every .py file in the installed vllm package to
$PROBE_OUT/manifest.txt. Runs inside the image as the invoking user; writes
only under the mounted /out.
"""
import hashlib
import json
import os
import pathlib
import sys

import vllm

out = pathlib.Path(os.environ["PROBE_OUT"])
root = pathlib.Path(vllm.__file__).parent


def sha(relative: str) -> str:
    try:
        return hashlib.sha256((root / relative).read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


versions = {"python": sys.version.split()[0], "vllm": vllm.__version__,
            "vllm_path": str(root)}
for module in ("torch", "flashinfer"):
    try:
        versions[module] = __import__(module).__version__
    except Exception as exc:  # recorded, not fatal: the hashes are the evidence
        versions[module] = f"import failed: {exc}"
print(json.dumps(versions, indent=1))
print(f"runner_sha256 {sha('v1/worker/gpu/model_runner.py')}")
print(f"block_table_sha256 {sha('v1/worker/gpu/block_table.py')}")
with (out / "manifest.txt").open("w") as fh:
    for path in sorted(root.rglob("*.py")):
        fh.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                 f"{path.relative_to(root)}\n")
