#!/usr/bin/env python3
"""How far a given revision's ``vllm_module_name`` is from the attested one.

The in-image arm (``tests/test_serving_name_mapping.py`` inside
``vllm/vllm-openai:latest``) establishes that THIS revision's replay equals
vLLM's own ``WeightsMapper`` on every probe name.  So the distance from any
other revision to the runtime is its distance to this one, and that needs no
image -- which is what makes the #108 divergence countable rather than argued.

usage: experiments/vllm_module_name_distance.py [git-rev, default master]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from mapper_probes import SYNTHETIC_TABLES, probe_names            # noqa: E402
from tessera.serving.contract import vllm_module_name as attested  # noqa: E402


def other(rev):
    """The named revision's ``vllm_module_name``, execed straight out of git.

    Against the code as committed, not against a retelling of it.
    """
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:src/tessera/serving/contract.py"],
        capture_output=True, text=True, check=True).stdout
    body = ("from typing import Any, Mapping\n"
            + src[src.index("def vllm_module_name("):src.index("def classify_construction(")])
    ns: dict = {}
    exec(compile(body, rev, "exec"), ns)  # noqa: S102 -- a committed revision of this file
    return ns["vllm_module_name"]


def call(fn, entry, name):
    try:
        return fn(entry, name)
    except ValueError:
        return None


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "master"
    was = other(rev)
    tables = dict(SYNTHETIC_TABLES)
    for path in sorted((ROOT / "docs" / "measurements" / "construction").glob("*.json")):
        tables[path.stem] = json.loads(path.read_text())["hf_to_vllm_mapper_unstacked"] or {}
    total = 0
    for name, table in tables.items():
        entry = {"hf_to_vllm_mapper_unstacked": table}
        diffs = [(n, call(was, entry, n), call(attested, entry, n))
                 for n in probe_names(table)]
        diffs = [d for d in diffs if d[1] != d[2]]
        total += len(diffs)
        example = f"  e.g. {diffs[0][0]!r}: {rev} {diffs[0][1]!r} vs runtime {diffs[0][2]!r}" \
            if diffs else ""
        print(f"{name}: {len(diffs)}{example}")
    print(f"TOTAL names where {rev} disagrees with the attested replay: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
