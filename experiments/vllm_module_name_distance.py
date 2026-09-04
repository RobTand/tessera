#!/usr/bin/env python3
"""How far a given revision's ``vllm_module_name`` is from the attested one.

The in-image arm (``tests/test_serving_name_mapping.py`` inside
``vllm/vllm-openai:latest``) establishes that THIS revision's replay equals
vLLM's own ``WeightsMapper`` on every probe name.  So the distance from any
other revision to the runtime is its distance to this one, and that needs no
image -- which is what makes the #108 divergence countable rather than argued.

usage: wf108-distance.py [git-rev, default master]
"""
import itertools
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tessera.serving.contract import (  # noqa: E402
    _MAPPER_FIELDS_REPLAYED, vllm_module_name as attested)

SYNTHETIC = {
    "substr_twice": {"orig_to_new_substr": {".block.": ".layer."}},
    "prefix_chain": {"orig_to_new_prefix": {"model.": "language_model.",
                                            "language_model.": "lm."}},
    "suffix_chain": {"orig_to_new_suffix": {".a_proj": ".b_proj", ".b_proj": ".c_proj"}},
    "substr_then_prefix_then_suffix": {
        "orig_to_new_substr": {"decoder.": "layers."},
        "orig_to_new_prefix": {"model.": "language_model.model."},
        "orig_to_new_suffix": {".gate_up": ".gate_up_proj"}},
    "a_dropping_prefix": {"orig_to_new_prefix": {"model.visual.": None,
                                                 "model.": "language_model.model."}},
    "a_dropping_substr": {"orig_to_new_substr": {".mtp.": None}},
    "a_dropping_suffix": {"orig_to_new_suffix": {".inv_freq": None}},
}
LEAVES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
          "mlp.experts.3.down_proj", "attn.qkv", "a_proj", "gate_up")


def probes(table):
    names = {f"model.layers.0.{leaf}" for leaf in LEAVES}
    names |= {f"model.visual.blocks.1.{leaf}" for leaf in LEAVES}
    keys = [k for field in _MAPPER_FIELDS_REPLAYED for k in (table.get(field) or {})]
    for key in keys:
        names |= {key, f"{key}layers.0.mlp.down_proj", f"model.layers.0{key}",
                  f"model.layers.0.{key}mlp.{key}down_proj",
                  f"prefix{key}middle{key}suffix"}
    for a, b in itertools.permutations(keys, 2):
        names.add(f"{a}model.layers.0{b}")
    return sorted(names)


def other(rev):
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:src/tessera/serving/contract.py"],
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
    tables = dict(SYNTHETIC)
    for path in sorted((ROOT / "docs" / "measurements" / "construction").glob("*.json")):
        tables[path.stem] = json.loads(path.read_text())["hf_to_vllm_mapper_unstacked"] or {}
    total = 0
    for name, table in tables.items():
        entry = {"hf_to_vllm_mapper_unstacked": table}
        diffs = [(n, call(was, entry, n), call(attested, entry, n)) for n in probes(table)]
        diffs = [d for d in diffs if d[1] != d[2]]
        total += len(diffs)
        example = f"  e.g. {diffs[0][0]!r}: {rev} {diffs[0][1]!r} vs runtime {diffs[0][2]!r}" \
            if diffs else ""
        print(f"{name}: {len(diffs)}{example}")
    print(f"TOTAL names where {rev} disagrees with the attested replay: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
