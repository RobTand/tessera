"""Rerun via PB with --snapshot-ref additional-api-base-20261016.

The reference is the exact old module source, executed independently of new
imports. It is not a comparison between post-move aliases.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

BASE = "d2e98b3cb2fcb6fdc27a552442ea021ee02da8ff"
parser = argparse.ArgumentParser()
parser.add_argument("--reference-only", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--pytest", nargs=argparse.REMAINDER)
args = parser.parse_args()
commit = subprocess.check_output(["git", "rev-parse", "additional-api-base-20261016"], text=True).strip()
assert commit == BASE, commit
raw = subprocess.check_output(["git", "show", commit + ":src/tessera/serving/dense_ownership.py"])
old = types.ModuleType("frozen_old_dense_ownership")
sys.modules[old.__name__] = old
exec(compile(raw, "exact-base/dense_ownership.py", "exec"), old.__dict__)
leaves = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj", "q_a_proj",
          "kv_a_proj_with_mqa", "wk", "weights_proj", "o_proj", "gate_proj", "up_proj", "w1", "w3", "down_proj")
prefixes = ("model.layers.0.self_attn.", "model.layers.0.self_attn.indexer.",
            "model.layers.0.mlp.", "model.layers.0.mlp.shared_experts.",
            "model.layers.0.mlp.experts.0.", "model.layers.0.feed_forward.")
configs = [None, {}, {"layer_types": ["linear_attention"]}, {"layer_types": ["full_attention"]},
           {"layer_types": ["deepseek_sparse_attention"]}, {"q_lora_rank": None},
           {"text_config": {"layer_types": ["linear_attention"], "q_lora_rank": None}},
           {"text_config": {"layer_types": ["full_attention"], "q_lora_rank": 16}}]
cases = []
for prefix in prefixes:
    for leaf in leaves:
        for architecture in (None, "Qwen3ForCausalLM", "Glm5NextForConditionalGeneration"):
            for config in configs:
                for names in (None, (), (prefix + "b_proj.weight",)):
                    cases.append((prefix + leaf + ".weight", architecture, config, names))

def outcome(fn, *args, **kwargs):
    try:
        return {"value": fn(*args, **kwargs)}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}

def observe(owner):
    observed = [outcome(owner.fused_module, name, architecture, config=config, tensor_names=names)
                for name, architecture, config, names in cases]
    observed.extend(outcome(owner.role_name, name) for name in
                    ("bare", "bare.weight", "x.y.weight", "x.y", "x.weight.weight", ""))
    groups = tuple((target, members) for _pattern, target, members in old.GLM_FUSED)
    observed.append({"value": owner.glm_fused_groups() if hasattr(owner, "glm_fused_groups") else groups})
    return observed
reference = observe(old)
result = {"base": commit, "reference_source_sha256": hashlib.sha256(raw).hexdigest(),
          "cases": len(reference), "old_outcomes": reference}
if not args.reference_only:
    current = observe(importlib.import_module("tessera.metadata"))
    assert current == reference, next((index, a, b) for index, (a, b) in enumerate(zip(reference, current)) if a != b)
    result["new_outcomes"] = current
    result["equal"] = True
Path(args.output).write_text(json.dumps(result, sort_keys=True))
print(json.dumps({key: value for key, value in result.items() if key not in ("old_outcomes", "new_outcomes")}, sort_keys=True))

if args.pytest:
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", *args.pytest]))
