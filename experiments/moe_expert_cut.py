#!/usr/bin/env python3
"""Cut a routed-MoE checkpoint down to its first ``--experts`` experts per layer.

WHY THIS EXISTS.  Issue #5's last open item is a **served** census and KL on a
routed-MoE Tessera checkpoint, and the only routed-MoE model this box can serve
is ``GLM-5.3-Flash-4layer``: 288 experts x 3 stacks = 21.74 G routed parameters,
about 3.75 GPU-hours of encode before a single token is served.  That is a cost
the serving path has to earn, not one it should be asked to pay before anyone
knows whether the path works at all.

So this writes a checkpoint that is the SAME MODEL CLASS, the SAME tokenizer and
the SAME real weights, with the expert dimension narrowed: experts ``0..N-1`` of
every routed stack are kept verbatim, the router's rows are narrowed to match,
and ``n_routed_experts`` is set to ``N``.  Nothing is synthesised and nothing is
re-scaled.  At ``N=16`` the routed body is 1.21 G parameters -- about 12 encode
minutes -- and the whole checkpoint is small enough that a teacher and a student
serve can both be run in one night.

WHAT IT IS NOT.  The cut model is not GLM-5.3-Flash: dropping 272 of 288 experts
changes which expert a token routes to, so its *generations* are not the parent
model's and no quality claim about GLM may be read off it.  What survives the
cut is exactly what issue #5 asks about -- the model class, the loader, the
expert-parameter mapping, the wire names, the serving route and the arithmetic
of the tile -- and a student/teacher KL on the cut model measures the error the
Tessera expert route introduces on THIS model's experts, against a BF16 teacher
of the same cut.  Say "the 16-expert cut" wherever the number is quoted.

    moe_expert_cut.py SRC OUT --experts 16
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

EXPERT_LEAF = re.compile(r"^(?P<stack>.*\.experts)\.(?P<expert>\d+)\.(?P<rest>.+)$")
#: The two router tensors whose FIRST axis is the expert axis.  ``noaux_tc``
#: scoring carries a per-expert bias beside the router weight, and a router
#: narrowed on one but not the other selects experts that are no longer there.
ROUTER_SUFFIXES = (".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--experts", type=int, default=16)
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    keep = args.experts
    index_path = args.src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    args.out.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    dropped = kept = narrowed = 0
    total_bytes = 0
    for shard in sorted(shards):
        payload = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in sorted(shards[shard]):
                match = EXPERT_LEAF.match(name)
                if match is not None and int(match.group("expert")) >= keep:
                    dropped += 1
                    continue
                tensor = handle.get_tensor(name)
                if name.endswith(ROUTER_SUFFIXES):
                    if tensor.shape[0] <= keep:
                        raise SystemExit(
                            f"{name} has {tensor.shape[0]} rows, fewer than --experts {keep}")
                    tensor = tensor[:keep].clone()
                    narrowed += 1
                payload[name] = tensor.contiguous()
                new_map[name] = shard
                kept += 1
        if not payload:
            continue
        save_file(payload, str(args.out / shard), metadata={"format": "pt"})
        total_bytes += (args.out / shard).stat().st_size
        print(f"  {shard}: {len(payload)} tensors")
        del payload

    # An index that names a shard this cut did not write is a checkpoint that
    # cannot be loaded, so the map is rebuilt from what was actually written.
    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=1) + "\n")

    config = json.loads((args.src / "config.json").read_text())
    text = config.get("text_config", config)
    before = text.get("n_routed_experts")
    text["n_routed_experts"] = keep
    if text.get("num_experts_per_tok", 0) > keep:
        raise SystemExit(f"num_experts_per_tok {text['num_experts_per_tok']} exceeds --experts {keep}")
    (args.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
                  "special_tokens_map.json", "video_preprocessor_config.json"):
        source = args.src / extra
        if source.exists():
            shutil.copy2(source, args.out / extra)

    print(f"n_routed_experts {before} -> {keep}; kept {kept} tensors, dropped {dropped}, "
          f"narrowed {narrowed} router tensors; {total_bytes / 2**30:.2f} GiB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
