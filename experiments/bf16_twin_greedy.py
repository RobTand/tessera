"""Does an ordinary loader serve the twin?  Load it as a plain checkpoint, greedily.

The twin's claim is that it is an *ordinary* BF16 checkpoint: no plugin, no
custom kernel, no ``quantization_config`` -- the tiles the wire decodes to,
under the source's own tensor names.  ``bf16_twin_check.py`` proves the bytes
and the structure; this proves a stock loader agrees, which is the only direct
evidence that a served gate can run on the twin later.  It is deliberately not
a vLLM serve: ``AutoModelForCausalLM.from_pretrained(twin)`` with nothing else
set is the strongest available statement of "nothing special is needed".

Run::

    PYTHONPATH=src python experiments/bf16_twin_greedy.py \\
      --source /home/rob/models/Qwen3-0.6B \\
      --twin /mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r6-twin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ("The capital of France is", "def fibonacci(n):")


def run(directory: str, tok, prompts, tokens: int) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.bfloat16).cuda().eval()
    got = {}
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=tokens, do_sample=False)
            logits = model(ids).logits[0, -1].float()
        top = torch.topk(torch.log_softmax(logits, -1), 5)
        got[prompt] = {
            "completion": tok.decode(out[0, ids.shape[1]:]),
            "top5": [(tok.decode([int(i)]), round(float(v), 3))
                     for v, i in zip(top.values, top.indices)],
        }
    del model
    torch.cuda.empty_cache()
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--twin", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.source)
    result = {"source": a.source, "arms": {}}
    for label, directory in [("bf16 source", a.source)] + [(d, d) for d in a.twin]:
        result["arms"][label] = run(directory, tok, PROMPTS, a.tokens)
        print(f"--- {label}", flush=True)
        for prompt, row in result["arms"][label].items():
            print(f"  {prompt!r} -> {row['completion']!r}", flush=True)
            print(f"    top5 {row['top5']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
