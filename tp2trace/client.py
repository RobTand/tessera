#!/usr/bin/env python3
"""tessera#1185: request client for the TP2 prefill trace (OFF REPO).

Runs INSIDE the pinned image (transformers + vLLM API on localhost).
Builds token-id prompts from the served artifact's own tokenizer so the
measured request is exactly 8192 prompt tokens:

  --mode warmup   a few short id-array prompts (allocator + kernel warmup)
  --mode measure  one prompt of exactly --tokens ids, --max-tokens out

The completions API takes the prompt as token ids, so no text retokenizes
differently on the server. Every response id is discarded; only status and
token counts are printed.
"""

import argparse
import json
import sys
import urllib.request


def post(url, payload, timeout_s):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.load(resp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mode", required=True, choices=("warmup", "measure"))
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        trust_remote_code=True)
    seed = ("The quick brown fox crosses the quiet river at dawn. " * 4000)
    ids = tok.encode(seed, add_special_tokens=True)
    while len(ids) < args.tokens:
        ids = ids + ids
    ids = ids[:args.tokens]
    assert len(ids) == args.tokens, len(ids)
    with urllib.request.urlopen(args.url + "/v1/models",
                                 timeout=30) as resp:
        model = json.load(resp)["data"][0]["id"]
    print(f"[tp2trace] client {args.mode}: model={model} "
          f"{len(ids)} ids, max_tokens={args.max_tokens}", flush=True)

    for rep in range(args.repeats if args.mode == "warmup" else 1):
        out = post(args.url, {"model": model, "prompt": ids,
                              "max_tokens": args.max_tokens,
                              "temperature": 0.0}, timeout_s=1200)
        choice = out["choices"][0]
        print(f"[tp2trace] client done rep={rep} "
              f"prompt_tokens={len(ids)} "
              f"finish={choice.get('finish_reason')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
