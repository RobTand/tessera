#!/usr/bin/env python3
"""Why the decode regime refuses on a hybrid SSM model, in one serve.

``kl_tool dump --regime decode`` proves every scored position is an M=1 forward
from ``usage.prompt_tokens_details.cached_tokens`` and refuses otherwise. On
LFM2.5-8B-A1B (conv/SSM + attention) it refused at the first scored position:
17 prompt tokens, 0 from the prefix cache. Two explanations fit that -- the
stride does not match the serve's KV block size, or the hybrid model does not
reuse blocks at all -- and this script tells them apart.

It measures, per request, how many prompt tokens the serve served from its
cache, for prefixes of one corpus chunk. Run it against a live serve started
with ``--enable-prompt-tokens-details``; the answers are in the receipt
``docs/measurements/moe-evidence-debt-2026-09-04.md`` section 6.

    docker run -d --name probe --gpus all --ipc=host -p 8000:8000 \
      -v /mnt/shared:/mnt/shared <image> vllm serve <bf16-model> \
      --served-model-name kl-target --host 0.0.0.0 --port 8000 \
      --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.35 \
      --max-logprobs 1024 --enforce-eager --enable-prompt-tokens-details \
      --trust-remote-code
    hybrid_prefix_cache_probe.py http://127.0.0.1:8000/v1/completions <corpus>

Requests are issued strictly sequentially, as the decode regime issues them:
the order is what makes the answer interpretable, because each request leaves
its own end state behind for the next one.
"""
from __future__ import annotations

import json
import sys

import requests

DEFAULT_CORPUS = "/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json"
PREFIX_LENGTHS = (17, 33, 65, 129, 257, 385, 512)


def _usage(payload: dict) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        raise SystemExit(
            "the serve returned no usage.prompt_tokens_details.cached_tokens; "
            "start it with --enable-prompt-tokens-details, or the M=1 claim "
            "this probe is about cannot be checked at all")
    return int(usage["prompt_tokens"]), int(cached)


def _post(url: str, body: dict) -> dict:
    response = requests.post(url, json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else "http://127.0.0.1:8000/v1/completions"
    corpus_path = argv[2] if len(argv) > 2 else DEFAULT_CORPUS
    contract = json.loads(open(corpus_path).read())
    bos = contract["bos_token_id"]
    chunk = list(contract["chunks"][0])
    full = ([bos] + chunk) if bos is not None else chunk
    print(f"corpus {corpus_path}\nchunk 0 length {len(full)}  bos={bos}")

    warm = _post(url, {"model": "kl-target", "prompt": full, "max_tokens": 1,
                       "temperature": 0.0, "add_special_tokens": False})
    print(f"warm-up (the one prefill the decode regime performs): {_usage(warm)}")

    lengths = tuple(length for length in PREFIX_LENGTHS if length <= len(full))
    for length in lengths:
        payload = _post(url, {"model": "kl-target", "prompt": full[:length],
                              "max_tokens": 1, "temperature": 0.0, "logprobs": 8,
                              "return_tokens_as_token_ids": True,
                              "add_special_tokens": False})
        prompt_tokens, cached = _usage(payload)
        print(f"  L={length:4d}  prompt_tokens={prompt_tokens:4d}  "
              f"cached_tokens={cached:4d}  rows_forwarded={prompt_tokens - cached}")

    # The discriminator: a repeat of a prefix the serve has already ANSWERED,
    # rather than one it merely holds attention blocks for.  Derived from the
    # lengths actually probed, so a shorter chunk does not silently skip it.
    repeat_at = max(length for length in lengths if length <= len(full))
    for attempt in (1, 2):
        payload = _post(url, {"model": "kl-target", "prompt": full[:repeat_at],
                              "max_tokens": 1, "temperature": 0.0, "logprobs": 8,
                              "return_tokens_as_token_ids": True,
                              "add_special_tokens": False})
        prompt_tokens, cached = _usage(payload)
        print(f"  repeat {attempt} at L={repeat_at}: prompt_tokens={prompt_tokens} "
              f"cached_tokens={cached}  rows_forwarded={prompt_tokens - cached}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
