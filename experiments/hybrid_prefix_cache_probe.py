#!/usr/bin/env python3
"""Why the decode regime refuses on a hybrid SSM model, in one serve.

``kl_tool dump --regime decode`` proves every scored position is an M=1 forward
from ``usage.prompt_tokens_details.cached_tokens`` and refuses otherwise. On
LFM2.5-8B-A1B (conv/SSM + attention) it refused at the first scored position:
17 prompt tokens, 0 from the prefix cache. Two explanations fit that -- the
stride does not match the serve's KV block size, or the hybrid model does not
reuse blocks at all -- and this script tells them apart.

It measures, per request, how many prompt tokens the serve served from its
cache, for prefixes of one corpus chunk, and then runs the A/B that decides
what ``kl_tool --decode-prime`` must send: priming ``full[:L-1]`` before
scoring ``full[:L]``, against priming ``full[:L]``. Run it against a live serve
started with ``--enable-prompt-tokens-details``. The measured answers are in
the receipt ``docs/measurements/hybrid-decode-prime-2026-09-05.md``; section 6
of ``docs/measurements/moe-evidence-debt-2026-09-04.md`` is the earlier,
partly superseded write-up.

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

# The serve's KV block size (vLLM ``--mamba-block-size``, 16 by default on this
# model). Reuse is granted in whole blocks, so a prefix whose length is
# ``1 mod BLOCK`` is the only one that can leave exactly one row to forward.
BLOCK = 16

# One chunk per A/B cell: each cell needs a prefix the serve has never seen, and
# reusing a chunk would let the previous cell's state answer the next one.
# ``variant`` names what the unscored prime asks for -- "A" is ``full[:L-1]``,
# "B" is ``full[:L]`` -- and the scored request is ``full[:L]`` in both.
VARIANT_CELLS = (
    ("A", 1, 129), ("A", 2, 17), ("A", 3, 257),
    ("B", 4, 129), ("B", 5, 17), ("B", 6, 257),
)


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


def _ask(url: str, prompt: list[int], *, scored: bool) -> tuple[int, int]:
    """One request. ``scored`` is the only difference the prime relies on.

    An unscored request asks for no ``logprobs``, so it is warm-up shaped and
    cannot contribute a scored position; it still runs the forward, which is
    the whole point of priming with it.
    """
    body = {"model": "kl-target", "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "add_special_tokens": False}
    if scored:
        body.update({"logprobs": 8, "return_tokens_as_token_ids": True})
    return _usage(_post(url, body))


def _show(tag: str, usage: tuple[int, int]) -> None:
    prompt_tokens, cached = usage
    print(f"  {tag:38s} prompt={prompt_tokens:4d} cached={cached:4d} "
          f"rows={prompt_tokens - cached:4d}")


def _chunk(contract: dict, index: int) -> list[int]:
    bos = contract["bos_token_id"]
    tokens = list(contract["chunks"][index])
    return ([bos] + tokens) if bos is not None else tokens


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else "http://127.0.0.1:8000/v1/completions"
    corpus_path = argv[2] if len(argv) > 2 else DEFAULT_CORPUS
    contract = json.loads(open(corpus_path).read())
    full = _chunk(contract, 0)
    print(f"corpus {corpus_path}\nchunk 0 length {len(full)}  "
          f"bos={contract['bos_token_id']}")

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
    # rather than one it merely holds attention blocks for.  It must be
    # ``1 mod BLOCK``: a request that ends on a block boundary leaves its state
    # one whole block short, so repeating THAT one forwards BLOCK rows and
    # would read as a failure to reuse when it is not.  Derived from the
    # lengths actually probed, so a shorter chunk does not silently skip it.
    repeat_at = max(length for length in lengths if length % BLOCK == 1)
    for attempt in (1, 2):
        _show(f"repeat {attempt} at L={repeat_at}",
              _ask(url, full[:repeat_at], scored=True))

    # And the same repeat at a length that is a whole number of blocks, which
    # is the row that misleads: it reuses all but one block, not all but one
    # token.  Printed on purpose, because the earlier form of this probe
    # repeated only here and its BLOCK-row output was read as "no reuse".
    boundary_at = max(length for length in lengths if length % BLOCK == 0)
    for attempt in (1, 2):
        _show(f"repeat {attempt} at L={boundary_at} (a block multiple)",
              _ask(url, full[:boundary_at], scored=True))

    # The A/B.  Both variants leave the serve holding a state for the scored
    # request to resume from; only one of them leaves it at L-1, and which one
    # is not derivable from vLLM's documentation -- hence a measurement.
    for variant, index, length in VARIANT_CELLS:
        if index >= len(contract["chunks"]):
            continue
        cell = _chunk(contract, index)
        if length >= len(cell):
            continue
        prime_at = length - 1 if variant == "A" else length
        print(f"chunk {index}, L={length}: variant {variant} -- prime "
              f"full[:{'L-1' if variant == 'A' else 'L'}] (unscored), then "
              f"score full[:L]")
        _show(f"prime full[:{prime_at}]", _ask(url, cell[:prime_at], scored=False))
        _show(f"scored full[:{length}]", _ask(url, cell[:length], scored=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
