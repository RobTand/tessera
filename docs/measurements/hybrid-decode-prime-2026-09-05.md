# Reaching M=1 on a hybrid conv/SSM serve: what the prefix cache will and will not resume

**Date** 2026-09-05 · **Host** sparky (GB10, sm_121) · **Issue** tessera#192

`kl_tool dump --regime decode` refused on LFM2.5-8B-A1B. This receipt records
what the pinned serve actually does with a repeated prefix, which of two primes
lifts the refusal, why the other route (a serve flag) does not exist on this
runtime, and the one measured row the rule fitted here does not explain.

Everything below was measured on the digest the two `routed_moe` cells name.
A measurement on any other digest attests a different runtime.

## 0. Identities

| | |
|---|---|
| image | `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c` |
| vLLM | `0.28.1rc1.dev397+gfd4a15126.d20260904` (from the serve banner, `serve_align.log:18`) |
| model | `/mnt/shared/models/LFM2.5-8B-A1B-BF16` (BF16 source, served as `kl-target`) |
| corpus contract | `/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json`, sha256 `c8eeabde8fce06c78073d5f2c3783ed217d2dd96f6f98a6a565920b1e7ce6ff5` |
| logs | `/home/rob/tessera-runs/ts192/` (`probe_align.log`, `prime_variants_align.log`, `prime_anomaly_align.log`, `prime_512_align.log`, `probe_all.log`, `serve_align.log`, `serve_all.log`) |
| instrument | `/home/rob/dq-runs/kl_tool.py`, md5 `8d70629f6df55e560fe5ad845e3a66dd` on sparky **and** sparklina after the change (`c88a91b7f300870272fedb59235c40ca` before it); `kl_estimator.py` unchanged at `0a48e5c8014f9d7f91cc89b8866830cb` |

Serve command (arm 1; arm 2 adds `--mamba-cache-mode all`):

```
docker run -d --name kl192 --gpus all --ipc=host -p 8000:8000 \
  -v /mnt/shared:/mnt/shared eugr/spark-vllm@sha256:0afec8d4... \
  vllm serve /mnt/shared/models/LFM2.5-8B-A1B-BF16 \
    --served-model-name kl-target --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.35 \
    --max-logprobs 1024 --enforce-eager --enable-prompt-tokens-details \
    --trust-remote-code
```

The engine reports `block_size=16`, `mamba_block_size=16`,
`enable_prefix_caching=True`, `mamba_cache_mode='align'`, and
`Padding mamba page size by 300.00% to ensure that mamba page size and
attention page size are exactly equal` (`serve_align.log:53`). One serve held
the GPU at a time; both containers were removed and the GPU returned to
5.79 W / 0 % before this was written.

## 1. Why `cached_tokens` is trusted as a *joint* reuse count

The whole argument rests on `rows forwarded = prompt_tokens - cached_tokens`.
On a hybrid model that is only sound if `cached_tokens` counts the reuse the
**recurrent** state also grants, not only attention-block reuse. Two measured
rows rule out an attention-only reading:

* `L=17` after a 513-token warm-up reports `cached=0`. The warm-up left
  attention blocks covering the whole chunk; an attention-only counter would
  have reported 16.
* A fresh 512-token prompt `Q`, issued unscored and then scored, reports
  `cached=0` on the *second* request (`prime_512_align.log`) — again with the
  first request's attention blocks present.

Both rows are the recurrent state refusing what attention would have allowed,
so the number tracks the binding constraint. That is what makes the row count
sound.

## 2. What the sweep does today (arm 1, `mamba_cache_mode=align`)

`experiments/hybrid_prefix_cache_probe.py`, one serve, strictly sequential
(`probe_align.log`):

```
warm-up (the one prefill the decode regime performs): (513, 0)
  L=  17  prompt_tokens=  17  cached_tokens=   0  rows_forwarded=17
  L=  33  prompt_tokens=  33  cached_tokens=  16  rows_forwarded=17
  L=  65  prompt_tokens=  65  cached_tokens=  32  rows_forwarded=33
  L= 129  prompt_tokens= 129  cached_tokens=  64  rows_forwarded=65
  L= 257  prompt_tokens= 257  cached_tokens= 128  rows_forwarded=129
  L= 385  prompt_tokens= 385  cached_tokens= 256  rows_forwarded=129
  L= 512  prompt_tokens= 512  cached_tokens= 384  rows_forwarded=128
  repeat 1 at L=512: prompt_tokens=512 cached_tokens=496  rows_forwarded=16
  repeat 2 at L=512: prompt_tokens=512 cached_tokens=496  rows_forwarded=16
```

Read the middle rows in order: each resumes from its *predecessor's* end,
aligned down to a block. L=33 resumes from 17→16, L=65 from 33→32, L=129 from
65→64. The decode sweep visits L ∈ {1, 17, 33, …} exactly once each and
ascending, so every scored request can only resume one stride behind itself
and forwards `stride+1` rows. The warm-up's 513-token prefill leaves attention
blocks but no resumable recurrent state at an interior position, which is why
the first scored position sees zero.

So the refusal is not a stride mismatch and not "the model does not cache". It
is that **the sweep never asks for the same prefix twice**, and on this
architecture only a prefix the serve has already *answered* is resumable.

## 3. The A/B that chooses the prime (`prime_variants_align.log`)

Six fresh chunks — a used chunk would let the previous cell answer the next.
Variant A primes `full[:L-1]`, variant B primes `full[:L]`; the prime is
unscored (no `logprobs`), the scored request is unchanged in both.

| variant | chunk | L | request | prompt | cached | rows |
|---|---:|---:|---|---:|---:|---:|
| A | 1 | 129 | prime `full[:128]` | 128 | 0 | 128 |
| A | 1 | 129 | scored `full[:129]` | 129 | **0** | **129** |
| A | 2 | 17 | prime `full[:16]` | 16 | 0 | 16 |
| A | 2 | 17 | scored `full[:17]` | 17 | **0** | **17** |
| A | 3 | 257 | prime `full[:256]` | 256 | 0 | 256 |
| A | 3 | 257 | scored `full[:257]` | 257 | **0** | **257** |
| B | 4 | 129 | prime `full[:129]` | 129 | 0 | 129 |
| B | 4 | 129 | scored `full[:129]` | 129 | **128** | **1** |
| B | 5 | 17 | prime `full[:17]` | 17 | 0 | 17 |
| B | 5 | 17 | scored `full[:17]` | 17 | **16** | **1** |
| B | 6 | 257 | prime `full[:257]` | 257 | 0 | 257 |
| B | 6 | 257 | scored `full[:257]` | 257 | **256** | **1** |

**Variant A does not work: 3/3.** #192's follow-up recommended it, reasoning
that a request ending at `L-1` (a multiple of 16 for every L in the stride-16
set) would leave a state there. It does not: a request ending exactly on a
block boundary leaves nothing the next request can resume from at that
boundary. **Variant B works: 3/3**, and is what `--decode-prime` sends.

The objection that motivated A — that re-issuing the scored prefix makes the
prime indistinguishable from a scored request in a served-request histogram —
is answered by **shape** rather than by length. The prime carries no
`logprobs`, so it is warm-up shaped, cannot contribute a scored position even
by accident, and a histogram separates the three populations exactly
(§5: 8 warm-ups, 248 primes, 256 scored).

## 4. The rule this fits, and the row it does not explain

Fitted to the twenty measured rows, the serve behaves as if a request of
length `P` leaves a resumable state at `P-1` **when `P-1` is a whole number of
blocks**, and a later request resumes from the longest such state that is a
strict prefix of it. That reproduces §2, §3 and:

```
prime_anomaly_align.log
  unscored full[:128]                prompt= 128 cached=   0 rows= 128
  scored   full[:113]                prompt= 113 cached=   0 rows= 113
  scored   full[:129]                prompt= 129 cached= 112 rows=  17
  scored   full[:129] again          prompt= 129 cached= 128 rows=   1

prime_512_align.log
  unscored R[:497]                   prompt= 497 cached=   0 rows= 497
  scored   R[:497]                   prompt= 497 cached= 496 rows=   1
```

**The residue — one class, two instances, observed and unexplained.**
Repeating a prompt whose length is an *exact multiple* of the block reports
`cached = length - 16`, not 0 and not `length - 1`:

```
probe_align.log      repeat 1 at L=512  cached=496  rows=16
probe_align.log      repeat 2 at L=512  cached=496  rows=16
prime_512_align.log  Q[:512] issued a third time  cached=496  rows=16
```

That is reuse of all-but-one-block. It is recorded as observed, not explained;
it is **outside the sweep's request set**, because every scored prefix the
decode regime issues is `1 mod 16` and every prime is that same length. The
fake in `tests/test_kl_tool_decode_regime.py` states in its docstring that it
does not reproduce this row, so nobody reads the fake as vLLM's algorithm.
This is why the probe now prints a repeat at both a `1 mod 16` length and a
block multiple: an earlier form repeated only at the block multiple, and its
16-row output was read as "no reuse".

## 5. The primed dump (the deliverable)

```
CUDA_VISIBLE_DEVICES="" .../python kl_tool.py dump --model kl-target \
  --url http://127.0.0.1:8000/v1/completions \
  --corpus-contract .../corpus_n8_s512.json --role teacher \
  --teacher-label BF16-LFM2.5-8B-A1B --top-k 1024 \
  --regime decode --decode-prime --out teacher_decode_primed.npz
```

Unprimed on the same live serve it refuses, with the new message:

```
position 3: the serve forwarded 17 rows, not 1 (49 prompt tokens, 32 from the prefix cache)
```

Primed: `dumped 256 positions x k<=1024 (decode regime)`, **26.635 s**.
Regime record (`teacher_decode_primed.meta.json`):

| field | value |
|---|---|
| `regime.name` | `decode` |
| `regime.stride` | 16 |
| `regime.scored_positions` | 256 |
| `regime.warmup_prefills` | 8 |
| `regime.prime.enabled` | `true` |
| `regime.prime.requests` | 248 (`L=1` is already a one-row forward and is not primed) |
| `regime.prime.scored_requests_unchanged` | `true` |
| `regime.rows_per_scored_forward` | 1 |
| `regime.cached_tokens_min` / `max` | 0 / 496 |
| `metric.topk_coverage_min` / `mean` | 0.8593973158537677 / 0.9856009961524046 |

Artifact: `/home/rob/tessera-runs/ts192/teacher_decode_primed.npz`, also copied
to `/mnt/shared/tessera-runs/ts192/` so the student compare can run from either
box. sha256 `3bfa14efb864cee15d57e9b7935b710241c587b9c15010a660c3956e4d5e7470`;
`kl_tool fingerprint` → `9148fb60a86788cd09b7302e1512bbb8fb84cacf1f5701994a1e724d5c4233c7`.

This is the **first LFM teacher half taken on the cells' pinned digest**: the
existing prefill teacher dump was taken on image `337dae6b` / vLLM 0.28.0, so
it cannot pair with a student dump taken here.

## 6. There is no serve-side route on this runtime (arm 2)

`vllm serve --help=mamba-cache-mode` offers `{align, all, none}`, and `all` is
the mode whose description says it keeps the state for every position. It does
not do that here. A second serve, identical but for `--mamba-cache-mode all`,
produced a **byte-identical probe table** (`probe_all.log` vs
`probe_align.log`). The runtime says why, in its own log:

```
serve_all.log:22 non-default args: {..., 'mamba_cache_mode': 'all', ...}
serve_all.log:26 WARNING [cache.py:330] Mamba cache mode 'all' is deprecated
    and will be removed in an upcoming release.
serve_all.log:27 WARNING [config.py:636] Hybrid or mamba-based model detected
    without support for prefix caching with Mamba cache 'all' mode: falling
    back to 'align' mode.
```

Line 22 attests the flag was received; line 27 attests it was discarded. So
outcome 2 of #192 — "the runtime can be asked for the regime directly" — is
closed by the runtime, not by argument, and the refusal message in `kl_tool.py`
quotes line 27 rather than asserting the conclusion (AGENTS.md: a claim about
another runtime is attested, never asserted).

## 7. What this does not claim

* The decode-regime *student* dump does not exist yet, so no cell's `grade`
  changes. Leg A is unblocked and half-done: the teacher half is on disk, the
  student serve and the compare remain.
* `--decode-prime` is **off by default**, so the dense decode receipts
  (`tessera-decode-regime-kl-2026-09-03.md`,
  `tessera-compiled-decode-kl-r6-2026-09-04.md`) reproduce request for request.
  They are attention-only Qwen artifacts where one request already resumes at
  `L-1` and no prime is needed.
* The prime doubles the request count. The dump is HTTP-bound; 256 positions
  took 26.6 s.
* Nothing here measures MoE quality, full-vocabulary KL, or a compiled serve.
