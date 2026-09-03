# The window GEMV, served: census, two-arm KL and latency (2026-09-03)

> **STATUS: IN PROGRESS.** Sections 1, 2 (method), 5 and 6 are established and
> final. The numbers in §2 (counts), §3 (KL) and §4 (latency) are **pending** —
> the campaign was queued behind six other agents' jobs on sparky's GPU lock at
> the time of writing. Every pending number is marked `PENDING`; nothing in this
> document is a placeholder standing in for a measurement that was taken and
> disliked.

**What this is.** Issue #10 wired `fp8_gemv.streamed_apply` into the streamed
`TESSERA_FP8` route and proved it bit-exact against the torch decoder at load.
Everything #10 measured was in-process: the kernel against `torch._scaled_mm` at
M = 1, a read-bandwidth figure, an agreement bound. **None of that is a serving
result**, and #83 exists to stop those numbers being cited as one. This document
is the served measurement: a route census in four mode x regime combinations, a
two-arm KL at byte-identical bytes, and latency read from the serving engine.

## 1. What was served, and that it is the exported bytes

One checkpoint, two arms, byte-identical **by construction rather than by
assertion**.

| | |
|---|---|
| Checkpoint | Qwen3-0.6B, `quant_method: tessera`, `TESSERA_FP8` on all 112 declared modules / 196 units |
| Grid / wire | E4M3, CHANNEL plane, window body, `window_bits = 14`, `q256 = 1024`; wire 4.0708 bpp, resident 8.025 bpp |
| Box | sparky — NVIDIA GB10, compute capability 12.1 (sm_121) |
| Runtime | vanilla vLLM 0.28.0, torch 2.13.0+cu130, python 3.12.3, image `vllm/vllm-openai:latest` |
| Tessera | `tessera` / `tessera.serving` 0.1.0 |

`armA` and `armB` under `/home/rob/tessera-runs/ts83/` are `cp -al` hardlinks of
the same export. Every file is **the same inode**, so the two arms cannot differ
in their bytes even if someone re-exported in between — and the driver re-checks
the inode at start, because the confound being guarded is precisely a re-export
between the two serves:

```
model.safetensors  inode 11665664 in armA, armB and the source export
  sha256 ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad
config.json
  sha256 e529fa91cd929d0c77f5e93d3ca2840ffcd1ba649f4af52e71f3bf47c573356b
```

**No re-export was performed for the second arm.** Re-exporting is how a
comparison stops being a comparison.

### The two arms

**Arm A** — the extension builds, and the GEMV lane prepares. Verified offline
before any GPU time that no unit can be refused *by name*: all 196 units parse as
`(body WINDOW, plane CHANNEL, window_bits 14, rates {4}, initial_state None)`,
inside `SUPPORTED_RATES = (1, 2, 4)` and `WINDOW_BITS_SUPPORTED = (14,)`.

**Arm B** — the arm the GEMV replaces. `TORCH_EXTENSIONS_DIR` points at a
**read-only mount**, so `kernel_window_gemv._ext()`'s `os.makedirs` raises before
nvcc is ever consulted, `prepare_fp8_gemv` fails, and `fp8_route` takes the
published fallback (`ext.NATIVE_EXTENSIONS`'s `when_unavailable` -> `torch_window`).
`CUDA_HOME` is untouched and `TMPDIR` stays writable, so this is the smallest
perturbation that reaches the fallback. That matters: `prepare_fp8_gemv` raising
is the *same state* a rate-3 wire or a shard start state reaches on a normal
serve, so arm B is the route's real fallback and not a test double.

This mechanism was **verified in the serving image, not assumed** — because if
`TESSERA_LANE_DOCKER_EXTRA` had failed to reach the container, arm B would have
served with the lane enabled on the same bytes and the delta would have come out
~0, which reads exactly like "the lane is neutral":

```
TORCH_EXTENSIONS_DIR=/ext-ro   (the wrapper's later -e wins over the base one)
TMPDIR=/ext                    (still writable: only the JIT build is refused)
makedirs refused as designed: OSError: [Errno 30] Read-only file system:
                              '/ext-ro/tessera_window_gemv'
```

### What is *not* held equal, stated rather than glossed

**"Same vLLM session" is not achievable here and is not claimed.** Extension
residency is a process fact, so the two arms are necessarily two processes. Held
equal: one image, one teacher npz, one corpus contract, one box, back to back, on
one inode. The arms also get **separate vLLM compile caches** — a measurement
workaround for #91, not a fix (§5).

## 2. The census

Method. `tools/tessera_route_census.py` loads the checkpoint through `vllm.LLM`,
drives one many-row forward and one one-row forward, and reads the route record
every Tessera module wrote from inside the worker. No log line is parsed.

### What a census verdict does and does not prove

Established from the code *before* spending GPU time, by running
`fp8_gemv.census_expected`:

```
eager    decode -> {(gemv, window_gemv), (_scaled_mm, torch_window), (_scaled_mm, window_gemv)}
eager    batch  -> {(_scaled_mm, torch_window), (_scaled_mm, window_gemv)}
compiled decode -> {(_scaled_mm, torch_window), (_scaled_mm, window_gemv), (combined pair)}
compiled batch  -> identical to compiled decode
```

The torch pair `(torch._scaled_mm, torch_window)` is admitted in **every** set.
Three consequences, all load-bearing for how the counts below are read:

1. **Arm B passes the census**, needing neither `--allow-fallback-decoder` nor any
   code change. The fallback is a legitimate route, not a failure.
2. **`verdict: served` therefore cannot tell the two arms apart.** It is a
   structural pass. Engagement is reported from the per-`(symbol, decoder)`
   histogram counts, cross-checked against the serve log's refusal count
   (`grep -c "the window GEMV lane"`: 0 for an engaged arm, 112 for the fallback).
3. **Under a compiled forward the census is weaker still**: `decode` and `batch`
   admit an identical set, and the combined pair is stamped whatever runs
   underneath. **The compiled census proves dispatch, not launch.** Kernel names
   in the torch profile are the launch evidence, which is why §4 carries a profile
   and not only a number.

### Counts

`served/total` per `(mode, regime)`, all four combinations for arm A plus the two
arm-B streamed arms. Arm B is not run resident because the resident route never
reaches the GEMV lane, so a resident arm B is the same serve as a resident arm A
by construction.

**All six censuses read `verdict: served`, 112/112 declared modules in both
phases, with `problems: []`.** Every route reports policy `TESSERA_FP8:<mode>`
and activation contract `fp8_per_token_dynamic` — the contract is unchanged by
the lane, which is the point (§5, #88).

| arm | mode | regime | verdict | served/total | decode `(symbol, decoder)` | prefill `(symbol, decoder)` |
|---|---|---|---|---|---|---|
| armA | resident | eager | served | 112/112 | `(_scaled_mm, torch_window)` | `(_scaled_mm, torch_window)` |
| armA | resident | compiled | served | 112/112 | `(_scaled_mm, torch_window)` | `(_scaled_mm, torch_window)` |
| armA | streamed | eager | served | 112/112 | **`(tessera_window_gemv::gemv, window_gemv)`** | `(_scaled_mm, window_gemv)` |
| armA | streamed | compiled | served | 112/112 | `(_scaled_mm+gemv, torch_window+window_gemv)` | `(_scaled_mm+gemv, torch_window+window_gemv)` |
| armB | streamed | eager | served | 112/112 | `(_scaled_mm, torch_window)` | `(_scaled_mm, torch_window)` |
| armB | streamed | compiled | served | 112/112 | `(_scaled_mm, torch_window)` | `(_scaled_mm, torch_window)` |

Read down the decode column, four facts:

1. **The GEMV runs, and the census names it.** `armA/streamed/eager` decode is the
   only cell reporting the symbol `tessera_window_gemv::gemv` — on all 112
   modules. That is the observation #10 never took.
2. **Its prefill is not the GEMV, by the lane's own rule.** `armA/streamed/eager`
   prefill reports `(_scaled_mm, window_gemv)`: the lane *prepared* (hence its
   decoder), the many-row forward went to the materialised GEMM (hence
   `_scaled_mm`), exactly as `GEMV_MAX_M = 8` says it must. **So the two arms
   differ in both phases, not just decode** — in prefill arm A decodes through
   the lane's kernel while arm B decodes in torch, and only then do both call
   `_scaled_mm`.
3. **`resident` is untouched by the lane** — the same `(_scaled_mm, torch_window)`
   as arm B. This is why a resident arm B was not run: it would be the same serve.
4. **The compiled rows do separate the arms**, arm A stamping the combined pair
   and arm B the torch pair, because no lane was prepared in arm B. That is still
   **dispatch, not launch**: the combined pair is stamped statically for every M.

The serve logs agree independently, and this is the cross-check that does not go
through the census at all — count of `"the window GEMV lane did not prepare"`:

```
0    census-armA-resident-eager.log        0    census-armA-streamed-eager.log
0    census-armA-resident-compiled.log     0    census-armA-streamed-compiled.log
112  census-armB-streamed-eager.log      112  census-armB-streamed-compiled.log
```

**0 in every arm A, exactly 112 in every arm B** — one refusal per declared
module, each `OSError: [Errno 30] Read-only file system`. The arms are the two
states the route is meant to have, and nothing else differs.

A preflight pair of resident censuses taken at master `c71f37b` (before this
branch's `experiments/` commits) read `served`, 112/112 in both phases, on
`TESSERA_FP8:resident` with decoder `torch_window`. They are kept at
`preflight/` and are **not** part of the reported set: all six above are taken at
one commit so a reader does not have to check that two commits describe the same
`src/`.

## 3. The served KL, two arms, byte-identical bytes

PENDING. Method: `kl_tool.py dump` against each served arm, then `compare`
against the image-matched teacher `qwen_teacher_bf16_v028.json.npz`, corpus
contract `corpus_qwen_n8_s512.json` (**Qwen-tokenized** — the default contract is
GLM-tokenized and `kl_tool` refuses the mismatch; the refusal was not worked
around). Mean, `confident`, top-1 agreement **and the p99/max tail** are reported
for each arm: a lower mean KL can hide a heavier tail, and for a lane that is
meant to be arithmetically equivalent to the path it replaces, tail movement is
the interesting result rather than the mean.

## 4. The latency

PENDING, and **scoped down deliberately**: the four streamed arms
(armA/armB x eager/compiled) are taken, the two armA-resident arms are not.
Resident never reaches the GEMV lane, so those two serves would have bought
context rather than evidence, at ~30 minutes of a box five other agents were
queued for. The streamed arms are kept because they carry the **profile**, and
which kernels launched is not load-sensitive even when the timing is.

Method, and why it is not #10's bench: every number is read from the
**serving process**. `vllm:time_to_first_token_seconds` and
`vllm:time_per_output_token_seconds` off `/metrics`, differenced across each
driven window, so they describe the requests driven and nothing else; client wall
clock is recorded beside them as a cross-check, never as the headline.

Two instruments, both required. The in-process torch profile says where time went
inside the run; the Netdata series says whether the box was loaded, which no
in-process tool can see. **On GB10 `gpu_utilization` is non-diagnostic** — it
means "a kernel is resident", not "the SMs are working" — so the box-side reading
is **power against the ~140 W envelope**, ranked as work per joule.

**Idle baseline: 17.0 W median** (mean 17.67, min 12, max 23.7) over
2026-09-03 21:54–22:24 UTC, `nvidia_smi.gpu_power_draw`, NVIDIA GB10. Arm power is
reported as watts **above this baseline**. That baseline was not taken on an
unused box: for its whole window five agents' jobs held sparky's GPU lock and the
GPU never left ~17 W. **Holding the GPU lock is not the same as loading the GPU**,
which is also why each arm's window is checked for foreign load rather than
assumed quiet.

### The box was contended, and every latency number says so

**This is the load the latency half was taken under, and it is not a quiet box.**
sparky ran at load average 33–68 on 20 CPUs (1.7–3.4 runnable processes per core)
throughout this session: five jobs queued on the GPU lock and eight concurrent
encodes from other agents. **The GPU lock serialises GPU jobs, not CPU-bound
ones**, so holding it is not evidence the box was quiet — the same point the
17 W idle baseline makes from the other direction.

A host-driven latency number taken at that load is noise. So `os.getloadavg()` is
recorded at both ends of every timed window into each receipt, with the CPU count
so it reads as a ratio, and each receipt carries its own `contended` verdict at a
threshold of one runnable process per core. **It is a label, never a filter**:
the numbers are reported either way, because a contaminated measurement that says
it is contaminated is useful while one that does not is worse than none.

Read the latency section accordingly — a **contended TTFT/TPOT is not evidence of
a latency win or loss**, and none is claimed from it. What the profile's kernel
names establish (which kernels launched, under eager and under a compiled
forward) is *not* load-sensitive and stands on its own. The census and the KL are
correctness measurements — which decoder ran, and the KL of byte-identical bytes
— and contention changes how long they take, not what they say.

### A confound caught before it was measured

The load driver built one token-id prompt per `(prompt_tokens, max_tokens)` pair
and sent it n times, against a serve running vLLM's default
`enable_prefix_caching=True`, with the warmup already driving the 512-token prompt
twice. **Every "prefill" request would have been a full prefix-cache hit**: the
many-row forward this A/B exists to compare would never have run, and the reported
TTFT would have been the cost of a cache lookup. Fixed in `cee31a3` by perturbing
the ids per request from a counter monotonic across the whole run. Turning prefix
caching off was rejected as the fix — it would have made the latency serve a
different serve from the one whose KL is reported. No measurement is retracted by
this: it was caught while queued, before the campaign ran.

## 5. Mechanism notes

**#88 — a design doc forbade the dispatch master ships.**
`docs/design/window-gemv-a-side.md` §5 item 1 read "Neither may dispatch the GEMV
inside `TESSERA_FP8`'s `apply()`", which `fp8_route.py` does. The code is right
and the doc was stale: the route quantises, hands the GEMV values, and the
executed activation contract stays `fp8_per_token_dynamic`. What ships is A8
arithmetic executed by a W?A16 kernel. Doc fixed on sight (`3a38a38`); #88 records
it.

**#91 — the compile-cache key does not see the GEMV lane.** `config.py` declares
only `serve_mode` into vLLM's compile-cache identity. The two streamed arms trace
structurally different graphs — arm A a single `tessera::fp8_streamed_apply` node,
arm B a window decode plus `torch._scaled_mm` — from byte-identical traced
sources, so they would share one cache slot. Worked around here with a separate
`VLLM_CACHE` per arm. **That is a measurement workaround, not a fix**; #91 owns
the fix.

**No contract change was needed.** `runtime_contract.json` already carries
`tessera_e4m3_k1_dense_sm121_{decode,batch}_scaled_mm_w8a8` at rung 1024. No wire
change, no schema minor, no `encoder_profile_id` change was made.

## 6. Where the numbers live

Under `/home/rob/tessera-runs/ts83/`:

- `arm-hashes.txt` — the sha256s above; `armA/`, `armB/` — the hardlinked arms.
- `census-<arm>-<mode>-<regime>.json` + `.log` — the six censuses.
- `preflight/` — the two master-commit resident censuses, excluded from the set.
- `kl_tessera_ts83-<arm>-streamed-<regime>.json` — the KL receipts.
- `latency-<arm>-<mode>-<regime>.json` + `prof-<arm>-<mode>-<regime>/` — the
  engine histograms and the chrome traces.
- `power-baseline.json` — the idle series above.
- `pytest-after.txt` — the suite line.

Drivers, in the worktree: `experiments/window_gemv_served.sh` (census + KL),
`experiments/window_gemv_latency.sh` (profiled latency serve),
`experiments/window_gemv_load.py` (load driver),
`experiments/window_gemv_trace_summary.py` (chrome-trace kernel aggregation).
