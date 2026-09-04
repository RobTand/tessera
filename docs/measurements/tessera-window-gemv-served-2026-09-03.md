# The window GEMV, served: census, two-arm KL and latency (2026-09-03)

> **STATUS: the census and the KL are measured and final. The KL does not
> reach the kernel. The latency is deferred.** §2 (census) is complete: six
> serves, all four mode x regime combinations, 112/112. §3 (KL) is complete for
> three of four arms, **and it does not measure what #83 asked for**: the dump
> scores prompt positions, so every scored forward is 512 rows and routes past
> the GEMV kernel (`GEMV_MAX_M = 8`) in both arms. The arms are bit-identical
> there, which establishes that the lane does not perturb the many-row path -- a
> real null, and a smaller claim than "bit-exact as served". **The served KL of
> the GEMV kernel itself is not taken by this campaign**; §3 says what would take
> it. All four arms are measured: both regimes give **0 of 8,380,400 differing
> values** between the arms, and the eager-vs-compiled difference is identical in
> the arm that uses the lane and the arm that does not, so it belongs to the
> compiled forward (#16) and not to this lane.
> §4 (latency) is **deferred to a quiet box** and states why:
> sparky ran at load 33-68 on 20 CPUs and went into swap, the two arms taken
> differ by 8x where the kernel difference is at most ~2x, and no latency claim is
> made from them. One deliverable is **missing rather than deferred** and §4 says
> so plainly: the original campaign captured no profile at all, because the
> harness enabled the profiler with an environment variable vLLM 0.28 does not
> have. **That is now fixed and the key arm re-taken**: armA/streamed/compiled
> shows `window_gemv_kernel<14, ...>` launching 9,016 times, which closes the
> compiled census's dispatch-not-launch limit. The other three arms still have no
> trace.
> Nothing in this document is a placeholder standing in for a measurement that was
> taken and disliked.
>
> **Since this was written, both gaps have owners and one of them has a
> number.** The GEMV's own served KL was taken by the follow-up campaign filed
> as #102, which gave `kl_tool.py` a decode regime and re-ran these two arms on
> this inode at M = 1: mutual `KL >= 0.012111` at 91.02% top-1 over 256 scored
> forwards, receipt at
> `docs/measurements/tessera-decode-regime-kl-2026-09-03.md`, with the arms'
> disagreement filed as #110. The latency A/B §4 defers is tracked by #109.
> That campaign is **eager**. The compiled half -- what vLLM runs by default,
> and where this campaign's only KL is §3's prefill-regime null -- was filed as
> **#113** and measured the next day: on these same two arms, compiled, the
> decode regime reads mutual `KL >= 0.012585` at 88.67% where the prefill
> regime reads `0.000000` at 100.00% -- a number that sits **below** the same-artifact rebuild delta (`KL >= 0.019423` decode, arm A re-served, one lane state, two builds), so it is **not** yet a lane-attributable separation, receipt at
> `docs/measurements/tessera-compiled-decode-kl-2026-09-04.md` §7. Nothing is
> restated here as this campaign's result; the pointers are here so a reader who
> lands on this receipt is not left at its gaps.
>
> **So, exactly:** #83's census (all four mode x regime combos, 112/112) and its
> two-arm served KL (eager/streamed, decode regime, #102: mutual
> `KL >= 0.012111` at 91.02% top-1, arms 0.436065 / 0.432477 against the BF16
> teacher) are measured and on master; the served latency A/B is unmeasured
> (#109), the compiled-serve decode-regime KL is measured off-campaign (#113:
> mutual `KL >= 0.012585` at 88.67%, below a `0.019423` same-artifact rebuild
> delta, so not yet lane-attributable), and which arm is closer to BF16 at
> M = 1 is open (#110) -- more open than before, since the arms' ordering flips
> between eager and compiled at 256 positions and a rebuild of one arm moves
> its teacher KL 5.3x further than the flip. **Nothing here is a served
> latency or speed claim for the lane.**

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
   in a torch profile are the launch evidence, and the original campaign captured
   no profile at all — the harness enabled it with an environment variable vLLM
   0.28 does not have (§4, fixed in `dd42ff4`). **That gap is now closed for the
   compiled arm**: a re-run of `armA/streamed/compiled` under the fix shows
   `window_gemv_kernel<14, ...>` launching **9,016 times** (§4). So the compiled
   verdict is backed by launch evidence after all. The eager arm never depended on
   it: its decode phase reports `(tessera_window_gemv::gemv, window_gemv)` at
   `M1:...` shapes, a symbol only the kernel provides.

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

1. **The GEMV runs, at the shape it exists for, and the census names it.**
   `armA/streamed/eager` decode is the only cell reporting the symbol
   `tessera_window_gemv::gemv` — on all 112 modules, `state: served`, at shapes
   `M1:N1024:K2048`, `M1:N1024:K3072`, `M1:N4096:K1024`, `M1:N6144:K1024`. **M = 1.**
   The same serve's prefill regime records `torch._scaled_mm` at `M64:...`. That
   is the observation #10 never took, and the shapes are what make it more than a
   label: the kernel was entered in the regime it was written for, not merely
   selected. (It is also, read the other way, why §3's KL says nothing about the
   kernel — the KL's scored forwards are M = 512.)
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

> **READ THIS FIRST: the GEMV kernel was not on the KL path, in either arm --
> but the two arms were still two different programs.** The result below is a
> real, exact null over 4,190,200 values, and what it establishes is a *served
> native-vs-torch decode identity*: at M = 512 arm A went
> `fp8_gemv.streamed_apply` -> `ext.window_decode` while arm B went
> `tessera_prepared.decode()` in pure torch. That is worth having and it is
> what section 2's census shows. It is **not** the served KL of the window
> GEMV, which is the thing this campaign set out to measure. `kl_tool.py dump` scores prompt positions -- it sends
> `max_tokens: 1` with `prompt_logprobs` over a 512-token prompt
> (`kl_tool.py:262`) -- so every one of the 4088 scored positions comes from a
> **512-row prefill forward**. `GEMV_MAX_M = 8`, so M = 512 routes to the
> materialised `torch._scaled_mm` in **both** arms, which this campaign's own
> census records directly: armA's prefill phase reports
> `(torch._scaled_mm, window_gemv)`, and `census_expected`'s eager *batch* set
> contains no `gemv` symbol at all. During the KL dump the two arms ran the same
> GEMM over the same materialised FP8 bytes.
>
> **So the served numerical output of the GEMV kernel is not measured by this
> campaign.** That is a gap in the deliverable, stated rather than papered over,
> and section 4's missing trace compounds it: the strongest evidence here that
> the kernel ran at all is the eager census decode record (and, under compile,
> the recovered trace in §4 showing 9,016 `window_gemv_kernel` launches), and it
> is good
> evidence as far as it goes: `symbol: tessera_window_gemv::gemv`,
> `decoder: window_gemv`, `state: served`, **112 modules**, at shapes
> `M1:N1024:K2048` and friends -- M = 1, which is the regime the kernel exists
> for. The same serve's prefill regime records `torch._scaled_mm` at `M64:...`.
> So the kernel ran, at the right shape, on every module. **What it computed, as
> served, has not been compared to anything.**

**What the two-arm KL does establish: the lane's presence does not perturb the
path it does not serve.** On the eager pair the two
arms' served logprob dumps are **bit-identical** -- 0 of 8,380,400 values differ,
in both the top-K token ids and their logprobs:

| eager pair | arrays | values compared | differing | max abs delta |
|---|---|---|---|---|
| **eager** armA vs armB, `ids` (int32, 4088x1025) | 1 | 4,190,200 | **0** | 0 |
| **eager** armA vs armB, `lps` (float32, 4088x1025) | 1 | 4,190,200 | **0** | 0 |
| **compiled** armA vs armB, `ids` | 1 | 4,190,200 | **0** | 0 |
| **compiled** armA vs armB, `lps` | 1 | 4,190,200 | **0** | 0 |

Both regimes, then: **0 of 8,380,400 values differ in each**, on byte-identical
bytes, with the GEMV lane refusing 0/112 modules in arm A and 112/112 in arm B in
both.

**Build provenance, which makes the eager pair the cleanest comparison here.**
The two eager arms carry the **same `build_fingerprint`**, `eaeedd6b3ee434be…` —
byte-identical weights *and* an identical build, differing only in whether the
lane was reachable. The two compiled arms carry different fingerprints
(`d62c417b…`, `f58a4e78…`), which is expected and deliberate: each arm compiles
into its **own `VLLM_CACHE` root**, because vLLM's compile-cache key does not see
the GEMV lane (#91) and a shared cache would let one arm serve the other arm's
compiled artifacts. Separate roots are what keep the compiled A/B honest; the
cost is that the compiled pair is not build-identical, only weight-identical.

All six censuses and every arm in this document were produced from **one
checkout at branch commit `3d9b2dd`** (off master `c71f37b`), before #100 added
`image_digest` to the build identity. No arm straddles that change, and nothing
was rebased mid-campaign.

`ids` is the **top-1024 set plus the prompt token at each of 4088 scored
positions**, so its ordering tracks the logits directly: any numerical difference
between the two paths would have reordered ties and shown up as differing ids
long before it moved a mean. It did not.

**Under M > 8, identity is the expected outcome, not a surprise.** Both arms
decode the same wire to the same FP8 weights and hand them to the same
`_scaled_mm`; the honest reading is that this **confirms the lane's weight
materialisation is exact on the many-row path** and that installing the lane
costs nothing there. That is worth having -- a lane that perturbed the path it
does not serve would be a defect -- but it is a smaller claim than "the window
GEMV is bit-exact as served", and this document does not make the larger one.

The arms are not vacuously the same object: the census records a different
decoder in each, and the serve logs record the GEMV lane refusing **112 of 112**
modules in arm B and **0 of 112** in arm A. The routes differed. On the forwards
the KL actually scored, the arithmetic could not.

### The KL, for completeness

Against the image-matched teacher, all four figures agreeing to full float
precision on the eager pair, as bit-identical dumps require:

| arm | regime | KL `all` mean | `confident` mean | p99 | max |
|---|---|---|---|---|---|
| armA (GEMV) | eager | 0.46599389451679424 | 0.3845310749133978 | 2.8054884270366567 | 7.624929 |
| armB (torch) | eager | 0.46599389451679424 | 0.3845310749133978 | 2.8054884270366567 | 7.624929 |
| armA (GEMV) | compiled | 0.4668730966935983 | 0.3867850368604267 | 2.7715860806290746 | -- |
| armB (torch) | compiled | 0.4668730966935983 | 0.3867850368604267 | 2.7715860806290746 | -- |

Arm A vs arm B, eager: **delta +0.000000, ratio 1.0000x, on the mean and on the
tail alike** -- on prefill-scored positions both arms served through
`_scaled_mm`. The tail is reported beside the mean because a lower mean KL can
hide a heavier tail; there is no tail movement to report either.

**The eager/compiled difference is now shown NOT to be this lane's, rather than
argued.** armA compiled reads 0.46687 against armA eager's 0.46599 -- and armB
compiled reads **0.4668730966935983, identical to armA compiled to the last
digit**, with bit-identical dumps. The divergence is therefore the same size in
the arm that uses the GEMV lane and the arm that does not: it is a property of
the compiled forward, not of the lane. The two dumps differ almost
everywhere (only 71,564 of 4,190,200 ids equal), which is what a compiled forward
reordering a top-1024 set looks like and *not* a corpus mismatch -- both dumps
carry the same corpus `source_sha256` and the same 4088 scored positions. This
divergence is present in the arm that uses the GEMV lane and belongs to the
eager-vs-compiled question tracked separately in #16; it is reported here and not
chased. It landed on 0.46687 -- lane-neutral under compile too. This is a question about
compiled **materialisation**, not about the GEMV lane, for the same reason the
rest of this section is: the scored forwards are M = 512 in both arms. Completing
the table does not close the gap named at the top of this section.

### What would close it

A decode-regime dump: feed each sequence's prefix incrementally so that, with
prefix caching on, every scored position is an **M = 1** forward and therefore
routes to the GEMV kernel in arm A and to torch decode in arm B. The teacher must
be re-dumped the same way and the metric identity shown unchanged, or the
comparison is against a differently-produced reference. That needs a serve, a
teacher rebuild and a change to `kl_tool.py` (which lives outside this repo), so
it is filed rather than fixed here.

**And that is what closed it, filed as #102 and measured the same night.**
`kl_tool.py` grew an opt-in `--regime decode` in which every scored position is
an M = 1 forward, verified per request from
`usage.prompt_tokens_details.cached_tokens` rather than asserted, with the
teacher re-dumped in the same regime and cross-regime `compare` refused
outright. Re-run over these two arms on this inode, the decode regime reads
mutual `KL >= 0.012111` at 91.02% top-1 over 256 positions where the prefill
regime above reads `0.000000` at 100.00%, and the route trace shows why: 28 672
`tessera_window_gemv::gemv` launches on the decode dump's scored forwards and
zero on the prefill dump's. Against the BF16 teacher in that regime, arm A reads
`KL >= 0.436065` and arm B `KL >= 0.432477` — **two arms, two KLs**, which is
what #83 asked for. That receipt declines to say which arm is closer to BF16 at
256 positions, and files the arms' M = 1 disagreement as #110; read it there,
not here. The receipt is
`docs/measurements/tessera-decode-regime-kl-2026-09-03.md`.

**It closed it eager, and only eager.** Both of #102's arms served with
`--enforce-eager` (its own §5, and `enforce_eager: True` in both serve logs).
The A/B protocol this section points at is explicit that eager is half of it --
`docs/design/window-gemv-a-side.md` §5 item 5, "Eager **and** compiled, both
residency modes" -- and the compiled half is where the production configuration
lives, because vLLM compiles by default. The compiled arms of *this* campaign
have a KL, but it is the prefill-regime null above and it is uninformative for
the same `GEMV_MAX_M` reason -- `kl_tessera_ts83-arm{A,B}-streamed-compiled.json`
read `0.4668730966935983` at 62.67% top-1 over 4088 positions in **both** arms,
identical to the last digit, exactly as the eager prefill pair reads
`0.46599389451679424` in both. So the compiled serve's decode-regime KL for this
lane is **unmeasured**, and it is filed as **#113** rather than left in this
paragraph. What was *not* in doubt there is engagement: §4's recovered trace
shows the GEMV launching 9 016 times under a compiled forward with zero CUTLASS
GEMM launches in every purely-decode bin. **#113 supplied the missing number
the next day**: compiled, on these arms, mutual `KL >= 0.012585` at 88.67% over
256 M = 1 positions against `0.000000` at 100.00% over 4088 prefill ones --
a number that sits **below** the same-artifact rebuild delta (`KL >= 0.019423` decode, arm A re-served, one lane state, two builds), so it is **not** yet a lane-attributable separation -- and
the compiled prefill dumps of *this* campaign carry the same
`kl_tool.py fingerprint` as #113's, so the null above is an identity across
campaigns as well as across arms
(`docs/measurements/tessera-compiled-decode-kl-2026-09-04.md`).

### Method

Method: `kl_tool.py dump` against each served arm, then `compare`
against the image-matched teacher `qwen_teacher_bf16_v028.json.npz`, corpus
contract `corpus_qwen_n8_s512.json` (**Qwen-tokenized** — the default contract is
GLM-tokenized and `kl_tool` refuses the mismatch; the refusal was not worked
around). Mean, `confident`, top-1 agreement **and the p99/max tail** are reported
for each arm: a lower mean KL can hide a heavier tail, and for a lane that is
meant to be arithmetically equivalent to the path it replaces, tail movement is
the interesting result rather than the mean.

## 4. The latency

**Result: DEFERRED to a quiet box, and no latency claim is made from this
campaign.** Two arms were taken before the box degraded; both are reported below,
both are labelled contended, and **no A-vs-B ratio is computed from them.** This
is the honest outcome rather than a missing one: the numbers exist, they are
recorded, and they are not evidence about the lane. **The ratio this section
does not compute is tracked as #109**, which carries the discipline it needs:
an idle-power trace recorded before the run starts, both arms in one process and
one session, and power read against the ~140 W envelope rather than
`gpu_utilization`.

### What was measured, and why it is not a result

| arm | regime | TTFT | TPOT (engine) | ITL | decode wall/req | peak load1/cpu | swap in use |
|---|---|---|---|---|---|---|---|
| armA (GEMV) | eager | 70.34 ms | 43.541 ms (n=12) | 43.541 ms (n=1524) | 5,602 ms | 3.413 | not in receipt |
| armB (torch) | eager | 541.69 ms | 349.367 ms (n=12) | 349.367 ms (n=1524) | 44,913 ms | 2.49 | not in receipt |

A third receipt, `latency-armA-streamed-compiled.json`, exists because that arm
was re-run for its **trace**; it was taken at load 48-56 and its timings are not
reported here for the same reason the two above carry no ratio. Kernel names are
not load-sensitive; timings are.

Both receipts in the table predate `c6d6064`, so **neither carries a swap
reading** and the
column says so rather than inventing one. The swap figures quoted below are
box-level, from `free -g` on sparky at 19:30 and 19:41 local (13 of 15 GiB, then
10 of 15), which brackets armB's window (23:23-23:32 UTC = 19:23-19:32 local) and
not armA's (19:18-19:19 local). Scoped that way deliberately: it is enough to say
armB ran into swap and not enough to put a number in armA's row.

**The two arms differ by 8x. The kernel difference cannot be 8x** -- #10's
in-process bench puts it at 1.28-2.08x at M=1 -- so what this table measures is
the box, not the lane. Reporting `43.5 ms vs 349.4 ms` as a lane result would
have been the single worst error available in this task.

**And the load average alone would have pointed the wrong way.** armB, the arm
that was 8x slower, had the *lower* peak load1/cpu (2.49 vs 3.413). The run queue
was not what slowed it: armB ran while 13 of 15 GiB of swap were in use and
sparky was thrashing, after thirteen agents started full test suites at once.
A reader given only the load numbers would have concluded the slower arm was the
less contended one. That is why the receipt now records available memory and swap
beside the load average, and why `contended` is true on either leg (`c6d6064`).

### The box-level instrument agrees, and it is the one that settles it

`nvidia_smi.gpu_power_draw`, node sparky, NVIDIA GB10, over each receipt's own
`marks_utc` window:

| arm | window (UTC) | GPU power | fraction of ~140 W envelope |
|---|---|---|---|
| armA/eager | 23:18:31-23:19:39 | 17-31 W | 12-22% |
| armB/eager | 23:23:31-23:32:33 | ~44 W sustained | ~31% |
| idle baseline | 21:54-22:24 | 17.0 W median | 12% |

**Neither arm was GPU-bound.** armA sat within a few watts of the 17 W idle floor
for its whole window -- the GPU was very nearly doing nothing while the client saw
5.6 s per request. This is exactly the reading principle 15 exists for: on GB10
`gpu_utilization` would have reported a resident kernel and said nothing, while
power against the envelope says the work was not on the GPU at all. Both arms'
time went to host contention.

Work per joule is deliberately **not** ranked here. Ranking it would require the
work to be the thing consuming the joules, and it demonstrably was not.

### The profile: missing for the campaign, then recovered for the compiled arm

**RECOVERED.** After the fix below landed, `armA/streamed/compiled` was re-run and
**did** produce a trace. Under the compiled forward, on the engine-core rank0
trace:

| kernel | launches | device time |
|---|---|---|
| `window_gemv_kernel<14, 16, 1, unsigned short, false, false, false>` | **9,016** | 149.86 ms |
| `window_decode_kernel<14, 4, unsigned char>` | 784 | 237.74 ms |
| per-token FP8 quant | 9,800 | 44.42 ms |
| `scaled_mm`/CUTLASS GEMM | **448** | 29.83 ms |
| flash attention | 4,088 | 19.86 ms |

**The GEMM row was first published as 3,136, and that was my summariser's bug,
not the serve's.** `window_gemv_trace_summary.py` tested its `cutlass|scaled_mm|
gemm` pattern *before* its attention pattern, and flash-attention kernels are
templated on CUTLASS types -- `cutlass::bfloat16_t` is inside their own mangled
name -- so 2,688 attention launches were counted as GEMMs. The number mattered:
3,136 GEMMs against 9,016 GEMV launches reads as *the compiled forward is partly
falling back at M=1*, which is precisely the question this trace was taken to
settle. Fixed in the tool (bucket order, with the reason in the source) and the
summary regenerated.

**This closes the gap §2 point 4 names.** The compiled census proves dispatch;
these are kernel names in a chrome trace, which is launch. The window GEMV
*executes* under a compiled forward, 9,016 times, at the wire's window width
(the `14` template parameter is `WINDOW_BITS`, and `WINDOW_BITS_SUPPORTED = (14,)`).
The eager arm never needed this -- its census record carries the
`tessera_window_gemv::gemv` symbol at M1 shapes directly -- so with this trace the
launch question is answered in **both** regimes.

**And there is no fallback at M=1 -- the time axis says so, not the total.** A
total cannot separate "these GEMMs are the prefill" from "the compiled forward
falls back to a GEMM at M=1"; the two readings of one number are opposite
conclusions about the lane. Splitting the 1.446 s of kernel activity into 24 bins
(`--phases 24`) separates them, because the GEMV only exists at M=1, so any bin
holding GEMV launches is a decode bin:

| bins | window_gemv | window_decode | scaled_mm/CUTLASS | attention |
|---|---|---|---|---|
| with GEMV activity (decode) | 9,016 | 71 | **41** | 3,884 |
| without it (prefill / materialisation) | 0 | 713 | **407** | 204 |

The 41 are not steady-state decode: they fall in bins 2, 8 and 11 only, which are
the bins that *also* hold `window_decode` launches -- boundary bins straddling a
prefill and the decode that follows it. In every bin that is purely decode (3-7,
12-17) the CUTLASS GEMM count is **exactly 0** while the GEMV runs 784 times per
bin. Under a compiled forward, at the shape the kernel exists for, nothing else
is doing that work.

*An observation the trace volunteers, recorded rather than chased:* the largest
single device-time kernel in the run is not in any of these buckets. It is a
cuBLAS bf16 GEMV (`internal::gemvx::kernel<int, int, __nv_bfloat16, ...>`), 50
launches for **263.9 ms** -- more device time than the entire window-GEMV bucket.
An unquantised bf16 Linear at M=1 has that shape, but the trace does not name the
module and this document does not attribute it. It is flagged because it bears on
what a decode-latency comparison of this lane could ever show: if that kernel is
on the decode path, the lane is not what dominates it.

Summary regenerated with `experiments/window_gemv_trace_summary.py`; the JSON is
`trace-armA-streamed-compiled.json`.

**Still missing: the other three arms**, whose serves predate the fix. That is a
smaller gap than it was -- the arm that carried the only unanswerable question is
the one recovered -- but it is a gap, and no trace-based statement is made about
armB or about the eager arms beyond what their censuses carry.

### Why it was missing, and the fix

**No trace was captured for any arm of the original campaign.** The serve was asked for profiling with
`-e VLLM_TORCH_PROFILER_DIR=/prof`, and **vLLM 0.28 has no such variable** --
profiling moved onto a config object reached from the command line as one json
argument, `--profiler-config`. The engine did not fail; it warned
(`Unknown vLLM environment variable detected: VLLM_TORCH_PROFILER_DIR`) and
served without profiling, so `/start_profile` was never registered and every
driver POST returned 404 into a receipt field. Fixed in `dd42ff4`, verified
against the image (`arg_utils.py:1652` adds the flag;
`entrypoints/serve/profile/api_router.py:21` is the route it enables) rather than
guessed, and an empty trace directory is now said out loud at the end of a run.

**This costs a stated deliverable and the doc says so rather than working
around it.** A compiled route record stamps the combined `(symbol, decoder)` pair
from the trace whatever runs underneath it, so the compiled census proves
**dispatch, not launch**. A kernel name in a chrome trace is the only thing that
proves the GEMV kernel actually ran under a compiled forward. The original
campaign captured none; the re-run above captured one, for `armA/streamed/
compiled`. Section 2's compiled verdict for **that** arm is now backed by
launches; the compiled verdict for `armB/streamed/compiled` still rests on
dispatch alone.

### Scope, stated

The two armA-resident arms were not taken. Resident never reaches the GEMV lane,
so those serves would have bought context rather than evidence, at ~30 minutes of
a box nine other agents were queued for. armB's latency arms were dropped once
the deferral was decided: they exist only for an A-vs-B ratio this report does not
compute, and the census plus the 112 refusals already establish armB's route.

Method, and why it is not #10's bench: every number is read from the
**serving process**. `vllm:time_to_first_token_seconds` and
`vllm:request_time_per_output_token_seconds` off `/metrics`, differenced across each
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
only `serve_mode` into vLLM's compile-cache identity.  **That is no longer
true, and it was drift rather than error:** it held at this work's merge base
`c71f37b`, and #91 has since landed `compile_identity.py::note_traced_dispatch`
(called from `serving/fp8_route.py:337` and `serving/bf16_route.py:755`), so the
traced dispatch is in the identity.  A re-run therefore no longer needs a
separate `VLLM_CACHE` root per arm to keep the two arms' compiled builds
apart. The two streamed arms trace
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
- `latency-<arm>-<mode>-<regime>.json` — the engine histograms. The paired
  `prof-<arm>-<mode>-<regime>/` directories exist and are **empty**: no trace was
  captured, for the reason given in §4. They are left in place rather than
  deleted so the gap is visible where a reader goes looking for it.
- `prof-armA-streamed-compiled/` — the one non-empty trace directory
  (`rank0.*.pt.trace.json.gz` is the engine-core trace with the kernel names),
  summarised into `trace-armA-streamed-compiled.json`.
- `power-baseline.json` — the idle series above.
- `campaign.log`, `gapfill.log` — the run logs, including every arm that failed
  and why.
- `pytest-after.txt` — the suite line.

The logprob dumps the bit-identical comparison is computed from are **not** under
that directory — they are `/mnt/shared/tessera-kl/qwen_tessera_ts83-<arm>-streamed-<regime>.json.npz`,
beside the teacher `qwen_teacher_bf16_v028.json.npz` and each dump's
`.meta.json` (corpus sha, tokenizer identity) and `.build.json` (which compiled
build served it). The comparison is two `numpy` loads and an `!=` count; it is
reproducible from those files alone, without a serve.

Drivers, in the worktree: `experiments/window_gemv_served.sh` (census + KL),
`experiments/window_gemv_latency.sh` (profiled latency serve),
`experiments/window_gemv_load.py` (load driver),
`experiments/window_gemv_trace_summary.py` (chrome-trace kernel aggregation).
