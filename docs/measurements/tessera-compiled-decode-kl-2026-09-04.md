# The window GEMV's served KL under a compiled forward (2026-09-04)

**Outcome.** The lane's two-arm served KL now exists in the configuration vLLM
actually serves. #83's census and #102's decode-regime KL were both taken
**eager** (`--enforce-eager`), and the A/B protocol they answer to asks for
"eager **and** compiled" (`docs/design/window-gemv-a-side.md` §5 item 5). This
receipt takes the compiled half, filed as **#113**.

**On byte-identical bytes -- one inode, `11665664`, 846 726 118 B -- with both
serves compiled (`compiled_forward: true`, `eager: false`):**

| regime | armB against armA | top-1 agreement |
|---|---|---|
| prefill (512-row scored forwards) | `KL >= 0.000000` | 100.00 % (4088 positions) |
| **decode (M = 1 scored forwards)** | **`KL >= 0.012585`** | **88.67 % (256 positions)** |

The eager pair (#102) reads `0.012111` at 91.02 % and `0.000000` at 100.00 %
on the same two arms. So the finding #110 records eager -- the GEMV and its
torch fallback do not agree at M = 1 as served -- **holds under the compiled
forward, at the same size.** It is not an artifact of eager dispatch.

**This receipt makes no latency or speed claim.** It is a KL campaign; the
lane's served latency A/B is #109 and is still unmeasured.

---

## 1. The two arms, and that they are two lane states

The arms are #83's hardlinked pair: `armA` and `armB` under
`/home/rob/tessera-runs/ts83/` are one inode, checked again at the start of
this run (`decode_regime_campaign.sh` refuses otherwise). Arm B serves the same
bytes with a **read-only extensions root**, so `prepare_fp8_gemv` raises before
nvcc is consulted and the route takes its published `torch_window` fallback --
the same state a rate-3 wire reaches, not a test double.

The serves say so per module, not in aggregate:

| | arm A | arm B |
|---|---|---|
| `the window GEMV lane did not prepare` lines | **0** | **112** |

Arm B's 112 all read `OSError: [Errno 30] Read-only file system:
'/ext-ro/tessera_window_gemv'`, one per module, and the model has 112 quantised
Linears. So every module in arm B is on the fallback and every module in arm A
is on the lane.

**They also did not share a compiled graph.** vLLM's AOT cache key differs
between the arms -- `329f7035…f982a12` (A) against `0c7ab274…e09689aa` (B) --
which is #91's `compile_identity.note_traced_dispatch` doing its job: the
per-module dispatch set rides in `additional_config["tessera"]`, so the lane
fact reaches the key. Without it the second serve could have replayed the
first's graph, and the two arms would have been one arm wearing two names.

Both build sidecars agree on everything else: `serve_mode: streamed`,
`vllm_version: 0.28.0`, image `vllm/vllm-openai:latest`
= `sha256:61fc8a896b0a…`, `custom_ops: ['none']`, `rms_norm: ['native']`.

## 2. What attests the lane here, and what does not

**Not the route trace.** Under a compiled forward the dispatch's Python body
runs at trace time, so the histogram would describe compilation rather than
serving; `_RouteTrace` is eager-only by contract and now enforces it. Both
arms' trace deltas read, verbatim:

```
--- startup (cumulative at the first snapshot) ---   (no launches)
--- decode-dump  (delta from startup) ---            (no launches)
--- prefill-dump (delta from decode-dump) ---        (no launches)
```

That is the correct reading of an honest instrument declining, and it is the
whole reason this receipt leans on the chain below instead. **No compiled GEMV
launch count is quoted here.**

**What does attest it**, in four links:

1. `fp8_gemv.streamed_apply` is a `torch.library.custom_op(mutates_args=())`,
   so its `M <= GEMV_MAX_M` branch runs at **runtime** inside an opaque node --
   Dynamo does not fold it at trace time;
2. the decode regime establishes M = 1 per request from
   `usage.prompt_tokens_details.cached_tokens`, or the dump refuses;
3. `cudagraph_capture_sizes` on this stack is `[1, 2, 4, 8, 16]`, so an M = 1
   decode forward captures at size 1;
4. and the served discriminator is the KL itself: the decode pair separates
   (0.012585, 88.67 %) exactly where the prefill pair is a perfect null
   (0.000000, 100.00 %), over bytes that differ in nothing but whether the lane
   could build.

Engagement under compile was independently established before this campaign:
`docs/measurements/tessera-window-gemv-served-2026-09-03.md` §4's recovered
Nsight trace shows `window_gemv_kernel<14, ...>` launching 9 016 times under a
compiled forward, with **zero** CUTLASS GEMM launches in every purely-decode
bin. This receipt adds the number that trace could not give: whether the two
arms compute the same thing.

## 3. Against the BF16 teacher, and the thing that flipped

Both arms are scored against the same decode-regime BF16 teacher
(`qwen_teacher_bf16_v028_decode`, 256 positions, dumped eager for #102).

| | arm A (GEMV lane) | arm B (torch fallback) | A - B |
|---|---|---|---|
| **compiled**, decode, 256 positions | `0.434323` (top-1 64.06 %) | `0.437112` (60.55 %) | **-0.002789** |
| eager (#102), decode, 256 positions | `0.436065` (63.67 %) | `0.432477` (62.11 %) | **+0.003588** |
| compiled, prefill, 4088 positions | `0.466873` (62.67 %) | `0.466873` (62.67 %) | 0 |
| eager, prefill, 4088 positions | `0.465994` (63.23 %) | `0.465994` (63.23 %) | 0 |

**The sign of A - B flips between the two dispatch regimes**, and that is this
receipt's most useful contribution to **#110**. #102 declined to say which arm
is closer to BF16 because 256 positions are too few; this is the measurement
that shows the caution was right rather than merely prudent. At this sample
size the ordering is not stable across a change that is not supposed to alter
the lane at all, so **no arm is declared closer to BF16 here either**. #110
needs more positions, not another dispatch regime.

**Do not read the compiled row against the eager row as a quality comparison.**
The teacher is one fixed eager BF16 dump in both campaigns, so an eager-vs-
compiled difference in a student folds in the compiled forward's own divergence
(**#16**, measured at ~0.2473 on the stock route). Within one row -- same
regime, same teacher, same serve configuration -- the A-vs-B contrast is clean;
across rows it is not.

## 4. The prefill regime is bit-identical, across arms *and* across campaigns

`kl_tool.py fingerprint` hashes the scored values plus the metric identity and
nothing else. All four compiled prefill dumps -- this campaign's two arms and
#83's two arms, taken eleven hours and one code change apart -- carry one hash:

```
qwen_ts113_compiled-armA_prefill.json.npz            b1901077a4b5fc08d366c3aa972b498407d4b4dbef992de23977244e72ebbf88
qwen_ts113_compiled-armB_prefill.json.npz            b1901077a4b5fc08d366c3aa972b498407d4b4dbef992de23977244e72ebbf88
qwen_tessera_ts83-armA-streamed-compiled.json.npz    b1901077a4b5fc08d366c3aa972b498407d4b4dbef992de23977244e72ebbf88
qwen_tessera_ts83-armB-streamed-compiled.json.npz    b1901077a4b5fc08d366c3aa972b498407d4b4dbef992de23977244e72ebbf88
```

while the two compiled **decode** dumps do not:

```
qwen_ts113_compiled-armA_decode.json.npz             08a5512b61db8b3f3161d69395a25c7269dfa805f2703fd53ed36a59f1b091d7
qwen_ts113_compiled-armB_decode.json.npz             bb461f528efd3b9ab472435690260d87ef68593c3e8299f4589ce93fdde8e777
```

Two things at once. First, #83's compiled null is an **identity**, not a small
number rounded to zero. Second -- and this is the control for §5's code change
-- #83's compiled arms served with `TESSERA_ROUTE_TRACE` **unset**
(`tessera_plugin_served.sh` never sets it) and this campaign's served with it
**set**, and the scored values are identical to the last logprob. Enabling the
trace on a compiled serve does not perturb what the serve computes.

## 5. Two failed attempts, and the defect the first one found

This campaign took three attempts on the PrismaBuild pool. Both failures are
recorded because the first is a real defect and the second is what a shared box
does.

**Attempt 1 -- the trace killed the serve.** With `TESSERA_ROUTE_TRACE` set,
the compiled serve never came up:

```
torch._dynamo.exc.Unsupported: Unsupported context manager
  Explanation: Dynamo does not know how to enter a `lock` context manager
from user code:
  .../vllm/model_executor/layers/linear.py:576  in forward
  /work/src/tessera/serving/fp8_route.py:389    in apply -> emit_route
  /work/src/tessera/serving/telemetry.py:271    in count -> with self._lock
RuntimeError: Engine core initialization failed.
```

vLLM 0.28 captures with `aot_compile_fullgraph`, and Dynamo cannot enter a
`threading.Lock` under a full-graph capture. `emit_route` swallows exceptions
"so telemetry can never break a request", but this one is raised while
*compiling* the traced body rather than while running it, so the guard never
saw it. `_RouteTrace`'s docstring already said "EAGER ONLY" for the right
reason; it did not behave that way. Fixed in the same branch: `count` returns
on `torch.compiler.is_compiling()` **before** the lock, which Dynamo constant-
folds, so the method is dead code under capture and the eager path is unchanged
byte for byte. Pinned by
`tests/test_route_trace.py::test_a_compiled_forward_can_emit_without_killing_the_serve`,
which compiles a forward calling `emit_route` with `fullgraph=True,
backend="eager"` -- the same capture, on CPU: **without the guard 1 failed, 8
passed**, failing on `Unsupported` at `telemetry.py with self._lock`; **with it
9 passed**. The serve-side proof is attempt 2's log, which reached "Compiling a
graph for compile range (1, 2048) takes 16.71 s" and "saved AOT compiled
function" -- past the point attempt 1 died. §4's fingerprint identity is the
control that the fix changed no number.

**Attempt 2 -- a shared box.** vLLM's own memory-profiling assertion:
`Initial free memory 98.54 GiB, current free memory 102.34 GiB ... other
processes sharing the same container release GPU memory while vLLM is
profiling`. Another campaign's container was exiting while this one profiled.
Not diagnosed further and not fixed: on GB10 the GPU and host share one pool,
so any concurrent process freeing memory looks the same to that assertion, and
the serve lock cannot serialise host processes it does not own. Attempt 3 ran
clean. The crashed attempt's log and its partial compile cache are kept at
`/home/rob/tessera-runs/ts113/crashed-with-trace/`.

## 6. Method, and where everything is

One pool action (`fba48546150a`, three attempts, executed on sparky), two
serves, sequential, each taking the box serve lock in turn. The teacher was not
re-dumped: it is #102's decode-regime BF16 dump, the same fixed reference both
eager arms used.

```
ARMTAG=compiled TESSERA_LANE_EAGER=0 TESSERA_KL_DUMP_PREFIX=qwen_ts113 \
RUNS=/home/rob/tessera-runs/ts113 experiments/decode_regime_campaign.sh arms
```

`TESSERA_LANE_EAGER=0` is the convention `tessera_plugin_served.sh` and
`window_gemv_served.sh` already use; `ARMTAG`/`TESSERA_KL_DUMP_PREFIX` exist so
this run does not overwrite #102's dumps. Both default to #102's exact values,
so the eager campaign is reproducible from the same two scripts.

| what | where |
|---|---|
| campaign / arm wrapper | `experiments/decode_regime_campaign.sh`, `experiments/decode_regime_kl.sh` |
| logs, traces, KL JSON, crashed attempt | `/home/rob/tessera-runs/ts113/` |
| dumps + build sidecars | `/mnt/shared/tessera-kl/qwen_ts113_compiled-arm{A,B}_{decode,prefill}.*` |
| the eager pair this is matched against | `docs/measurements/tessera-decode-regime-kl-2026-09-03.md`, `/home/rob/tessera-runs/ts102/` |

## 7. What remains unmeasured

* **Which arm is closer to BF16 (#110).** Unchanged, and now with a reason
  rather than a caution: the ordering flips between eager and compiled at 256
  positions.
* **Served latency (#109).** Nothing here is a latency or speed claim.
* **The resident mode compiled.** The resident route never reaches the GEMV
  lane, so a resident arm B is the same serve as a resident arm A by
  construction; the census covers it and a KL over it would be an identity
  check.
* **An allocated checkpoint (#104).** As #102: every allocated checkpoint we
  hold carries a column rate outside `SUPPORTED_RATES = (1, 2, 4)`. This is a
  uniform-rate checkpoint.
* **A compiled GEMV launch count.** Structurally unavailable from the route
  trace, and §2 says what stands in its place.
