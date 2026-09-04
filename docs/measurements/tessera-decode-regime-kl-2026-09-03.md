# A served KL in the decode regime, and why the prefill one could not see a GEMV (2026-09-03)

**Outcome.** `kl_tool.py` grew a second, opt-in regime. In `--regime decode`
every scored position is an **M = 1 forward**, verified against the serve's own
cached-token accounting rather than asserted, and the tool now **refuses** to
compare a decode-regime dump with a prefill-regime one.

Validated on the **`TESSERA_FP8` route** (E4M3 grid, CHANNEL plane, window body,
`TESSERA_SERVE_MODE=streamed`) on a Qwen3-0.6B Tessera checkpoint, whose
streamed decode path reaches the window-GEMV lane. Issue #104 establishes that
no *allocated* checkpoint we hold can take that lane at all (column rates
outside `SUPPORTED_RATES = (1, 2, 4)`), so this is the route a checkpoint on
disk actually reaches.

**The headline, on byte-identical bytes (one inode, `11665664`, 846 726 118 B):**

| regime | scored forwards ran | armB vs armA | top-1 agreement |
|---|---|---|---|
| prefill (what #83 measured) | `torch._scaled_mm` in **both** arms | `KL >= 0.000000` | 100.00 % (4088 positions) |
| decode (this receipt) | `tessera_window_gemv::gemv` in arm A, `torch._scaled_mm` in arm B | `KL >= 0.012111` | 91.02 % (256 positions) |

The prefill regime returns a perfect null on a pair of serves that were
provably **two lane states** -- the GEMV extension built in one and refused in
the other, attested by each serve's own startup route sweep. The scored
forwards in that regime ran the *same* kernel in both arms, which is precisely
why the null is uninformative rather than reassuring. That is the defect in
#102, measured.

---

## 1. The two regimes, stated

| | prefill (default, unchanged) | decode (new, opt-in) |
|---|---|---|
| request | `max_tokens: 1`, `prompt_logprobs: K` over the whole 512-token chunk | one request per position: `prompt = chunk[:L]`, `max_tokens: 1`, `logprobs: K`, `return_tokens_as_token_ids: true`, issued **sequentially** |
| scored position | every prompt position of the chunk | the sampled position only |
| rows in the scored forward | the chunk length (512) | **1** |
| positions | 8 x 511 = 4088 | 8 x 32 = 256, at prefix lengths 1, 17, ... 497 |
| prefills performed | 8 (one per chunk, and they are the scored forwards) | 8 (one per chunk, scoring nothing: they only fill the prefix cache) |
| how M is established | by construction | from `usage.prompt_tokens_details.cached_tokens`, per request, or the dump refuses |

The mechanism is vLLM's prefix cache. It hits on whole KV blocks, so a request
whose prefix is block-aligned but for its last token recomputes exactly one
row; the stride is therefore the serve's block size (16) and not a choice.
`--enable-prompt-tokens-details` is required on the serve, because without it
vLLM omits the cached-token field and the M = 1 claim would be unverifiable --
the dump refuses rather than record an unchecked number.

Requests are sequential, and that is part of the definition: two in flight at
once would be batched into one many-row forward by the scheduler.

The regime block a decode dump carries (from the BF16 teacher's payload):

```
"name": "decode",  "stride": 16,  "scored_positions": 256,  "warmup_prefills": 8,
"rows_per_scored_forward": 1,
"rows_verified_by": "usage.prompt_tokens_details.cached_tokens on every scored request",
"cached_tokens_min": 0,  "cached_tokens_max": 496,
"positions_are_subset_of_prefill": true
prefix_lengths[0] = [1, 17, 33, ..., 481, 497]
```

`cached_tokens_max = 496` is the L = 497 request: 496 tokens served from the
prefix cache, one row computed. `cached_tokens_min = 0` is the L = 1 request,
which has nothing to cache and is one row anyway.

## 2. (a) What the serve actually executed

`TESSERA_ROUTE_TRACE=<abs path>` makes the serve keep a launch histogram keyed
by route **and problem shape** -- the question the per-module "latest dispatch"
record cannot answer. Snapshots are taken at three points and differenced
(`experiments/route_trace_delta.py`). Arm A, verbatim:

```
--- decode-dump  (delta from startup) ---
  TESSERA_FP8:streamed M1:N1024:K2048 tessera_window_gemv::gemv window_gemv  launches=7168     modules=28
  TESSERA_FP8:streamed M1:N1024:K3072 tessera_window_gemv::gemv window_gemv  launches=7168     modules=28
  TESSERA_FP8:streamed M1:N4096:K1024 tessera_window_gemv::gemv window_gemv  launches=7168     modules=28
  TESSERA_FP8:streamed M1:N6144:K1024 tessera_window_gemv::gemv window_gemv  launches=7168     modules=28
  TESSERA_FP8:streamed M512:N1024:K2048 torch._scaled_mm window_gemv         launches=224      modules=28
  TESSERA_FP8:streamed M512:N1024:K3072 torch._scaled_mm window_gemv         launches=224      modules=28
  TESSERA_FP8:streamed M512:N4096:K1024 torch._scaled_mm window_gemv         launches=224      modules=28
  TESSERA_FP8:streamed M512:N6144:K1024 torch._scaled_mm window_gemv         launches=224      modules=28
--- prefill-dump  (delta from decode-dump) ---
  TESSERA_FP8:streamed M512:N1024:K2048 torch._scaled_mm window_gemv  launches=224      modules=28
  TESSERA_FP8:streamed M512:N1024:K3072 torch._scaled_mm window_gemv  launches=224      modules=28
  TESSERA_FP8:streamed M512:N4096:K1024 torch._scaled_mm window_gemv  launches=224      modules=28
  TESSERA_FP8:streamed M512:N6144:K1024 torch._scaled_mm window_gemv  launches=224      modules=28
```

The accounting closes exactly. The model has 28 layers x 4 Linear shapes:

* decode stage, M = 1: `7168 = 256 scored positions x 28 layers`, on all four
  shapes, so **28 672 GEMV launches = 256 forwards**, every one of them the
  scored forward of a position;
* decode stage, M = 512: `224 = 8 x 28`, i.e. **the eight per-chunk warm-ups**
  and nothing else. That is the prefill-only warm-up the regime performs, and
  it scores nothing (the warm-up request carries no `logprobs` field at all, so
  it cannot contribute a position even in principle);
* prefill stage: the same `224 = 8 x 28` at M = 512, and **zero launches at
  M = 1**. Not one scored forward of the prefill regime touched the GEMV lane.

The decode dump runs first, on a cold prefix cache, precisely so that count is
explainable; reversed, the prefill dump's 512-token prompts would have cached
the whole corpus and the warm-ups would have vanished into cache hits.

Startup accounts for the rest: 28 launches at each of M in {1, 2, 8, 16} and 56
at M = 2048, which is vLLM's memory-profile and dummy-run sweep, taken before
either dump and subtracted from both.

Arm B's histogram is the same shape with a different symbol everywhere:

```
--- decode-dump  (delta from startup) ---
  TESSERA_FP8:streamed M1:N1024:K2048 torch._scaled_mm torch_window    launches=7168     modules=28
  ... (all four shapes)
  TESSERA_FP8:streamed M512:N1024:K2048 torch._scaled_mm torch_window  launches=224      modules=28
```

and the serve log says why, per module:

```
[tessera-serving] WARNING: model.layers.9.mlp.down_proj: the window GEMV lane did not
prepare (OSError: [Errno 30] Read-only file system: '/ext-ro/tessera_window_gemv');
serving streamed through the torch window decode instead
```

So the two arms are **two genuine lane states**, not one wearing two names.
This is the check #91/#104 could not make, and the reason a two-arm number is
now interpretable.

## 3. (b) The prefill regime did not move

A dump file cannot be byte-compared across runs: its meta carries
`produced_at_utc`, `elapsed_s`, `argv` and `host`. `kl_tool.py fingerprint`
hashes the scored values plus the metric identity and nothing else. The
before-state is a dump frozen on disk **before the tool was touched** (#83's
arm A, taken 2026-09-03 19:09 with the pre-change instrument); the after-states
are this campaign's two prefill dumps, taken with the patched tool off the same
bytes and the same corpus:

```
--- qwen_tessera_ts83-armA-streamed-eager.json.npz     (pre-change tool)
fingerprint fb7a26adf0461132b8039f6741d99117564d9471c5c2b7a32f7711cd39eead8c
  regime = prefill
  ids_sha256 = 2e901ead27306ad3d4b23a428c584977c49e83579c868e12a37f7d0581245448
  logprobs_sha256 = 0b0fb509a9b90081b47505ffe815f1700f619d77c3e9d3935c8d68cfec170d8d
--- qwen_ts102_ts102-armA_prefill.json.npz             (patched tool)
fingerprint fb7a26adf0461132b8039f6741d99117564d9471c5c2b7a32f7711cd39eead8c
--- qwen_ts102_ts102-armB_prefill.json.npz             (patched tool)
fingerprint fb7a26adf0461132b8039f6741d99117564d9471c5c2b7a32f7711cd39eead8c
```

One hash across the pre-change tool and the patched tool, and across both arms.
Two things at once: the prefill mode is unchanged to the last logprob, and #83's
null is restated as an identity rather than as a count of differing values.

The pre-change dump carries no regime field and reads as `regime = prefill`, so
every number frozen on disk before today stays comparable and no receipt is
invalidated.

## 4. (c) Non-comparability, and why it is a refusal

A decode-regime number and a prefill-regime number over one corpus **are not
the same metric**. Same weights, same conditioning, different executed kernels
and a coarser position set. So the regime rides *inside* `metric_identity` and
prints next to every number, and `compare` refuses the cross pair:

```
$ kl_tool.py compare qwen_teacher_bf16_v028.json.npz qwen_ts102_ts102-armA_decode.json.npz
REFUSED: cross-regime compare.
    teacher qwen_teacher_bf16_v028.json.npz regime=prefill
    student qwen_ts102_ts102-armA_decode.json.npz regime=decode
  A regime says WHICH FORWARD produced each scored position (prefill: one many-row
  forward over the chunk; decode: one M=1 forward per position). Two regimes run
  different kernels over different position sets, so their difference is not a
  property of the two artifacts. Re-dump the reference in the student's regime --
  there is no flag to override this, because a number produced by overriding it
  would not mean anything.
  exit status: 1

$ kl_tool.py compare ... --allow-mismatch
REFUSED: cross-regime compare.
  ... (identical)
  exit status: 1
```

`--allow-mismatch` does not reach it, deliberately: everything under that flag
is an *alignment doubt* a caller may accept and stamp, while a cross-regime
pair is two metrics. The teacher was therefore **re-dumped in the decode
regime** (`qwen_teacher_bf16_v028_decode`, 256 positions) and every decode
number below is against that reference.

## 5. (d) The two-arm number

Both arms serve the same file through the same inode, eager, `--enforce-eager`,
image `vllm/vllm-openai:latest` = `sha256:61fc8a896b0a…`, each taking the box
serve lock in turn. Each dumps both regimes off its own serve.

**Mutual, arm B against arm A, same bytes:**

```
--- decode: armB against armA, byte-identical bytes ---
positions=256  top1_agree=91.02%
  ALL            KL >= 0.012111   (<= 4.209370 at the declared floor 3.72e-44)
  CONFIDENT      n=94 (37%)  KL >= 0.006282   (<= 0.581994)

--- prefill: armB against armA, byte-identical bytes ---
positions=4088  top1_agree=100.00%
  ALL            KL >= 0.000000   (<= 3.297790 at the declared floor 3.72e-44)
  CONFIDENT      n=1552 (38%)  KL >= 0.000000   (<= 0.693444)
```

**Against the BF16 teacher, each regime against its own teacher:**

| | arm A (GEMV lane) | arm B (torch fallback) |
|---|---|---|
| decode, 256 positions | `KL >= 0.436065`, top-1 63.67 % | `KL >= 0.432477`, top-1 62.11 % |
| prefill, 4088 positions | `KL >= 0.465994`, top-1 63.23 % | `KL >= 0.465994`, top-1 63.23 % |

The prefill row is the same number to six decimals in both arms -- as it must
be, since the trace says both arms ran `torch._scaled_mm` on every scored
forward. The decode row separates them.

**The position-matched contrast.** A decode number and a prefill number differ
for two reasons at once: which forward ran, and which positions were scored.
`experiments/decode_regime_subset.py` removes the second by restricting the
prefill pair to the decode regime's own 256 positions:

```
"prefill_on_decode_positions": {"positions": 256,  "kl_lower_mean": 0.4318166601479351,
                                "top1_agree_pct": 63.28}
"prefill_on_all_positions":    {"positions": 4088, "kl_lower_mean": 0.4659938945167942,
                                "top1_agree_pct": 63.23}
```

So on the *same* positions the prefill regime reads 0.4318 and the decode
regime reads 0.4361 for arm A: the two regimes agree closely on the artifact's
quality. **The regime change is not a quality claim.** What it changes is which
kernel is under test, and therefore whether a two-arm A/B can see anything at
all.

### A finding this turned up, filed as #110

The two arms are **not** identical as served: mutual `KL >= 0.012111`, 91.02 %
top-1 agreement over 256 M = 1 forwards. The existing bit-exactness receipts
([[tessera-kernel-lane-span2]], [[tessera-window-kernel-fp8]], "bit-exact
196/196") are about the **decoded weight tile** matching `materialize_fp8`, not
about the two GEMMs agreeing: arm A multiplies through the window table with
its own accumulation, arm B runs `torch._scaled_mm` on materialised fp8. A
difference of this size is consistent with different accumulation rather than a
defect -- but it had never been measured on a served path, because until today
no served KL scored an M = 1 forward. **Filed as #110** rather than carried in
a report: it is a lane question and this change is the instrument. This receipt
does not claim which arm is closer to BF16 (256
positions is too few, and the two decode numbers, 0.4361 and 0.4325, straddle
in the direction that would be surprising).

## 6. The tests, run against the code as it was

`tests/test_kl_tool_decode_regime.py` loads `kl_tool.py` by path, so pointing
`KL_TOOL_DIR` at a copy of the instrument taken *before* it was patched runs
the same suite against the old code:

```
$ KL_TOOL_DIR=/home/rob/tessera-runs/ts102/kl_tool_pre \
      python3 -m pytest -q tests/test_kl_tool_decode_regime.py
10 failed, 1 passed in 0.24s
$ python3 -m pytest -q tests/test_kl_tool_decode_regime.py   # the installed tool
11 passed in 0.06s
```

The one that passes on both is `test_prefill_request_body_is_unchanged`, which
asserts the prefill request body byte for byte. That is the point of it: it is
the test that must *not* be able to tell the two tools apart.

Among the ten failures is `TypeError: unsupported format string passed to
NoneType.__format__` -- a pre-existing defect in `compare`'s legacy print,
which raises whenever no position is confident. It was fixed in place along
with the regime work rather than filed, and the test that surfaced it is kept
as its regression guard.

`tests/test_route_trace.py` likewise, against `telemetry.py` one commit
earlier. vLLM loads a general plugin in **both** the API-server process and the
engine core; only the latter counts anything, and the former's flusher would
have replaced a full histogram with an empty one -- a census reading zeros off
a lane that ran, which is the same silent-and-directional failure the trace
exists to rule out:

```
$ pytest -q tests/test_route_trace.py     # telemetry.py at 9fa0679, no guard
>       assert [e["launches"] for e in entries] == [1], "the histogram survived"
E       AssertionError: the histogram survived
E       assert [] == [1]
$ pytest -q tests/test_route_trace.py     # with the guard
8 passed in 0.59s
```

The whole in-branch suite touched by this work:

```
$ pytest -q tests/test_route_trace.py tests/test_kl_tool_decode_regime.py \
      tests/test_serving_dispatch.py tests/test_serving_contract.py \
      tests/test_experiment_import_roots.py tests/test_issue_refs.py
95 passed in 0.93s
```

## 7. Method, and where everything is

Three serves, sequential, one box, all GPU work through the PrismaBuild pool
(`pbrun`, action `e70313132a38`, executed on sparky in 672 s):

1. `experiments/decode_regime_campaign.sh teacher` -- the BF16 teacher
   **re-dumped in the decode regime**. A decode student against the prefill
   teacher would be a comparison against a differently-produced reference, and
   §4 is the tool refusing exactly that;
2. arm A -- the streamed FP8 route with the window-GEMV extension available;
3. arm B -- the same bytes, one inode, with a **read-only extensions root**, so
   `prepare_fp8_gemv` raises before nvcc is consulted and the route takes its
   published `torch_window` fallback. That is the same state a rate-3 wire
   reaches, so arm B is the route's real fallback rather than a test double.

Each arm dumps **both** regimes off its own serve (decode first, cold cache)
and snapshots the route trace at three stages. There is deliberately no greedy
smoke: one would add a prefill and fifteen decode forwards to a histogram whose
whole point is that every launch is attributable.

| what | where |
|---|---|
| campaign / arm wrapper | `experiments/decode_regime_campaign.sh`, `experiments/decode_regime_kl.sh` |
| the three checks in §3-§5 | `experiments/decode_regime_evidence.sh`, `experiments/decode_regime_subset.py` |
| trace differencing | `experiments/route_trace_delta.py` |
| logs, traces, KL JSON | `/home/rob/tessera-runs/ts102/` |
| dumps | `/mnt/shared/tessera-kl/qwen_ts102_ts102-arm{A,B}_{decode,prefill}.json.npz`, `qwen_teacher_bf16_v028_decode.json.npz` |
| the instrument's diff | `docs/measurements/tessera-decode-regime-kl-2026-09-03.diff` (`kl_tool.py` and `kl_estimator.py` live at `/home/rob/dq-runs/`, outside any repo) |
| pre-change instrument | `/home/rob/tessera-runs/ts102/kl_tool_pre/` |

## 8. What remains unmeasured

* **The window-GEMV lane on an allocated checkpoint.** Issue #104: every
  allocated checkpoint we hold carries a column rate outside
  `SUPPORTED_RATES = (1, 2, 4)`, so the lane cannot prepare on one. This
  receipt validates the harness and the lane on a uniform-rate checkpoint.
* **The compiled forward, now filed as #113.** Both arms here served with
  `--enforce-eager`, so every served KL this lane has is eager, while vLLM
  compiles by default. The route trace is eager-only by construction: under
  `torch.compile` the dispatch's Python body runs at trace time, so a count
  would describe compilation, not launches (`route_shape` yields `M*` there and
  the trace's own `note` says so). A decode-regime dump under a compiled serve
  is not blocked -- `experiments/decode_regime_kl.sh` takes `TESSERA_LANE_EAGER=0`
  for it -- but its trace would not attest the shapes, so the attestation there
  is `compile_identity.note_traced_dispatch` (which op each module traced, and
  in the compile-cache key, #91) plus the fact that `streamed_apply` is a
  `custom_op(mutates_args=())` whose `M <= GEMV_MAX_M` branch runs at runtime
  inside an opaque node, with the mutual KL itself as the served
  discriminator.
* **The NVFP4 and BF16 routes**, and MoE. Not touched here.
* **Which arm is right.** §5's closing note, filed as **#110**. 256 positions
  on one 0.6B checkpoint is a signal to chase, not a verdict.
* **Cost.** Per scored position the decode regime is ~26x the prefill one: on
  arm A's serve the decode dump reported 45.0 s at its last chunk for 256
  positions (0.176 s each) against 27.3 s for 4088 (0.0067 s each). Both
  figures are the tool's own cumulative wall clock and include a first-chunk
  cost the decode path pays and the prefill path does not (35.9 s of the 45.0
  s is chunk 1; chunks 2-8 run 1.3 s each) -- not attributed here, and not a
  perf claim. It is a targeted instrument for lanes that only serve small M,
  not a replacement default, which is why it is opt-in.
