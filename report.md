# #91 — the serve lane is not in the compile-cache key

Branch `muse/ts-91-cachekey`, off master `82cdf51`.

## 1. What the defect is, restated from the code

`serving/config.py:332` declares exactly one Tessera fact into vLLM's
compile-cache key:

    declare_compile_identity(serve_mode=self._mode)

but `fp8_route.apply()` branches on `getattr(layer, "tessera_gemv", None)`, a
plain Python attribute that Dynamo resolves at **trace** time. A streamed serve
that has the window-GEMV extension traces one opaque
`tessera::fp8_streamed_apply` node; a streamed serve that does not traces a
window decode plus `torch._scaled_mm`. Both have `serve_mode == "streamed"`,
and the traced *sources* are byte-identical, because the branch is data and not
code. So nothing the cache key reads differs, and the two graphs share one slot.

`bf16_route.py:759` has the same shape and is not named in the issue.

Three facts about the pinned runtime make this real rather than theoretical, and
they were read out of vLLM 0.28 inside the pinned image:

* AOT compile is the default path (`VLLM_USE_AOT_COMPILE` is on for torch
  >= 2.10 with the compile cache enabled), and the AOT key is computed
  **before** Dynamo runs, from `aot_compile_hash_factors`
  (`compilation/caching.py:573`) plus `_model_hash_key(forward)`; the cache
  directory is `VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/{key}`
  (`compilation/decorators.py:520-552`). Nothing about the trace can enter a key
  computed before the trace.
* The load-side guard does not close it. `_verify_source_unchanged` checks the
  file list carried by the **saved** artifact (`source_info.inlined_sources`),
  not the file set the loading process would trace. Two runs over identical
  files therefore both pass it, whichever graph is inside.
* `additional_config` is the only Tessera-controlled key input; it is hashed as
  `json.dumps(..., sort_keys=True)` (`config/vllm.py:520`), so a fact placed
  there does reach the key. The timing works: `process_weights_after_loading`
  runs inside `load_model` (`gpu_worker.py:450`), which precedes the first
  worker-side `compute_hash` (`:487`) and `compile_or_warm_up_model` (`:694`).

## 2. What I reproduced, and what I only argued

**Reproduced (measured, on the pinned runtime, receipts committed).** The
*computed key* — the fallback the brief explicitly allows. `experiments/
ts91_compile_key_states.py` runs in the pinned `vllm/vllm-openai:latest` image,
builds a real `VllmConfig`, declares `serve_mode="streamed"`, walks 112
stand-in modules through four lane states, and asks **vLLM's own**
`aot_compile_hash_factors` for the key.

| state | master key | fixed key |
|---|---|---|
| every module on the GEMV lane | `3c4a1a5a…` | `85abb8e3…` |
| every module on the torch fallback | `3c4a1a5a…` | `233b6eae…` |
| mixed: odd layers refused | `3c4a1a5a…` | `0716f2fc…` |
| mixed: even layers refused | `3c4a1a5a…` | `49dcd56d…` |

Master: **1 distinct key for 4 different graphs.** Fixed: **4.** One state
computed twice gives the same key in both trees. Receipts:
`experiments/results/ts91_compile_key_states_{master,fixed}.json`.

**Reproduced on GPU (route wiring).**
`tests/test_serving_fp8_gemv.py::test_the_two_streamed_lanes_declare_two_compile_identities`
drives the real streamed route twice — once with the extension, once with
`kernel_window_gemv._ext` patched to raise — and the declared identities differ,
and repeat within a state. 3 passed in 135 s on GPU.

**Not established at the time of writing.** The end-to-end serve reproduction —
two streamed arms over one `VLLM_CACHE` root landing in the same key directory
and replaying the wrong graph. `experiments/ts91_chain.sh` runs it as four
censuses (A→cacheX, B→cacheY, B→cacheX, A→cacheY), with per-chain cache roots
*and* per-chain JIT extension dirs so two concurrent chains cannot contaminate
each other. Both chains are **queued in the PrismaBuild pool** on sparklina
(`pbrun --gpu`, jobs `ae7b06d9c893` before / `0a090ffa1b67` after) and had not
been scheduled when this was written. §6 says how to read the result.

Two things the serve chain adds that the CPU tier cannot give:

* the **consequence** — what a wrong replay costs — which is the part the CPU
  measurement is silent about by construction;
* one fact about the **fix**, not the bug: the CPU measurement mutates a record
  and hashes the same `VllmConfig` in one process, so it cannot show that in a
  real worker the dict `note_traced_dispatch` writes into at
  `process_weights_after_loading` is the same object `compute_hash` reads at
  `gpu_worker.py:487`/`:694`. `after-A→X != after-B→Y` is that check.

## 3. The fix, and why it is per module

`compile_identity.note_traced_dispatch(prefix, op)` records, per module, which
op that module's `apply()` will dispatch, and folds the set into the record that
`declare_compile_identity` already put into `additional_config["tessera"]`:

    "traced_dispatch": "tessera::fp8_streamed_apply=56+torch._scaled_mm=56#d4300f60a0994ba6"

The route calls it from `process_weights_after_loading` — **the point where the
lane is decided** — for both residencies, and the op name it passes is the same
string constant the `torch.library.custom_op` decorator registers
(`fp8_gemv.STREAMED_APPLY_OP`, `bf16_route.STREAMED_APPLY_OP`), so the key can
never name an op the dispatcher does not.

**Per module, not a per-checkpoint boolean — and this is measured, not
asserted.** The GEMV refusals are per unit (rate != 3-wire, `L != 14`, a shard
start state, no toolchain), so a checkpoint can be mixed. The two mixed states
in the table above have **identical histograms** (56 + 56) and different
digests: a boolean, or a count, would call those two different forwards one
forward and hand each the other's graph. The rendered value keeps both halves —
the histogram is human-readable in a `config.json`, the digest is what makes the
set exact.

The value is a `sha256` prefix, not `hash()`: Python's string hash is salted per
process, and a cache key that changed on every restart would be a different bug.

## 4. Test evidence

`tests/test_serving_compile_identity.py` (12 passed, 2 skipped, CPU):

* two lane states produce two identities; one state reproduces its own,
  whatever order the modules are walked in;
* the mixed pair shares a histogram and still differs (`rpartition("#")`);
* the fact is a stable digest across processes;
* nothing is declared when nothing was declared into (a plain `serve_mode`
  serve is byte-identical to master's record);
* a second `VllmConfig` starts a fresh accumulation, so one process serving
  twice does not smear one arm's lane onto the next;
* `test_vllm_hashes_the_two_lane_states_apart` runs vLLM's own hash over the two
  records.

`tests/test_serving_fp8_gemv.py` — the GPU test above, 3 passed.
`tests/test_serving_bf16_gemv.py::test_a_per_unit_refusal_changes_the_compile_identity`
— the per-unit half, using R=7 (outside the lane's support set) against R=4, so
it needs no toolchain patch.

Suite: see §7.

## 5. Off-task fixes (one line each)

* `bf16_route.py` had the identical latent bug (`:759`); wired the same way,
  same commit as the fp8 one (`70fd1b4`, `568190f`).
* `bf16_route.__all__` did not export `STREAMED_APPLY_OP` while `fp8_gemv`'s
  did — exported (`ce8e786`).
* `docs/design/window-gemv-a-side.md` §5 (the A/B protocol, which asks for
  eager **and** compiled arms — exactly where the shared slot bites) now says
  the lane is in the key, and that #83's per-arm `VLLM_CACHE` root is
  measurement hygiene rather than the fix, so nobody deletes it as "fixed by
  #91" (`ce8e786`).

**Recorded here, not filed and not fixed** (one defect, and I could have fixed
it — the reason I did not is scope, so it is stated rather than deferred
silently): `fp8_gemv.census_expected(compiled=True)` returns `combined | batch`
for **both** regimes (`fp8_gemv.py:387-389`), so a compiled census admits the
torch pair on every module and passes even if every module silently fell back to
it. That is the same blindness as the cache key, one stage later, and it is why
§6's chain must be read off the log rather than off the census verdict.

The fix is now short, and shorter than it was before this change:
`compile_identity.traced_dispatch()` returns exactly the per-module
`{prefix: op}` map the census would need, in the same process the census runs in
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) — so a discriminating compiled expectation
reads that instead of a static set, with no telemetry redesign. It is a change
to the census contract and its tool, which is a second diff and would swamp this
one; it belongs to whoever owns the census expectation.

## 6. How to read the chain when it lands, and one trap

The fact to read is the **directory name** under
`cache-X/torch_compile_cache/torch_aot_compile/` after arm A, against the one
under `cache-Y` after arm B. Equal = the collision, on a real serve.

One alternative explanation had to be ruled out first, and it was, by
measurement rather than by reading: arm B forces the no-GEMV state by pointing
`TORCH_EXTENSIONS_DIR` at a read-only root, so if that variable were one of
vLLM's env key factors, a difference in the two keys would be the *forcing
method* and not the lane. It is not a factor. In the pinned image,
`envs.compile_factors()` starts from vLLM's own env vars minus an ignore list
(its docstring), `TORCH_EXTENSIONS_DIR` is not among its keys, and
`hash_factors(compile_factors())` is byte-identical with it set to `/ext-ro`,
set elsewhere, and unset (`5720a79df14f37f5` in all three). So on the master
tree, equal directory names mean the collision and unequal ones would need a
different explanation than arm B's method.

For the crossed arm (B over A's cache) expect a **loud** failure at the first
forward, not a silent wrong answer: `aot_compiled_fn(...)` is not inside a
`try`. If that is what happens, the issue's "silent in the direction that
matters" is too strong for this mechanism, and the report should say so. The
census verdict is not the evidence either way — an AOT load skips the trace, so
"no module reports a route" fires for a *correct* replay too.

## 7. Suite

The list is computed, not judged: `tools/impacted_tests.py --ref master...HEAD`
(taken from master; this branch predates it) walks the import graph and returns
everything reverse-reachable from the diff — **verdict `narrowed`, 15 changed
files, 12 test files**:

    tests/test_fused_member_rungs.py       tests/test_serving_dispatch.py
    tests/test_serving_bf16_gemv.py        tests/test_serving_fp8_gemv.py
    tests/test_serving_bf16_route.py       tests/test_serving_fp8_route.py
    tests/test_serving_compile_identity.py tests/test_serving_loader_gates.py
    tests/test_serving_contract.py         tests/test_serving_moe_dispatch.py
    tests/test_serving_name_mapping.py     tests/test_serving_sharding.py

All twelve, one run, on sparklina through the PrismaBuild pool
(`pbrun.py --gpu`), so the GPU tests actually executed rather than skipping:

    242 passed, 3 skipped, 14 warnings in 140.73s     exit code 0

No failures, so nothing needed re-running against master. (Two earlier runs of
subsets — 187 passed / 3 skipped CPU-only, 147 passed / 2 skipped on GPU — are
superseded by this one.)

**Pre-fix failure lines**, each measured against the `82cdf51` tree with the
branch's test file copied into it:

* `tests/test_serving_compile_identity.py` (7 added tests) — collection fails:
  `ImportError: cannot import name 'DISPATCH_FACT' from
  'tessera.serving.compile_identity'`.
* `test_serving_fp8_gemv.py::test_the_two_streamed_lanes_declare_two_compile_identities`
  and
  `test_serving_bf16_gemv.py::test_a_per_unit_refusal_changes_the_compile_identity`
  — `2 failed in 1.45s`, both on
  `AttributeError: module 'tessera.serving.compile_identity' has no attribute
  'reset_for_tests'`.

Those three lines say the *fact* is new, which is honest but weak: an API error
is not proof a test catches a defect. The behavioural pre-fix line is the
measurement in §2 — the same question the tests ask (do two lane states key
apart?), put to master's own source through vLLM's own hash function, answered
**one key for four states**.

## 8. Rebase

The branch is based on `82cdf51`; master moved to `0ef73ed` while this ran.
`git merge-tree --write-tree 0ef73ed HEAD` is clean — the intervening commits
touch `experiments/`, `tests/test_pair_grid_*`, `AGENTS.md` and the issues
snapshot, and no serving file. Not rebased, because the before-arm of the
reproduction is pinned to `82cdf51`.

## 9. Consultations

One advisor call (planning and again before writing this). No Fable agent was
needed: the hard part was the pinned runtime's key path, which is a read, not a
judgement.
