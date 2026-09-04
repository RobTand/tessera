# Stamping which compiled build served an arm (2026-09-02)

**Outcome.** Every serve that produces a KL dump now writes a machine-readable
build identity beside it (`<dump>.build.json`,
`tessera.serving.build_identity`), and the reader
(`experiments/serving_compile_divergence.py`) refuses the two rows that make a
claim about the build. The identity is a digest of the **contents** of the
compile-cache slot, not the AOT key the issue proposed: the key does not
distinguish the two builds the receipt was written from. Verified offline
against those two surviving caches — same key, 196 autotune records each,
different digests. The `TORCHINDUCTOR_DETERMINISTIC` knob is wired
(`TESSERA_SERVE_DETERMINISTIC=1`), off by default, exercised in tests, and
**not** measured on a live serve; see §4.

Closes the provenance half of **#30**. It does not close the determinism half.

- Module: `src/tessera/serving/build_identity.py` (stdlib only; no torch, no
  vLLM — it runs on the host after the container is gone).
- Wrapper helper: `experiments/build_identity.sh`, sourced by
  `experiments/serve_and_dump_kl.sh` and `experiments/tessera_plugin_served.sh`.
- Reader: `experiments/serving_compile_divergence.py` (schema bumped to
  `tessera.serving_compile_divergence/2`; every row gained a `build` block).
- Tests: `tests/test_serve_build_identity.py` (18, all offline,
  `CUDA_VISIBLE_DEVICES=""`).

## 1. Why the obvious stamp would have been worse than nothing

`docs/measurements/serving-compile-divergence-2026-09-02.md` §2–3 measured
this: a compiled vLLM artifact **replayed** is bit-identical (0.000000 /
100%); the same graph **rebuilt** is not (0.017117 / 95.65%), because 120 of
196 autotuned Triton kernels chose a different `XBLOCK`/`num_warps` on the
second build.

Issue #30 proposed stamping `grep -oE 'torch_aot_compile/[0-9a-f]+'`. Both of
those builds live under **one** key, `15957ad9…`, with byte-identical
`cache_key_factors.json` — vLLM keys the cache by its own *inputs*, and the
autotune outcome is not an input. A key-matched stamp would therefore have
certified a rebuild as a replay: provenance that reads like a guarantee and is
not one.

What identifies the build is what inductor wrote into that slot. The
fingerprint is a sha256 over

* `inductor_cache/**.best_config`, with `time_taken_ms` and
  `triton_cache_hash` removed — those record the benchmark, not the choice, and
  74 of the 196 real records differed only there, so fingerprinting them would
  cry wolf on every stamp;
* plus `inductor_deterministic`, `serve_mode`, `eager`, the image and the vLLM
  version the log reports.

The backbone `computation_graph.py` differed between the two builds too
(`fused_add_rms_norm.maybe_inplace` 42 → 0), and it is recorded — as
**provenance**, deliberately outside the fingerprint. vLLM writes
`Using cache directory: …/backbone` only when it *compiles*: measured on the
two real arms, the rebuild log carries one such line and the replay log zero.
Fingerprinting it would therefore have given one build two fingerprints and
refused the "build replayed by a second serve" row this stamp exists to
certify. The autotune digest separates the two real caches on its own, so
nothing is lost. `test_the_serve_that_built_it_and_the_serve_that_replayed_it_
are_one_build` pins it.

## 2. The offline receipt

Both caches from 2026-09-02 survive on sparky. Read-only, no serve, no GPU:

```
                    records  autotune digest   computation_graph
vllm-cache            196    04525ea7cbedbe3c  fecc8be059b2e83f
vllm-cache-fresh      196    dbeb2b8b4e6ba94d  40aa82a574a97fec
```

Same key `15957ad9…` on both sides; the digests separate them. That pair is
pinned as `test_the_two_surviving_caches_are_told_apart` (skips cleanly on a
box without the directories).

Through the production path, on the two arms' own serve logs and their own
cache roots:

```
$ python3 -m tessera.serving.build_identity stamp \
    --log .../serve_qwen_tessera_k2-resident-graph.log \
    --cache-root .../vllm-cache --serve-mode resident --eager 0 --out chain.build.json
build identity -> chain.build.json  (aot 15957ad9…; fingerprint a4ddc1e047c8411c; complete=True; fresh_compiles=1)
$ ... --log .../serve_qwen_tessera_k2-resident-graph-fresh.log --cache-root .../vllm-cache-fresh ...
build identity -> fresh.build.json  (aot 15957ad9…; fingerprint 3a4060ad47d0c89f; complete=True; fresh_compiles=0)

$ python3 -m tessera.serving.build_identity compare chain.build.json fresh.build.json --require same
REFUSED: ... the two arms served a different compiled build (aot);
any difference between them is a difference of the compiler as much as of the weights
exit=4
```

Those are the two arms whose logits read 0.017117 / 95.65% apart. One key, two
fingerprints, and the refusal fires.

## 3. What is stamped, and what is not

Derived from the runtime's own log lines, verified against real serve logs
under `/home/rob/tessera-runs/tsplugin`:

| field | source line |
|---|---|
| `aot_keys_loaded` | `decorators.py:311` `Directly load AOT compilation from path …` |
| `aot_keys_saved` | `decorators.py:708` `saved AOT compiled function to …` |
| `reload_failures` | `decorators.py:321` `Compiling model again due to a load failure …, reason: …` |
| `backbone_keys` | `backends.py:1094` `Using cache directory: …/backbone` |
| `fresh_compiles` | `backends.py:1155` `Dynamo bytecode transform time` |
| `vllm_version` | `core.py:122` `Initializing a V1 LLM engine (v0.28.0)` |

`identity` is what the fingerprint covers; `provenance` (log path, timestamp,
cache root, backbone slot, `fresh_compiles`, reload reasons) is deliberately
outside it, so two honest replays of one build fingerprint alike.

**A partial stamp refuses.** `tessera_plugin_served.sh` already pins one
`$VLLM_CACHE` across every arm, so its sidecars are complete.
`serve_and_dump_kl.sh` mounts no cache root by default — the arms it was
written for are eager, and adding a default mount would change serving
behaviour mid-campaign — so a **graph-mode** arm through that wrapper stamps
`complete: false` unless the operator sets the new opt-in
`TESSERA_KL_VLLM_CACHE`. An incomplete record refuses to certify sameness
*or* difference; it does not quietly compare AOT keys.

**Every dump now on disk is unstamped** and reads as such. The reader reports
that in the row and in `problems`, and still computes the KL: an archive that
cannot be read is not a safer archive. Re-run on the existing dumps, the two
`build_vs_build` rows now say in as many words that nothing on disk supports
their claim.

## 4. The determinism knob: wired, off, unmeasured

> **Correction from `inductor-determinism-knob-2026-09-04.md` (#16).** The knob
> has since been measured on GPU in the pinned serving image, and it does less
> than this section assumes. Two builds *with the flag set as this wrapper sets
> it* — an env var — still chose different `R0_BLOCK`/`num_warps` on a second
> `torch.compile` entry in the same process, because the flag does not reach that
> second entry: the first entry's kernels transcribe `'deterministic': True` and
> the second's transcribe `False`. So "two otherwise identical arms that differ
> on the flag get different fingerprints" is still true and still useful, but the
> flag being *on* in a fingerprint does not mean the whole build was compiled
> under it. Two consequences for what is written below: `deterministic_effective`
> certifies on `fresh_compiles > 0`, and a build with more than one entry is
> certified on the strength of its first; and the "Not measured: whether two
> builds under the flag actually agree" bullet below now has a partial answer —
> on a two-graph probe they did not. Whether a *serve* is affected turns on
> whether vLLM's backbone compiles its submodules inside one Dynamo entry, which
> that receipt states as open with the experiment that settles it.


`TESSERA_SERVE_DETERMINISTIC=1` forwards `-e TORCHINDUCTOR_DETERMINISTIC=1`
into the serve and stamps the flag into the identity, so a campaign cannot
silently mix arms built with it and without: two otherwise identical arms that
differ on the flag get different fingerprints and are refused as one build.

Three things are tested; one is not, and cannot be here.

* **The env var is the live knob.** Two subprocesses on the installed torch:
  unset → `torch._inductor.config.deterministic` is `False`, `=1` → `True`. If
  a torch bump renames or drops it, that fails instead of the campaign stamping
  a flag that decides nothing. Scope: this runs on the *host* torch
  (2.10.0+cpu), not the torch inside the serving image — it proves the env var
  is live on a torch, not on the one that serves. Nothing here reads the
  container's torch, and no serve log records its version.
* **It enters the fingerprint** (above).
* **It has a silent no-op, and the stamp catches it.** The tell is the same
  whether or not vLLM's compile-cache key covers `config.deterministic` (the
  surviving backbone slot's `cache_key_factors.json` does not mention it, but
  one artifact is not the runtime's own table, so this is not asserted): if the
  flag forces a rebuild, `fresh_compiles > 0` and the claim stands; if a warm
  `$VLLM_CACHE` hands back an artifact autotuned *without* the flag,
  `fresh_compiles == 0` is the tell:
  `deterministic_effective()` is False there and `require_deterministic_build()`
  refuses, naming the warm cache.
* **Not measured: whether two builds under the flag actually agree.** That is
   the receipt #30 asks for and it needs two live serves from emptied caches,
   which this change did not run. Until it exists the flag stays off, and the
   practice remains the one that was measured to work: never empty the compile
   cache mid-campaign, and now, stamp the build.

*Addendum 2026-09-03 (issue #30, determinism half: instrument verified, receipt
still CPU-only).* `experiments/inductor_determinism_probe.py` now records the
parameters each report was compiled under (`params`: hidden width, shapes,
compile mode, device, identical in every child build) and carries the
post-compile config-reset finding machine-checked (`knob.summary`), pinned by
`tests/test_inductor_determinism_knob.py`. Re-measured on the host torch
(2.11.0+cu130, GPU hidden, `TESSERA_PROBE_HIDDEN=256`): the env var reaches
inductor at import (`on.at_import` [True, True]), reads False after a compile
ran either way (`summary.on.resets_after_compile: True`), both arms' two
fresh-cache builds agree bitwise, and both arms record zero `.best_config`
records -- the CPU path has no Triton autotuning, so whether the flag
suppresses device benchmarking is still unanswered. The campaign decision
stands: the flag stays off until two K2 resident builds from empty caches with
the flag are served and compared at 0.000000.

---

## Addendum, 2026-09-03: the identity gained a field, so the numbers above moved

`docs/measurements/serving-compile-dispatch-2026-09-03.md` established that
vLLM 0.28 flips two dispatch defaults together on "is inductor going to run"
-- `custom_ops` and `ir_op_priority` -- and that pinning both makes a compiled
serve bit-identical to an eager one (KL 0.000000, top-1 100.00%, 4088
positions), while pinning either alone does nothing at all. A build identity
that did not record which of those a serve resolved was blind to a difference
worth 30% of the top-1 predictions, so the resolved dispatch is now part of
both the identity and the fingerprint, and `require_same_dispatch` refuses a
pair that differs on it.

Three things in this receipt are therefore superseded rather than wrong:

* **The field list** (§1) does not include the dispatch block. It should be
  read as the field list of the identity *before* this addendum.
* **The two quoted fingerprints** (`a4ddc1e047c8411c`, `3a4060ad47d0c89f`) no
  longer reproduce: the fingerprint hashes one more field now. The *relation*
  they were quoted to demonstrate -- that the two arms served different
  compiled builds and that `compare --require same` refuses them -- is
  unchanged, and that is what the example is for.
* **The test count** (18) is now higher; `tests/test_serve_build_identity.py`
  is the authority, not this line.

Everything in §1 about why the naive stamp would have been worse than nothing
stands unchanged, and the reason this addendum exists rather than an edit is
that the numbers above were really produced, on the bytes named, at the time
named. A receipt that is quietly rewritten to stay true is no longer a
receipt.
