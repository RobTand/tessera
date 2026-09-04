# The eager-vs-compiled gap, and why one build of one graph was not another (2026-09-02)

Two open serving questions, both answered from artefacts already on disk: the
position dumps under `/mnt/shared/tessera-kl` and the two vLLM compile caches
under `/home/rob/tessera-runs/tsplugin`. No serve was started for either.

**Result, in one line each.**

* **The eager-vs-compiled gap is deterministic**, not run-to-run scatter. Of the
  thirteen eager/compiled pairs that cover the three checkpoints served by both
  lanes, **twelve land on exactly three numbers** — six decimals, the same p99
  and the same top-1 agreement, from serves in separate containers that never
  shared a process. The thirteenth is the one arm whose compiled artifact is
  known to be a different build (§3). The gap is a property of *(weights,
  build)*, and it is large: 0.0269 on the E4M3/FP8 route, **0.2445** on the
  E2M1x2/NVFP4 route.
* **The one non-reproducible build is explained.** Under one AOT key,
  `15957ad9…`, **120 of 196** autotuned Triton kernels chose a different
  tiling in the second build. The plugin receipt named that as the *likely*
  mechanism and said so; it is now counted rather than guessed.
* **Issue #16's premise table is contaminated** and so is the sentence it came
  from. `0.0176` is not an eager-vs-compiled number: it is the
  plugin-vs-Gridbook *mutual* KL on the NVFP4 route. The three numbers the issue
  puts in one column measure two different quantities on three different
  checkpoints. `docs/measurements/tessera-allocated-served-2026-09-02.md` is
  corrected in this commit.

Harnesses: `experiments/serving_compile_divergence.py` (the KL matrix) and
`experiments/compile_build_forensics.py` (the two caches). Results:
`experiments/results/serving_compile_divergence.json`,
`experiments/results/compile_build_forensics.json`.

---

## 1. What 0.0176 actually is

`docs/measurements/tessera-serving-plugin-2026-09-02.md` §3 reports 0.0176 in
the **"mutual KL"** column of the twelve-arm plugin-vs-Gridbook table, for
`k2-resident-graph`. Its own JSON says the same:

```
$ python -c "import json; d=json.load(open('/home/rob/tessera-runs/tsplugin/kl_mutual_k2-resident-graph.json')); print(' '.join(d['argv']))"
/home/rob/dq-runs/kl_tool.py compare
  /mnt/shared/tessera-kl/qwen_gridbook_k2-resident-graph.json.npz
  /mnt/shared/tessera-kl/qwen_tessera_k2-resident-graph.json.npz
  --teacher-label-override qwen_gridbook_k2-resident-graph ...
```

Two *runtimes*, one regime. Not one runtime, two regimes. The allocated receipt
read it as "the single 0.0176 the plugin receipt records for a uniform E4M3
wire" — wrong on both counts: it is the **K2/NVFP4** wire, and it is a lane
comparison. Issue #16 inherited the table from that sentence.

The plugin receipt's own eager-vs-compiled numbers are in a different table, and
they are **0.026861** (E4M3) and **0.244223 / 0.244481** (K2).

## 2. The eager-vs-compiled matrix

`kl_tool.py compare`, eager dump against compiled dump, same artifact path, same
corpus contract (`cfbddc2c…`, 8×512, 4088 scored positions), same tokenizer
digest — the harness refuses a pair that fails any of those three.

```
== eager_vs_compiled ==
  comparison                                                         KL >=    top-1       p99  note
  plugin E4M3 resident                                            0.026861   90.34%    0.2081  FP8 route, q1024 uniform
  plugin E4M3 streamed                                            0.026861   90.34%    0.2081  FP8 route, q1024 uniform
  plugin K2 resident                                              0.244223   70.57%    1.4949  NVFP4 route, q896
  plugin K2 resident (fresh build)                                0.244481   70.03%    1.4963  NVFP4 route, second build
  plugin K2 streamed                                              0.244481   70.03%    1.4963  NVFP4 route, q896
  plugin mixed resident                                           0.118001   81.02%    0.9320  both routes in one body
  plugin mixed streamed                                           0.118001   81.02%    0.9320  both routes in one body
  gridbook E4M3 resident                                          0.026861   90.34%    0.2081  same bytes, other lane
  gridbook E4M3 streamed                                          0.026861   90.34%    0.2081  same bytes, other lane
  gridbook K2 resident                                            0.244481   70.03%    1.4963  same bytes, other lane
  gridbook K2 streamed                                            0.244481   70.03%    1.4963  same bytes, other lane
  gridbook mixed resident                                         0.118001   81.02%    0.9320  same bytes, other lane
  gridbook mixed streamed                                         0.118001   81.02%    0.9320  same bytes, other lane
  allocated 4.0 resident                                          0.028838   90.75%    0.2259  FP8 route, four rungs
  uniform R1006 resident                                          0.020028   92.42%    0.1488  FP8 route, one rung
  stock twin K2 (vanilla vLLM NVFP4)                              0.247301   70.43%    1.4084  materialised NVFP4 bytes, no plugin
```

Read the first thirteen rows as three groups.

| checkpoint | measurements | KL >= | top-1 | p99 |
|---|---:|---:|---:|---:|
| E4M3 q1024 (FP8 route) | 4 of 4 agree | **0.026861** | 90.34% | 0.2081 |
| mixed (84 FP8 + 28 NVFP4) | 4 of 4 agree | **0.118001** | 81.02% | 0.9320 |
| K2 q896 (NVFP4 route) | 4 of 5 agree | **0.244481** | 70.03% | 1.4963 |
| K2 q896, the odd build | 1 of 5 | 0.244223 | 70.57% | 1.4949 |

Four measurements per checkpoint, from **two lanes** (Gridbook and the Tessera
plugin, separate containers, separate processes, separate compiled artifacts)
and **two residency modes**, and they land on the same six decimals including
the p99 and the top-1 agreement. Two conclusions follow directly:

* **The gap does not vary run to run.** Whatever it is, it is reproduced exactly
  by an independent serve. The three numbers in issue #16's table are not three
  samples of one quantity with scatter between them.
* **It is not a property of the route alone.** Within the FP8 route the gap is
  0.0269, 0.0288 and 0.0200 on three different checkpoints (uniform q1024, the
  four-rung allocated 4.0, uniform R1006) — those are different weights, so a
  different number is what a deterministic gap *should* produce. Across routes
  it is an order of magnitude: 0.0269 (all-FP8) → 0.1180 (25% NVFP4 modules) →
  0.2445 (all-NVFP4).

The fifth K2 row is the exception, and §3 is what it turns on. The stock twin
(vanilla vLLM on the same wires materialised to NVFP4 compressed-tensors, no
plugin at all) reads 0.2473 — a third runtime on a third kernel path landing
within 1.2% of the plugin's NVFP4 number, which is what says the size of this
gap belongs to compiling an NVFP4 forward on this model rather than to anything
Tessera does.

**Build-to-build and lane-to-lane, for contrast:**

```
== build_vs_build ==
  plugin K2 resident compiled: chain build vs fresh build         0.017117   95.65%    0.1922  two builds of one graph
  plugin K2 resident compiled: build replayed by a second serve   0.000000  100.00%    0.0000  one build, served twice

== lane_vs_lane ==
  K2 resident compiled: plugin vs gridbook                        0.017591   95.65%    0.1841  the receipt's 0.0176
  K2 resident compiled (fresh build): plugin vs gridbook          0.000000  100.00%    0.0000  the rebuild
  K2 resident eager: plugin vs gridbook                           0.000000  100.00%    0.0000  the eager control
  E4M3 resident compiled: plugin vs gridbook                      0.000000  100.00%    0.0000  the FP8 route's compiled arm
```

A compiled artifact replayed is **bit-identical**. A compiled artifact rebuilt
is not. That is the whole reproducibility story in two rows.

## 3. The build that differed, read off the two caches

Both caches survive: `vllm-cache` (the twelve-arm chain's, shared across every
container) and `vllm-cache-fresh` (emptied at 15:43:19, rebuilt by
`fresh_k2rg.sh`). Both hold `torch_compile_cache/36d07e6697/rank_0_0/backbone`
and `torch_compile_cache/torch_aot_compile/15957ad9…`.

```
$ python experiments/compile_build_forensics.py \
    --a /home/rob/tessera-runs/tsplugin/vllm-cache \
    --b /home/rob/tessera-runs/tsplugin/vllm-cache-fresh \
    --key 36d07e6697 --aot-key 15957ad9e7a7... --census
cache_key_factors identical : True
computation_graph identical : False
  vllm_ir op census (a -> b):
    fused_add_rms_norm.default                          14 ->    56  <-- differs
    fused_add_rms_norm.maybe_inplace                    42 ->     0  <-- differs
    rms_norm.default                                    57 ->    57
autotune records in both    : 196 (only in a 0, only in b 0)
  byte-identical            : 2
  same choice, new timing   : 74
  DIFFERENT TUNING CHOICE   : 120

campaign census: 21 backbone graphs
  fused_add_rms_norm (maybe_inplace, default) -> how many graphs:
    (0, 56): 4
    (2, 54): 4
    (42, 14): 1
    (52, 4): 4
    (56, 0): 8
```

Three findings, in order of how much weight they carry.

**(a) The key did not distinguish the two builds.** `cache_key_factors.json` is
byte-identical. vLLM's cache key is doing exactly what it promises — it keys the
*inputs* — and neither it nor `compile_identity.py`'s residency fold can key an
*output* that the compiler chose non-deterministically. Nothing here is a defect
in the identity hook; the hook separates modes, and it did.

**(b) 120 of 196 autotuned kernels chose a different tiling.** Every
`.best_config` record is present in both builds at the same content-addressed
path (0 only in either side), so the same kernels were compiled; 120 of them
were compiled to a different schedule. Examples, chain build → fresh build:

| record | changed |
|---|---|
| `inductor_cache/24/ad9b72d0…` | `XBLOCK` 8 → 2, `num_warps` 8 → 2 |
| `inductor_cache/2e/13daf901…` | `XBLOCK` 512 → 1024, `num_warps` 8 → 4 |
| `inductor_cache/2e/851dd70e…` | `XBLOCK` 4 → 8, `num_warps` 4 → 8 |
| `inductor_cache/2q/b987ef92…` | `XBLOCK` 2 → 4, `num_warps` 2 → 4 |

A further 74 records are the *same* choice with a different measured
`time_taken_ms`, and 2 are byte-identical. The records carry their own
measurement, which is what identifies the mechanism: the choice is made by
timing candidates on the device at compile time, so it depends on what else the
box was doing. Three other workers were loading this box during that stage and
the rebuild's own first attempt was killed mid-profile by one of them. A
different `XBLOCK`/`num_warps` is a different reduction tree, so it is different
floating-point arithmetic, so it is a different logit — which is the observed
0.017117.

The plugin receipt called this "the likely mechanism and it is not proven here".
It is now measured: **61% of the autotuned kernels under one key changed.**

> **Two corrections from `inductor-determinism-knob-2026-09-04.md` (#16).**
> First, the sentence above ("a different reduction tree … which is the observed
> 0.017117") joins a measured count to a measured KL by an inference nobody
> checked.  On the pinned torch, two fresh builds on an idle box produced a
> record differing in `R0_BLOCK` 1024 → 4096 and `num_warps` 8 → 16 whose
> **outputs were bitwise equal** — the fp32 reduction difference vanished under
> the bf16 store.  So 120 is an upper bound on how many of the differing records
> moved a logit, not a count of them; the 0.017117 stands, its attribution to
> all 120 does not.  Second, "Three other workers were loading this box" is not
> what makes this reproduce: across two runs of that probe the same record
> diverged on an **idle** box (5.46 W of a ~140 W envelope) and *agreed* on a
> loaded one (74.24 W), so load is neither necessary nor sufficient.

**(c) The dumped graph also differs, and this one is not yet explained.** The
same key's `computation_graph.py` records 42 of 56 `fused_add_rms_norm` calls at
the `maybe_inplace` overload in the chain build and 0 in the fresh build. In the
pinned vLLM 0.28,
`compilation/passes/ir/inplace_functionalization.py::VllmIRInplaceFunctionalizationPass`
rewrites every `maybe_inplace` node to `default` ("the maybe_inplace overloads
have the same signature as the default overload so the pass simply replaces the
called overload"), so a fully-passed graph carries none. The campaign census
says the split is not stable in either direction: 21 backbone graphs from one
campaign take **five** distinct splits, from all-56-unfunctionalized to
all-functionalized.

What this is **not** claimed to be: the cause of the 0.017117. It changed no
autotuned kernel — all 196 sit at identical content-addressed paths in both
builds — and by the functionalization pass's own docstring the two overloads
dispatch to one implementation, differing only in whether the output may alias a
donated input. Both of those are readings, and neither is a measurement of the
op's numerics: a custom op is an opaque extern call that emits no Triton kernel,
so identical kernel paths cannot tell you whether the extern call read a
different downstream buffer.
It is recorded because a cache directory that reports two different graphs under
one key is a reproducibility fact in its own right, and because the dump is what
a future reader will reach for. Establishing whether the dump is a genuine
compile difference or an artefact of *when* `print_readable` runs relative to
the passes — and, if it is genuine, whether the aliasing changes any downstream
read — needs a vLLM-internals answer this receipt does not have; filed as
**#29**.

*Addendum 2026-09-03 (#29 closed as dump artefact, no vLLM serve needed).*
`experiments/results/compile_build_forensics.json` (committed beside this
receipt) already contains the deciding count: all 22 graphs in the census name
exactly 56 `fused_add_rms_norm` + 57 `rms_norm` call sites, with only the
overload suffix varying across the five splits. A pure rewrite pass is
deterministic given identical inputs (byte-identical `cache_key_factors.json`),
so two different suffix mixes under one key cannot both be the pass's output --
the dump records pass progress at write time (piecewise submodule compilation
/ partially-mutated `split_gm`), not the artifact that ran. The AOT slot that
did run holds the same 196 records at identical paths in both builds, so there
is no second source of build-to-build variation to add to the A/B rule of §4.
`experiments/compile_build_forensics.py` now reports the overload-normalized
comparison alongside the raw one, so the next such pair reads as what it is.

## 4. The rule this buys, and the knob it names

**For A/Bs.** Both arms in one regime, or the comparison is measuring the
compiler. The gap is deterministic, so it is quotable as a route-level error
bar, and it is far too big to absorb silently:

| route | eager-vs-compiled KL >= | top-1 |
|---|---:|---:|
| TESSERA_FP8 (E4M3 wire → W8A8) | **0.020 – 0.029** | 90–92% |
| TESSERA_NVFP4 (E2M1x2 wire → W4A4) | **0.2445** | 70% |
| mixed body (25% NVFP4 modules) | **0.1180** | 81% |

For scale: the NVFP4 route's compile gap, 0.2445, is *larger than the entire
KL-vs-BF16 of a good 4-bit arm* — the allocated 4.0 body reads 0.3485 and the
byte-matched uniform reads 0.1746. A cross-regime A/B on this route is not a
noisy comparison; it is a different measurement.

**For reproducibility.** Two things, one measured and one named.

*Measured:* a compiled artifact replayed is bit-identical (0.000000 / 100%). So
a measurement campaign must **never empty its compile-cache root mid-campaign**,
and every receipt should record the AOT key its arms served
(`grep -oE 'torch_aot_compile/[0-9a-f]+' <serve log>`). The twelve-arm chain
already shared one cache root by design; what it did not do is stamp the key
into each dump's provenance, which is why the odd arm took a forensic session to
pin down. Filed as **#30**.

*Follow-up (same day, `docs/measurements/serve-build-identity-2026-09-02.md`):*
the stamp exists, and it is **not** the AOT key this paragraph asks for — the
key is the same on both sides of §3, so a key-matched stamp would have
certified the rebuild as a replay. What is stamped is a digest of the cache
slot's contents, which separates the two caches above (`04525ea7…` vs
`dbeb2b8b…`).

*Named, not tested:* the pinned torch has a knob for exactly the mechanism §3(b)
measures. `torch/_inductor/config.py:1007-1009` in torch 2.13.0+cu130:

```python
# A deterministic mode that skips any on device benchmarking in Inductor
# if we know they affect numerics.  WARNING: Expect perf hit in this mode.
deterministic = os.getenv("TORCHINDUCTOR_DETERMINISTIC") == "1"
```

That is read off the pinned build, and it is all this receipt claims: the
option exists and its docstring describes this mechanism. **No arm here was
served with it**, so whether it actually makes two builds of this graph agree is
unmeasured, and it is not switched on anywhere — turning it on would change the
numerics of every future arm relative to every arm already measured, which is a
campaign-level decision, not a worker's. #28 carries it.

---

## Appendix: what was compared

Every dump is `prismaquant.kl_position_dump/2` on the Qwen corpus contract
`cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d` (8 × 512,
4088 scored positions, tokenizer `/home/rob/models/Qwen3-0.6B`), top-1024
support, teacher-student intersection, lower bound by the data-processing
inequality. `serving_compile_divergence.py` refuses any pair whose corpus
contract or tokenizer digest disagrees, and refuses an `eager_vs_compiled` or
`build_vs_build` pair whose two dumps name different `artifact_path`s — the
comparison that would look like a result and be a mislabelled one.
