# The inductor determinism knob, measured: it governs one compile, not one process

**Date** 2026-09-04 · **Box** sparky (GB10, sm121, driver 595.84) · **Image**
`vllm/vllm-openai@sha256:61fc8a896b0a…` · **torch** 2.13.0+cu130 ·
**Issue** #16 (second defect: compiled builds are not reproducible)

## 0. Why this exists, and what it is not

`docs/measurements/serving-compile-divergence-2026-09-02.md` §3b counted the
mechanism behind the one non-reproducible compiled arm: under a single AOT key,
with byte-identical `cache_key_factors.json`, **120 of 196** autotuned Triton
kernels chose a different `XBLOCK`/`num_warps` on the second build, and the
rebuild read KL 0.017117 / 95.65% top-1 against its own predecessor.  §4 of that
receipt named `TORCHINDUCTOR_DETERMINISTIC=1` as the remedy and
`docs/measurements/serve-build-identity-2026-09-02.md` §4 wired it as
`TESSERA_SERVE_DETERMINISTIC=1` — while recording, in both places, that **no arm
was ever served with it**.  #30 was closed on the wiring; the knob itself stayed
a name.

This receipt measures the knob.  It does **not** serve anything: the arms here
are a two-graph toy in the pinned image, not two vLLM backbones from empty
caches, and section 4 says what that leaves open.

## 1. The matched set

`experiments/inductor_determinism_probe.py`, two arms, two builds per arm, each
build in a **child process of its own** with a **fresh
`TORCHINDUCTOR_CACHE_DIR`** — the flag is read once, at inductor's import
(`torch/_inductor/config.py:1007-1009`: `deterministic =
os.getenv("TORCHINDUCTOR_DETERMINISTIC") == "1"`), so a build that inherits a
warm process inherits the wrong answer.  The only difference between the arms is
that environment variable.

Each child does **two** `torch.compile`s in one process: an rms-norm-shaped
block at hidden 8192, then the same block at hidden 4096.  That second compile
is the whole point of section 3.

**The box was idle.**  `nvidia-smi` immediately before the run read **5.46 W**
of a ~140 W envelope, and pbrun placed the action on sparky and it finished in
25 s.  This matters because the earlier receipt attributed the 120/196 to a
loaded box, and section 2 says that attribution is not needed.

Result: `experiments/results/inductor_determinism_probe.json`
(schema `tessera.inductor_determinism_probe/1`).

## 2. Two fresh builds diverge on an idle box

| arm | autotune records | in both builds | same choice, new time | **different choice** | outputs bitwise equal |
|---|---|---|---|---|---|
| `off` | 2 / 2 | 2 | 1 | **1** | yes |
| `on`  | 1 / 1 | 1 | 1 | 0 | yes |

The off arm's divergent record, verbatim from the two builds:

```
off0  cg/f62c0da5….best_config  {"XBLOCK": 1, "R0_BLOCK": 1024, "num_warps": 8,  …,
                                 "time_taken_ms": 63, "triton_cache_hash": "M4TG4EN3…"}
off1  cg/f62c0da5….best_config  {"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 16, …,
                                 "time_taken_ms": 64, "triton_cache_hash": "EXFMVC7U…"}
```

`configs_hash` is identical (`8a75395a…`), so the candidate set was the same and
only the pick differed; the two measured times that decided it are 63 ms and
64 ms apart in a *different* direction from the choice.  **So a fresh compile
picks a different tiling from a fresh compile on a quiet box.**  Contention is
sufficient to cause this, not necessary, and #16's second defect does not need a
loaded box to reproduce.

**A scope correction the earlier receipt should carry.**  §3b of
`serving-compile-divergence-2026-09-02.md` reasons "a different
`XBLOCK`/`num_warps` is a different reduction tree, so it is different
floating-point arithmetic, which is the 0.017117".  Here the tiling changed and
`outputs_bitwise_equal` is **true** — the fp32 reduction differences vanish under
the bf16 store.  The 0.017117 is measured; its attribution to those 120 records
is inferred, and nobody has classified which of the 120 are numerics-affecting.
That does not overturn the rebuild number, it prices it: the count of differing
records is an upper bound on the count that mattered.

## 3. The finding: the flag governs the first compile only

The flag plainly reaches the compile it reaches.  In the `on` arm the
first compile's kernels carry, in their own `triton_heuristics` metadata,
`'deterministic': True` and `has_loadstore_with_contiguous_rdim: True`, and the
first compile's `.best_config` record is **gone** — algorithm selection stopped
benchmarking (`torch/_inductor/runtime/benchmarking.py:158` `may_ban_benchmarking`,
`torch/_inductor/select_algorithm.py:3840,3901` `pick_deterministic_choice`).

Then it stops.  The generated kernels record their own shape, which says which
`torch.compile` emitted them (`'x': 8, 'r0_': 8192` is the first, `'x': 256,
'r0_': 4096` the second):

| build | first-compile kernels | their `'deterministic'` | second-compile kernels | their `'deterministic'` |
|---|---|---|---|---|
| `off0` | `ok/coknxgc3`, `qz/cqz5ad6o` | False | `cg/ccg3diw3`, `dk/cdk7yvnc` | False |
| `on0`  | `hm/chmuejdl`, `ss/cssclfga` | **True** | `ik/cikh4ylt`, `cg/ccg3diw3` | **False** |

Three observables agree, and they are independent of each other:

1. **The kernel's own record of the flag.**  The `'deterministic'` key in the
   emitted metadata is written at `torch/_inductor/codegen/triton.py:6200` as
   `"deterministic": config.deterministic or config.batch_invariant` — it is a
   *direct transcript of the config value at the moment that kernel was
   generated*, not an inference about it.  In the `on` builds the first
   compile's kernels transcribe `True` and the second compile's transcribe
   `False`.  (`has_loadstore_with_contiguous_rdim` comes from the same block,
   `:6251-6252`, under the same condition.)
2. **The codegen.**  The second compile's kernels are byte-identical between the
   flag-off and flag-on builds.  All four builds emit `cg/ccg3diw3…py` verbatim,
   and their remaining second-compile kernel hashes to `cbdddc82d2ce551e` in all
   four once the one line that differs — the absolute `# kernel path:` comment,
   which contains the build directory's own name — is dropped.
3. **The surviving autotune.**  Exactly one `.best_config` survives under the
   flag, and it is the second compile's `cg/f62c0da5…` — the *same record that
   flipped* in the off arm — still carrying a device-measured `time_taken_ms`
   (74, 73).  The flag removed the benchmark for the first compile's `qz/79ad…`,
   which had been stable across both off builds, and left the benchmark running
   on the one that had not been.  Under the flag that record could not have been
   written: `may_ban_benchmarking` does not skip a non-vetted benchmark, it
   **raises** `RuntimeError("In the deterministic mode of Inductor, we will avoid
   those benchmarkings…")`.  A `time_taken_ms` in the record is proof the flag
   was not in force when it was measured.

The config attribute agrees but is the weakest of the four: it reads `True` at
import in both `on` builds and **`False` after the first compile** in both
(`knob.reads.on`; `knob.summary.on.resets_after_compile` is `True`).  It is
listed last on purpose — an attribute read after the fact could be an artifact
of how the config module reports state, which is why the conclusion rests on
what the compiler *wrote*.

That the two `on` builds happen to agree on that surviving record (both
`R0_BLOCK` 4096 / `num_warps` 16, both `triton_cache_hash EXFMVC7U…`) is **one
sample of a still-device-timed choice**, not the flag working.  It is the same
value `off1` picked without the flag.

**So, on the pinned torch: `TORCHINDUCTOR_DETERMINISTIC=1`, set the way a serve
sets it, is in force for the first inductor compile in the process and not for
the ones after it.**  A serve that compiles N graphs from an empty cache gets a
deterministic graph 1 and N−1 benchmarked ones.

**What this does to a gate we already have.**
`src/tessera/serving/build_identity.py:476` `deterministic_effective()` certifies
a build on `inductor_deterministic and fresh_compiles > 0`.  Under the finding
above, a record with `fresh_compiles > 1` is certified on the strength of its
first compile alone.  The gate is **not changed here**, because changing it needs
the number section 4 says does not exist yet: how many fresh inductor compiles a
vLLM backbone build actually performs in one process.  The one compile-dispatch
serve log that says anything (`serve_qwen_dispatch_compiled.log:29`) reports
`num_artifacts=3 num_submods=29` — and it *loaded* that AOT artifact rather than
building it, so it is a count of what was replayed, not of fresh compiles.
Named here, and in #16, rather than acted on.

## 4. What was not measured

- **No serve.**  Two vLLM serves of one checkpoint from **empty**
  compile-cache roots under `TESSERA_SERVE_DETERMINISTIC=1`, compared at
  0.000000 — the receipt #16's third bullet asks for — still does not exist.
  The GPU pool refused the placement for it tonight (`pbrun: no live worker can
  run this action … demand {'gpu': 1, 'mem_gb': 16}` on the only box holding this
  worktree), and a contended serve would not have been worth taking.
- **Scale.**  The graph here autotunes 2 records; the served backbone autotuned
  196.  An `off` arm that agrees on a toy is not evidence a real build is
  reproducible, and the two arms here agree on outputs *bitwise* — so this probe
  cannot say what a tiling flip costs in KL.
- **The reset's mechanism.**  That the flag stops applying across the first
  compile is measured three ways above; *which* line in torch clears it is not
  located.  A grep of the pinned build finds **no assignment to
  `config.deterministic` anywhere in `torch/_inductor`** (top level plus one
  directory down) — every one of the fourteen sites is a read.  So it is not a
  plain write, it is the config module's own state machinery, and this receipt
  states the behaviour and not its cause.
- **Whether reasserting helps.**  `experiments/inductor_determinism_probe.py` now
  carries a third arm (`on_reassert`, schema `/2`) that puts the attribute back
  before the second compile, which would separate "the flag stopped applying"
  from "the second graph would not have used it anyway".  It has not run; the
  numbers above are the `/1` two-arm run.

## 5. Provenance

Every number in this receipt is from one run:
`experiments/results/inductor_determinism_probe.json`, produced 2026-09-04
01:38:28–01:38:43 EDT on sparky through `pbrun` action `f8e771c3fc97`
(`executed on sparky in 25s`), inside the pinned image resolved by
`PYTHONPATH=src python3 -m tessera.serving.runtime_image pin`.  The per-record
and per-kernel details in sections 2 and 3 were read back off the run's own
cache roots.  The torch line numbers are from a grep of the same image
(`pbrun` action `4884c24517a9`).  The 120/196 count, the 0.017117 rebuild KL and
the 95.65% top-1 are quoted from `serving-compile-divergence-2026-09-02.md` and
carry that receipt's provenance, not this one's.

Tests touched by the change that carries this receipt:
`tests/test_issue_refs.py tests/test_audit_doc_claims.py
tests/test_serve_build_identity.py tests/test_inductor_determinism_knob.py`
— **44 passed in 43.06 s** on sparky (`pbrun` action `189a47296cf3`).
