# The inductor determinism knob, measured: as a serve would set it, it does not survive the first compile

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
caches, and section 5 says what that leaves open.

## 1. The matched set

`experiments/inductor_determinism_probe.py`, two builds per arm, each build in a
**child process of its own** with a **fresh `TORCHINDUCTOR_CACHE_DIR`** — the
flag is read once, at inductor's import (`torch/_inductor/config.py:1007-1009`:
`deterministic = os.getenv("TORCHINDUCTOR_DETERMINISTIC") == "1"`), so a build
that inherits a warm process inherits the wrong answer.

Each child does **two** `torch.compile`s in one process: an rms-norm-shaped block
at hidden 8192, then the same block at hidden 4096.  That second compile is the
whole point, because a vLLM serve compiles many graphs in one process.

| arm | what differs |
|---|---|
| `off` | control, `TORCHINDUCTOR_DETERMINISTIC=0` |
| `on` | `TORCHINDUCTOR_DETERMINISTIC=1`, set the way a serve sets it (an env var) |
| `on_reassert` | the same, plus `config.deterministic = True` put **back** on the config object before the second compile |

Two runs, both on sparky through `pbrun`, both in the pinned image:

| run | schema | arms | GPU power before | pbrun action | result |
|---|---|---|---|---|---|
| 1 | `/1` | `off`, `on` | **5.46 W** of a ~140 W envelope (idle) | `f8e771c3fc97`, 25 s | `experiments/results/inductor_determinism_probe_two_arm.json` |
| 2 | `/2` | `off`, `on`, `on_reassert` | **74.24 W** (another job resident) | `bdd06c2953ef`, 36 s | `experiments/results/inductor_determinism_probe.json` |

The load difference is not cosmetic and is corroborated inside the data: the same
autotune candidates took 63–74 ms to benchmark in run 1 and 135–151 ms in run 2.

## 2. Two fresh builds diverge, and the box being busy is not the reason

The divergent record is the same one in both runs — `cg/f62c0da5…`, the second
compile's — and the flip is the same flip:

| run | box | arm | `cg/f62c0da5…` build a | build b | diverged? |
|---|---|---|---|---|---|
| 1 | idle, 5.46 W | `off` | `R0_BLOCK` 1024, `num_warps` 8 (63 ms) | **4096, 16** (64 ms) | **yes** |
| 1 | idle, 5.46 W | `on` | 4096, 16 (74 ms) | 4096, 16 (73 ms) | no |
| 2 | busy, 74 W | `off` | 1024, 8 (141 ms) | 1024, 8 (143 ms) | no |
| 2 | busy, 74 W | `on` | 1024, 8 (151 ms) | **4096, 16** (150 ms) | **yes** |

`configs_hash` is identical throughout (`8a75395a…`), so the candidate set never
changed and only the pick did.  Two things follow.

**Box contention is sufficient, not necessary.**  The off arm diverged on an
**idle** box (run 1) and agreed on a loaded one (run 2).  §3b of the earlier
receipt explained the 120/196 by "three other workers were loading this box";
that explanation is not needed and is not what makes this reproduce.

**A scope correction the earlier receipt should carry.**  §3b reasons "a
different `XBLOCK`/`num_warps` is a different reduction tree, so it is different
floating-point arithmetic, so it is a different logit — which is the observed
0.017117".  Here the tiling changed and `outputs_bitwise_equal` is **true**, in
every arm of both runs — the fp32 reduction differences vanished under the bf16
store.  The 0.017117 is measured; its attribution to those 120 records is
inferred.  120 is an upper bound on how many moved a logit, not a count of them.

## 3. The finding: the flag does not survive the first compile

The flag plainly governs the compile it reaches.  In the `on` arm the **first**
compile's kernels carry `'deterministic': True` and `has_loadstore_with_contiguous_rdim`
in their own `triton_heuristics` metadata, and the first compile's `.best_config`
record is **gone** — algorithm selection stopped benchmarking
(`torch/_inductor/select_algorithm.py:3840,3901` `pick_deterministic_choice`;
`torch/_inductor/runtime/benchmarking.py:158` `may_ban_benchmarking`).

Then it stops.  `arms.*.kernel_meta_deterministic_by_phase`, run 2:

| arm | first compile's kernels | second compile's kernels | `.best_config` records per build |
|---|---|---|---|
| `off` | `False` | `False` | 2 |
| `on` | **`True`** | **`False`** | 1 (the second compile's) |
| `on_reassert` | **`True`** | **`True`** | **0** |

Three observables agree, and they are independent of each other:

1. **The kernel's own record of the flag.**  The `'deterministic'` key is written
   at `torch/_inductor/codegen/triton.py:6200` as `"deterministic":
   config.deterministic or config.batch_invariant` — a *transcript of the config
   value at the moment that kernel was generated*, not an inference about it.
   Under `on`, the first compile transcribes `True` and the second `False`.
2. **The codegen.**  In run 1 the second compile's kernels are byte-identical
   between the flag-off and flag-on builds: all four builds emit
   `cg/ccg3diw3…py` verbatim, and their other second-compile kernel hashes to
   `cbdddc82d2ce551e` in all four once the one differing line — the absolute
   `# kernel path:` comment, which contains the build directory's own name — is
   dropped.  The flag left no trace on the second compile's source.
3. **The surviving autotune, and the choice it made.**  Exactly one
   `.best_config` survives under `on`, the second compile's, still carrying a
   device-measured `time_taken_ms`.  Under the flag that record could not have
   been written: `may_ban_benchmarking` does not skip a non-vetted benchmark, it
   **raises** `RuntimeError("In the deterministic mode of Inductor, we will avoid
   those benchmarkings…")`.  And in run 2 that record is the one that **flipped
   between the two flag-on builds** — the precise failure the flag exists to
   prevent, happening with the flag set.

The config attribute agrees but is the weakest leg: it reads `True` at import in
every `on` and `on_reassert` build and **`False` after the first compile** in all
of them (`knob.reads`; `knob.summary.*.resets_after_compile` is `True`).  It is
listed last on purpose — an attribute read after the fact could be an artifact of
how the config module reports state, which is why the conclusion rests on what
the compiler *wrote*.

**`on_reassert` is the control that separates the two explanations.**  Putting
`config.deterministic = True` back before the second compile removes **every**
device-timed record in both builds (0 and 0) and makes both phases transcribe
`True`.  So the second graph was not "a graph that would not have used the flag
anyway": it would have, if the flag had still been set.  **The flag stopped
applying.**

**Therefore: `TORCHINDUCTOR_DETERMINISTIC=1`, set as an environment variable the
way `TESSERA_SERVE_DETERMINISTIC=1` sets it, governs the first inductor compile
in the process and not the ones after it.**  A serve that compiles N graphs from
an empty cache gets a deterministic graph 1 and N−1 benchmarked ones, and run 2
shows one of those N−1 flipping.

## 4. What this does to a gate we already have

`src/tessera/serving/build_identity.py:476` `deterministic_effective()` certifies
a build on `inductor_deterministic and fresh_compiles > 0`.  Under section 3, a
record with `fresh_compiles > 1` is certified on the strength of its first
compile alone.

The gate is **not changed here**, because changing it needs a number that does
not exist yet: how many fresh inductor compiles a vLLM backbone build actually
performs in one process.  The one compile-dispatch serve log that says anything
(`/home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_compiled.log:29`)
reports `num_artifacts=3 num_submods=29` — and it *loaded* that AOT artifact
("Directly load AOT compilation from path …", `:30`) rather than building it, so
it counts what was replayed, not fresh compiles.  Named here and in #16 rather
than acted on.

## 5. What was not measured

- **No serve.**  Two vLLM serves of one checkpoint from **empty** compile-cache
  roots under `TESSERA_SERVE_DETERMINISTIC=1`, compared at 0.000000 — the receipt
  #16's third bullet asks for — still does not exist.  The pool refused the GPU
  placement for it (`pbrun: no live worker can run this action … demand {'gpu':
  1, 'mem_gb': 16}`) and then queued 38–42 deep for an hour on the only box that
  can see this worktree; a serve taken under that would have measured the queue.
- **Whether reasserting is the fix.**  `on_reassert` shows the flag *can* govern
  a later compile if something re-sets it.  Nothing sets it inside a vLLM serve,
  and this receipt does not propose a patch that would — a producer-side
  monkeypatch of another runtime's compiler is exactly the kind of claim
  principle 14 refuses to let a receipt assert.
- **Scale.**  This graph autotunes 2 records; the served backbone autotuned 196.
  An arm that agrees on a toy is not evidence a real build is reproducible, and
  outputs here were bitwise equal in every arm of both runs — so this probe
  cannot say what a tiling flip costs in KL.
- **The reset's mechanism.**  *Which* line clears the flag is not located.  A
  grep of the pinned build finds **no assignment to `config.deterministic`
  anywhere in `torch/_inductor`** (top level plus one directory down) — all
  fourteen sites are reads.  So it is not a plain write; it is the config
  module's own state machinery, and this receipt states the behaviour, not its
  cause.

## 6. Provenance

Two runs, both on sparky, both inside the image resolved by `PYTHONPATH=src
python3 -m tessera.serving.runtime_image pin`
(`vllm/vllm-openai@sha256:61fc8a896b0a…`, torch 2.13.0+cu130, sm121, driver
595.84):

* run 1 — `pbrun` action `f8e771c3fc97`, `executed on sparky in 25s`, 2026-09-04
  01:38:28–01:38:43 EDT, GPU 5.46 W before →
  `experiments/results/inductor_determinism_probe_two_arm.json` (schema
  `tessera.inductor_determinism_probe/1`).
* run 2 — `pbrun` action `bdd06c2953ef`, `executed on sparky in 36s`, 2026-09-04
  02:51 EDT, GPU 74.24 W before / 73.92 W after →
  `experiments/results/inductor_determinism_probe.json` (schema `/2`).

Per-record and per-kernel details in sections 2 and 3 were read back off each
run's own cache roots.  Torch line numbers are from a grep of the same image
(`pbrun` action `4884c24517a9`).  The 120/196 count, the 0.017117 rebuild KL and
the 95.65% top-1 are quoted from `serving-compile-divergence-2026-09-02.md` and
carry that receipt's provenance, not this one's.

Tests touched by the change that carries this receipt: `tests/test_issue_refs.py
tests/test_audit_doc_claims.py tests/test_serve_build_identity.py
tests/test_inductor_determinism_knob.py` — **44 passed in 43.06 s** on sparky
(`pbrun` action `189a47296cf3`).
