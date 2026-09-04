# The inductor determinism knob, measured: set as a serve sets it, it does not reach a second compile entry

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
whole point, because a vLLM serve compiles more than one graph in one process.
Note the shape of this probe, because section 3 finds it matters: two graphs
reached through **two `torch.compile` entries**, which is not the only way a
serve gets to more than one graph.

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

**Box contention is neither necessary nor sufficient.**  The `off` arm — the
like-for-like pair — diverged on an **idle** box (run 1, 5.46 W) and *agreed* on
a loaded one (run 2, 74.24 W).  §3b of the earlier receipt explained the 120/196
by "three other workers were loading this box"; that explanation is not needed,
does not predict either of these two runs, and is not what makes this reproduce.

**A scope correction the earlier receipt should carry.**  §3b reasons "a
different `XBLOCK`/`num_warps` is a different reduction tree, so it is different
floating-point arithmetic, so it is a different logit — which is the observed
0.017117".  Here the tiling changed and `outputs_bitwise_equal` is **true**, in
every arm of both runs — the fp32 reduction differences vanished under the bf16
store.  The 0.017117 is measured; its attribution to those 120 records is
inferred.  120 is an upper bound on how many moved a logit, not a count of them.

## 3. The finding: the flag governs the compile it reaches, and not the next one

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

**Therefore, exactly what was measured: `TORCHINDUCTOR_DETERMINISTIC=1`, set as
an environment variable the way `TESSERA_SERVE_DETERMINISTIC=1` sets it, does not
survive to a second `torch.compile` entry in the same process.**  The second
entry benchmarks, and in run 2 its choice flipped between two flag-on builds.

**What that costs a *serve* depends on a layer this probe cannot see.**  The two
compiles here are two `torch.compile` wrappers — two Dynamo entries, one inductor
compile under each.  A vLLM backbone is the other shape: **one** Dynamo entry
whose `VllmBackend` splits the graph and runs several inductor compiles beneath
it, which is the shape the compile-dispatch serve log reports when it says
`num_submods=29 num_artifacts=3`
(`/home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_compiled.log:29`;
that line describes a build it then *loaded* from AOT at `:30`, so read it for
the graph's shape, not as a count of fresh compiles).  Two hypotheses fit every
observable above and they differ in what a serve gets:

* the reset fires **per Dynamo entry** — then a backbone's submodule compiles all
  sit inside the first entry, all get the flag, and a serve that makes one Dynamo
  entry is covered;
* the reset fires **per inductor compile**, or at first codegen (which the
  first-compile `True` transcript is equally consistent with) — then a backbone
  gets submodule 1 deterministic and the rest benchmarked.

This probe does not discriminate them: it changed both layers at once.  The
discriminator is small and named in section 5.

## 4. What this does to a gate we already have

`src/tessera/serving/build_identity.py:476` `deterministic_effective()` certifies
a build on `inductor_deterministic and fresh_compiles > 0`.  Under section 3, a
record with `fresh_compiles > 1` is certified on the strength of its first
compile alone.

The gate is **not changed here**, and the reason is the open question at the end
of section 3, not a missing count.  `fresh_compiles` counts lines matching
`"Dynamo bytecode transform time"` in the serve log
(`build_identity.py:113,157`) — that is a count of **Dynamo entries**, which is
exactly the layer whose relationship to the reset is unresolved.  So:

* if the reset fires per Dynamo entry, `fresh_compiles > 1` is precisely the
  right refusal and this gate is one measurement away from correct;
* if it fires per inductor compile, `fresh_compiles == 1` certifies a backbone
  whose submodules after the first were benchmarked, and the count the gate would
  need is one the serve log does not print at all.

Picking a threshold before knowing which is true is guessing.  Named here and in
#16 rather than acted on.

## 5. What was not measured

- **No serve.**  Two vLLM serves of one checkpoint from **empty** compile-cache
  roots under `TESSERA_SERVE_DETERMINISTIC=1`, compared at 0.000000, still does
  not exist.  Being exact about why, since "the box was busy" is the kind of
  excuse this project distrusts: **it was never submitted.**  Through the window
  this receipt was measured in, the pool ran 38–42 ready and 6 claimed against 3
  workers on `sparky` — the only box that can see this worktree — and the small
  GPU probe above waited in that queue twice (`pbrun: gave up waiting for
  922705a786a2`, and one refusal, `pbrun: no live worker can run this action …
  required tags ['sparky'] … demand {'gpu': 1, 'mem_gb': 24}`, from tagging it
  `--here`).  A pair of serves is minutes of exclusive GPU each; taken against
  that queue it would have been a contended measurement of a compile whose whole
  subject is how compile-time device timing varies with contention.  So it was
  not taken.
- **Whether reasserting is the fix.**  `on_reassert` shows the flag *can* govern
  a later compile if something re-sets it.  Nothing sets it inside a vLLM serve,
  and this receipt does not propose a patch that would — a producer-side
  monkeypatch of another runtime's compiler is exactly the kind of claim
  principle 14 refuses to let a receipt assert.
- **Scale.**  This graph autotunes 2 records; the served backbone autotuned 196.
  An arm that agrees on a toy is not evidence a real build is reproducible, and
  outputs here were bitwise equal in every arm of both runs — so this probe
  cannot say what a tiling flip costs in KL.
- **Which layer resets, and therefore what a serve gets.**  Section 3's two
  hypotheses are not separated here.  The experiment that separates them is
  small and needs no serve: compile two submodules through
  `torch._inductor.compile_fx.compile_fx` inside **one** Dynamo entry (or with
  no Dynamo entry at all) in one process with the flag set, and read the
  `deterministic` key out of each one's `triton_heuristics` metadata.  Both `True`
  means the reset is per Dynamo entry and a one-entry serve is covered; the
  second `False` means every backbone submodule after the first is benchmarked.
  Until that runs, `deterministic_effective()` stays as it is (section 4).
- **The reset's mechanism.**  *Which* line clears the flag is not located.  A
  two-level grep of the pinned build (`torch/_inductor/*.py` and
  `torch/_inductor/*/*.py`) finds **twelve direct reads of `config.deterministic`
  and no assignment to it**.  That grep cannot see an indirect write —
  `config.patch(deterministic=…)`, a `setattr`, a `load_config` dict — and did not
  look for one, and it did not descend past one directory.  So: not a plain write
  at those twelve sites; the rest is the config module's own state machinery, and
  this receipt states the behaviour, not its cause.

## 6. So what does #16's third bullet actually get?

#16 asks, third bullet, for "a pinned inductor cache for measurement runs so a
build is reproducible."  The determinism knob was a *proposed* way to get that
without pinning anything; as wired it is now measured not to be one.  The
mechanism that **is** measured to work is the pinned cache itself, and it already
ships:

* **Replay from one cache root is bit-identical.**  `historical-compiled vs
  compiled` reads KL 0.000000 / 100.00% top-1 with the cache root present for
  both builds (`serving-compile-dispatch-2026-09-03.md` §3, "What this does not
  settle"), and the eager control `historical-eager vs eager` is 0.000000 as
  well.
* **Which build served an arm is stamped, not assumed.**
  `tessera.serving.build_identity` digests the *contents* of the compile-cache
  slot beside every KL dump, because the AOT key does not distinguish the two
  builds the divergence receipt was written from
  (`serve-build-identity-2026-09-02.md` §1).
* **A campaign that rebuilt anyway is refused, not silently compared.**
  `require_same_build()` (`build_identity.py:421`) fails a pair whose compiled
  builds differ, `require_same_dispatch()` (`:433`) fails a pair that resolved
  vLLM's dispatch pair differently, and since this branch a shell can reach both
  through `build_identity compare --require {same,distinct,same-dispatch}`.

So the operational answer is: pin the cache root for the length of a campaign,
never empty it mid-campaign, and let the stamp refuse the pair when it happened
anyway.  What is **not** available is a compiled build that reproduces *from an
empty cache*, and this receipt is the reason the flag named for that job does not
deliver it.

## 7. Provenance

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
