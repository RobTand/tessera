# tessera #100 -- pin the serving image by digest, and refuse on mismatch

Branch `muse/ts-100-vllmpin`, worktree `/home/rob/tmp/musefix/ts-100-vllmpin`.
Not pushed, not merged.

## The headline finding, first: no pull was needed, and here is why

The issue's premise -- "the two boxes hold different bytes under `:latest`" --
is a **local-image-id artefact**. Both GB10s already carry the chosen digest.

```
sparky     .Id sha256:61fc8a896b0a...  RepoDigests ["vllm/vllm-openai@sha256:61fc8a896b0a..."]
sparklina  .Id sha256:89154ef00dd1...  RepoDigests ["vllm/vllm-openai@sha256:61fc8a896b0a..."]
```

`sha256:61fc8a896b0a...` is a **manifest digest**, not an image id. Sparky runs
docker 29 with the containerd snapshotter, where `.Id` *is* the manifest
digest; sparklina runs overlay2, where `.Id` is the config digest. Same bytes,
two ids. `docs/measurements/tessera-serving-plugin-2026-09-02.md` section 9 had
already measured exactly this and warned that "ranking two boxes as different
runtimes off the local Id would have been wrong".

Consequences, and none of them changes the decision:

* **The chosen digest is right and is satisfiable on both boxes today.** No
  substitution, no stop-and-report.
* **Step 4 of the brief is already satisfied.** Nothing was pulled onto
  sparklina -- there was nothing to pull, and its load average was 80. The
  sequencing constraint ("do not land a hard refusal before sparklina holds the
  pin") is met vacuously, so the gate is hard from the first commit.
* **The gate had to be built around this or it would have been worse than
  nothing.** A gate comparing `.Id` -- the literal reading of "resolves the
  local image id" -- would have refused sparklina forever for holding identical
  bytes, and a refusal that permanently disables one box is not a fix. The
  check is membership of the pin in `RepoDigests`.

Verified on both boxes with the real CLI against the real daemon (sparklina
exercised through a temporary copy of the module + contract, since the branch
lives on sparky; the copy was removed afterwards):

```
sparky     resolve vllm/vllm-openai:latest -> reason "pinned", rc 0, local_id 61fc8a89...
sparklina  resolve vllm/vllm-openai:latest -> reason "pinned", rc 0, local_id 89154ef0...
```

## Where the single pin lives

`src/tessera/serving/runtime_contract.json`, `versions.attested_on.image`,
now `vllm/vllm-openai@sha256:61fc8a896b0a...` instead of the floating tag.
Contract bumped v9 -> v10 with a changelog entry saying what changed and why.

That field was already this package's statement of *what runtime it was
attested on* (principle 14), so putting the pin there means moving the pin
requires re-attesting. `src/tessera/serving/runtime_image.py` is the only
reader. **No script, test or doc holds a second copy of the digest**, and
`test_the_pin_is_the_contract_field_and_nothing_else_holds_it` enforces that
over `git ls-files` -- exempting prose, because a measurement doc recording the
digest its serve ran under is a receipt, not a copy of the pin (one such doc
exists: `docs/measurements/tessera-ldlq-window-served-2026-09-02.md`).

`pinned_reference()` also **refuses a tag in the contract itself**, so the gate
cannot be made vacuous by an edit that still reads like a pin.

## What the refusal does, and how a program reads it

`experiments/runtime_image.sh` is sourced by every wrapper that starts a
container. `runtime_image_require IMAGE`:

* resolves `IMAGE` against the local daemon and compares the pin to
  `RepoDigests`;
* on mismatch **or absence**, returns 2. Every call site is written
  `runtime_image_require "$IMAGE" || exit 2`, so a wrapper without `set -e`
  refuses too;
* prints the machine-readable record on **stdout** and the prose on **stderr**.
  Both, always: a gate only a human can read gets skipped by the script that
  should have honoured it.

The record a program reads:

```json
{"schema":"tessera.runtime_image/1","refused":true,"reason":"image_pin_mismatch",
 "requested":"vllm/vllm-openai:latest","requested_tag":"latest",
 "pinned":"vllm/vllm-openai@sha256:61fc...",
 "repo_digests":["vllm/vllm-openai@sha256:0000..."],
 "resolved_digest":"sha256:0000...","local_id":"sha256:cccc...","gated":true,
 "fix":"docker pull vllm/vllm-openai@sha256:61fc..."}
```

In Python the same object is `RuntimeImageError.payload`. `reason` is one of
`pinned`, `image_pin_mismatch`, `image_absent`, `not_pinned_repository`.

**The gate runs before `serve_lock_acquire` in every wrapper that takes the
lock.** A wrapper that is going to refuse must not first take the box's one
serve lock and make fourteen other agents queue behind a failure.

**Scope, and it is a judgment I am flagging rather than burying.** The pin
governs one repository: `vllm/vllm-openai`. `serve_and_dump_kl.sh` and
`nvfp4_moe_oracle_probe.sh` default to `prismaquant/glm53-mia-sm121:487ecf187`,
which is a different runtime with no pin; those are **resolved and stamped, not
refused**, because gating them against a pin that does not exist would break
GLM serves to enforce nothing. Widening the pinned set is Rob's call, not an
inference; the module says so and the code takes one line to change.

## Which receipts now carry the id

`build_identity.py` gained two fields, and the split between them is the point:

* `identity.image_digest` -- the **resolved manifest digest**. In `identity`,
  so it is inside `build_fingerprint`: this is what ran.
* `provenance.image_local_id` -- the local docker id. Deliberately **not** in
  `identity`: it differs between the two GB10s for identical bytes, so
  fingerprinting it would make every cross-box comparison refuse itself.

`identity.image` (the reference the wrapper asked for) is unchanged, so a
receipt now records both what was requested and what it resolved to.

The digest reaches the stamp through `RUNTIME_IMAGE_DIGEST` /
`RUNTIME_IMAGE_LOCAL_ID`, which `runtime_image_require` sets and
`build_identity_stamp` forwards -- no signature change, so no caller churn, and
a wrapper that did not gate stamps neither field rather than a wrong one.

Receipts covered: **`experiments/serve_and_dump_kl.sh`** and
**`experiments/tessera_plugin_served.sh`** (the two that write build sidecars).
The four wrappers that write no sidecar -- `serve_smoke_graph.sh`,
`gridbook_lane_served.sh`, `tessera_plugin_run.sh`,
`nvfp4_moe_oracle_probe.sh` -- gate and echo `image <ref> -> <digest> (local id
<id>)` into their own output, which is the receipt they have.

All six wrappers that call `docker run`, and the five campaign wrappers that
exported `TESSERA_KL_IMAGE=vllm/vllm-openai:latest`
(`stock_lane_served.sh`, `rotation_serve_arms.sh`, `ldlq_lut_serve.sh`,
`ldlq_lut_chain.sh`, `bf16_route_served.sh`), now default to
`$(runtime_image_pin)` -- so the common path *runs by digest* rather than being
verified into it.

### Two things I deliberately did not change

* `experiments/allocated_serve_2026-09-02/chain_{allocated,probe}.sh` still
  set `TESSERA_KL_IMAGE=vllm/vllm-openai:latest`. They are dated records of a
  run, they route through the now-gated harness, and rewriting a record to
  match today's rule is the kind of edit that makes history lie. They will
  pass while `:latest` resolves to the pin and refuse loudly if it stops
  doing so, which is correct in both directions.
* `docs/measurements/*` and `experiments/results/*.log` naming `:latest` are
  receipts of what ran. Untouched.

## Test evidence

See the "Tests" section of the branch report handed to the coordinator; the
suite log is `/home/rob/tmp/ts100-suite.log`.

New file `tests/test_runtime_image_pin.py`, 12 tests. It pins the **rule**,
never the digest:

| test | what it pins |
|---|---|
| `..._pin_is_a_digest_reference_not_a_tag` | the pin matches `repo@sha256:<64 hex>` |
| `..._contract_field_and_nothing_else_holds_it` | no second copy in any acting file |
| `..._tag_in_the_contract_is_refused_rather_than_honoured` | the gate cannot be edited into vacuity |
| `..._tag_resolving_to_other_bytes_is_refused_not_warned` | mismatch raises; payload carries `reason` + `fix` |
| `..._absent_image_is_the_same_refusal_with_the_same_fix` | absence is a refusal, not a pass |
| `..._local_id_never_decides_anything` | the cross-box case: two ids, one verdict |
| `..._unpinned_repository_is_stamped_not_refused` | Mia's image survives |
| `..._every_wrapper_that_starts_a_container_gates_...` | every `docker run` wrapper is behind the gate |
| `..._shell_helper_refuses_and_prints_json_...` | the wrapper's own control flow stops, JSON on stdout |

No test shells out to `docker`: the daemon's answer is injected, which is what
lets the two-box case be tested at all when there is only one box in a test
run. The live cross-box check was done by hand and is recorded above.

`tests/test_serve_build_identity.py` was asserting the literal tag string; it
now reads `pinned_reference()` and additionally asserts `identity.image_digest`
and `provenance.image_local_id`.

## For the coordinator: two merge hazards

### A fingerprint discontinuity

`identity` gained a key, so `build_fingerprint` changes shape at this commit.
`compare()` between a pre-pin receipt and a post-pin one reports
`differs: ["image", "image_digest"]` and `require_same_build` refuses. That is
correct -- the old receipt recorded nothing about what ran -- but **a campaign
that straddles this merge will have arms that refuse to compare**. Re-stamp or
re-serve the older arm, or complete the campaign before merging.

### Merging rewrites scripts that are executing right now

Fourteen agents are serving off `/home/rob/tessera`, and bash reads a script
incrementally rather than slurping it: a wrapper part-way through a serve can
execute a spliced file if `serve_and_dump_kl.sh` and its siblings are rewritten
under it. Every wrapper in this change is one of those. That is a merge-timing
call to price, not a defect in the change; it argues for merging into a quiet
window rather than beside a live campaign.

## Off-task fixes

* `docs/status/2026-09-01-where-tessera-stands.md:734` and
  `src/tessera/serving/bf16_route.py:74` called serves taken under the floating
  tag "the pinned image". Now "the stock vLLM 0.28 image", which is what those
  censuses can honestly claim. (The brief named
  `docs/tessera-one-format.md:297`; I read it, and it says "vanilla
  `vllm/vllm-openai` v0.28.0" with no claim of pinning -- nothing to fix there.
  These two are the real instances.)
* `docs/ARCHITECTURE.md` gained section 4.4a saying what "the pinned runtime"
  now denotes, per its own maintenance rule.
* `experiments/inductor_determinism_probe.py:41`'s docstring `docker run`
  example named the floating tag; it now names the pin by reading it.

## What this does not claim

Which digest the pre-2026-09-03 receipts were taken under is still not
recoverable from what is written down. The pin is the point at which
provenance *starts* being knowable, not a retroactive statement about the
archive.
