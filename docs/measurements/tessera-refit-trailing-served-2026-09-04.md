# The trailing refit's objective, served (2026-09-04)

**Status: measured and closed. The exporter flag, the pair instrument and the
encoder-drift finding are measured and stand; the served A/B completed, and
its receipts are retained in
[branch-recovery-2026-09-05.md](branch-recovery-2026-09-05.md), which
supersedes the three `PENDING` sections below (kept as history of the queued
run; quote numbers from the recovery receipt, not from them). The closing
note is at the end of this document.**

The action is `experiments/refit_trailing_run_all.sh`, queued on the PrismaBuild
pool twice — `aadd46b6525d…` at `gpu 1 / mem_gb 40 / cpu 4`, and
`b19a7b789fd2…` at `mem_gb 24` once the first was found unplaceable — both with
checkout `/home/rob/tmp/wf75` and tag `sparky`. Either one runs the whole
chain, and the chain is idempotent (the export skips an existing twin, the
serve an existing npz), so whichever lands first does the work and the other
re-verifies. It exports the control, runs the pair check, and **serves only if
the pair check passes**: same-day arms whose codes differ would mean the
encoder is not deterministic, and a served number on top of that is two
treatments again, arriving by a different door. It leaves
`/mnt/shared/tessera-runs/refit-trailing/DONE` carrying the verdict lines and
copies the three receipts beside it; `pb-queue/done/<key>.json` (or
`failed/<key>.json` on a non-zero exit) holds the whole log under
`detail.stdout`. `experiments/refit_trailing_fill_doc.py` then fills the three
sections from those receipts.

Neither had been placed as of 05:30 on 2026-09-04: fifteen GPU items were
queued for sparky's two slots, eight of them 1300–1600 denied passes deep, and
sparklina's GPU was held out of pool. **The worktree must not be removed until
one of them lands** — both actions name it as their cwd.

**What this is.** tessera#75's fair pair, taken to the one leg a screen cannot
supply. The measurement half is on master (`experiments/refit_trailing_pair.py`,
merged `9add21d`): at the wire, matched pass count, swapping only the
**trailing** refit's objective to the full `H` is **0.9191x** on six dense
Qwen3-0.6B units (6 of 6) and **0.9999x** on the six GLM-5.3-Flash experts.
The GLM leg *clears* the 1.00x gate, and at that magnitude the honest reading
is "does not regress GLM", not "improves it".
`tessera.control.assert_plane_promotion` then refused the arm on exactly one
leg:

> the served KL measures arm None, not the promoted arm 'B-Jac ...' -- a served
> number for a different arm is not evidence for it

This is that leg, and nothing else. **No default moves here.**

## The blocker that stood between the screen and the serve

`ActivationSource.refit_objective_trailing` landed with tessera#103 — the
field, the exported config, the merge guard's `SHARED_ACTIVATION` — but no
exporter could set it. `experiments/export_tessera_serving.py` plumbed
`--refit-metric` alone, so the arm was expressible in a measurement script and
not in a checkpoint. `--refit-metric-trailing` is that flag; unset is the
uniform schedule, byte for byte the encode that was already there.

Pinned end to end by `tests/test_refit_trailing.py`: the flag must reach the
recorded `activation_aware` block (wire manifest *and* twin manifest) and it
must reach the bytes without changing their length. Both hold on the real
artifact: `bjac-tessera/tessera_serving_manifest.json` records
`refit_objective: "h^1.0"`, `refit_objective_trailing: "hessian"`, and its
totals are `wire_bytes 220301312` / `on_disk_bytes 220443566` — the incumbent's
figures to the byte.

## The 2026-09-02 bytes are not this A/B's control

The obvious A arm was the shipped LUT-plane incumbent, `ldlqH1-stock-twin`,
served at KL **0.5310275686796917** on 2026-09-02 and again on 2026-09-04 at
01:26 to the last digit. It is the wrong A, and the pair instrument is what
said so rather than a reading of the log.

`experiments/refit_trailing_bytes.py` classifies every shared tensor of two
stock twins. #75's pair is `T R_h T R_h T R_h T R_H` against
`T R_h T R_h T R_h T R_h`, and the encoder's loop is trellis-then-refit
(`src/tessera/encode.py:2715`), so the last trellis pass runs *before* the
trailing refit and the two arms' **codes must be identical** — which is what
the merged screen measured (`codes_sha256` equal on all six units, arms B-Jac
and B-GS). Against `ldlqH1` they are not:

| | `ldlqH1` (2026-09-02) vs `bjac` (2026-09-04) |
|---|---|
| `.weight_packed` | 115 same, **81 different** |
| `.weight_scale` | 0 same, 196 different |
| `wire_bytes` | equal (220301312 both) |
| verdict | **NOT the matched pair** |

Receipt: `experiments/results/refit_trailing_bytes_ldlqH1_vs_bjac.json`.

81 units of moved codes cannot be the trailing objective's doing. They are the
encoder's: `ldlqH1-tessera/tessera_serving_manifest.json` records
`git: "unknown"` and `written: 2026-09-02T19:05:27`, and roughly forty commits
have touched `src/tessera/encode.py` and its neighbours since — the exact LUT
table fit (`f2f319d`), the epsilon-derived swap-accept test (`2f6a15a`), the
raised swap budget (`56b4a26`), the block-landed refit (`3c35ee0`), Lloyd's
fixed point (`c175c7a`, `072155f`), among others. Serving 2026-09-02 bytes
against 2026-09-04 bytes measures the trailing objective **and** that drift:
two treatments and no control, which is the failure
[[two-treatments-are-not-a-control]] records.

So the control is re-exported, not quoted: `a4h1` is the uniform `h^1.0` x4
schedule built by the same checkout, on the same day, as `bjac`. The
2026-09-02 bytes stay in the run as a *drift reading*
(`compare-drift` → `experiments/results/refit_trailing_encoder_drift.json`),
which is what they are now evidence of.

## The arms

| | A (control) | B-Jac (the arm) |
|---|---|---|
| name | `a4h1` | `bjac` |
| inner refits | `h^1.0` x4 | `h^1.0` x3 |
| trailing refit | `h^1.0` | **full `H`, Jacobi** |
| wire | E2M1x2, `q256=896` (the 4-bit TCQ cap), LUT16 plane | same |
| LDLQ | sigma 1.0, block 32 | same |
| encoder | `wf/ts-75` @ this commit | same |

Everything that is not the trailing refit's objective is held fixed: the same
source (`/home/rob/models/Qwen3-0.6B`), the same Hessian capture
(`/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt`), the same static A4 input
scales, the same teacher payload
(`/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json.npz`), the same corpus
contract (`corpus_qwen_n8_s512.json`, the Qwen tokenizer's), the same pinned
image (`vllm/vllm-openai@sha256:61fc8a89…`), the same
`gpu_memory_utilization` (0.15, ~18 GB of 121 — the twin is 0.84 GB and the
rest is KV cache for 8x512 tokens), eager, one box, one day. The 2026-09-02
incumbent was served at 0.30; that is one more reason its 0.5310 is a drift
reading here and not this pair's bar, which is A's own KL.

## The matched pair, on the artifacts

PENDING — `experiments/results/refit_trailing_bytes.json`, `a4h1` against
`bjac`. It must read codes identical on all 196 units, plane moved, wire
lengths equal. If the codes differ there too, the pair theory is wrong and
that is the finding, not a footnote.

## Served

PENDING — `/mnt/shared/tessera-runs/refit-trailing/kl_a4h1.json` and
`kl_bjac.json`, both `prismaquant.kl_compare/2`, both against the teacher
above.

## The gate, verbatim

PENDING — `experiments/results/refit_trailing_pair_gate_served.json`, run with
`--served-arm B-Jac --served-kl-json …/kl_bjac.json --served-bar-json
…/kl_a4h1.json`. A separate file from the merged
`refit_trailing_pair_gate.json`, which is the screen's own verdict and stays
the record of what a screen earns.

## Scope

One model (Qwen3-0.6B, dense), one wire (E2M1x2 at the `q256=896` TCQ cap, LUT16
plane), one `(sigma, block)` = (1.0, 32), one Hessian capture, one corpus, one
box, prefill regime, eager. The screen's cross-check is six GLM-5.3-Flash
experts, which are a different input distribution and not a serve. Nothing here
licenses a default.

One more seam, and it is the same one this page found in the incumbent's bytes.
The screen's ratios (0.9191x, 6/6, GLM 0.9999x) were measured at `9add21d`'s
encoder, and **eleven** commits have touched `src/tessera/{encode,trellis}.py`
since — including the exact LUT table fit (`f2f319d`) and the block-landed
refit (`3c35ee0`), which are the machinery this pair exercises. The served leg
below is built by today's encoder. So the gate is fed a screen and a serve
taken at different encoders; that is weaker than the drift the incumbent
carried (the arms *within* each leg are matched), but it is the same kind of
gap, and re-running the screen at this checkout is what would close it.


## Closure recovered on 2026-09-05

The campaign above completed. Its three original result files are byte-identical
to the surviving receipts and are now retained with the two served KL JSONs in
[the recovery receipt](branch-recovery-2026-09-05.md). This supersedes the
`PENDING` status and the requirement to keep the old worktree for an in-flight
job. The historical mean improves but the tail worsens; the method remains
opt-in and this is not a current-encoder qualification. The completed
fixed-namespace `refit_trailing_run_all.sh` launcher is retired.
