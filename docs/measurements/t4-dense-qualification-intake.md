# Current-owner T4 dense qualification intake

## Disposition

**Incomplete; no new qualification or admission.** This is an intake audit, not
`t4-dense-qualification.json`,
not a substitute report, and not a publication request. No GPU numerical or timing
campaign was launched. Defaults, static FP4 semantics, qualified menus and the
PrismaQuant pin are unchanged.

Inspected checkout: `50afcb4a44d0bbac26d25930d36a4681b8a98f91`.
Served Tessera reference: `17b3fdb396410698c9a142615345f046d2f94a5d`.
The qualification producer is identical in those two sources. Source equality
alone does not establish the identity of a loaded native build.

## Existing producer and exact gaps

`experiments/t4_code/t4_fused_qualify.py` already provides v3 reports, production
`nvfp4_route` intake/execution, current-wire numerical comparison, configurable
shapes and explicit row/column/replicated TP cuts. Do not replace those mechanisms.
However:

- `dense_frame` packs one `weight` role, `load_dense_route` passes one output
  partition, and `dense_cells` selects `tessera_a4_roles[0]`. These do not prove
  ordered merged-role assembly or replicated small-role handling.
- Global-shape TP cuts cannot be applied blindly to a construction-local shape.
  KDA12576x4096 is already TP2-local, with ordered role heights
  q/k/v=4096 each, b=32, f_a/g_a=128 each. Both ranks need execution evidence,
  output offsets and per-role comparisons using the production output views and
  static scales. Neither a single 12576-row weight nor 12448/12416 receipts
  establishes that property.
- Existing selected-population validation is not an authoritative owner inventory.
  No complete owner-to-case reconciliation or current v3 report packet was found
  in the inspected checkout documentation.

The construction geometry reference remains
`docs/measurements/construction/2026-10-07/glm53-flash-4layer-selected-t8.json`;
its T8 measurements are not T4 qualification.

## Dependency intake (not a claim of global absence)

The supplied issue titles identify existing owners. Searches of checkout/served
documentation and named entries in the shared measurement store did not locate
completed issue-linked deliveries for the following work. This audit cannot
establish their full execution status or exclude differently named external
artifacts. Existing follow-up tessera#1502 already tracks delivery; do not create
a competing campaign or duplicate follow-up.

| Existing owner | Required consumption before completion |
| --- | --- |
| tessera#1468 | Authoritative owner-derived coverage/reuse manifest: MLP, shared experts, attention, heads, vision, small/tail roles, construction-local TP2 KDA; exact pending populations and scoped refusals. |
| tessera#1470 | Source/loaded-build-bound corrected measurements and matched-byte baselines for uncovered cells, including failed/losing measurements. |
| tessera#1469 | Observation **and execution** native activation-byte receipts bound to the measured input/static-scale contract. |
| tessera#1462 / tessera#1472 | Existing arithmetic owner's accepted disposition and compatibility with the executing build; consume tessera#1213–tessera#1216 outcomes rather than repairing arithmetic here. |
| tessera#1241 | Direct-consumer outcome and source/build compatibility with qualification execution. |
| tessera#1471 | Canonical publication/handoff disposition for only the exact measured, admitted scope, after receipt validation. |

Historical candidates found during intake are not admitted reuse:

- `/mnt/shared/tessera-measurements/issue1018-publication-20261008/revision1/final/delivery.json`
  describes partial historical tables and unchanged publication bytes; it does
  not deliver current WINDOW/L14 dense qualification.
- `experiments/results/t4_activation_quantizers_historical_20261007.json` is a
  pinned historical quantizer attestation, not capture of the present cells at
  both boundaries.
- `/mnt/shared/tessera-measurements/kernels-t4-fp4-mma-attestation-sol-20261007/corrective-integrated-sparky/attestation.json`
  explicitly retains `arithmetic_qualified: false`. Passing finite probes cannot
  be promoted to completed arithmetic-owner approval.
- Historical R768 census/stock reports do not bind new dense cells to actual
  launches, profiles, both-device power or work-J.

These absolute paths identify external intake candidates only; they are not
portable packet receipt references. No SHA-bound receipt is asserted here.

## Completion boundary

After compatible owner deliveries arrive, extend only missing qualification
adapters for ordered module roles, local dimensions and attribution. Retain the
portable packet with embedded producer argv/reports and relative SHA256-bound
receipts. Correctness needs all eight q128–q1024 widths, onehot/random M1 and
multirow cases, actual per-role numerical/output-placement and eager/capture/
replay checks, including both KDA ranks. Body timing needs
M=1,2,4,8,16,64,512,2048,4096,8192; heads use the declared sampled-decode-position
domain, not an invented M8192 full-vocabulary prefill requirement. Record every
measured cell's admission separately from measurement coverage. Preserve failures
and scoped refusals without admitting them. Rank-local qualification is not
proof of distributed serving.

No GPU pilot is authorized or submitted by this intake: existing measurement
ownership must first be reconciled. Any genuinely missing GPU experiment must
first record its decision, why CPU cannot decide it, the smallest targeted pilot,
expected GPU minutes and compatible reuse plan; run that bounded pilot before
sweeps. CPU structural checks cannot supply actual launch/resource, numerical,
activation-byte or energy evidence.

## Verification

Command (x86):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_t4_dense_qualification_evidence.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: **2 failed**, both at `PACKET.is_file()` for missing
`docs/measurements/t4-dense-qualification.json`; no skips, NO CUDA, zero device
allocations. This reproduces the absent deliverable, not a kernel numerical
failure or a defective acceptance test. O1–O5 remain incomplete.

Cheap x86 guards were also run with the same environment/options, selecting
`tests/test_issue_refs.py tests/test_impacted_tests.py`: **212 passed**, no skips,
NO CUDA, zero device allocations. No GPU-arm check was run; the missing portable
packet is independent of arm, and no GPU execution conclusion follows.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"972313d2676a7e7fe9864980781a101c8e836587223adcbb52878e5d9349b1c4","size":1775}},"operation":"dd5088bcf2b12618bf2bf6baf3d668fd349d43cbba953d2568457273db2cf905","receipt":"0f23ff998ee02defbda1a0ed60f205a17f2bc8bbf10d68c97a653de36a792b60"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7c43e20590ffc45b7fb24456c3c642e198bcb34603f1ac2008f51265eec42b13","size":3657}},"operation":"df3d494ae445d16ab7654d0bd162163e155ba511785e1b44b080827039c3de10","receipt":"79ac9667967b6464b61aa84ca84d8c7b74521174bdf91070829650430076316b"}]
