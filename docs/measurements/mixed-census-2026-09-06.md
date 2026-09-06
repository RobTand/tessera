# Mixed dense/routed census gate qualification

PrismaQuant [#253](https://github.com/RobTand/prismaquant/issues/253),
base `6faa5ce314cadeee8a190cbeadcf6cde3a333efb`, 2026-09-06.
This is CPU gate qualification on synthetic receipts, not a model run,
runtime attestation, wire audit, performance measurement or quality result.

The gate now joins planned dense weight tensors to their fused owners and
checks them alongside complete routed expert stacks. The synthetic full-mix
fixture has 22 E4M3/q1024 stacks containing 2,112 expert projections,
six E2M1x2/q896 matrices in four dense owners, and 60 BF16-grid/q1792
matrices in 48 dense owners: 74 owners and 2,178 matrices in total.
Plain `"BF16"` and `"PASSTHROUGH"` plan entries remain passthrough obligations.

The existing exporter fusion rule was moved unchanged into the lightweight
`tessera.serving.dense_ownership` module; the exporter retains its public
wrapper. Reusing only role labels was insufficient: swapping equal-shaped
source roles between owners or splitting one fusion into two owners must
also refuse. Per-structure launch and cell checks preserve unattested
populations. A synthetic routed-only attestation never promotes dense owners.
No runtime cells were added. The existing `export_identity.options.plan`
binding is still required, including for a direct export given a truthful seal.

All execution used PrismaBuild. Exact commands, demands, source snapshots,
terminal paths, CAS result digests, and verbatim surface/skip lines are in
[qualification.json](data/mixed-census-2026-09-06/qualification.json).
Terminal exit status and published CAS payload bytes were checked independently.

| Run | PB action prefix | Result |
|---|---|---|
| Before mixed support | `07587ee990c9` | 13 failed at the routed-only restriction; rc 1 |
| Before dense-owner binding | `fffc98ce16e2` | 2 failed with `DID NOT RAISE ValueError`; rc 1 |
| Final focused census tests | `23c3226b99ed` | 77 passed, 0 skipped, 0 uncollected modules; rc 0 |
| Selected 48-file suite | `17f45352438f` | 1,137 passed, 2 failed, 72 skipped; rc 1 |
| Only failed files on pristine base | `70beb82a5d1e` | Same 2 failures, 15 passed, 3 skipped; rc 1 |
| Python compile checks | `12833df530be` | rc 0 |

The focused run used four xdist workers with `--dist worksteal`; the selected
suite used 16. Both ran on dl380g10 with Python 3.14.4 and CPU PyTorch 2.11.0.
Native thread counts were bounded to one per worker. Neither exercised CUDA.
The first red run used the torch-free interpreter on Sparky and reported 70
uncollected third-party-dependent modules elsewhere in the repository; all
13 selected mixed-census cases were collected and failed at the intended gate.

The two selected-suite failures are unchanged-base translator failures:
`test_fused_member_rungs.py` pins an obsolete literal import spelling, and
`test_menu_selection_requirement.py` lacks the newly required model config
fixture. Both reproduce on pristine `6faa5ce` and were already corrected
upstream by [Tessera PR #371](https://github.com/RobTand/tessera/pull/371),
commits `96763b4` and `5c2e4fd`. This branch retains the requested base and
does not duplicate those fixes.

The selected-suite skips include encoder/CUDA tests, absent model and served
artifact roots, and the unsupported E2M1 reader range; their full reasons are
retained in the qualification record. The CPU result does not qualify those
surfaces. Actual mixed-model runtime census and quality measurements belong
to the separate campaign.

## Delivery rebase, 2026-09-06

The implementation was subsequently rebased onto master
`a9eb572e1b90b17f716562192910681e65430fba` for
[Tessera #372](https://github.com/RobTand/tessera/issues/372).
The only conflict was the architecture provenance header; both the upstream
dense translator entry and the mixed census entry were retained. Executable
changes applied without conflicts. The original `6faa5ce` evidence above is
preserved as measured history; the serving campaign's frozen source is separate.

PB action `82de4959c03543878108a421bd6fac3c2705b9e93b3ddb8037337445a836e626`
tested rebased commit `bf59f624ca597a587bcc1647d3d6f125bc65b205`:
**94 passed, 3 skipped, 0 uncollected modules, exit 0** in 42.15 seconds.
It covered the three focused census files and the two previously failing
translator files, whose failures are fixed in the new base. It ran on dl380g10,
Python 3.14.4 and PyTorch 2.11.0+cpu, with eight xdist workers using
`--dist worksteal` and native threads bounded to one. The three skips were
verbatim `needs a CUDA device`; no tests allocated on CUDA. Actual terminal
status and CAS payload size/hash were verified; the complete command and
receipt identities are appended to the qualification record.

The coordinator owns the full integration suite on the eventual merge result.
This prerequisite does not close PrismaQuant #253 or claim actual mixed-model
runtime, performance or quality qualification.


## Coordinator integration result

Both full populations ran once through PrismaBuild on the same effective source
`8cebd5e`, using eight GPU-visible workers on GB10 and sixteen CPU workers on x86,
with native/build threads bounded to one. GPU: 3,587 passed, one failed,
25 skipped, one expected failure; 486 tests allocated on the device. CPU:
3,061 passed, one failed, 549 skipped. Neither population had uncollected
modules. The ledger retains both original nonzero exits and exact source
agreement; the missing-artifact, absent-vLLM and two-device skips are not
claimed as covered by this run.

Both failures were the same documentation check: the offline tracker snapshot
did not yet include the two new references in this measurement document.
Commit `7018fa2` refreshed that snapshot from GitHub. The affected check then
passed all three tests, zero skips/uncollected, in PB action
`3936ac3d43d44a7168464b8fa14faa7e125fcf6f540e5fb6ef898e058c7f3542`.
No executable code changed for that correction, so the full populations were
not repeated. Original failures remain failed records; this is their explicit
disposition, not a rewritten green suite.

Exact actions, measured snapshots, output hashes and the correction are in
[data/mixed-census-2026-09-06/integration.json](data/mixed-census-2026-09-06/integration.json).
The full joined receipt and per-worker populations are under
`/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/census-integration-373`.
