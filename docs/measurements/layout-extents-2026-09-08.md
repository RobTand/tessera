# Payload-free layout pricing — 2026-09-08

`PlaneExtent` and `TerminalExtent` own the same geometry, validation and byte
arithmetic used by the writer's `PlaneDescriptor` and `TerminalRecord`.
The latter add payload digests and retain canonical serialization. The new
`build_plane_extents` and `build_terminal_extent` APIs avoid placeholder
payloads; `calculator.terminal_rate` uses them. No wire recipe, serving gate,
encoder identity or format-menu decision changed.

## Exactness

All 43 rows of the existing byte audit are identical before/after: 23 encodes
(18 shape and five real-weight/Hessian cases), eight layout rows and 12 release
rows. These include legacy/current layouts and shard/release conditions. The
encode-only run does not claim a decode corpus from local external artifacts.
The dedicated extent matrix additionally compares counts, offsets, widths,
alignment and terminal sizes for both layouts, completion, release, row scale,
shards, arity, span and partial refinement. A large geometry is priced with
payload allocation and hashing forbidden. Extents have no serialization or
content-identity API.

The 18 new cases failed before implementation, PB `0e7ee63155c3`: missing
`build_plane_extents`/`build_terminal_extent`, and
`tests/test_layout_extents.py:13: AssertionError: pricing allocated or hashed a payload`.
They passed afterwards (`e9c20bd12bc4`) and in both complete test populations.

## Paired CPU profile

The workload is 1,537 calculator calls at shape `(2048,4096)`, q256 256–1792,
window body L12, cap7, and row scales. It is a calculator measurement, not the
complete PrismaQuant menu, an encoder, model capture or served latency.
PrismaBuild admitted one paired measurement on Sparklina, Python 3.12.3,
affinity `[5,6]`, native threads one, 4 GiB physical reservation. Each arm ran
in a fresh process under cProfile; order was before/after/after/before.

| Arm | Seconds |
| --- | --- |
| Before 1 | 13.5221 |
| After 1 | 10.5140 |
| After 2 | 10.4376 |
| Before 2 | 13.4059 |

Mean time fell from 13.4640 to 10.4758 seconds (22.19%) under this instrument.
The before profiles attribute 2.930/2.924 seconds to 13,833 payload SHA256 calls;
those calls are absent after. All four exact rational-rate outputs have SHA256
`f9047f158813155db6fa3a117d329be2c1eddf97c5cfb03a0662f250e92c27ac`.
The baseline is `01d48d8d1767c972684f81ec0c1ea137da8ca9c8`; measured optimized
source is `25ebe277e` (full identity in the receipt).

Netdata retains 50 points per series on both Sparks for the exact paired
window. Sparklina averaged 5.25% user CPU, 0.59% system CPU, 0.93% I/O wait,
zero swap I/O and 4 W GPU power. Sparky concurrently ran the separately
admitted GLM capture and averaged 46.27 W GPU power. The timed calculator uses
no GPU; no GPU saturation, GPU energy gain or end-to-end speedup is claimed.

## Test populations

| Population | Mode | Result |
| --- | --- | --- |
| DL380, Torch 2.11.0 CPU, 202 selected files | PB fanout: 20 shards × two xdist workers, worksteal, native threads one, 6 GiB/shard | Initial full run: 3,548 passed, three failed, 583 skipped; zero missing modules |
| CPU corrections | Two PB shards, two workers each | Seven collection tests and three issue-reference tests passed |
| Sparklina, stock vLLM image, Torch 2.13.0+cu130, GB10 | Ten preferred CPU cores, worksteal, strict CUDA, native threads one | 4,171 passed, eight skipped, one expected failure; zero missing modules |

The CPU failures were two nested tests inheriting the launcher's
`PYTEST_ADDOPTS=--dist=worksteal` while explicitly disabling xdist, and the new
issue reference missing from the offline snapshot. The launcher now passes
options only to the parent invocation; the issue snapshot is refreshed.
Only the affected files were repeated. A pristine baseline run of those files
was also submitted (`6ce5cb1e22c4`); its outcome is retained with the evidence.
CPU skips retain the device-dependent surface, missing reference roots and
optional tools; their full verbatim histogram is in the audit. CPU results do
not claim CUDA coverage.

The complete CUDA population exercised 508 tests that allocated on device and
had no missing-artifact skips. Its eight skips, verbatim:

- 2: `e2m1-tcq-lut-release does not cut 4 ways along columns`
- 2: `e2m1-tcq-lut-release does not cut 8 ways along columns`
- 2: `needs two CUDA devices`
- 1: `E2M1 publishes no reader range`
- 1: `this checkout IS shared, so the refusal cannot fire here`

The vLLM container ran directly under Rob's vLLM exemption. Its immutable image
is `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
It used the previously qualified git/header/user bootstrap and scoped pytest
8.4.2/xdist 3.8.0. The complete frozen source is
`af2c20360fcdc5245a2f18a2602ca49adc86faff`; all ten workers and an independent
1,141-file byte/mode audit agree on effective source
`a75f6e425876c7828f1b5424c8ead61c3a2b3089b2ab2142ae7a4e03b756a14c`.
Exit was zero, the surface publication hash matches the log, and the owned
container was removed. This does not establish multi-device coverage or a new
served quality/performance result.

## Evidence and retained attempts

All paths below are relative to
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/`.

- `extents-root-audit-01.json`: canonical CAS receipts/payloads, complete source
  snapshot comparisons, byte artifacts and paired profile/Netdata hashes.
- `extents-suite-root-audit-01.json`: all 20 CPU shard outcomes, two corrections,
  actual skip histograms, native source verification and artifact hashes.
- `extents-byte-before-01.json`, `extents-byte-after-01.json`: byte matrices;
  PB actions `79f6d19f5c38`, `50efe85a6bda`; official diff `8b690a38d295`.
- `extents-calculator-profile-02/`: four profiles, exact rates, both-host raw
  Netdata and result; PB `63b55f9e8edb`.
- `extents-vllm-native-01/`: exact invocation, exit, log, container ID, complete
  population and ten worker shares. PrismaQuant reference is frozen at `42e9f19a58`.

Failed setup attempts remain evidence: an external test launcher was refused
before submission and moved into the source closure; the first profile omitted
baseline version metadata (`6d6be424455a`), and a follow-up refused its existing
output directory (`ff77947d6410`). No timing was produced by either. The qualified
run uses a new directory and archives `pyproject.toml` with the baseline.
The first native PB attempt (`e7eeab2e2cc6`) had 4,115 passes, one issue-reference
failure and 19 skips; strict CUDA refused missing reference roots. Its original
surface and log are retained. The qualified container supplies those roots and
vLLM dependencies rather than weakening the check.

The observed PB test-fanout gap is filed as RobTand/prismabuild issue 387:
`pbtest` cannot declare GPU demand or forward population options. PB continues
to own CPU file partitioning; no second scheduler was introduced.
