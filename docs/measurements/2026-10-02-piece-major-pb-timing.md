# Piece-major timing resumed through PrismaBuild (#739 / PR 868)

## Reused acceptance and execution cutover

The earlier numeric V4 receipt remains accepted, not rerun: SHA-256
`0e35d25f968b7683e069d5ebc47e4ee4ed36af062b70e62ad92e59b608fba4ec`,
with bitwise mode 0/1/2 and final outputs at M1/M2048. The 53 timing-control
passes and exact preflight `a4109cd6056fc812a314d24528f9a9d38d57a752d9af6eaba097817217b568eb`
are also reused. These do not establish served speed, graph or energy gains.

Current policy supersedes the handover's historical benchmark exemption:
**batch benchmarks use PrismaBuild even when they import vLLM custom ops**.
The retained direct-held-FD mode is an input/containment implementation, not
permission to run outside PB. `piece_major_pb_action.py` imports only the
qualified `run_direct_arm` and `owned_cleanup`; the old paired main is not run.
The controller lives in the repository so PB snapshots its executable bytes.
An initial external-helper submission was refused before admission and produced
no execution. There is no second dispatcher, native rebuild or numeric rerun.

The controller commit is `88ab364a708507072524d151050c98c9a2f58067`.
The original sealed benchmark/harness at `c8a776c0e0b33f32b6bf570a49c73a25caf7df48`
is unchanged by that commit. Both layouts still use ELF
`4d693c2621333eb36b32b21479e0283e95236e8864923fad9d1f15bc1487091d`.
The frozen source, captured IDs, seeded BF16 activations, uniform routing
weights and exact original benchmark argv/environment/native bank are retained.

Artifacts live under
`/mnt/shared/astra-resume-20261002/t8_performance/piece-major-common-c236b7aa/`:

- `PB-TIMING-PACKET-v2.json`, SHA-256
  `c2f39ccec98f1946e52ba8975dd648758c65736e589d8706ee1a675ca4f53a49`;
- original `DIRECT-TIMING-WINDOW-v1.json`, unchanged SHA-256
  `5d23bf912d74c6643b38a945dce139879ab9df963873439c4f2f84e73828b7c7`;
- explicit coordinator authorization `PB-TIMING-GO-v2.json`;
- controller, preparation, guard, terminal and both-host telemetry evidence in
  `pb-timing-v1-99cd5da71437/`, with `timing` linking to the original output
  `timing-v1-99cd5da71437/` rather than repeating the measurement.

CPU preparation action
`56b7cf1dac090fedb50f3f03d934e417b065b8c4bed758697140b4da3014561f`
finished on DL380G10, exit 0, 0.74 seconds. No GPU was visible and no benchmark
or test population was run. Actual stdout and full published-generation
`PrismaBuildCAS.lookup` verified the result; local claim
`21cf15c5970f01dbe911b8515234f10865ff7c68f132ca0f0e183d64bdf0fa39`
binds payload `d1551d125d11b60dea7bf0130bd1f49a336b9a7d258ce57be7c43fc79f20e99e`.

## Authorized finite measurement population

Coordinator GO authorizes exactly one PB measurement/exclusive/here attempt
on Sparky: CPU 2, total UMA budget 16 GiB, GPU subset estimate 8 GiB,
native threads 1, no retries. The existing container keeps its 16 GiB hard
limit and no extra swap. The existing owner bounds execution to 600 seconds,
with wrapper TERM/15-second kill grace; PB timeout is 900 seconds including
post-run read-only telemetry collection. The launch requires 40 GiB available,
and its active guard requires 24 GiB available and full PSI avg10 below 20.
Both Sparks were quiet before submission; no artificial competing load was
created. At 03:10:46 UTC, MemAvailable was 118,980,087,808 bytes on Sparky and
124,059,451,392 on Sparklina; full PSI avg10 was zero on both. There were no
Docker containers or GPU compute processes. PB's complete queue census was empty.

Measurement action
`f10b6eaa4e0f92bb230302c8f8ea3d2c08645a3d26efe7104093d5a610e30a17`
was published at Unix 1790997053.6140316 and claimed on Sparky at
1790997097.9992228. Its source snapshot is
`9616b5f9c994b91667136dd1176226bf134ce601`, parent the controller commit above.
The finite population is legacy/PM/PM/legacy at each M in [1, 2048], eight
cells total; each has 10 warmups, 30 CUDA-event observations, five profiler
calls with a raw Torch trace, and a 30-second power/clock/temperature series.
CUDA graphs are disabled. Scope is the real A8SE L10 TP2 rank-0 operator,
not full-model throughput or served activation replay. Energy/work-per-joule
remains HOLD pending independent clock, coverage and instrument agreement.


## Terminal result and CPU-only PB evidence analysis

The single measurement finished with program and outer PB exit 0 in 279.934
seconds (finished Unix 1790997378.9244442). No retry, repeated qualification or
follow-up GPU action was launched. The qualified owner verified exact container
`8e0c2c90a1124e8428bc00caa5f7c04a345a22d7ca8dd11e9645aba6d19c16eb`
already absent after cleanup, with token `99cd5da71437b7926ee5965778e4a9c1`.
The attempt preserved its assigned CPU affinity [5, 6].

CPU-only PB evidence analysis action
`e728845b7495184a468f1c56b9d4d5d47d72dbbd939c010918e8a8258f2bfcfe`
finished on DL380G10 with exit 0 in 1.00 second, no visible GPU. It verified
all eight cells, 30 finite positive raw CUDA events each, exact conventional
medians, eight raw trace hashes and five calls for each routed native kernel
in each trace, 30-second raw power/clock series, accepted input hashes, nine
unchanged held original file identities and one unchanged native bank.
The bound successful benchmark checks every cell's final output bits against
the accepted numeric proof before timing; intermediate numeric qualification
was deliberately not repeated. Both arms retained 1,872,827,160 resident bytes.

| M | Legacy cell medians (ms) | PM cell medians (ms) | Mean-of-cell-medians legacy / PM (ms) | Observed operator latency delta | Observed operator calls/s ratio |
|---|---|---|---|---|---|
| 1 | 0.506304, 0.556048 | 0.497168, 0.498336 | 0.531176 / 0.497752 | -6.29% | 1.06715x |
| 2048 | 13.780048, 13.890848 | 12.509808, 12.385024 | 13.835448 / 12.447416 | -10.03% | 1.11151x |

These are observations of this bounded ABBA operator window, not a statistical
population, clock-controlled promotion or served speedup. M1's legacy arms
show a 9.83% first-to-last median increase; the 6.29% mean contrast must not
hide that drift. M2048's legacy first-to-last drift is 0.80%. There is no
served activation replay, full-model throughput, graph or quality claim.

Before/after in-process profiler self-device times, preserving both control
and both candidate cells, are:

| M | Legacy us/call | PM us/call | Mean self-device-time delta |
|---|---|---|---|
| 1 | 395.5110, 395.7124 | 335.0932, 334.4342 | -15.38% |
| 2048 | 13640.8354, 13774.4708 | 12320.4646, 12239.8368 | -10.41% |

Raw traces retain full kernel names and per-call counts; the profiler is not
substituted for raw CUDA-event operator timing. The event and profile contrasts
are directionally consistent, but host/launch overhead and thermal/clock drift
are visible rather than normalized away.

## Host, clock and energy disposition

All eight steady-power cell windows and the action window retain both-Spark
power, CPU, memory and swap-I/O raw HTTP Netdata responses, queries, native
collection cadence and accepted/rejected bucket intervals. Supplemental MCP
provider responses retain action-wide disk I/O and SM clocks on both Sparks.
No GPU-utilization percentage is interpreted as saturation.

Typical per-cell Netdata accepted coverage is 24 seconds of the 31-second
integer-bounded request (three 8-second groups); straddling groups are rejected.
Action-wide coverage is 270 of 281 requested seconds (27 10-second groups).
The raw responses expose these gaps: successful collection is not complete
coverage. Event-phase SM clocks and the 10 Hz in-process clocks vary by cell;
M1 power-loop clocks range 2444-2535 MHz, and M2048 clocks range 2190-2431 MHz.
Temperature spans 46-84 C across the measured steady loops. Sparklina's
supplemental SM-clock response stays at 2145 MHz, unlike Sparky's varying clocks.
There is no clock-controlled A/B or independently qualified cross-instrument
agreement. **Energy and work-per-joule remain HOLD.** No loop power quotient
is promoted to an energy result.

The pressure guard retained 279 samples, minimum MemAvailable
112,758,484,992 bytes and maximum full memory PSI avg10 8.51, inside the approved
24 GiB / <20 bounds. PB's cgroup scope peak was 5,374,832,640 bytes, with
3,963,170,816 current bytes in its terminal observation; process I/O covered
nine observed processes, 4,180,303,872 read bytes and 9,052,160 write bytes.
Box-level UVM residual peak was 7,873,806,336 bytes. These are different accounting
domains: none alone is an attributed GPU allocation peak or proof of a leak.
The root-owned reboot occurred only after terminal/cleanup/evidence preservation;
no native bank, result or historical evidence was removed.

## Exact evidence and receipts

- Benchmark receipt `timing-v1-99cd5da71437/bench_t8r.json`:
  `3ce7df07c5acc2646174f8296c401a1e2e119ddd0dba736249d4404290403fe2`.
- Both-host raw collector `pb-timing-v1-99cd5da71437/netdata.json`:
  `586d00f298bbc7a9983b2265f3293f586022cacbb275c4d4d5c5741a7303410c`.
- Owner CPU-only PB evidence analysis `pb-timing-v1-99cd5da71437/TIMING-REVIEW.json`:
  `499374194aec6d04d70ebf38395630dc8548d6d7a25425d309af30a1ed6a577e`.
- Measurement local CAS claim:
  `df67ce906da9468acd4379f5557d306fc1473511ad9df10c813b9680b945daee`,
  payload `5d2d40806a8087180e66bdebbc6ac846a17ddfcd823c1273b94bb29dba65a154`.
- CPU-assessment local CAS claim:
  `568897d9fbc09a00e6cf2933becbf0cb9f022bf3141f3bdc7522989611e4bfcc`,
  payload `87d247c966ed4b1bafa5aac80b58f67ad778d3f0f314008713cb4d70fa51ef4c`.
- `pb-timing-v1-99cd5da71437/CAS-VERIFIED.json` retains actual receipts,
  resource observations and supplemental Netdata hashes. All three actions'
  actual logs, claim/payload checks and full published-generation
  `PrismaBuildCAS.lookup` were verified. A submission acknowledgement was not
  treated as completion.

The source-sealed original harness is intentionally preserved, including its
historical direct-mode terminology; current execution policy is the PB-only
contract above. Root owns independent critical QA, draft PR acceptance and merge.
No merged/default-on or deployment claim is made by this record.


## Independent reviewer disposition

The owner-run `e728845b...` action above is CPU-only PB evidence analysis, not
an independent reviewer. A separate performance reviewer rederived the primary
receipt hashes, all CUDA events and raw traces in PB action
`934597c6b6b74c89f6383dc2f174075223d44a340c757e393a08dbfdcdfc2976`
(DL380G10, exit 0). The reviewer then verified the measurement, owner analysis
and reviewer action through full published SDK CAS lookup in action
`24c72e6ca2a460364f7a62b4242f6f8c549b14c8aae83b7db5981f1e792166ea`
(DL380G10, exit 0).

Reviewer evidence:
`/mnt/shared/astra-resume-20261002/t8_performance/INDEPENDENT-PM-ANALYSIS.json`,
SHA-256 `47e874a8874639f7f5ca6e13f365de036426233bf1e1763c212ebfb04b98ced9`;
CAS evidence `INDEPENDENT-PM-CAS-VERIFIED.json`, SHA-256
`ea811c4a451ad400768761d0554921784b92ae2b199f5d61a7adf7d5057608fe`.

The reviewer accepts sealed numeric/identity evidence and the scoped
single-window operator observations. It does not accept stable, served or
clock-controlled speedup, energy results or default promotion. M1's early/late
paired latency reductions are 1.804% / 10.379%, versus 9.825% legacy drift;
M2048 paired reductions are 9.218% / 10.840%. Native mode0/mode2 profile savings
corroborate a layout-specific kernel effect, but warming/DVFS through 84 C is
still a confound. The reviewer identifies mode0 gate/up (7.14-7.18 ms, roughly
58% of PM self-device time), then down (4.14-4.16 ms), as the next measured
performance priorities. This is prioritization, not authorization for another
GPU run. Energy remains HOLD; defaults remain off.


## Next controlled M2048 preregistration (no GPU authority)

`docs/measurements/2026-10-02-piece-major-controlled-latency-protocol.json`
contains a complete conditional experimental design, SHA-256
`556859fb5552b2a26a449a6f6cf1af25ee6b289358f29ed60eac56d61bb2caee`.
It is explicitly **not a runnable or qualified action and not GPU GO**.
The independent reviewer accepts the design conditionally, not execution.
The current eight-cell harness is preserved; it does not accept this M2048-only
latency phase, and narrow versioned adaptation plus CPU qualification are launch
prerequisites rather than falsely inherited from the old controls.

The causal question is reported-SM-matched, thermally equilibrated M2048 layout
latency independent of arm order. Three balanced ABBA/BAAB block pairs produce
six independently conditioned processes, 24 cells and 720 raw CUDA events.
The second pair reverses block order. Each cell retains the original 10 warmups
and 30 observations. Original numeric V4, native bank and eight Torch profiles
are reused; no per-cell 30-second power heaters, numeric rerun or duplicate
five-call profiler/NCU population is included.

Proposed experimental predicates are predeclared, not historical facts:
preceding 30-second valid sensor window, all reported-SM samples within 1% of
a frozen reference, temperature span <=2 C, first/last10s temperature median
difference <=1 C; scored cells maintain that bound, matched arm mean clocks
within 1% and temperature medians within 2 C, unchanged throttle reasons and
no newly asserted thermal throttle. Unscored alternating-arm conditioning is
capped at 180 seconds per block; failure is retained, not retried or excluded.
No ad-hoc NTP, GPU-clock, driver or power-policy changes are authorized.

The reference rule uses the pooled median from the first qualifying unscored
30-second window before any scoring. The first ABBA block commits the frozen
reference through PB's produced-output contract; subsequent five actions require
supported `--after PRODUCER:TEMPLATE_ID` and its manifest-bound identity. They
never poll an unsealed shared file, mint their own target or add a scheduler.
All blocks keep the same reference/device/driver/source/input/native identity.
Fresh processes do not erase common physical-device thermal history.

Native sensor cadence and delivery lag must be qualified against the actual
subsecond scored cells (historically about 0.4-0.6 seconds), not merely the
conditioning window. The design's proposed sampling-resolution objective is
at least ten genuine native timestamped observations per cell, with maximum
native update interval plus delivery lag <=one tenth of the shortest cell and
native history bracketing both boundaries. Polled duplicates/interpolated 1s
series or before/after NVML values cannot satisfy that proof. These predicates
support sampled *reported* SM matching only; they do not establish effective
cycle frequency or exclude all between-sample DVFS changes. An unavailable
qualified instrument is an explicit readiness failure, not a fallback claim.

CUDA events/same-host monotonic time are the primary latency domains. PB#1440
postboot diagnosis still has gateway-only NTP192.168.1.1/reference481E2358 and
no trusted independent fleet-clock alignment; energy remains HOLD. The design
keeps cross-host telemetry diagnostic, recording true source cadence, clock
mapping and accepted/rejected coverage rather than inventing energy precision.
All block/pair contrasts and ABBA-minus-BAAB interactions will be reported.
Three pairs are minimum descriptive variance/order controls, not guaranteed95%
significance. No pooled-event bootstrap or outcome-driven extension is allowed.

### Exact separate counter question and capability evidence

A separate, unauthorized counter question asks why PM M2048 mode0 gate/up still
accounts for about58% of device time. It is not bundled into latency scoring;
profiler-instrumented durations never enter the CUDA-event statistics.

CPU-only PB preparation
`344d23228906e59db6353b18889254e466613e2f7e0d4d801c20e27722eb1056`
finished on DL380G10, exit0, 2.37 seconds. It decoded the existing CAS archive
of historical sm121 action `7ae4bc6c8ce2312a85636e33ed39658953a88fcb640f25df2e317bb091fabb2f`,
without any kernel/profile rerun. Actual named metrics were recovered from its
wide raw CSV, SHA `eccb437abf39fcdbd3ad700d1c78d37ac250902359d7b5ea4414ec43fe574f6a`.
Inventory `pb-timing-v1-99cd5da71437/RETAINED-SM121-COUNTER-NAMES.json` has SHA
`d4944d0c20635f449aedfc562a261fa61e1cdab65acfeaec3d4691f70a5af01a`;
preparation receipt `d79d8fa1ad2636b3dcbcadf61515cf3b5c60dd15844b0bf878397c201048a59d`
was checked with actual logs/payload and full published SDK CAS lookup.
This is capability evidence from distinct `0f953b69...` ELF, not PM performance.
The old and PM populations share GB10/sm121 and driver595.91.07, but exact
current metric/tool support remains a prelaunch check.

The sealed metric names include L2 read/write sectors, sysmem-sector proxies,
tensor activity, achieved occupancy, executed instructions and barrier/long-
scoreboard/short-scoreboard/MIO/wait attribution. **No `dram__` transaction
metric occurs in the retained raw inventory.** L2/sysmem proxies are not renamed
DRAM counters; true supported DRAM evidence is a prerequisite for such a claim.
The exact selected names are in the JSON protocol, not guessed aliases.
NCU logical launch/replay/cache/pass identity must be recorded separately under
its own exact GO; one logical launch may entail multiple physical tool replays.

Initial CPU preparations `b961c174...` (missing published SDK on PYTHONPATH)
and `c5ffea08...` (expected long-form metric header, actual CSV is wide) both
failed before producing capability evidence. Their actual negatives are retained;
explicit SDK injection and wide-header inspection produced the valid result above.
These are CPU inspection repairs, not GPU measurement retries.

CPU static design check
`1ecdb9d486759d1cf40a0bc9bc75c8e48409508ba90606ac2545430f9d56aae8`
finished on DL380G10, exit0, 0.94 seconds; full SDK CAS receipt
`bb332a87fee4402205388767c6392c96695aeed054e6389f987c927e73294f7a` verified.
It checked the original design's six balanced orders/720 observations, exact
M2048 seed1605385075/accepted input and final-bit hashes, actual selected metric
membership and absent DRAM names. Its output
`pb-timing-v1-99cd5da71437/CONTROLLED-LATENCY-PLAN-CPU-CHECKED.json` has SHA
`43dbd3ac2e14a39d03a3490b6527fec2c5da0a1dab2f36f1650b9bf6ca3f2569`.
That check binds the preserved v1 design SHA
`9053ff5be285c7a329c088408c838b128b59be93378ce493d5dcad3bba2e8598`;
subsequent reviewer sealing constraints add PB reference dependencies and explicit
subsecond resolution to v2 above. It is not new harness/sensor qualification.
No controlled GPU submission has occurred; the parent issue remains open,
the PR remains draft, and PM remains default off.


## Superseding actionable descriptive repeatability adaptation

Root explicitly separated useful repeated same-host A/B evidence from unsupported
perfect clock-control requirements. The latest JSON design now has SHA
`a4789d8f4e19375acb1af11e7ef07bd7b95f30cabf824d6f592c1f83abccecdb`
and schema `tessera.pm_m2048_descriptive_repeatability_design.v1`.
The old conditional clock-matched v1/v2 designs remain immutable external
history, not readiness gates for this descriptive experiment. No 1% frequency,
device-native sensor-lag or plateau claim is imposed or fabricated.

The existing benchmark/protocol/wrapper now implements a narrowly versioned
`repeatability` phase: only M2048, fixed60s unscored alternating-arm conditioning,
three balanced ABBA/BAAB pairs, four cells/block, 10 warmups and30 fresh events
per cell. Six independently started finite PB blocks produce720 raw events.
All events remain in execution order, including outliers; all available polled
power/SM/temperature/throttle readings/errors remain. No retrospective exclusion,
adaptive conditioning, clock mutation or result-driven retry is allowed.
Temperature/order/DVFS uncertainty is reported rather than normalized away.

The accepted numericV4 input/final-bit proof is bound and checked before scoring;
original source/native/readset semantics remain unchanged. The eight accepted
Torch profiles are content-bound mechanism evidence for the same binary and
inputs, explicitly reused rather than represented as fresh-window profiles.
No duplicate numeric, Torch or NCU population or per-cell30s heater is added.
Fresh both-host Netdata retains action, conditioning and every event window,
including raw diskIO/SM/temperature/pressure responses and per-series errors.
Subsecond windows may have no native Netdata group; that is unobserved coverage,
not a fabricated high-rate reading. Energy and general/served speed remain HOLD.

The new exact phase's causal CPU control was PB
`5090c5b00a025ac784fe3ef685dd379257f401ce26f2e0029f725eaca0fcac58`:
9 failures /11 passes on the old owner (unknown new protocol/options/profile
binding and rejected wrapper phase). The implemented owner qualified PB
`79a9ac1d45aa812a0b4633dd61157616404dd231e676cdf6ccabdfae1b97d8ed`:
31 passes, zero skips/uncollected and zero CUDA-allocated tests onDL380G10,
CPU2/native1, xdist worksteal, 6.88s outer/4.69s pytest. This is narrow CPU
control evidence, not GPU or CUDA-surface qualification; the accepted prior53
controls/numeric qualification were not redundantly rerun. Actual logs, payload
and full published SDK CAS lookup verified receipt
`52bda9f93a9f60d9832509f1f98975d28ed47863f17ff14a0734050e12683e82`.

Existing containment remains600s with15s wrapper grace/PB900, CPU2/UMA16GiB/
GPUsubset8GiB/native1 and exact owned-CID cleanup. No new scheduler is added;
PB's existing campaign interface owns six scientific block actions. No clock
reference dependency is needed because these blocks make no clock-matching
claim. Exact source-bound per-block packets and real-input CPU preflight precede
independent review and explicit rootGPU GO. Nothing here authorizes a new GPU
submission, deployment/default/pin/cell change or automatic issue739 closure.


## Reviewer-found boundary-observer retention repair

The independent reviewer accepted the executable descriptive design, not
clock/thermal-equilibrium/served/energy/default qualification, and found that
initial/final NVML query failures could discard already captured event arrays.
The observation owner now uses one guarded recording path for boundary and
background queries. Query failures remain timestamped diagnostic errors;
completed event arrays are returned unchanged, not suppressed or rerun. The
controller also labels a failed block whose benchmark receipt was never published:
partial in-process events were not retained. A failure does not authorize retry.

Actual causal CPU action
`eb0ad81ecd2104e0f960e26ff2de7df460b9a22c1fae35c1d9d1e32d75b0228f`
failed both initial/final-query cases on the old owner. The repaired observation,
thread cleanup and conditioning controls passed action
`94bc0f7eb17344828fdd0c70b045e3e31c83d616c96125ee0dc6f3ffbb32e519`:
5 passes, zero skips/uncollected/CUDA allocations, CPU2/native1/xdist on healthy
Sparklina, 2.61s outer/1.76s pytest. Actual logs, claim/payload and full published
SDK CAS lookup were verified. This narrowly checks the changed error path,
not a repeated31-case suite or numeric/profile/GPU population.

Sparky and DL root filesystems fell below the standing5% free-space floor;
their durable maintenance holds remain operative. SharedNAS source/evidence
storage remains healthy (~44.25% free), so this reachable source repair was
made there and CPU validation submitted from healthySparklina using an existing
configured NVIDIA Sync identity with strict known-host checking, no vault-agent
retry, credential copying, administrator bypass or deletion. The earlier8e1/c073
bundle is retained unlaunched and superseded by new source-bound packets; it must
not be launched against changed helper bytes. No new GPU authority is implied.


## Exact normal-nonzero terminal classification

A second reviewer check found that the explicit measurement-retention label
covered raised exceptions but not an ordinary nonzero benchmark return code.
The classifier now runs in `finally` before terminal publication for all normal
and exceptional endings. It states receipt-present versus unpublished partial
in-process data without converting a failed return into success or retry.

Actual controller-main causal CPU action
`456e29cb21213931cb6b115bbf74fe7341142ccbadfdd97ee7bad53be223d575`
was1 failed/1 passed: ordinary return1 omitted the required retention label,
while the exception case had it. After the repair, exact main-entry controls
passed `07efd0a030a9ebc62585aead82fc91781068cd65384e5a9e13009a90e8028dc3`:
2 passes, zero skips/uncollected/CUDA allocations on healthySparklina,
CPU2/native1/xdist, 2.50s outer/1.70s pytest. Owner calls were inert CPU fixtures;
this is terminal-contract evidence, not actual CUDA/benchmark qualification.
Actual logs, claim/payload and full published SDK CAS lookup were verified.
No prior31/5-case suite, numeric/profile or GPU population was repeated.
Source-corrected packets supersede the unlaunched beeb bundle; exact new hashes,
independent review and rootGO remain required before any six-block submission.


## Six exact descriptive blocks: actual terminal and containment evidence

The reviewed final executable source is `0a3ddc70da489264446c626e0928f4bd05b7997e`.
The ready bundle SHA is
`e2e6005d5584a4ac2b035daa7c60bf5a6e787322c800688819f5f1e8194f1161`;
it supersedes the retained unlaunched source-corrected packets above. The real
per-block Root GO records are bound in `AUTHORIZED-CAMPAIGN.json`, SHA
`c709cd4fcc8d704866751cd4c83a9cb5fd91a3ed39b20059f0fcfac71b28c06d`.
Root's conditional GO was exercised only after supported disk recovery, a fresh
Sparky root-free floor of at least5%, owned holds OPEN, fresh memory/PSI fit,
quiet both Sparks and the Original CUDA campaign's actual terminal cleanup.
No source/kernel/native/default/clock mutation or diagnostic replay was added.

The six existing-PB campaign actions each completed in one attempt,
program/PB return0 on Sparky, with a retained benchmark receipt:

| Pair/block | Action key | Canonical CAS receipt SHA |
|---|---|---|
| 1/1 | `69c48d7605636730070b23707563b5933a21a8845100d840e0ab660f46238b55` | `87212035021be46814dbc8fa91afdb2273666a7634d33f584715aca645bfbe73` |
| 1/2 | `f514036155b8fa0b0555a3efaa3661185686af4fa4d9029af496473a2faf37aa` | `6523240bdcae1239389c9e9564f7522fe7c479ca180caf069f0f42662aee359b` |
| 2/1 | `8366197f6088017b2876bbd127aef5ee736988413204716da11eb6e50c1f83b3` | `a02dfc88c59fd8990e93f49857cd609e63d6c9d546da1c668f42b24f179b2d52` |
| 2/2 | `893ad64f99b0ddc32eae3c5f3fc811090fe76123af65b5aefa60a1e3b2817c4c` | `5cced0ef0ea41a052638f31d6fe9bc86e7e3ae970c7531555e7ac6cf78d5cf53` |
| 3/1 | `0a7534583346f4aa765698844752b675b80fb81566a1f26756cf65ccafcc1b1a` | `0e24736244ceb54a2157508fe30e36bac4ffa83ec8e6a8a2ae63e299602a003e` |
| 3/2 | `e70a2b9745ef7fc8ef910b039e1f7f5e58092c8960f1b3d2ab6244f222441a68` | `5d5f56842719ad8e0bc8f38c280e38d0700b3fb1858ca9cfe74a287c2e3903a0` |

The immutable evidence directory is
`/mnt/shared/astra-resume-20261002/t8_performance/piece-major-common-c236b7aa/repeatability-ready-0a3ddc70/`.
`SIX-ACTUAL-CAS-VERIFIED.json`, SHA
`7b0e5e8c1f1c31e153db096527045054bb60d3a7e4f3d49882aa34beb72cf7be`,
preserves all six actual action records, full published-generation CAS lookups,
and local-claim/payload hash checks. The full lookup verifies the receipt's
attestation; a hash-only claim check is not substituted for that verification.
The attempts used unchanged published generation `d028dfee920b-1790960385-1d815cff72d1`,
not the subsequently selected c437 SDK4 maintenance generation. SDK3 suffices
for this held-original-FD manifest-reader transport; no SDK4 injected lease is claimed.

PB actually executed blocks in chronological order1/1,3/1,1/2,3/2,2/1,2/2.
The predefined pair identities remain intact, but pairs1 and3 are nonadjacent.
Fresh contained processes and fixed60s conditioning do not erase that shared
physical device's intervening thermal history or establish independent clocks.
Do not relabel this as three adjacent paired windows.

All six exact owned CIDs were already absent; all six PB process-live counts
were zero, and their exact broker cgroup directories were absent. The complete
queue showed no remaining owned READY/CLAIMED row. A fresh Docker/GPU-process
inventory was empty; Sparky root availability remained98,664,050,688 bytes of
1,968,362,958,848 bytes, above the standing5% floor. The both-Spark measurement
interval was explicitly released to the coordinator and other owners, without
retry, cancellation or another measurement window. Root's subsequent SDK4
maintenance holds new submissions; that is distinct from a scientific result.

Terminal success establishes retained population and containment, not the
preregistered raw-event/paired contrast or fresh telemetry coverage. Those
claims required the single independently owned receipt-only CPU reduction
completed below; no duplicate owner reduction, numeric qualification,
Torch/NCU profile or new GPU population is authorized by this record. The
original13.835448→12.447416ms observation keeps its original single-window
scope. Issue739 remains open, PR868 remains draft, PM remains default off,
and clock qualification, served/general speed and energy remain HOLD.


## Completed independent descriptive repeatability acceptance

The independent reviewer accepted the exact6-block/24-cell/720-event population
for **scoped descriptive M2048 repeatability**, not a causal clock-controlled,
served, energy or production-adoption result. One receipt-only CPU action,
`1ca9902c1b64f0b9e03142020ab533530dec42bb4383e6d493e8df0362513ce5`,
actually executed return0 onDL380G10 in3.395695s, CPU1/mem1GiB/native1. No
Torch dependency, newGPU work, duplicate owner analysis, numeric qualification,
Torch/NCU profile or old CPU control population was executed for this reduction.

The reducer is source-bound at Git
`4426ac34a5b37469d86f38691ffbf356850d4aa2`, file
`t8_performance/independent-pm-review/repeatability_review.py`, SHA
`b6425bddf5f08714b4ea3f7c8705f09c1b454ad3de4a08bfa3696f3e8f191b06`.
Its exact219-range retained-metadata manifest SHA is
`5b00b91f9f366d4af32967b0f997220de06fee1ad57574569df884c327eb2b5c`.
It used the actualSDK4 injected context/current c437 helper and public
covers/acquire/openFD/close/release, with pin
`752af8d11c87be2e13af9de8d8b3f303` and ref
`78521c6a03bc4f64828435a16ad9f03a`, closed/released. This proves the completed
metadata reader's public lease, not a retroactive SDK4 lease for the six original
SDK3/direct-held-FD GPU attempts, nor immutable original tensor-provider authority
for an experimental trusted native-code artifact.

It verified all six original full CoreCAS attestations/canonical claims/payloads,
source/protocol/numeric/input/final-bit bindings, retained native ELF4d693c,
nine stable original file identities per block, and1,872,827,160 resident bytes
per arm. The original eight accepted Torch profiles are explicitly reused;
their old durations are not called fresh-window profiles.

| Pair/block | Order | Legacy mean of two cell medians (ms) | PM mean of two cell medians (ms) | Descriptive latency reduction |
|---|---|---|---|---|
| 1/1 | ABBA | 13.606424 | 12.363664 | 9.133626% |
| 1/2 | BAAB | 13.596272 | 12.522104 | 7.900460% |
| 2/1 | BAAB | 13.656608 | 12.501448 | 8.458615% |
| 2/2 | ABBA | 13.563072 | 12.226640 | 9.853461% |
| 3/1 | ABBA | 13.668440 | 12.369192 | 9.505459% |
| 3/2 | BAAB | 13.773632 | 12.262216 | 10.973258% |

Here A=legacy andB=piece-major. Every cell's statistic is the conventional
median of its30 retained raw CUDA events; each arm's block mean is the
arithmetic mean of its two cell medians. The preregistered pair statistic is
**the mean of the ABBA and BAAB block percentage reductions**, not the ratio
of pooled arm medians:

| Predefined pair | Mean block percentage reduction | ABBA-minus-BAAB order contrast (percentage points) |
|---|---|---|
| 1 | 8.517043% | +1.233167 |
| 2 | 9.156038% | +1.394845 |
| 3 | 10.239359% | −1.467799 |

Mean of the three pair contrasts is **9.304147%**; descriptive sampleSD is
**0.870658 percentage points**, pair range8.517043–10.239359%. All six blocks
favoredPM, with block reductions7.900460–10.973258%. These are descriptive
sampled operator contrasts, not a confidence interval or guaranteed significance.
The720 events are not720 independent runs. Actual chronological order remains
1/1,3/1,1/2,3/2,2/1,2/2: pairs1/3 are nonadjacent and correlated physical
thermal/device history is retained. No hypothetical adjacent pairing or
thermal/clock normalization replaces the actual observations. The earlier
13.835448→12.447416ms result remains its separate original single-window result.

### Fresh telemetry and non-promotion boundary

Scored host-polled temperatures span75–79C and reported SM clocks2281–2431MHz;
PB action temperature peaks span76–80C. Conditioning block2/2 retains five
throttle-mask4 observations. All captured NVML observation-error arrays were
empty, but requested100ms host polling and repeated reported clocks do not
establish native sensor-update cadence, lag bounds, matched effective clocks,
thermal equilibrium or absence of unobserved throttle events. Raw timestamps,
reason masks and readings remain in the accepted packet, without exclusions.
CPU-PSI peak11.92 is retained as host evidence, not subtracted from timing.
Sparklina had noGPU workload in the reviewed interval and reported7–8W;
that is explicit both-host context, not a fabricated zero-power measurement.

Both-host raw Netdata includes each action, conditioning and scored window,
actual collection/group cadence, errors and rejected/empty groups. **Accepted
Netdata power coverage is0s for every subsecond scored cell on both hosts.**
That is unobserved native-bucket coverage, not0W, not an interpolated
per-cell energy estimate, and not proof that high-rate NVML and Netdata agree.
Energy/work-per-joule therefore remainsHOLD. No GPU-utilization percentage is
called saturation; resident bytes, scope accounting and sensor domains are not
silently conflated. Clock qualification, general/served throughput, full-model
quality/activation replay, graph qualification and production adoption remain
unqualified/HOLD. PM stays defaultOFF, PR868 draft and issue739 OPEN.

### Actual acceptance and analysis evidence

Accepted independent report:
`/mnt/shared/astra-resume-20261002/t8_performance/INDEPENDENT-PM-REPEATABILITY-REVIEW.json`,
SHA `d479eb81d02765d2b441402ea99ed2e35c27b38db88eb53eef885f14938a6977`.
It retains all720 raw events, per-cell observations, all six source/input/CAS
bindings, both-host coverage/errors and chronology. Canonical analysis CAS
receipt `c2667e269ae45c868feaf27094287529f0c9a7ba5acb5d4de14fc44f611acd2e`
binds payload `052317229dd63013ae99578fd4c903a66f9784d50ae40510f399c1c74f83cdb1`
and local claim `944f966481ef14d61ddc35a36599758045316076c71d8641f2f245c8394af1ec`.
The reviewer's MCP claim check reported attestation_verified=null and was not
represented as attestation verification. A separate **read-only current
published CoreCAS lookup** verified this analysis's full attestation
`2dd85ce55ce4377b4fae81efdc83ed12690cd36809eedd52dbfad6e293b8f5fe`.
That lookup is evidence inspection, not a duplicate reduction or qualification.
Its record is `repeatability-ready-0a3ddc70/ANALYSIS-ACTUAL-FULL-CORE-VERIFIED.json`,
SHA `98ae3dae0341aab114e9050a28408cf82974cb1b654c295d6188deb6f7e6ac55` under the existing piece-major evidence root.

## Current-base implementation review, 2026-10-03

The current-master merge at `3f2974fc` resolved only the additive architecture
stamp conflict; native kernel source remains `c236b7aa`, with no change to the
qualified PM reader, layout, build flags or benchmark caller. This is review
for a bounded default-off implementation landing, not a new scientific gate.
The frozen numeric and six-block evidence above retains its exact identities
and scope. Issue739, serving/default/pin adoption and energy remain OPEN/HOLD.

Current-base PB CPU `725e6454` recorded 209 passed, 60 skipped, one failure:
the AST-isolated parser test did not put its real sibling protocol on its
import path. The skips were 53 CUDA kernels, two dual-CUDA-device cases and
five absent checkpoint cases; zero tests allocated CUDA. The test-only repair
also routed its frozen artifact spelling through the existing root registry.
`8102c9ce` then passed all parser and issue-reference cases (24 passes) but
found the separately retained timeout-owner path bypassing that registry.
The timeout control now uses a named box artifact and truthfully skips where
that original owner is absent. PB `ec5f1cd3` exercised the actual retained
owner and box-root gate: 8 passed, zero skips/uncollected/CUDA allocations.
These overlapping CPU runs are not added into a fictitious full-suite result.
The earlier draft/no-merge statements describe the historical review stage;
normal required checks and exact-head acceptance still govern any landing.


## Next concrete lever: default-off mode0 dual-B fragment scheduling (#739)

On accepted base7437b569e16b42a7bf2ca13c25f343ab77ac6c42, the retained
PM M2048 Torch profile identifies mode0 gate/up at roughly58% of native device
time (~7.14–7.18ms). This selects a measured kernel priority, not an unmeasured
bandwidth diagnosis. The next implementation loads both independent B column
groups into consumer registers before the first group's MMA. It applies only to
MMA8 routed mode0, one-run R4, retaining each accumulator's K/G/mi/even/odd order,
with no producer/history/barrier/shared layout, wire/Params, mode1/2 or dense/two-run
schedule change. Additional operand register lifetime is an explicit possible
cost; no speedup is inferred before new source/ELF/numeric/profile evidence.

`TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH` is exact0/1, default0 and frozen
at routed_fused import. Only MMA8 receives its compiler define, and the loader
checks the actual native export. Off/on use separate processes/extension roots;
no common cache API, native module name, decoder/default or contract row changes.
Build/benchmark wrappers forward that build-scoped choice. The architecture and
`experiments/t8r_speed/mma8_gate_up_dual_b_design.json` keep the scope and matched
numeric/profile/both-host/work-per-joule requirements explicit.

Causal CPU action
`68c5cc0fc7e3dd1712fd4b79adb832669eca147b955e89739442cbcb82e80fc4`
ran on the original owner and failed10/passed5: the new frozen choice was absent,
unknown choices were accepted, and wrong compiled scheduling was not refused.
This real red run is not relabeled a pass; full successful CoreCAS lookup returns
null for its failed action. Changed controls passed once in action
`45f91e8022f390901845c0be46e18cf4ed303390e4da1d606bee8cd2349362ed`:
15passes/0skips/0uncollected/0CUDA-allocated, CPU2/native1/xdistworksteal on actual
capacity-eligibleSparklina, pytest1.71s/outer2.604418s. The deliberate worker
constraint records the below-floorSparky exclusion and existing healthy CPU
interpreter dependency, not arbitrary GPU load. Actual logs/terminal plus full
published CoreCAS verified canonical receipt
`222717a34a9992b694dcb39d770cc9ce853dfb100f849308a9e29dbc1101a836`,
payload `57f35d2d224e0a73c320d05d2de6b78aceaa518e44dcc85a6a1f8e2529d824d4`
and attestation `844bb9d043b01c2fdee25abb11b80e15a5adea4466fa4f6d9adb8d8a9556297f`.
These CPU controls qualify the changed build-choice/refusal contract only.

T16 remains the sole common PM/T4/T16/T8 composition/native-build owner. Fresh
unique all-family banks, actual compiler/recipe/source/PyInit/SASS identities and
changed-source correctness are required before any matched performance quantum;
oldc236/4d/51f6 evidence is never restamped onto the new composition. Numeric/safety
admission must preserve approved science and actual coordinator/worker5%floor
plus honest working allowance; Root's explicit conditional CPU-build decision
requires real canonical ROOT/MAX positive scratch demands and settled unsafeSP
zero-offer proof. Stale/unknown/refused predicates stop that launch. No cleanup,
image pull, old numeric/profile/control replay, new GPU/performance/energy/default/
serving/deployment authorization is implied by this source/CPU milestone.


### Exact wrapper empty-choice repair (independently closable child #893)

Commissioned parent review found that shell `${choice:-0}` and `-z` forwarding
coerced/dropped explicitly empty declarations while the Python owner refuses
empty choices. Both real wrapper paths now distinguish unset from empty:
build defaults only unset to0, benchmark forwards every set value (including
empty/invalid) to the strict owner. No code/native/performance change is inferred.
Ten new actual-shell-path CPU controls (inert image/Docker only) caused
`752355d9398c5da924f149087cc8e0767d10ab2a0ec420fca93389ca5c7065f1`
to fail the two explicit-empty cases/passed8. After the two-line repair,
`c5f2f2722968e88d539bf4f0c2a1bfcaf513b928ab19d4de9c5cb2274ff0eadd`
actually passed10/0skip/0uncollected/0CUDAallocated onSparklina, CPU2/native1/
xdist; pytest1.82s. Full current published CoreCAS verified the actual ending,
manifest/payload and attestation. The prior15 Python-owner controls and all
historic native/profile/numeric populations were not repeated. Source-only
child893 may close independently; parent739/native/performance/serving outcome
remains OPEN and all native/GPU/performance/default/pin/energy gates remain.
