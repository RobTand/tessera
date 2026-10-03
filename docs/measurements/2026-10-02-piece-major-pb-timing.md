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
