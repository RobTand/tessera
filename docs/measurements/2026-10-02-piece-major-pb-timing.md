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
