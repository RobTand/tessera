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
