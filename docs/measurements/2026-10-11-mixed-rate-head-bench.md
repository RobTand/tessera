# The fused routed window kernel at HEAD: every rung from R768 to R1152 (tessera#694)

Issue: RobTand/tessera#694, item 2 of #690. Source: branch `tessera-694` at
`f709862e2` (the kernel, loader and lane predicates of `origin/master`
`4c0fd6aca`; the branch adds only measurement tools, tests and this note).
Kernel `src/tessera/serving/csrc/routed_fused_window.cu`, sha256
`8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f`: every
run below recorded this sha, and it is the sha of the file on master. Contract
v66. Hosts: sparklina and sparky (GB10, sm_121, 140 W envelope), each run
`--exclusive` at priority 0. Image
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`
(torch 2.13.0+cu130, vLLM 0.30.1rc1.dev336).

## Status: read this first

- **Reach holds.** At this head every rung from R768 to R1152 builds the
  production routed adapter (`FusedRoutedWindowMoE`, the E4M3 MMA library,
  class dispatch) and runs it on the device. The GPU oracles pass with zero
  skips: 427 routed tests and 297 dense tests ([Oracles](#oracles-at-head)).
- **The 1.5x bar holds for five rungs and is missed by two.** Against the
  served R1024 (piece-major words), R768, R832, R896, R928 and R960 stay within
  1.42x on every measure in every run. R1088 and R1152 reach 1.50x to 1.57x on
  device time (graph replay and profiler) at their worst M, and 1.48x to
  1.52x on the eager forward wall. The release's own R1088 stack (17 of the 42
  routed layers) reads 1.47x to 1.52x on graph replay at M 1 to 8 and at most
  1.498x on the eager wall ([Time over R1024](#time-over-r1024)).
- **Against R1024 with legacy words, every rung holds.** The largest ratio
  is 1.395x (R1152, eager wall at M 512).
- **Why.** The served R1024 is 7% to 17% faster than R1024 with legacy words on
  the eager wall and 12% to 21% faster on graph replay (run A against run C).
  Two-run launches cannot take the changes that made it faster, and they
  execute 1.6x the instructions of the one-run launch ([Where the time
  goes](#where-the-time-goes)).
- **Decision for Rob.** The bar has two readings. Against R1024 as #694
  measured it on 2026-09-29 (legacy words), it holds with a margin of 7% or
  more. Against the served R1024, R1088 and R1152 miss by up to 5% on device
  time. Either accept the first reading, or fund kernel work on the two-run
  path. This note moves neither reading.
- **Not measured.** R1216 and the other (4, 5) offsets; any M between 8 and
  512; piece-major words for a two-run stack (the kernel cannot read them);
  a served decode; a route census at this head ([Route
  census](#route-census-at-head)); the compact adapter ("before") at HEAD,
  because the route has no compact substitute any more
  (`fused_routed_window_supported`): the 2.0x to 3.9x of
  `docs/measurements/2026-09-28-mixed-rate-fused-window.md` stand.

## What was run

Four timing jobs on the same source and kernel, one process each, and one NCU job:

| Run | Action | Host | What it holds |
|---|---|---|---|
| A | `fd310391` | sparklina | eight synthetic rungs, then the release artifact's R1024, R1088 and R832 stacks (real wires) |
| B | `83e461bb` | sparky | the eight synthetic rungs again |
| C | `01d0c3f8` | sparklina | the eight synthetic rungs with `TESSERA_ROUTED_PIECE_MAJOR=0`, so R1024 has legacy words |
| power | `a4ab5500` | sparky | the eight synthetic rungs at M 8 and 512, 30 s power windows, with Netdata |

**Synthetic rungs** (`experiments/t8r_speed/bench_rung_forward.py`). Each rung
is the production routed owner on random words at the TP2 rank's shapes: 288
experts, hidden 4096, 1024 intermediate columns, top 8, balanced routing
(token t picks experts (8t + j) mod 288). The column placement is the grammar's
own schedule (`grammar.bresenham_rate_schedule`), so the run table, column
order, block descriptors and word count are the wire's; `tests/
test_bench_rung_forward.py` holds them to the independent bitstream packer for
every rung from 256 to 2048 in steps of 32. R1024 takes piece-major words
where the intake does, and every other rung keeps legacy words, as the loader
does. The call is the one `bench_t8r.build_routed` returns, so a cell here and a
cell there time the same code.

**Real wires** (`experiments/t8r_speed/bench_t8r.py`). The T8R release
artifact's layer 10 (R1024, one run, piece-major words), layer 11 (R1088,
rates 4 and 5, legacy words) and layer 42 (R832, rates 3 and 4, legacy words),
through the production intake. The Sep-28 artifact predates required expert
classes, so the bench derives the legacy uniform metadata and checks the loaded
run tables against it (see [Legacy fixture shim](#legacy-fixture-shim)). The
artifact has no other routed rung, which is why the other rungs are synthetic.

**Cells.** M = 1 to 8, 512, 2048 and 8192, 10 warm-up and 30 timed
iterations. Per cell: the output sha256; the eager CUDA-event wall time
(median); for M <= 8, the call replayed from a captured CUDA graph (the
decode path's proxy); the torch.profiler device time per call; NVML board
power over a 3 s loop of back-to-back calls. A forward pass walks the rungs
and the Ms upward and a reverse pass walks them downward, and a cell's time
is the mean of the two passes' medians, so a drift in clock or temperature
moves both halves of a ratio. The spread |F - R| / mean is at most 2.6% for
every rung except R768 (median 3.5%, at most 12.6%). A ratio is a rung's time
over R1024's at the same M in the same run. A cell above the bar is bold.

## Time over R1024

Largest ratio over M, per rung and run (A and B: R1024 has piece-major
words, the served layout; C: R1024 has legacy words):

eager forward wall:

| Rung | A | B | C |
|---|---:|---:|---:|
| R768 | 1.080 at M 8192 | 1.090 at M 8192 | 1.018 at M 8192 |
| R832 | 1.357 at M 8 | 1.369 at M 8 | 1.258 at M 512 |
| R896 | 1.350 at M 8 | 1.349 at M 512 | 1.243 at M 512 |
| R928 | 1.369 at M 8 | 1.370 at M 7 | 1.261 at M 512 |
| R960 | 1.396 at M 8 | 1.375 at M 512 | 1.269 at M 512 |
| R1088 | **1.517** at M 8 | 1.497 at M 512 | 1.378 at M 512 |
| R1152 | 1.479 at M 8 | **1.508** at M 512 | 1.395 at M 512 |

graph replay (M <= 8):

| Rung | A | B | C |
|---|---:|---:|---:|
| R768 | 1.048 at M 8 | 1.061 at M 1 | 0.947 at M 7 |
| R832 | 1.394 at M 8 | 1.405 at M 7 | 1.265 at M 8 |
| R896 | 1.365 at M 8 | 1.376 at M 2 | 1.245 at M 8 |
| R928 | 1.393 at M 8 | 1.402 at M 4 | 1.268 at M 8 |
| R960 | 1.419 at M 8 | 1.397 at M 1 | 1.256 at M 8 |
| R1088 | **1.549** at M 8 | **1.503** at M 4 | 1.373 at M 7 |
| R1152 | **1.503** at M 8 | **1.520** at M 8 | 1.343 at M 7 |

profiler device time:

| Rung | A | B | C |
|---|---:|---:|---:|
| R768 | 1.052 at M 8192 | 1.102 at M 8192 | 1.007 at M 8192 |
| R832 | 1.387 at M 8 | 1.400 at M 7 | 1.248 at M 512 |
| R896 | 1.369 at M 8 | 1.380 at M 1 | 1.245 at M 512 |
| R928 | 1.405 at M 8 | 1.408 at M 4 | 1.263 at M 7 |
| R960 | 1.413 at M 8 | 1.396 at M 1 | 1.264 at M 512 |
| R1088 | **1.571** at M 8 | **1.522** at M 512 | 1.376 at M 512 |
| R1152 | **1.512** at M 8 | **1.526** at M 7 | 1.379 at M 512 |

Counts of cells above the bar in A and B (11 cells for the eager wall and the
profiler, 8 for graph replay): R1088 has 1 and 0 on the wall, 7 and 2 on graph
replay, 8 and 6 on the profiler; R1152 has 0 and 1, 1 and 3, 1 and 5; no other
rung has one. In C no rung has one.

Reading, with the checks that bound it:

- **Two hosts agree.** Runs A (sparklina) and B (sparky) differ by at most
  0.05 in the largest ratio of any rung, and both put the same two rungs over
  the bar on device time.
- **The harness is slightly pessimistic.** Against the real stacks in the same
  job (run A, tables below), the synthetic R1088 is 0% to 5% slower than the
  real one (3% to 5% at M 2 to 8 on graph replay and the profiler), the
  synthetic R1024 is 1% to 5% slower than the real one (13.7% at M 1 on the
  eager wall, where the call is host bound), and synthetic and real R832
  agree within 2.3%. So the synthetic ratio of R1088 exceeds the real one by up
  to 0.05 on device time (graph, M 8: 1.549 against 1.524). The synthetic
  ratios of R1152 may overstate by a similar amount; no real R1152 wire exists.
- **R1024 holds.** The real R1024 stack reads 0.4277, 0.7568, 1.2881 and
  2.3846 ms at M 1, 2, 4 and 8 in the 2026-10-10 run (`f33afcf0`) and 0.4255,
  0.7561, 1.2954 and 2.3912 ms in run A; at M 512, 2048 and 8192 the two runs
  differ by 0.7%, 0.5% and 1.6%. The branch changes no kernel, loader or lane
  predicate, so R1024 cannot differ by construction.
- **Large M is not at risk.** At M 2048, the chunk size of the 2500 tok/s
  headline, no rung exceeds 1.39x on the eager wall or the profiler against the
  served R1024 (run B, R1088, profiler: 1.390), and at M 8192 none exceeds
  1.31x.

### Run A

Time over R1024 (piece_major words), eager forward wall:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.982 | 1.259 | 1.241 | 1.274 | 1.257 | 1 | 1.323 | 1.307 |
| 2 | 1.020 | 1.303 | 1.284 | 1.317 | 1.311 | 1 | 1.431 | 1.354 |
| 3 | 1.001 | 1.278 | 1.267 | 1.297 | 1.301 | 1 | 1.447 | 1.395 |
| 4 | 1.023 | 1.330 | 1.324 | 1.350 | 1.343 | 1 | 1.487 | 1.421 |
| 5 | 1.016 | 1.334 | 1.317 | 1.356 | 1.356 | 1 | 1.478 | 1.430 |
| 6 | 1.023 | 1.329 | 1.311 | 1.348 | 1.360 | 1 | 1.492 | 1.464 |
| 7 | 1.030 | 1.339 | 1.331 | 1.365 | 1.369 | 1 | 1.496 | 1.468 |
| 8 | 1.029 | 1.357 | 1.350 | 1.369 | 1.396 | 1 | **1.517** | 1.479 |
| 512 | 1.022 | 1.333 | 1.329 | 1.350 | 1.355 | 1 | 1.493 | 1.474 |
| 2048 | 1.023 | 1.244 | 1.249 | 1.283 | 1.276 | 1 | 1.381 | 1.356 |
| 8192 | 1.080 | 1.252 | 1.230 | 1.258 | 1.249 | 1 | 1.256 | 1.252 |
| max | 1.080 | 1.357 | 1.350 | 1.369 | 1.396 | 1 | **1.517** | 1.479 |

Time over R1024 (piece_major words), graph replay (M <= 8):

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.032 | 1.353 | 1.338 | 1.372 | 1.367 | 1 | 1.436 | 1.412 |
| 2 | 1.029 | 1.363 | 1.344 | 1.377 | 1.377 | 1 | **1.520** | 1.425 |
| 3 | 1.017 | 1.338 | 1.323 | 1.360 | 1.369 | 1 | **1.524** | 1.461 |
| 4 | 1.030 | 1.363 | 1.360 | 1.387 | 1.383 | 1 | **1.530** | 1.470 |
| 5 | 1.019 | 1.370 | 1.351 | 1.392 | 1.388 | 1 | **1.518** | 1.465 |
| 6 | 1.028 | 1.361 | 1.338 | 1.383 | 1.389 | 1 | **1.528** | 1.500 |
| 7 | 1.036 | 1.365 | 1.362 | 1.389 | 1.388 | 1 | **1.527** | 1.497 |
| 8 | 1.048 | 1.394 | 1.365 | 1.393 | 1.419 | 1 | **1.549** | **1.503** |
| max | 1.048 | 1.394 | 1.365 | 1.393 | 1.419 | 1 | **1.549** | **1.503** |

Time over R1024 (piece_major words), profiler device time:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.027 | 1.356 | 1.342 | 1.369 | 1.374 | 1 | 1.447 | 1.427 |
| 2 | 1.026 | 1.348 | 1.332 | 1.365 | 1.369 | 1 | **1.514** | 1.420 |
| 3 | 0.988 | 1.317 | 1.305 | 1.339 | 1.350 | 1 | **1.516** | 1.451 |
| 4 | 1.011 | 1.358 | 1.364 | 1.387 | 1.387 | 1 | **1.540** | 1.477 |
| 5 | 0.992 | 1.344 | 1.332 | 1.381 | 1.369 | 1 | **1.506** | 1.453 |
| 6 | 0.992 | 1.350 | 1.330 | 1.373 | 1.384 | 1 | **1.552** | 1.494 |
| 7 | 0.998 | 1.347 | 1.357 | 1.389 | 1.379 | 1 | **1.531** | 1.500 |
| 8 | 1.015 | 1.387 | 1.369 | 1.405 | 1.413 | 1 | **1.571** | **1.512** |
| 512 | 1.016 | 1.321 | 1.335 | 1.368 | 1.365 | 1 | **1.517** | 1.480 |
| 2048 | 0.985 | 1.235 | 1.228 | 1.266 | 1.258 | 1 | 1.369 | 1.360 |
| 8192 | 1.052 | 1.214 | 1.223 | 1.251 | 1.226 | 1 | 1.240 | 1.251 |
| max | 1.052 | 1.387 | 1.369 | 1.405 | 1.413 | 1 | **1.571** | **1.512** |

Real wires against synthetic, eager forward wall (real run fd310391):

| M | real R1088 / R1024 | synthetic R1088 / R1024 | real R832 / R1024 | synthetic R832 / R1024 | synthetic / real R1024 | synthetic / real R1088 | synthetic / real R832 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.498 | 1.323 | 1.418 | 1.259 | 1.137 | 1.005 | 1.009 |
| 2 | 1.422 | 1.431 | 1.325 | 1.303 | 1.025 | 1.031 | 1.008 |
| 3 | 1.412 | 1.447 | 1.321 | 1.278 | 1.025 | 1.051 | 0.992 |
| 4 | 1.445 | 1.487 | 1.349 | 1.330 | 1.017 | 1.047 | 1.003 |
| 5 | 1.473 | 1.478 | 1.364 | 1.334 | 1.033 | 1.036 | 1.010 |
| 6 | 1.459 | 1.492 | 1.347 | 1.329 | 1.013 | 1.035 | 0.999 |
| 7 | 1.493 | 1.496 | 1.373 | 1.339 | 1.024 | 1.025 | 0.998 |
| 8 | 1.483 | **1.517** | 1.377 | 1.357 | 1.015 | 1.039 | 1.001 |
| 512 | 1.483 | 1.493 | 1.385 | 1.333 | 1.028 | 1.035 | 0.989 |
| 2048 | 1.400 | 1.381 | 1.291 | 1.244 | 1.036 | 1.022 | 0.999 |
| 8192 | 1.286 | 1.256 | 1.284 | 1.252 | 1.031 | 1.007 | 1.006 |

Real wires against synthetic, graph replay (M <= 8) (real run fd310391):

| M | real R1088 / R1024 | synthetic R1088 / R1024 | real R832 / R1024 | synthetic R832 / R1024 | synthetic / real R1024 | synthetic / real R1088 | synthetic / real R832 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.481 | 1.436 | 1.387 | 1.353 | 1.034 | 1.002 | 1.009 |
| 2 | 1.497 | **1.520** | 1.391 | 1.363 | 1.025 | 1.040 | 1.004 |
| 3 | 1.471 | **1.524** | 1.351 | 1.338 | 1.015 | 1.052 | 1.005 |
| 4 | **1.506** | **1.530** | 1.392 | 1.363 | 1.024 | 1.041 | 1.003 |
| 5 | **1.512** | **1.518** | 1.406 | 1.370 | 1.035 | 1.040 | 1.009 |
| 6 | **1.512** | **1.528** | 1.373 | 1.361 | 1.021 | 1.032 | 1.013 |
| 7 | **1.515** | **1.527** | 1.401 | 1.365 | 1.026 | 1.034 | 1.000 |
| 8 | **1.524** | **1.549** | 1.402 | 1.394 | 1.023 | 1.039 | 1.017 |

Real wires against synthetic, profiler device time (real run fd310391):

| M | real R1088 / R1024 | synthetic R1088 / R1024 | real R832 / R1024 | synthetic R832 / R1024 | synthetic / real R1024 | synthetic / real R1088 | synthetic / real R832 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.490 | 1.447 | 1.388 | 1.356 | 1.027 | 0.998 | 1.003 |
| 2 | 1.493 | **1.514** | 1.398 | 1.348 | 1.029 | 1.044 | 0.992 |
| 3 | 1.465 | **1.516** | 1.352 | 1.317 | 1.018 | 1.054 | 0.992 |
| 4 | **1.506** | **1.540** | 1.395 | 1.358 | 1.021 | 1.044 | 0.994 |
| 5 | **1.519** | **1.506** | 1.404 | 1.344 | 1.043 | 1.034 | 0.999 |
| 6 | **1.502** | **1.552** | 1.375 | 1.350 | 1.018 | 1.052 | 1.000 |
| 7 | **1.521** | **1.531** | 1.395 | 1.347 | 1.028 | 1.035 | 0.992 |
| 8 | **1.522** | **1.571** | 1.406 | 1.387 | 1.014 | 1.047 | 1.001 |
| 512 | 1.484 | **1.517** | 1.385 | 1.321 | 1.025 | 1.048 | 0.977 |
| 2048 | 1.408 | 1.369 | 1.299 | 1.235 | 1.047 | 1.017 | 0.995 |
| 8192 | 1.276 | 1.240 | 1.273 | 1.214 | 1.039 | 1.010 | 0.991 |

### Run B (sparky)

Time over R1024 (piece_major words), eager forward wall:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.004 | 1.291 | 1.278 | 1.293 | 1.298 | 1 | 1.340 | 1.322 |
| 2 | 1.048 | 1.332 | 1.318 | 1.338 | 1.329 | 1 | 1.407 | 1.383 |
| 3 | 1.022 | 1.328 | 1.306 | 1.327 | 1.316 | 1 | 1.418 | 1.424 |
| 4 | 1.025 | 1.352 | 1.331 | 1.357 | 1.345 | 1 | 1.452 | 1.445 |
| 5 | 1.023 | 1.345 | 1.326 | 1.349 | 1.340 | 1 | 1.449 | 1.442 |
| 6 | 1.025 | 1.354 | 1.313 | 1.353 | 1.334 | 1 | 1.458 | 1.475 |
| 7 | 1.031 | 1.368 | 1.341 | 1.370 | 1.365 | 1 | 1.468 | 1.492 |
| 8 | 1.031 | 1.369 | 1.338 | 1.355 | 1.356 | 1 | 1.462 | 1.485 |
| 512 | 1.046 | 1.342 | 1.349 | 1.367 | 1.375 | 1 | 1.497 | **1.508** |
| 2048 | 1.044 | 1.273 | 1.258 | 1.291 | 1.292 | 1 | 1.387 | 1.379 |
| 8192 | 1.090 | 1.288 | 1.254 | 1.276 | 1.269 | 1 | 1.304 | 1.280 |
| max | 1.090 | 1.369 | 1.349 | 1.370 | 1.375 | 1 | 1.497 | **1.508** |

Time over R1024 (piece_major words), graph replay (M <= 8):

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.061 | 1.393 | 1.372 | 1.388 | 1.397 | 1 | 1.463 | 1.443 |
| 2 | 1.052 | 1.392 | 1.376 | 1.399 | 1.393 | 1 | 1.487 | 1.462 |
| 3 | 1.041 | 1.398 | 1.364 | 1.381 | 1.381 | 1 | 1.496 | 1.497 |
| 4 | 1.041 | 1.401 | 1.374 | 1.402 | 1.393 | 1 | **1.503** | 1.492 |
| 5 | 1.037 | 1.386 | 1.365 | 1.392 | 1.381 | 1 | 1.494 | 1.494 |
| 6 | 1.036 | 1.384 | 1.347 | 1.386 | 1.360 | 1 | 1.491 | **1.516** |
| 7 | 1.040 | 1.405 | 1.368 | 1.389 | 1.396 | 1 | 1.499 | **1.517** |
| 8 | 1.031 | 1.390 | 1.362 | 1.384 | 1.382 | 1 | **1.501** | **1.520** |
| max | 1.061 | 1.405 | 1.376 | 1.402 | 1.397 | 1 | **1.503** | **1.520** |

Time over R1024 (piece_major words), profiler device time:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.052 | 1.387 | 1.380 | 1.385 | 1.396 | 1 | 1.465 | 1.455 |
| 2 | 1.041 | 1.381 | 1.371 | 1.402 | 1.384 | 1 | 1.476 | 1.475 |
| 3 | 1.009 | 1.376 | 1.345 | 1.362 | 1.366 | 1 | 1.485 | 1.490 |
| 4 | 1.016 | 1.390 | 1.370 | 1.408 | 1.387 | 1 | **1.501** | 1.493 |
| 5 | 1.012 | 1.387 | 1.373 | 1.405 | 1.386 | 1 | **1.509** | **1.516** |
| 6 | 1.007 | 1.376 | 1.340 | 1.381 | 1.346 | 1 | **1.504** | **1.511** |
| 7 | 1.019 | 1.400 | 1.378 | 1.381 | 1.395 | 1 | **1.505** | **1.526** |
| 8 | 1.006 | 1.371 | 1.358 | 1.395 | 1.377 | 1 | **1.503** | **1.514** |
| 512 | 1.057 | 1.366 | 1.377 | 1.390 | 1.394 | 1 | **1.522** | **1.520** |
| 2048 | 1.020 | 1.271 | 1.248 | 1.286 | 1.290 | 1 | 1.390 | 1.384 |
| 8192 | 1.102 | 1.261 | 1.261 | 1.274 | 1.286 | 1 | 1.306 | 1.273 |
| max | 1.102 | 1.400 | 1.380 | 1.408 | 1.396 | 1 | **1.522** | **1.526** |

### Run C (R1024 with legacy words)

Time over R1024 (legacy words), eager forward wall:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.845 | 1.082 | 1.077 | 1.095 | 1.090 | 1 | 1.128 | 1.122 |
| 2 | 0.913 | 1.151 | 1.145 | 1.163 | 1.161 | 1 | 1.232 | 1.191 |
| 3 | 0.880 | 1.136 | 1.114 | 1.137 | 1.137 | 1 | 1.227 | 1.196 |
| 4 | 0.909 | 1.190 | 1.165 | 1.187 | 1.194 | 1 | 1.285 | 1.249 |
| 5 | 0.926 | 1.223 | 1.196 | 1.218 | 1.220 | 1 | 1.318 | 1.279 |
| 6 | 0.919 | 1.199 | 1.183 | 1.200 | 1.206 | 1 | 1.320 | 1.292 |
| 7 | 0.937 | 1.229 | 1.219 | 1.235 | 1.222 | 1 | 1.348 | 1.317 |
| 8 | 0.939 | 1.252 | 1.222 | 1.255 | 1.238 | 1 | 1.347 | 1.324 |
| 512 | 0.960 | 1.258 | 1.243 | 1.261 | 1.269 | 1 | 1.378 | 1.395 |
| 2048 | 0.979 | 1.191 | 1.186 | 1.207 | 1.201 | 1 | 1.293 | 1.289 |
| 8192 | 1.018 | 1.197 | 1.182 | 1.196 | 1.178 | 1 | 1.209 | 1.178 |
| max | 1.018 | 1.258 | 1.243 | 1.261 | 1.269 | 1 | 1.378 | 1.395 |

Time over R1024 (legacy words), graph replay (M <= 8):

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.857 | 1.126 | 1.111 | 1.138 | 1.127 | 1 | 1.182 | 1.163 |
| 2 | 0.905 | 1.186 | 1.175 | 1.198 | 1.195 | 1 | 1.277 | 1.228 |
| 3 | 0.881 | 1.157 | 1.134 | 1.170 | 1.163 | 1 | 1.261 | 1.222 |
| 4 | 0.910 | 1.215 | 1.194 | 1.209 | 1.222 | 1 | 1.314 | 1.280 |
| 5 | 0.924 | 1.245 | 1.215 | 1.229 | 1.239 | 1 | 1.336 | 1.299 |
| 6 | 0.920 | 1.216 | 1.193 | 1.220 | 1.218 | 1 | 1.339 | 1.315 |
| 7 | 0.947 | 1.250 | 1.230 | 1.260 | 1.242 | 1 | 1.373 | 1.343 |
| 8 | 0.942 | 1.265 | 1.245 | 1.268 | 1.256 | 1 | 1.372 | 1.343 |
| max | 0.947 | 1.265 | 1.245 | 1.268 | 1.256 | 1 | 1.373 | 1.343 |

Time over R1024 (legacy words), profiler device time:

| M | R768 | R832 | R896 | R928 | R960 | R1024 | R1088 | R1152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.850 | 1.117 | 1.107 | 1.134 | 1.121 | 1 | 1.180 | 1.162 |
| 2 | 0.890 | 1.175 | 1.173 | 1.194 | 1.191 | 1 | 1.275 | 1.219 |
| 3 | 0.847 | 1.148 | 1.129 | 1.170 | 1.159 | 1 | 1.257 | 1.205 |
| 4 | 0.894 | 1.218 | 1.207 | 1.227 | 1.233 | 1 | 1.340 | 1.307 |
| 5 | 0.898 | 1.244 | 1.219 | 1.241 | 1.235 | 1 | 1.343 | 1.302 |
| 6 | 0.884 | 1.214 | 1.212 | 1.214 | 1.224 | 1 | 1.340 | 1.325 |
| 7 | 0.903 | 1.241 | 1.218 | 1.263 | 1.231 | 1 | 1.368 | 1.332 |
| 8 | 0.908 | 1.239 | 1.228 | 1.258 | 1.245 | 1 | 1.364 | 1.339 |
| 512 | 0.941 | 1.248 | 1.245 | 1.237 | 1.264 | 1 | 1.376 | 1.379 |
| 2048 | 0.953 | 1.173 | 1.187 | 1.218 | 1.193 | 1 | 1.295 | 1.288 |
| 8192 | 1.007 | 1.159 | 1.176 | 1.200 | 1.185 | 1 | 1.199 | 1.181 |
| max | 1.007 | 1.248 | 1.245 | 1.263 | 1.264 | 1 | 1.376 | 1.379 |

### The release artifact's real wires

The same job (`fd310391`) timed the artifact's stacks. R1088 over R1024 reads
1.286x to 1.498x on the eager wall, 1.471x to 1.524x on graph replay and
1.276x to 1.522x on the profiler; R832 over R1024 reads at most 1.418x on every
measure. The 2026-10-10 run of the same stacks (`f33afcf0`) read R1088 at
1.276x to 1.479x (wall), 1.449x to 1.504x (graph) and 1.283x to 1.505x
(profiler), and R832 at most 1.404x. So the R1088 device-time excess is
repeatable on the real wires, and the 2026-10-10 reading of 1.505x at M 8 was
not noise.

## Where the time goes

Launch split of one forward (profiler microseconds per call, forward pass of
run A). The gate/up launch carries most of the excess: R1088's is 1.52x of
R1024's at M 1, 1.64x at M 8, 1.57x at M 512 and 1.44x at M 2048, against
1.46x, 1.48x, 1.45x and 1.34x for the down launch.

Launch split at M = 1 (forward pass, profiler us per call):

| Rung | gate/up us | / R1024 | down us | / R1024 | all kernels us | / R1024 |
|---|---:|---:|---:|---:|---:|---:|
| R768 | 205.6 | 1.033 | 111.6 | 1.037 | 354.2 | 1.027 |
| R832 | 276.5 | 1.389 | 153.4 | 1.426 | 467.4 | 1.356 |
| R896 | 275.5 | 1.384 | 149.6 | 1.392 | 462.6 | 1.342 |
| R928 | 282.1 | 1.417 | 152.3 | 1.416 | 471.9 | 1.369 |
| R960 | 282.6 | 1.419 | 153.4 | 1.427 | 473.7 | 1.374 |
| R1024 | 199.1 | 1.000 | 107.5 | 1.000 | 344.8 | 1.000 |
| R1088 | 303.4 | **1.524** | 156.5 | 1.456 | 499.0 | 1.447 |
| R1152 | 293.4 | 1.474 | 159.3 | 1.481 | 492.0 | 1.427 |

Launch split at M = 8 (forward pass, profiler us per call):

| Rung | gate/up us | / R1024 | down us | / R1024 | all kernels us | / R1024 |
|---|---:|---:|---:|---:|---:|---:|
| R768 | 1466.3 | 1.002 | 797.6 | 1.042 | 2306.9 | 1.015 |
| R832 | 2015.4 | 1.377 | 1093.1 | 1.428 | 3151.7 | 1.387 |
| R896 | 1999.5 | 1.366 | 1068.4 | 1.396 | 3111.5 | 1.369 |
| R928 | 2052.7 | 1.403 | 1096.1 | 1.432 | 3192.4 | 1.405 |
| R960 | 2068.5 | 1.413 | 1100.7 | 1.438 | 3212.8 | 1.413 |
| R1024 | 1463.5 | 1.000 | 765.5 | 1.000 | 2272.9 | 1.000 |
| R1088 | 2395.8 | **1.637** | 1130.9 | 1.477 | 3570.5 | **1.571** |
| R1152 | 2250.5 | **1.538** | 1141.9 | 1.492 | 3436.0 | **1.512** |

Launch split at M = 512 (forward pass, profiler us per call):

| Rung | gate/up us | / R1024 | down us | / R1024 | all kernels us | / R1024 |
|---|---:|---:|---:|---:|---:|---:|
| R768 | 6807.1 | 1.012 | 3764.2 | 1.026 | 10842.1 | 1.016 |
| R832 | 8874.7 | 1.319 | 4945.3 | 1.348 | 14096.0 | 1.321 |
| R896 | 8996.4 | 1.338 | 4972.8 | 1.356 | 14244.1 | 1.335 |
| R928 | 9260.9 | 1.377 | 5060.7 | 1.380 | 14600.1 | 1.368 |
| R960 | 9234.8 | 1.373 | 5059.9 | 1.380 | 14568.5 | 1.365 |
| R1024 | 6726.2 | 1.000 | 3667.6 | 1.000 | 10673.7 | 1.000 |
| R1088 | 10586.3 | **1.574** | 5323.1 | 1.451 | 16191.3 | **1.517** |
| R1152 | 10173.6 | **1.513** | 5348.9 | 1.458 | 15800.9 | 1.480 |

Launch split at M = 2048 (forward pass, profiler us per call):

| Rung | gate/up us | / R1024 | down us | / R1024 | all kernels us | / R1024 |
|---|---:|---:|---:|---:|---:|---:|
| R768 | 7408.3 | 0.993 | 4140.6 | 0.970 | 12551.4 | 0.985 |
| R832 | 9506.4 | 1.275 | 5219.4 | 1.222 | 15738.8 | 1.235 |
| R896 | 9426.4 | 1.264 | 5210.4 | 1.220 | 15645.7 | 1.228 |
| R928 | 9758.6 | 1.308 | 5356.7 | 1.255 | 16127.4 | 1.266 |
| R960 | 9651.5 | 1.294 | 5374.9 | 1.259 | 16029.9 | 1.258 |
| R1024 | 7458.5 | 1.000 | 4270.0 | 1.000 | 12740.2 | 1.000 |
| R1088 | 10729.2 | 1.439 | 5702.1 | 1.335 | 17439.5 | 1.369 |
| R1152 | 10479.3 | 1.405 | 5838.3 | 1.367 | 17329.3 | 1.360 |

**NCU** (`2f388d29` on sparklina, one call per rung and M, report
`ig694-20261011/ncu/t8r.ncu-rep`, table by `experiments/t8r_speed/
ncu_rung_table.py` from `ig694-20261011/ncu-import/rungs.raw.csv`). R768 is
the one-run rate-3 launch, R832 and R896 are the (3, 4)
pair, R1088 and R1152 the (4, 5) pair, R1024 is the piece-major rate-4 launch.

M = 1:

| Rung | Launch | Rates | us | Instructions (M) | Issue active % | SM throughput % | Barrier stall | Long-scoreboard stall | Short-scoreboard stall | Shared bank conflicts (M) | Registers |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R768 | gate/up | 3 | 244.6 | 25.7 | 29.8 | 40.8 | 6.28 | 0.29 | 0.88 | 5.9 | 96 |
| R768 | down | 3 | 136.3 | 13.8 | 28.6 | 37.6 | 6.92 | 0.45 | 1.15 | 3.0 | 96 |
| R832 | gate/up | 3/4 | 321.9 | 38.8 | 33.7 | 35.0 | 5.72 | 0.22 | 0.89 | 5.8 | 106 |
| R832 | down | 3/4 | 183.6 | 19.9 | 30.4 | 30.2 | 6.37 | 0.54 | 0.99 | 2.9 | 96 |
| R896 | gate/up | 3/4 | 313.2 | 38.1 | 34.2 | 34.7 | 5.40 | 0.22 | 0.90 | 5.5 | 106 |
| R896 | down | 3/4 | 178.1 | 19.5 | 30.7 | 29.9 | 6.17 | 0.53 | 1.04 | 3.0 | 96 |
| R1024 | gate/up | 4, piece-major | 220.4 | 23.9 | 30.0 | 41.7 | 6.14 | 0.23 | 1.38 | 5.6 | 96 |
| R1024 | down | 4, piece-major | 123.7 | 12.8 | 29.2 | 38.4 | 6.53 | 0.50 | 1.04 | 2.8 | 96 |
| R1088 | gate/up | 4/5 | 335.6 | 38.7 | 32.1 | 33.5 | 6.27 | 0.30 | 0.90 | 6.3 | 104 |
| R1088 | down | 4/5 | 185.1 | 19.7 | 30.0 | 29.8 | 6.68 | 0.53 | 0.91 | 3.1 | 96 |
| R1152 | gate/up | 4/5 | 320.4 | 38.8 | 34.1 | 34.6 | 5.58 | 0.25 | 0.89 | 6.2 | 104 |
| R1152 | down | 4/5 | 187.2 | 19.7 | 29.4 | 29.1 | 6.54 | 0.57 | 0.98 | 3.3 | 96 |

M = 512:

| Rung | Launch | Rates | us | Instructions (M) | Issue active % | SM throughput % | Barrier stall | Long-scoreboard stall | Short-scoreboard stall | Shared bank conflicts (M) | Registers |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R768 | gate/up | 3 | 7380.7 | 928.6 | 30.7 | 48.4 | 6.04 | 0.21 | 0.88 | 211.0 | 96 |
| R768 | down | 3 | 4175.4 | 497.2 | 29.1 | 44.1 | 6.81 | 0.41 | 1.14 | 107.1 | 96 |
| R832 | gate/up | 3/4 | 9813.7 | 1399.1 | 34.5 | 41.3 | 5.53 | 0.15 | 0.89 | 208.1 | 106 |
| R832 | down | 3/4 | 5597.7 | 717.6 | 31.2 | 35.6 | 6.17 | 0.46 | 0.99 | 104.5 | 96 |
| R896 | gate/up | 3/4 | 9637.8 | 1375.6 | 34.6 | 40.5 | 5.35 | 0.17 | 0.90 | 197.6 | 106 |
| R896 | down | 3/4 | 5475.6 | 704.0 | 31.3 | 35.1 | 6.02 | 0.48 | 1.04 | 108.6 | 96 |
| R1024 | gate/up | 4, piece-major | 6944.3 | 863.2 | 30.4 | 47.5 | 6.05 | 0.20 | 1.38 | 199.7 | 96 |
| R1024 | down | 4, piece-major | 3848.0 | 462.6 | 29.3 | 44.3 | 6.55 | 0.48 | 1.04 | 99.5 | 96 |
| R1088 | gate/up | 4/5 | 10502.5 | 1397.6 | 32.2 | 38.4 | 6.29 | 0.26 | 0.90 | 228.0 | 104 |
| R1088 | down | 4/5 | 5703.7 | 711.0 | 30.3 | 34.7 | 6.64 | 0.51 | 0.91 | 113.2 | 96 |
| R1152 | gate/up | 4/5 | 10178.0 | 1401.2 | 32.9 | 39.2 | 6.05 | 0.29 | 0.88 | 221.0 | 104 |
| R1152 | down | 4/5 | 5832.9 | 711.1 | 29.7 | 33.6 | 6.51 | 0.53 | 0.98 | 119.1 | 96 |

What the table says:

- **The excess is instructions, not memory.** At M 512 every two-run gate/up
  launch executes 1,376 M to 1,401 M instructions against 863 M for the
  one-run piece-major launch (1.59x to 1.62x), and the down launch 704 M to
  718 M against 463 M (1.52x to 1.55x). The one-run legacy launch (R768)
  executes 929 M (1.08x). The count is the same within 2% for R832, R896, R1088
  and R1152, so it follows the two-run path and neither the rates nor the share
  of high-rate columns.
- **The issue rate is the same, so time follows the count.** Issue-active is
  29% to 35% in every launch, and the barrier stall is 5.4 to 6.9 cycles per
  issue in every launch. Long-scoreboard stalls are 0.15 to 0.57 everywhere:
  the launches do not wait on memory.
- **(4, 5) is slower than (3, 4) at the same instruction count.** R1088's
  gate/up runs 10.5 ms at M 512 for 1,398 M instructions, R832's 9.8 ms for
  1,399 M. The (4, 5) launch has lower SM throughput (38% against 41%).
- **Hypotheses, not measurements.** The extra ~530 M instructions per gate/up
  launch at M 512 are about 900 per chunk, or about 110 per producer warp if
  the eight producer warps issue them all. That is the size of the column
  map's work (`col_map` runs once per half per chunk in `load_prev`, once in
  `issue_words`, and `publish_micro` picks one of four decode variants). Whether
  piece-major words for two-run stacks would also cut the instruction count
  is unknown; they cut R1024's time by 7% to 17% against legacy words.

Why the two-run path cannot take what R1024 took: piece-major words are
admitted for one run at rate 4 only (`routed_fused_window.cu:851`, the intake
at `moe_route.py:500`, `routed_fused.has_one_rate_four_run`), and the staged
stream history is single-run only (`routed_fused_window.cu:876`; #763 cut
R1024 by 7% to 15% and slowed every two-run unit by 1% to 9%, so #793 limited
it to one run). Piece-major words became the default for an eligible uniform
R4 stack on 2026-10-08 (`cd0137956`), and the staged history reached the
one-run kernel on 2026-09-30 (`f62fc6143`); #694's bar was set on 2026-09-28.

## Power per window, ranked by work per joule

Work is one forward, so forwards per joule is comparable across rungs. NVML
samples at 10 Hz over a 3 s loop of back-to-back calls (graph replay for
M <= 8). Every rung draws 51% to 61% of the 140 W envelope at M >= 2 (run A;
54% to 59% in run B). R1024 ranks first at every M in run A except M 1, and
in run B except M 1 and M 512; R768, which moves 25% fewer bytes, ranks first
there and is first or second everywhere. A (4, 5) rung is last or second to last
at every M in run A and at 10 of 11 Ms in run B.

Work per joule at M = 1:

| Rank | Rung | Mean W | Envelope fraction | Forwards per joule |
|---|---:|---:|---:|---:|
| 1 | R768 | 61.8 | 0.44 | 45.724 |
| 2 | R1024 | 74.0 | 0.53 | 39.134 |
| 3 | R832 | 75.5 | 0.54 | 28.450 |
| 4 | R896 | 77.2 | 0.55 | 28.132 |
| 5 | R928 | 75.9 | 0.54 | 27.799 |
| 6 | R960 | 77.3 | 0.55 | 27.359 |
| 7 | R1088 | 74.6 | 0.53 | 26.960 |
| 8 | R1152 | 76.7 | 0.55 | 26.698 |

Work per joule at M = 8:

| Rank | Rung | Mean W | Envelope fraction | Forwards per joule |
|---|---:|---:|---:|---:|
| 1 | R1024 | 73.8 | 0.53 | 5.830 |
| 2 | R768 | 80.8 | 0.58 | 5.242 |
| 3 | R896 | 77.7 | 0.55 | 4.101 |
| 4 | R928 | 76.5 | 0.55 | 4.070 |
| 5 | R832 | 78.5 | 0.56 | 4.055 |
| 6 | R960 | 77.0 | 0.55 | 4.046 |
| 7 | R1088 | 74.6 | 0.53 | 3.791 |
| 8 | R1152 | 76.5 | 0.55 | 3.787 |

Work per joule at M = 512:

| Rank | Rung | Mean W | Envelope fraction | Forwards per joule |
|---|---:|---:|---:|---:|
| 1 | R1024 | 71.9 | 0.51 | 1.300 |
| 2 | R768 | 80.4 | 0.57 | 1.153 |
| 3 | R928 | 74.8 | 0.53 | 0.917 |
| 4 | R832 | 78.3 | 0.56 | 0.893 |
| 5 | R960 | 76.7 | 0.55 | 0.891 |
| 6 | R896 | 80.1 | 0.57 | 0.883 |
| 7 | R1152 | 74.2 | 0.53 | 0.844 |
| 8 | R1088 | 75.2 | 0.54 | 0.829 |

Work per joule at M = 2048:

| Rank | Rung | Mean W | Envelope fraction | Forwards per joule |
|---|---:|---:|---:|---:|
| 1 | R1024 | 71.8 | 0.51 | 1.088 |
| 2 | R768 | 78.6 | 0.56 | 0.990 |
| 3 | R832 | 76.9 | 0.55 | 0.824 |
| 4 | R960 | 75.3 | 0.54 | 0.814 |
| 5 | R928 | 76.0 | 0.54 | 0.813 |
| 6 | R896 | 78.0 | 0.56 | 0.811 |
| 7 | R1152 | 76.2 | 0.54 | 0.759 |
| 8 | R1088 | 76.7 | 0.55 | 0.743 |

**Netdata, 30 s windows** (action `a4ab5500` on sparky; Netdata's
`nvidia_smi.gpu_power_draw` updates every 10 s, so a 3 s window sees at most
one sample and the longer windows are the cross-check). Mean power in watts
and forwards per joule by NVML and by the overlap-weighted Netdata series:

| M | Rung | NVML W | Netdata W | NVML forwards per joule | Netdata forwards per joule |
|---:|---|---:|---:|---:|---:|
| 8 | R768 | 82.1 | 76.5 | 5.141 | 5.513 |
| 8 | R832 | 80.5 | 77.7 | 3.881 | 4.022 |
| 8 | R896 | 80.7 | 78.7 | 3.945 | 4.045 |
| 8 | R928 | 80.4 | 82.7 | 3.859 | 3.750 |
| 8 | R960 | 80.5 | 79.2 | 3.899 | 3.965 |
| 8 | R1024 | 77.5 | 76.7 | 5.655 | 5.720 |
| 8 | R1088 | 78.5 | 78.9 | 3.548 | 3.529 |
| 8 | R1152 | 79.8 | 79.7 | 3.639 | 3.642 |
| 512 | R768 | 80.3 | 77.3 | 1.129 | 1.172 |
| 512 | R832 | 80.6 | 79.5 | 0.860 | 0.871 |
| 512 | R896 | 79.8 | 82.0 | 0.880 | 0.856 |
| 512 | R928 | 80.5 | 81.9 | 0.850 | 0.835 |
| 512 | R960 | 79.9 | 76.9 | 0.863 | 0.897 |
| 512 | R1024 | 77.5 | 74.5 | 1.230 | 1.278 |
| 512 | R1088 | 78.5 | 75.6 | 0.793 | 0.823 |
| 512 | R1152 | 79.2 | 77.8 | 0.784 | 0.798 |

NVML and Netdata agree within 3.0 W on 15 of the 16 windows and within 5.6 W
(7%) on the other (R768, M 8). Both rank R1024
first, R768 second, the (3, 4) rungs next and the (4, 5) rungs last, and
differ only in the order inside the (3, 4) group, which is inside the noise.
The whole-job Netdata window of run A, and of the other jobs, is in
`netdata.json`, `netdata-pm2.json`, `netdata-legacy.json` and
`netdata-power30.json` under `ig694-20261011/`.

## Dense and shared experts at HEAD

`27083c52` (2026-10-10, sparky, source `b2875875a4`, the same kernel sha) timed
all 12 Tessera dense and shared-expert groups of the release artifact at every M. Two serving paths appear. At M <= 8
the groups run the fused dense identity (`native_fused_window_dense_e4m3mma`,
`tessera::fused_window_dense`; the profiler shows `routed_fused_kernel<..., 2,
true, ...>` plus `dense_reduce_kernel`). At M >= 512 they serve the decode-once
copy through CUTLASS: resident E4M3 modules decode once at load by default and
serve M >= 256 from the copy (tessera#931, `MIN_M` 256). The fused-lane dense
evidence at HEAD is therefore the M <= 8 cells below plus the GPU oracles. The
profiler names are in the receipt's `bench_t8r.json`.

Eager wall (ms) of the fused dense identity, and the ratio to the R1024 module
of the same shape where the artifact has one (no bar applies to dense
modules):

| Group | M 1 | M 2 | M 4 | M 8 | Over the R1024 module, M 1 | M 8 |
|---|---:|---:|---:|---:|---:|---:|
| `shared_gate_up.R1024.L11` | 0.073 | 0.084 | 0.084 | 0.083 | 1 | 1 |
| `shared_gate_up.R1088.L10` | 0.086 | 0.087 | 0.085 | 0.084 | 1.188 | 1.008 |
| `shared_gate_up.R960.L13` | 0.084 | 0.085 | 0.084 | 0.085 | 1.163 | 1.017 |
| `shared_gate_up.R832.L25` | 0.085 | 0.086 | 0.084 | 0.084 | 1.176 | 1.007 |
| `shared_down.R1024.L12` | 0.067 | 0.066 | 0.067 | 0.068 | 1 | 1 |
| `shared_down.R1088.L10` | 0.071 | 0.070 | 0.070 | 0.070 | 1.054 | 1.041 |
| `shared_down.R960.L11` | 0.070 | 0.069 | 0.069 | 0.070 | 1.036 | 1.034 |
| `shared_down.R832.L17` | 0.070 | 0.070 | 0.070 | 0.070 | 1.039 | 1.033 |
| `dense_gate_up.R960.L0` | 0.227 | 0.228 | 0.229 | 0.228 | 1.320 | 1.321 |
| `dense_down.R1088.L0` | 0.185 | 0.184 | 0.183 | 0.183 | - | - |
| `dense_gate_up.R1024.L2` | 0.172 | 0.173 | 0.172 | 0.173 | 1 | 1 |
| `dense_down.R832.L1` | 0.181 | 0.181 | 0.181 | 0.180 | - | - |

Mixed-rate dense modules are slower than their R1024 sibling on the fused
lane: `dense_gate_up` R960 by 1.32x at every M, `shared_gate_up` by 1.16x to
1.19x at M 1 (about 1.00x to 1.03x at M >= 2) and `shared_down` by 1.03x to
1.06x. On device time they are slower still (kernel microseconds at M 1:
`shared_gate_up` 33.9 against 41.7 to 43.2, `shared_down` 23.1 against 30.3 to
31.2, `dense_gate_up` 130.3 against 185.7). The eager wall hides this where the
host path dominates.

## Oracles at HEAD

`tests/test_routed_fused_window.py` (`4ddd5eec`, sparky) and
`tests/test_dense_fused_window.py` (`0ca8d62a`, sparklina), at `f709862e2`
in the pinned image under `--strict-cuda`, through `experiments/
routed_fused_tests.sh`:

| File | Passed | Failed | Skipped | Allocated on the device | Seconds |
|---|---:|---:|---:|---:|---:|
| `tests/test_routed_fused_window.py` | 427 | 0 | 0 | 419 | 306 |
| `tests/test_dense_fused_window.py` | 297 | 0 | 0 | 290 | 209 |

The matrix covers every one-run rate 1 to 8 and every adjacent pair, with the
fp64 one-hot decode oracle at bound zero, the accumulation oracle at the
truncating-accumulator bound, the real GLM dense role shapes, and CUDA-graph
capture. The files and the kernel are byte-identical at the head of this
branch, so the receipts hold for it; the surface files are
`ig694-20261011/gpu-routed/surface.json` and `gpu-dense/surface.json`.

## Route census at HEAD

The committed stub-B receipt
(`experiments/results/glm53_u1_stub_b_fused_mixed_tp1_eager_census.json`)
records verdict `served` and 21 of 21 modules on the fused kernel in both
phases, on image X (`f8dbe1a0`, vLLM 0.28.1rc1.dev397) at tessera commit
`af429328a1`. It covers routed q256 896, 928, 1024 and 1088 (E4M3) and 1024
(BF16), and dense 832, 880, 960, 1024 and 1088. `tests/
test_glm_u1_census_cells.py` replays it against the packaged cell table and
asserts the acceptance set; `tests/test_route_census_module_space.py` holds
the module space.

It was not taken again at HEAD, and it cannot be: since the 2026-10-07
cutover (#1024) the routed route requires `expert_ids` and `expert_classes` in each routed scheme,
and stub B's five routed schemes carry neither. Action `a28e9dc8`
(CPU, this head) ran the scheme validator on the committed stub-B config and
got `ValueError: model.language_model.layers.<L>.mlp.experts: expert_ids is
required` for layers 3 to 7, and accepted the class-carrying R768 stub config
(`glm53_r768_stub_base_config.json`, census `...r768_stub_base_tp2_eager_
census.json`, verdict `served`, decoder `native_routed_window_classes_
e4m3mma`). A census at HEAD needs a class-carrying stub B, which is a
re-export. A serve census also runs vLLM, which Rob exempted from PrismaBuild
on 2026-09-06, and no agent of this pipeline starts it directly. The reach
evidence at HEAD is above: every rung from R768 to R1152, R928 included, builds
the class-dispatch fused adapter and runs it (the `launch_pair` of each rung in
the receipts is `native_routed_window_classes_e4m3mma`), and the dense oracle
covers every pair.

## Legacy fixture shim

The first bench submission of the real wires failed before any cell: the
Sep-28 artifact's schemes carry no `expert_ids`, which HEAD requires. For
layers 10, 11 and 42 every expert of a projection carries a byte-identical wire
length, so every expert shares one rate multiset; equal multisets under the
deterministic grammar schedule are identical run tables, hence one identity
class, which is the library's own reading of an int q256
(`scheme._expert_role_rungs`). The bench restates it through the library's
sorter (`_legacy_uniform_scheme_metadata`) and then checks every loaded run
table against the asserted rungs (`_check_routed_runs_against_classes`); the
loader refuses differing extents first. Production still refuses class-less
schemes (above). Tests: `tests/test_bench_t8r_routed_builder.py` (4 passed;
the 3 new cases fail on the unfixed file). The shim changes no product path.

## Receipts

Output root: `/mnt/shared/tessera-measurements/ig694-20261011/` (2026-10-10
rows under `ig694-20261010/`).

| What | PrismaBuild key | Output |
|---|---|---|
| Run A, synthetic rungs + real-wire controls | `fd310391` | `synthetic/`, `real/`, `netdata.json` |
| Run B, synthetic rungs, sparky | `83e461bb` | `synthetic-pm2/`, `netdata-pm2.json` |
| Run C, R1024 with legacy words | `01d0c3f8` | `synthetic-legacy/`, `netdata-legacy.json` |
| 30 s power windows, Netdata cross-check | `a4ab5500` | `power30/`, `netdata-power30.json` |
| NCU, six rungs at M 1 and 512 | `2f388d29` | `ncu/t8r.ncu-rep` |
| NCU report imports (2026-10-10 and this report) | `3b97e503` | `ncu-import/a2.raw.csv`, `rungs.raw.csv` |
| GPU oracle, routed fused window | `4ddd5eec` | `gpu-routed/` |
| GPU oracle, dense fused window | `0ca8d62a` | `gpu-dense/` |
| Scheme validator on the stub configs | `a28e9dc8` | the action's stdout |
| 2026-10-10 real-wire bench | `f33afcf0` | `ig694-20261010/routed-bench/` |
| 2026-10-10 dense bench | `27083c52` | `ig694-20261010/dense-bench/` |
| 2026-10-10 NCU of the real wires | `05458683` | `ig694-20261010/ncu/t8r.ncu-rep` |
| CPU: the bench tests, before and after the script | `4ffcc4fb` / `efb7d4e2` | pbtest reports |
| CPU: the tool tests | `2d9e78ed`, `a1d8d571` | pbtest reports |
| CPU: the shim tests, before and after | `e814cb54` / `4107ce01` | pbtest reports |

The tables above are the output of `experiments/t8r_speed/rung_report.py`
(and `ncu_rung_table.py --markdown`) on these receipts. A first routed bench
of the real wires (`e30220c4`, `9f6901e1`) failed on the class-less fixture
before any cell.
