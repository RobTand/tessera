# Control lifecycle harness determinism — tessera#1530

## Scope and source

The repair is confined to `tests/test_graph_attest_determinism_control.py` and
normative guidance in `AGENTS.md`. The protected regression and shared CpuArm
are unchanged; the broader scenario suite remains owned by tessera#1533.
No production deadline, admission rule, controller, rendezvous validator,
mismatch-retention assertion or restart/death assertion was relaxed.

Base HEAD: `a203ef481e02e1320e6121c9d677120dfadbf987`.
SHA-256 of the exact tested sources (uncommitted patch on that base):

- `tests/test_graph_attest_determinism_control.py`: `8ddf35f9534ced85d9529e24b1dc1e0766327b70d1ff5a18a6063b60c87500f3`
- `tests/test_graph_attest_control_clock_isolation.py` (unchanged): `ccf55ba42935beef1f2ea02c9adcc30ef497c741287d9c0727fae3fe4301a821`
- `AGENTS.md`: `858a4734a114eba0dfbbafdfea50783b325f086a64df534138b1296b89dff3fb`

Claims and both real controller processes now share a logical epoch. Only the
controller-local managed_window/rank_window clock references are replaced, not
Python's process-wide time module or the real Envelope/run_rank algorithms.
Logical protocol time stays fixed during these success-path simulations;
expiration is not their subject. Production rendezvous condition polls remain
intact. Ambient watchdog time still runs, including when the protected regression
advances ambient monotonic time at payload launch.

The local payload acknowledges readiness on a pipe, blocks on an owned stop
command (or controller pipe EOF), and acknowledges orderly termination.
Cleanup verifies that acknowledgement and exit status, then reaps, with kill/wait
in finally on failure. There is no self-expiring payload or one-second reap
assumption. A 120-second watchdog detects failure, not success. Existing real
controller, restart, PID-death and retained-mismatch assertions still execute.
No tests were removed or assertions weakened; no tautological tests were found
in the changed target.

## RED and implementation checks

Research's supplied RED operation `4a76effd7b6f4d7a3e55d483e7c6fc77b3cf8419e073a803f688eb36ca684492`
reported both deterministic ambient-clock cases failing with graph work deadline
exhaustion. This proves the clock dependency, not the exact trigger of the two
historical excusals; that attribution remains uncertain.

Three intermediate implementation checks stopped on their first failed
invocation, not retries of unchanged code: missing local subprocess import
(4 failed), incorrect inherited rank attribute (4 failed), and missing original
cleanup-record fields (2 failed, 2 passed). All had 71 deselected. Each cause was
corrected before the next check. The final five-run sequence below starts after
those corrections and contains no ignored failure.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d8ef8b43b4c0d330e7f141ba563b95e7e380283fcae954c15aeea7fe9dad65ab","size":23446}},"operation":"7ea41607cf27f817d270982feaa652485add5f41ad4cd1654b61ad142b589d39","receipt":"be02751b9a676cd52a809ec4c9b236a886349b5d233839893c0216118488a238"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"343125bcdd168e20c67059377b1ab197c62d4d107799b52e4aa1bd53622e33bd","size":23674}},"operation":"700dd33881d6adbc5a98d1a6b55becd8d6d801e49b1f68b0e037b03ad2c51e19","receipt":"586d46d5ac96e3b922a7067a0dbcf3ace385969a2582a28cca69f6f11855078b"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"304777d36e67f86b5fdb3756efa892b6bd1103a84a6b296a5101bc69abc0914a","size":7523}},"operation":"3b890cd40b2030eed42091f49e8576e2138d66a6db40481fd4f735b80f6be330","receipt":"c9a6ab4aa9efd750b287395c4cbe23606081d0ee45eaf3e7fbe6e06701d0739c"}]

## Fresh post-fix verification

All commands were plain commands through `run_tests` (already PrismaBuild), on
the x86 arm. Python 3.14 / CPU torch; every invocation reported NO CUDA,
zero skipped tests, zero uncollected modules and zero device allocations.
These are CPU protocol simulations, not CUDA numerical qualification. No GPU
run was requested: no CUDA operation is needed to decide this harness defect;
the configured GPU-arm evaluation will still exercise CPU simulations.

### Five consecutive executed invocations

```sh
for attempt in 1 2 3 4 5; do echo "ATTEMPT=$attempt"; PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_control_clock_isolation.py tests/test_graph_attest_determinism_control.py -q -k 'ambient_work_clock or (test_real_rank_lifecycle_restarts_both_ranks_and_retains_mismatch_without_failing and matching-off)' --durations=10 --basetemp="$TMPDIR/pytest" || exit $?; done
```

Each invocation executed exactly four tests: the protected ambient-clock cases
for control and pm, and the existing matching-off lifecycle cases for control
and pm. Each collected 75 tests, selected 4 and deselected 71. The loop exited
zero only after all five executed invocations:

| Attempt | Result | Reported duration (diagnostic only) |
| --- | --- | --- |
| 1 | 4 passed, 71 deselected | 8.65s |
| 2 | 4 passed, 71 deselected | 8.62s |
| 3 | 4 passed, 71 deselected | 8.17s |
| 4 | 4 passed, 71 deselected | 7.83s |
| 5 | 4 passed, 71 deselected | 10.20s |

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0003d1c79da62afa6d2dc5783379c180d784d5f7e6f0ee876b2034ae7c4886b9","size":6380}},"operation":"74f6e1f34eef4f99095c5455614693f7c8fe80e952e763ef27197401ad1d27a3","receipt":"8f559553272b92a4d9a64fd32f6e6812fddb618e957e231dafa4f3f0ffb32f9a"}]

### Separate bounded xdist run

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_control_clock_isolation.py tests/test_graph_attest_determinism_control.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

75 passed in 19.32s; all target and regression cases executed, including both
mismatched-off lifecycles. No deselection, skip or uncollected module.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a1db843e69e6fec79519d254256a7e7d5e9a71d71dc4358f0ef2c0e3494f00e7","size":2104}},"operation":"d7851bd33e1c0540e93b02f57726a3e22de9bbec02d96f843af70fc82aa36f53","receipt":"e96f86465f54e98a45e9e1d2ab35a8f56b7cf27d4c001d4db57933323ae0c977"}]

### Repository guards

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

212 passed in 149.38s, with no deselection, skips or uncollected modules.
`git diff --check` also passed. The final diff was inspected; no production or
protected acceptance file changed. The once-only full integration suite remains
with the coordinator.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"97d9023562c80585698ccfefad03f306b9d2d81f3f1f09e723e6c306289fe1df","size":1755}},"operation":"135dac11f54008e286956cffcab8fb54362fbbb5951c1865fd13240b14edb11b","receipt":"073e71709c8a78d91b62cccfa9bb6db8a65a64d621441a1a25cedce79c3658be"}]
