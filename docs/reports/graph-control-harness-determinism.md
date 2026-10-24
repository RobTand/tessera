# Control lifecycle harness determinism — tessera#1530

## Final repair scope and source

Repair base HEAD: `e4ccf3b14b29c802e4eb9ecc4bf6a1574b3f5adc`.
The exact tested files on that base have these SHA-256 digests:

- `tests/test_graph_attest_determinism_control.py`: `e29ba55196cd0c61fae332cf4c95df8bedea924b80f026483c20e7eed0c98a9d`
- `tests/test_graph_attest_determinism_lifecycle_regression.py`: `072cb1bd4ab66eb89bc28ec9fa2622eda4ac1b2f396341594305fb53308fc5a7`
- `AGENTS.md`: `7797c94e557ac7c4ac71924b90b21656fd98b44276ef07845f15f6c4a74ebc70`
- Protected `tests/test_graph_attest_control_clock_isolation.py` (unchanged): `ccf55ba42935beef1f2ea02c9adcc30ef497c741287d9c0727fae3fe4301a821`

The harness still uses real Envelope/run_rank controllers and rendezvous
validation. Claims and controller-local managed_window/rank_window references
share one logical epoch; ambient subprocess watchdogs use real time. The local
payload has readiness and stop/acknowledgement pipes, no self-expiry, and is
reaped after owned cleanup with kill/wait on failure. The repair additionally
closes pipes even after a failed handshake. Production serving deadlines and
admission rules are unchanged, as are the existing mismatch-retention and
restart/death assertions. Shared CpuArm and the scenario suite owned by
tessera#1533 are untouched. The --control-rank interface is preserved.

Review found that direct calls to the lifecycle bypassed its module's autouse
fixture, so PM baseline setup could try to import an unavailable box client on
the GPU arm. PM setup now owns a temporary local reader for the real maintained
validator and restores the previous benchmark.CLIENT in a guaranteed context.
A new direct-call test deliberately starts with an absent client source and
executes both controllers to prove this path is hermetic.

The teardown regression previously modeled TERM/wait/KILL rather than the
adapter's stdin stop protocol. It now models acknowledged stop, already-exited,
missing acknowledgement and unresponsive stop for both phases (8 cases).
It proves invalid/unresponsive stops cannot publish successful cleanup,
only unresponsive live children need escalation, and all paths reap and close
pipes. Its runpy capture now registers and restores direct assignments to
benchmark.CLIENT, both protocol clocks, sys.path, argv, Envelope and run_rank.
Two further cases prove restoration after normal exit and interrupted capture.
The lifecycle regression population is now 11 cases, up from the four broken
cases in the reviewed patch; no tests or meaningful assertions were removed.
No tautological tests were found. Normative deterministic-test guidance is now
present in AGENTS.md, including direct-call setup and runpy isolation.

## Earlier evidence and corrections

Research's RED demonstrated ambient-clock work-deadline exhaustion in both
protected cases. Exact attribution of the historical two excusals is still
uncertain. The previous implementation's x86-only passes did not establish
GPU-arm hermeticity and omitted the broken teardown regression. Its claimed
AGENTS.md digest did not match the reviewed tree; the hashes above supersede
that account. Review receipts established GPU failures before controller launch
and four teardown failures plus shared-state leakage, all rechecked against
source for this repair. The new results below supersede those failures, not
retries or ignored failures of unchanged code.

Research RED:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9e794944a93d5e929db182b3da7c9ff58371cd9d894238363dc340e7d2bedbea","size":17716}},"operation":"4a76effd7b6f4d7a3e55d483e7c6fc77b3cf8419e073a803f688eb36ca684492","receipt":"d57a5645f8c078b5921b45b078eb7c6279a5cc0b0d62d4cebba63b10a3af1a46"}]

Review GPU-arm O1 failure (1 failed, 74 passed):
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b4370035432eb8e55781752c059e337e216a9a8d59536557366ae7ce1c9dfe6f","size":5089}},"operation":"d406fd8c604a28dfe2533452d55e77e69b6e72b21ea98c5e763a0bbffaed83d0","receipt":"f2488f8594b692aaa82beb3826c61890fa8a8808567d072448543074e47f89e6"}]

## Fresh post-repair checks

All checks use plain commands through run_tests, already admitted PrismaBuild
actions. Native threads are limited to one; xdist uses two workstealing workers.
Temporary files are under each operation's TMPDIR. These are CPU simulations on
both arms, not CUDA numerical qualification. Device visibility does not change
that population; no strict-CUDA, installation, private interpreter or nested
launcher is used. No full integration suite is claimed.

### GPU-arm budget and decision (recorded before execution)

CPU cannot reproduce the configured pinned image's missing shared-client source,
Python/package surface or architecture. One targeted run of the 11 lifecycle
regressions first decides hermetic PM setup and teardown/isolation on that arm
(expected <1 GPU minute, zero device work). Only after it passes, execute the
required five-invocation sequence (4 cases each; expected <2 GPU minutes) and the
separate exact O1 bounded-xdist command (75 cases; expected <1 GPU minute).
Those decide the review's arm-specific O1/O2 failures, not CUDA qualification.
These minimal named populations are not a numerical sweep.

### Teardown, direct-call and isolation regressions

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_determinism_lifecycle_regression.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86: 11 passed, zero deselected/skipped/uncollected, zero device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1ed1fbcba0765eb79d04a0944c71d04cbd96f959a9d9a4fb6bf985be969d6583","size":2007}},"operation":"348594d28500f6e06e52db7f8551cecd2a33de3255d96ae763d28759ad4fb441","receipt":"bc027cb2a11006230d69fd0f6f4aae0e954af784a41823377eb6c37c46304b45"}]

GPU arm: 11 passed, zero deselected/skipped/uncollected, zero device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ad93b3a48997dbbd0d54f4e519010786c71a91c28c6921c228c2f370a4d2be1d","size":1081}},"operation":"c8d511f3ee854a54188f302e5c71556e98fe71760404813a95974fc542689c73","receipt":"ec02162cdc9139971cb1bb4467e0ee96ad127cfdb34fd6d967af59c3fc55fb5d"}]

### Required five consecutive executed invocations (O2)

```sh
for attempt in 1 2 3 4 5; do echo "ATTEMPT=$attempt"; PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_control_clock_isolation.py tests/test_graph_attest_determinism_control.py -q -k 'ambient_work_clock or (test_real_rank_lifecycle_restarts_both_ranks_and_retains_mismatch_without_failing and matching-off)' --durations=10 --basetemp="$TMPDIR/pytest" || exit $?; done
```

Each invocation selects exactly the protected control/PM ambient-clock cases
and the original matching-off-control/matching-off-pm lifecycles: 4 selected
of 75 collected, 71 deselected. The loop stops at the first failure.

Retained stdout was read back and confirms every ATTEMPT=1 through ATTEMPT=5
on each arm. Every attempt passed 4 tests and deselected 71, with zero skips,
uncollected modules or device allocations; both loops exited zero. Durations
below are diagnostics only, never assertions.

| Attempt | x86 result / duration | GPU-arm result / duration |
| --- | --- | --- |
| 1 | 4 passed, 71 deselected / 5.45s | 4 passed, 71 deselected / 2.21s |
| 2 | 4 passed, 71 deselected / 5.31s | 4 passed, 71 deselected / 2.21s |
| 3 | 4 passed, 71 deselected / 5.27s | 4 passed, 71 deselected / 2.07s |
| 4 | 4 passed, 71 deselected / 5.16s | 4 passed, 71 deselected / 2.11s |
| 5 | 4 passed, 71 deselected / 5.00s | 4 passed, 71 deselected / 2.26s |

x86 sequence:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"df82df0e8239585dbdb46ebafb8665d853a84b9241795c23dadecf70199ca1da","size":6379}},"operation":"dade24fa910d16f278e3bfcfcefddf509c9a283cbfec44e55f511a05f615aeef","receipt":"bfba648c3aa62c5debb5b612d2de83d5a97038f76f90a1e5824e8340b2cead3f"}]

GPU-arm sequence:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"22950c357c6f55dbe341e5737e05c3e86e5684a5fd5e9030e073c6fc463937e3","size":5380}},"operation":"bcdc4739bed68a7e05afd70d0abfdb14a6029998534ac749824dc055f9e51cef","receipt":"e2f01291fa2242edf7fab45206f9f5d060002e5a7e5ceb09489f873a85f28e28"}]

### Separate exact O1 bounded-xdist command

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_control_clock_isolation.py tests/test_graph_attest_determinism_control.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86: 75 passed in 14.90s. GPU arm: 75 passed in 4.91s. All 73 original
target cases plus both protected cases executed on each arm, including mismatch
retention. Zero deselection, skips, uncollected modules or device allocations.

x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"675bbf4106bfea456f9ee798de5b24620dcdc8cef594e05a9a2852b0458094df","size":2095}},"operation":"142f8eb071ddf5a14bd7aea1b8a49ef95ab409e7a8507336272408553d58c826","receipt":"a213a9ff0affdda50e369a81f74295a2ef10d4d5fdcf025826d25f95cc1bea6d"}]

GPU arm:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a4860029d2e265288f636dd14d26793f7401fad1ed034d2422d313202018b86e","size":2018}},"operation":"de3367098c9d6f09d2d4bd031bf59928470565f0d051dbc36bae1d33a64e503a","receipt":"da6cdd176c4084ba2e624095f3d7dce612760cd07d601d1916a4e71bfa2978df"}]

All GPU-arm checks completed within the predeclared budget. The visible GB10
and CUDA torch 2.13.0+cu130 still performed zero device allocations. x86 reported
CPU torch 2.11.0 and NO CUDA. Neither arm supplies CUDA numerical coverage here.

### Combined x86 population and mandatory repository guards

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_control_clock_isolation.py tests/test_graph_attest_determinism_control.py tests/test_graph_attest_determinism_lifecycle_regression.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

This covers cross-file worker reuse as well as both mandatory repository guards.
298 passed in 99.46s: 75 target/protected cases, 11 lifecycle regressions and
212 guard cases. Zero deselection, skips, uncollected modules or device
allocations. The final diff and source hashes were rechecked with
`git diff --check` and `sha256sum`; the protected source remains unchanged.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5f0bde30f1f3c2626fbc90da468eaeb4cee293d443f05a1ac61c121841f26eb5","size":2098}},"operation":"c6797eb24587eee5f0bdcf2d5d3c24f0a776340b8fdd63c8afb7bdc58a58ec03","receipt":"b06661ebac3fc48a6391530be26282be28bb6005c5646b7ce7308ff6aef26a71"}]
