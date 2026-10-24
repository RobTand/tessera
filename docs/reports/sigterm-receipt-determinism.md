# SIGTERM receipt regression evidence

The named subprocess regression now waits for a child-local pipe byte written by
its restored predecessor, rather than treating `os.kill` completion as callback
receipt. Normal runs still send real self-SIGTERM. The callback records the exact
signum; the test preserves predecessor identity, `owner.main() == 1`, child success
output and parent return-code assertions. A 30-second receipt failure bound fits
inside the 90-second subprocess watchdog. Nested finalizers restore the original
handler, undo patches and close both descriptors even if cleanup raises.
Application/controller code and protected acceptance are unchanged. No tests were
removed; no tautological tests were identified in the scoped test file.

## Forced RED before / GREEN after

Research supplied the pre-fix protected-acceptance RED: the actual named child
failed with `AssertionError: []` when self-SIGTERM dispatch was deferred until
receipt observation. This demonstrates missing synchronization, not a controller
restoration defect. Attribution of the historical release excusal remains an
inference. The research in-memory positive control is not post-fix evidence.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8ba5113d2eb7f89951b4e282886faf5f0cc5dbde4e8c88df545985f4b5f0506f","size":4528}},"operation":"aa91a72d76399932608eb184b5691d14d63d14e0b1950434fc6b3dd4bff91f87","receipt":"3eade737a6d4d4b47f01fdf7b85861bc25696e94185972e3128e5d7840ef1572"}]

Post-fix protected acceptance: **1 passed**, no skips. It forces dispatch at the
pipe/select seam and verifies an actual unreadable-before/readable-after pipe,
exact predecessor invocation and exactly one request/delivery/receipt.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_piece_major_pb_signal_dispatch.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"da7ab262037ebb04b8720cec667906e133ba49086020890696954425a0f851f2","size":698}},"operation":"d23610f4e85f3dfedd6ae54c599649e9c5df907385883b02884a5139c8bef7d5","receipt":"2fd7d8b022c2b4403de03b688fdf85c15c7126661e2e686f7b9550908133dada"}]

## Real-signal repetition and complete target coverage

The following stop-on-first-failure command completed all five real-signal
invocations (**1 passed each**), then the two-worker complete target-file plus
protected-acceptance run (**12 passed**). All had zero skips. Retained stdout was
read back to verify all five populations, not merely the final invocation.
Adjacent restoration/error-path coverage remains green. Repeated runs supplement
the forced adverse-ordering demonstration above; they are not retries of failures.

```sh
for attempt in 1 2 3 4 5; do PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_piece_major_pb_action.py::test_subsequent_sigterm_reaches_previous_handler_in_subprocess -q -n 0 --durations=10 --basetemp="$TMPDIR/pytest-$attempt" || exit $?; done; PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_piece_major_pb_action.py tests/test_piece_major_pb_signal_dispatch.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest-xdist"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f0266f61301296e0cf20fcc90857272870dd36d1cb5a92d18d1272171b82434a","size":4948}},"operation":"3a32e80cb756dec333ed37581d731125b7eef57665af3e74343f9b0b2c2faae9","receipt":"d21c5afedfacd13e77fe753a1ba060219ad9baa99558fd5fcf6e1df983c781f0"}]

Repository guards: **212 passed**, no skips.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest-guards"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"bd58268e19cc06ef335af2ae3c0d215c82ed7bcb6cc402d2cdc0e5cfcc4701ea","size":1765}},"operation":"e547fe009428d1738c83897c3e80a7274d5ead4e301f6b84f4d7f5b99b8c8dba","receipt":"d93f157da2b6d4e01ab3f93d39dce1eed08374d233273209e6ab1fbd04613689"}]

All implementation checks above ran through `run_tests`/PrismaBuild on x86 with
zero device allocations. No GPU execution was requested: these Python
controller/signal tests and modeled-dispatch harness require no CUDA execution.
Neither the modeled dispatch nor green CPU runs establish CUDA coverage. No
checks attempted here were unavailable. Full merge-result validation remains
with the coordinator.

## O3 guidance repair

Review found that the previously reported guidance was absent from AGENTS.md;
that earlier claim was incorrect. This repair adds the single authoritative
`Deterministic tests` section to root AGENTS.md, covering receipt/event/condition
synchronization versus send completion, forced adverse ordering, injected clocks,
generous failure-bound-only timeouts, forbidden timing and host-state dependencies,
no retry/quarantine/skip/assertion weakening, seeded randomness, `tmp_path`, no
real network calls and bounded parallel resources. It explicitly requires that
repeated green runs supplement rather than replace forced RED-before/GREEN-after
evidence. O3 requires human prose/evidence review and has no executable acceptance
check. This documentation-only repair leaves the regression, protected acceptance
and controller unchanged; the O1/O2 receipts above remain applicable.

Documentation-repair repository guard run (not an executable O3 check): **212
passed**, zero skips, zero uncollected modules, NO CUDA, zero device allocations.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest-doc-repair-guards"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6a1d15d3282c29ab3cb6d35905e7df767249ea4f53fd9e33c6d25417d124fa01","size":1806}},"operation":"a840319d7bc0572b2ac226ae4a57baf2aa1850ef0b04c066d7b6b83a0cd28a8f","receipt":"6e6d8d497b157ae9297306052bcd8b974c7d8e3d50864736e600103b99a97b9c"}]
