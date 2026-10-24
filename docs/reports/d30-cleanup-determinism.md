# D30 cleanup determinism implementation evidence

## Scope and source

Base: `70a409bd2ed22c5eea391890d7a8732c02e5b60d`.
Source SHA-256 (test implementation unchanged by the guidance repair):

- `tests/test_graph_attest_d30_memory_guard.py`: `610d3dbc9dc98fa38bf0f2c965bc18c963db382b5a7be2b2a256c87a26a3bbc9`
- `AGENTS.md` after guidance repair: `ce4db69d79514d3c19ee9d811e2098bcac1584f88f37cc35cac95e99a1c51697`
- Protected, unchanged `tests/test_d30_cleanup_test_contract.py`: `19a28b425d6230264f7ecad28a100b3cb040c61e690752dd887438fad126af95`
- Unchanged production `experiments/graph_attest_702/managed_window.py`: `39457455ac667685c8bbe900f51c1c4d04866fc2ad4b056eb0e1d3a422a6b842`

The three original test names remain discoverable. Only physical boundaries are
replaced: real `Envelope.run` and `_terminate` decide timeouts, TERM, grace,
conditional KILL and reaping. Signals change live group membership and return
codes; cooperative TERM schedules a completion event. The stubborn group has a
leader and descendant, both still present at KILL after ten controlled seconds;
a separate group remains untouched. The finite envelope forces KILL at absolute
instant 108 rather than full-grace instant 111. Records and propagated timeout
messages are asserted separately from the process model. Cooperative evidence
allows one controlled 50 ms poll of quantization, never host timing.

Removed obsolete child program strings, PID polling/death helpers and threading
import. No tests removed; no tautological tests identified in the target file.
Other existing tests, including delayed reap and Docker integration, remain.
The guidance was missing from reviewed revision
`cb487f68787c493c31b3bf3ae039051fa673a2ab`: its `AGENTS.md` SHA-256 was
`f8131409f937d773bc3ee67f1abf65baf77a84ecb09af4840228c9bccf9a3aa9`.
The previous claim that guidance had been added, and its recorded hash, were
incorrect. This documentation-only repair adds the normative **Deterministic
test authoring** section to `AGENTS.md`, preserving all existing policy and
submission/deployment instructions. It covers explicit readiness/events,
injected clocks, no host elapsed assertions or sleep-to-wait, generous failure
bounds, host/order/filesystem independence, seeded randomness, no network calls,
`tmp_path`, worker/native-thread limits, prohibited flake workarounds and real
policy exercised through observable boundary models, with the D30 example.
These requirements are verified by inspecting the section and its additive
diff; behavioral test passes do not prove authoring guidance is present.

## Guidance repair verification

Repair starts from reviewed revision `cb487f68787c493c31b3bf3ae039051fa673a2ab`
and changes only `AGENTS.md` and this report. The cleanup implementation,
protected acceptance file and production code are unchanged; the earlier
five-serial-plus-xdist receipts below still identify the same test source.
No tests were added, removed or weakened in this repair. No new GPU run or
whole-repository suite was needed for this documentation-only change.

Rechecked the protected probes and both repository guards on x86 using:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_d30_cleanup_test_contract.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed and read retained stdout: **215 passed**, zero skips/uncollected
modules, NO CUDA, zero device allocations. `git diff --check` also passed.
O3 itself is established by reading the normative section, not these probes.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c9d6158d044ad13c27d4e87e82eb2134436650a76b52103462cff90090ec091b","size":1923}},"operation":"aa0fe525a5edc004b7846869bd4a0f3302e3aad9ecc9f29e58027b2aa2851024","receipt":"b12fbfd460f60a3df297fca71f59c7dd98aadcd1dc4252270b3d9ebc7795a0d4"}]

## Prior RED and implementation correction

Research supplied a protected-probe RED: all three tests failed on unavailable
ambient boundaries, operation
`75446047a9d4814e2de7e31d42fd28fef08cc2fcd41748863c4df00e159c7af2`
(as recorded in the brief). This is boundary-dependency evidence, not a claim to
have reproduced the intermittent historical release failure.

The first implementation sequence aborted on serial run 1 (2 failed, 4 passed):
the additive fake epoch accumulated floating-point drift, yielding
`1800000001.0000002` instead of `1800000001`. Fixed the model to derive epoch from
its single monotonic clock; did not retry unchanged failures or assemble a
streak from partial runs. This was an implementation-model defect, not a
production or acceptance defect.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"729a74a30d81728e8578bb4682cb54787b0d7982c01ad23c9970731fb261bfac","size":7226}},"operation":"c9fcaa8a5bbc0d1179fcb44d0de1ffbbf942abf8c876aa8a10b673597680b2e8","receipt":"e09c2b8d6d34ea3b3da94d88ac977b30ea4f634a403eb91d2fc3dc03f7535785"}]

## Final required sequence

Executed through plain `run_tests` on x86; retained stdout independently read.
All five consecutive serial invocations passed 6 tests each (three named tests
plus their three isolated protected probes; 16 deselected), followed by bounded
xdist passing the same 6. No failures, skips or uncollected modules in this
sequence. NO CUDA; zero device allocations. No implementer GPU run: these are
Python cleanup dependencies, not CUDA behavior; the configured dual-arm review
checks remain responsible for GPU-arm assessment. No GPU numerical claim.

Exact command:

```sh
selection='cooperative_child_exits_within_grace_without_a_kill or finite_cleanup_deadline_shortens_the_kill_grace or term_ignoring_child_group_is_killed_only_after_the_full_grace'; for attempt in 1 2 3 4 5; do printf 'D30 executed serial run %s\n' "$attempt"; PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_d30_memory_guard.py tests/test_d30_cleanup_test_contract.py -k "$selection" -q -n 0 --durations=10 --basetemp="$TMPDIR/pytest-$attempt" || exit $?; done; PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_d30_memory_guard.py tests/test_d30_cleanup_test_contract.py -k "$selection" -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest-xdist"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"da6e1f9015cc0dbac71b52b643e8b23b95c7059f57ecd5db681b75adae3c7c0b","size":6339}},"operation":"a9113d9bb47fc7a03ce240b4b541b40f4e6e7274c639fe44ffe62e8555234b1e","receipt":"dcfa7f9105bd94a7cf9105e0391111062d43c05d09a5af2a758f2472b5a61256"}]

## Broader verification

Final-source probes plus issue-reference guards (including this report):
6 passed, zero skipped/uncollected, NO CUDA, zero device allocations.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_d30_cleanup_test_contract.py tests/test_issue_refs.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"659bb51463419640a17dc204121a7038c9d52cf3d2cfff410cfc4aa9e0eb1a16","size":1148}},"operation":"10b25bf360c15fc1397b652175fe4c8828523b6e6c6fc3dbc284cfaa943e8ac0","receipt":"b450badbce8e7cc27a67ee0265366e1d179fce9935a21d95944d5878501b40d8"}]

Before final tightening of two epoch-end assertions from default `approx` to
exact equality, the complete target, protected probes and both repository guards
passed: 234 passed, zero skipped/uncollected, NO CUDA, zero device allocations.
Only those two assertions changed afterward; the final sequence above exercises
both. No full repository suite was run (left to integration).

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_d30_memory_guard.py tests/test_d30_cleanup_test_contract.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4f27ecaa28c90fa35cac08c136c8b51888197300ce07f65bbfe3a52b489973f9","size":2015}},"operation":"c3fc875335b4e0d9e582f2346e33baf683236b6caf571b8913af3ee7f343a7fa","receipt":"2a3c36723c7cc51b4c2da5767f12a7dbe644169b5ec2a16381baaad0f08e97b5"}]
