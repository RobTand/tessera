# W2 owner-local capture and replay outcomes

Implementation on base `17b3fdb396410698c9a142615345f046d2f94a5d` for tessera#1259.
Only the W2 instrumentation/serving portions of historical reviewed head
`30942802d69e1cd0081c93bcd38b58c6802f7d51` were reused. Its producer/verifier
changes were not imported. Protected acceptance files are unchanged.
The additive contract is described in ARCHITECTURE.md; strict admission is W3.
No runtime repin, serving-default change, external publication, GPU campaign,
device completion or compiled qualification is claimed.

## Failing before (supplied research action)

The research action ran:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_hooks.py tests/test_graph_owner_publication.py tests/test_glm53_graphs.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed research result: 17 failed, 51 passed, no skips/uncollected modules or
device allocations, NO CUDA. Missing owner evidence, pre-return class counters,
and the final-bound-only legacy diagnostic were exercised. The competing-writer
case then stopped at its missing-evidence prerequisite; this is not a claim of
an independently reproduced race on the base. The acceptance module docstring
retains the earlier standalone 51-pass baseline receipt too.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c0a42a83de9d4c9c8353decb4ecf6f7000d4e68402f005c6b2467b05d440e17f","size":33418}},"operation":"ccf28e1ea30dc15a79d6225715a9dcc05214feb7c4c0816b099a23637243f135","receipt":"12c27da7fa47d5f9a93157067217b8cf983fe63816394949ec9680570ca01b3f"}]

## Initial implementation and independent review

The same command passed 68 tests on x86 after selective historical reuse,
with NO CUDA, zero skips/uncollected modules/device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"28f5a58a40a829e0a9256b0470169e560974fcb38dad2f5e0f119858e2594cfb","size":1674}},"operation":"1bead038dc85e2b42f7b8b5549de4c1deffe370b685088398e6c33fb83531785","receipt":"7083b508d4b92eece334f4b25a3f2705c57f32c19efb00dccc896a4e4777d0c2"}]

A separate-session read-only review identified four additional lifecycle gaps:
nested dummy calls inherited serving scope, intermediate capture maps could
outlive pruning by an outer override, fork children lost the daemon publisher,
and branch-selection failures escaped the inner replay observer. Four small
regressions in test_graph_owner_lifecycle.py reproduce these through the real
import finder/wrappers without replacing accounting. Before repair, the command
below failed all four (16.57 seconds, zero skips, CPU-only):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest tests/test_graph_owner_lifecycle.py -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7c609f39d62d3e9eef0e00c2de6fa9cd7534fb6ff4958dfce86ba99baebb502b","size":6762}},"operation":"764001e0541a27e81f029b063d9fb33698d005a67bff26792712c501a97d9aea","receipt":"7a2552898695d12cb7cf76899463159579df2a047cff77ef05c988e7782901b4"}]

The repairs explicitly scope dummy calls, use final outer inventory while keeping
inner failures sticky, lazily restart a process-local publisher, and compose replay
observation outside branch selection while learning actual bounds from its inner
call. After repair:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_lifecycle.py tests/test_graph_owner_hooks.py tests/test_graph_owner_publication.py tests/test_glm53_graphs.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

72 passed in 16.38 seconds on x86, NO CUDA, zero skips/uncollected modules/device
allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6f0a14586bcd040a0d0cb6a55f5b79b591eb05f1395c9a660729403069362db4","size":1690}},"operation":"88b3797187fa16ded000bc9e51b322d50819b6671ee3fe950be64b6621faeb95","receipt":"0276ba5a22b769f704c4a083b66a52074beba4f1abc9194fb8b17bfe90b973d1"}]

## Configured GPU-arm compatibility check (planned before submission)

Decision: check this same minimal CPU-only 72-test selection in the configured
Python 3.12 pinned GPU image, to catch interpreter/dependency-specific observer
or serving-wrapper regressions. CPU Python 3.14 cannot establish that environment
compatibility. Expected executor time below two GPU-host minutes, no device
allocations. This is not a numerical/capture GPU campaign. No sweep or full suite
was requested. This rationale was recorded before submission.

Completed: the same 72-test command above passed on the configured GPU arm in
5.05 seconds, torch 2.13.0+cu130, GB10 visible, zero skips/uncollected modules and
**zero device allocations**. The check establishes host-code compatibility only.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"540b30a8f871d796ba6720a9542b64bb65bb4d8080410e5ac257256c6d38fcd0","size":1596}},"operation":"85832141591bd0d372b999406bfeee2c111ff1dff869c4750faeb758746e94f3","receipt":"4deae366a12cc8b6b49c1e597655c52b490a7354ab97b3c0dfaf8ea9ddefd517"}]

## Final review and guards

A second, separate-session read-only review inspected the repaired diff, all
owner lifecycle/publication tests, ARCHITECTURE.md and this evidence, and read
the retained before/after action results. Verdict: PASS, no remaining blocking
findings. It confirmed all four earlier gaps repaired, serialized publication,
actual final inventories, successful-return-only owner credit, fork reset and
publisher restart, and preservation of W2/W3 and host/device boundaries. Its only
nonblocking follow-up was to replace this document's then-planned GPU check with
the completed result and review disposition, now recorded here. No review test
rerun or GPU numerical validation is implied.

Cheap x86 guards:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

212 passed in 269.57 seconds; NO CUDA, zero skips/uncollected modules/device
allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f8fdff0bcb0ee90a44cbe5320cf7f378784391e64c09397651d2600e771f8a4d","size":1835}},"operation":"5e26ad9487c89bd875423db6d6fd69a3e34aa30b0968db80278683b92550f55c","receipt":"dba0de2e94ae6154edba74104dd98c0919b79040b4888d4f97d47785980d3c4f"}]

`python3 tools/impacted_tests.py --ref 17b3fdb396410698c9a142615345f046d2f94a5d`
ran in this worktree: verdict `narrowed`, five changed paths, 499 selected test
files (including conservative unnamed-path consumers). That broad selection
was not run; verification here is the targeted contract/serving selection and
cheap guards above. Full merge-result integration belongs to the coordinator.
Final `git diff --check` passed, and protected acceptance files have no diff.
