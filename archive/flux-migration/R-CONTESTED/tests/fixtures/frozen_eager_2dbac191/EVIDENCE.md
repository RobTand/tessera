# Frozen eager-selector test data

This fixture supports selector-wiring regression only, not GPU performance,
numerical qualification, or an OS-level network sandbox claim.

## Generation and independent comparison

Implementation independently ran `git archive
2dbac1910c88254d9c6391f02a34c4b07e516803 src pyproject.toml
tests/test_piece_major_reader_boundaries.py` against the supplied readable
Tessera bare repository (research/export source only). The export was 4,362,240
bytes with 126 regular files. SHA-256 was
`91f3a340e9e14068e448bbc150a702d16e913cdf70fc487e7acbe0a17fb272fb`,
identical to the independently supplied research digest. No source was changed.

Compression used Python `gzip.GzipFile(filename='', mtime=0, compresslevel=9)`;
compressed size is 1,236,761 bytes, SHA-256
`547591c0f4ca1709c3ea0f28b9acfd3125a8e060458e920b31b48da992b7858d`.
The manifest records the command and both digests. The test loader separately
pins commit and digests so editing manifest labels cannot authorize substitute
source. Checks need neither the original object database nor network access.

## Pre-fix evidence

The protected acceptance module retains the research receipt and actual
history-free x86 failure: `CalledProcessError` 128,
`fatal: not a tree object: 2dbac191...`, operation
`06a65f9b56207fddbebb7cb7c31ad8afa332894063d6114772d13c8bcef5d2db`.
This establishes the Git-history defect, not the contents of an inaccessible
historical gate traceback.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"47e3f5c62466debe30cfa20f65e543f20b5697c38fee3a1bba17e8a0e8ba6b95","size":6160}},"operation":"06a65f9b56207fddbebb7cb7c31ad8afa332894063d6114772d13c8bcef5d2db","receipt":"4e440e35dddd7aaad1c89a46231abe558f9767e4d7ad2479ee62ab0a6fff16cd"}]

## Prior implementation evidence (historical, superseded below)

The following admission and outcomes describe the recovered implementation,
not this replan. The protected harness has since been corrected by research.

### GPU admission decision (before prior submission)

First run only the original frozen target and the offline snapshot probe (two
cases), on arm `gpu`, expecting approximately one minute. It decides whether
the configured GPU interpreter actually executes the fresh selector child with
fixture-owned imports and retained sealed third-party paths, without optional
prerequisite skips. CPU cannot establish configured-GPU execution. No CUDA
allocation or timing measurement is needed by these selector-wiring tests.
After this minimal probe, run the complete eager-levers module plus acceptance
coverage on each configured arm (48 cases per arm), expecting about one minute
on GPU. After the first GPU run exposed the protected snapshot harness defect
below, the remaining GPU run will deselect that one known failure (47 cases),
not repeat it or claim it passed. The complete x86 run includes all 48 cases
plus the repository's two guard modules. This checks preservation of all surrounding eager controls and all
fixture failure cases; it is not a broader GPU sweep.

## Post-fix outcomes

Source base: `a2269f9b42310d18ff1250504177bf1ea25b2048`; implementation
changes only the eager-levers test and this fixture directory. The runtime pin,
production selectors, local client handling, fresh-child success marker, three
flag-reader assertions and MIN_M/MIN_M-1 boundary are unchanged. No normative
architecture claim changed, so architecture documentation was not edited.

All commands used `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 python -m pytest`, `-q -rs -n 2 --dist worksteal
--durations=10 --basetemp="$TMPDIR/pytest"`. No installations or private
interpreters were used.

1. GPU minimal population: original target plus
   `tests/test_frozen_eager_fixture.py::test_actual_target_executes_offline_without_history`.
   **1 passed, 1 failed**, zero skips, zero device allocations, 3.72 seconds.
   The original frozen target executed successfully. The snapshot acceptance
   case fails before historical imports: CUDA Torch 2.13.0+cu130 imports
   cuda.bindings -> cuda.pathfinder -> ctypes.util.find_library("dl"), which
   invokes `/sbin/ldconfig -p`. The acceptance audit hook raises
   `AssertionError: external command in frozen child`. This is dependency
   initialization, not Git export or networking from the test. The GPU surface
   reports Python 3.12.3 and NVIDIA GB10. This is a protected harness defect
   under the configured dependency stack, not grounds to suppress an import
   error, skip the probe, or change production code. Minimal fixture correction:
   initialize the genuine Torch prerequisite before installing the external-
   command audit hook, keeping the hook active for all frozen source imports
   and selector work (and retaining the network hook). Not applied here.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8d327c3c2c7d4e874e7f92b027b6926dc7be35b961cbd855a27e9d4e99e93939","size":7016}},"operation":"2b3307a46c8e40a9cd99909b5c4ade91f997fa2213eac4cd0b88a9c8b1334e45","receipt":"82a7746ed3b9e6f9d68a747ae34f64e087ffaea882968e7970742ba334380b4a"}]

2. Complete x86 population: `tests/test_frozen_eager_fixture.py
   tests/test_graph_attest_eager_levers.py tests/test_issue_refs.py
   tests/test_impacted_tests.py`. **260 passed**, zero skips, zero device
   allocations, 84.16 seconds. Includes all 48 acceptance/eager cases and both
   guards. CPU Torch 2.11.0+cpu, no CUDA. The offline snapshot actually executed,
   not a dependency skip; its import-origin checks passed. Identity, all six
   invalid inputs, and all four rebound unsafe archives passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c0812513808ec19b7c61ecbaa3d7189dd3c79e39955a883e1e1a181120e0e016","size":1846}},"operation":"0d56ac9c5604a92ddea8aeb00e741d145c1af162954dc0366330e3e06bb1bfe7","receipt":"df8ac23d4bcc7192bb563fe1a4dfd029c4b356490ac5d1137b4308ae14c98a72"}]

3. GPU remaining population: `tests/test_frozen_eager_fixture.py
   tests/test_graph_attest_eager_levers.py --deselect
   tests/test_frozen_eager_fixture.py::test_actual_target_executes_offline_without_history`.
   **47 passed**, zero skips, zero device allocations, 3.88 seconds. Includes
   the complete eager-levers module and all remaining acceptance cases. The
   known snapshot failure was not repeated. This is not a claim of a green
   complete GPU acceptance run: O3 remains impossible as written under that
   protected audit hook and configured Torch initialization.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e4743985749c02dc0c3b997dc434d8a9875c486fb3ca2e22e88f05ea82907c1d","size":1580}},"operation":"996918e7fe7b3875992a0ef61018d82bfcbcc1225213d68b67b5fe66ed73752b","receipt":"c82b42b6f8cdf66ae60a65c497381e25ac0fb1336aceae4c5b5fa8e2497fb5cf"}]

Final diff inspection confirmed no protected acceptance edits or production
changes. `git diff --check` passed. These observations establish fixture
identity and selector wiring only, not numerical or GPU performance coverage.

## Replan implementation verification

Recovered only eager-levers and the three fixture artifacts from
914e032fe0b23986cb761624ce0bae476219e8de; protected acceptance was untouched.
Current base: e0029e8279bf5b443001dbf2db3d6cf41773b954. Independently
compared decompressed fixture bytes with the exact git archive command above
using locally available history (no fetch): byte-for-byte equal, 4,362,240
bytes, 126 regular files and six directories. Source tree is
4d0a7c9abc8c699d29d3772aa99bdbaab1754cd7. Compressed bytes remain
1,236,761 bytes; both SHA256 values above were independently rechecked.
Historical Git is used only for this one-time provenance comparison, not checks.

### Current GPU admission (before submission)

Run the original frozen target and the corrected offline snapshot acceptance
(two cases) first on gpu. Expected budget: less than one GPU minute. This
decides genuine configured CUDA prerequisite initialization and actual fresh
selector execution without skips; CPU cannot prove the CUDA import boundary.
If successful, run all acceptance and eager-levers cases (48) on gpu, budget
less than one GPU minute, to verify surrounding controls without deselection.
These tests intentionally allocate no device tensors and claim no numerical
or performance qualification.

### Current outcomes

Commands used the common prefix and options documented above, including -rs
to expose skip reasons. No packages were installed and no checks queried
historical Git. The revised protected acceptance file is unchanged.

- GPU targeted command selected the original frozen target and
  tests/test_frozen_eager_fixture.py::test_actual_target_executes_offline_without_history.
  2 passed in 8.12s, no skips or uncollected modules, zero device allocations.
  Genuine Torch 2.13.0+cu130 on NVIDIA GB10; snapshot and original selector ran.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"45143c4a95e92096677384c2fd5513edffa8b0bbae1fd8d708ae214a230746cb","size":703}},"operation":"4614ed6168e22d077acd30f6d9d13c5ae777f25d82a02489ccd997bc717bd7f2","receipt":"a4977308a089c6b16cc8c4190fba989f63e1ae467a83120330ef3eec79bd38fc"}]

- GPU complete command selected tests/test_frozen_eager_fixture.py and
  tests/test_graph_attest_eager_levers.py, no deselections: 48 passed in 7.87s.
  Zero skips, uncollected modules and device allocations. Same CUDA stack.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"dfa768774d8023b341580e43250eed0eb484acf324fbddff100971d3cddbf83a","size":1556}},"operation":"9a16ddc75367849896173a6f4a7ea4343ba8458dda9061a3313b6bc14705640e","receipt":"546fcb94e40ee99e58add7c2c5bb44b28328063070da140cd4874983f0872152"}]

- x86 complete command selected those same modules plus tests/test_issue_refs.py
  and tests/test_impacted_tests.py: 260 passed in 136.98s. CPU Torch 2.11.0+cpu,
  no CUDA, zero skips, uncollected modules and device allocations. There are
  no CPU dependency skip reasons to report: both selector probes executed.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"77a5325e66bf0eebdf2d65b038aa001b40713ca46073c26e6b23a12f5c80a34a","size":1847}},"operation":"6caae82b851b6670c726435b7cc7c30f4aac0fd388792d1c8d44d92ac618cdd6","receipt":"e548811cda2eada3613004aba7ec7235013862a79a1b8c080cbed2d6176a0506"}]

Replan pre-fix retained stdout was read and confirms Git exit 128, not a tree
object (1 failed, no skips, 9.63s):
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6799611417554c42e3a0a86ee791737724dc1126af99d8dc21d4fbb6390d5333","size":6345}},"operation":"44edb389c119c626d539f796f2271db82675cd13a1b2f3b27b1d24d389b78463","receipt":"b21b8468daf55981b4e1aa3d3c42bcf6269759bc900a3f2c46cfbe9209a9185e"}]

Normal impacted selection was rerun with:
python3 tools/impacted_tests.py --ref e0029e8279bf5b443001dbf2db3d6cf41773b954 --json
It returned narrowed, 419 tests including both acceptance and eager-levers.
It sees the one tracked test edit; new fixture files remain untracked because
the platform Git index is read-only (git restore failed creating index.lock).
The coordinator must rerun selection once new artifacts are tracked; no staging
or commit was performed. Full integration and the broader selected population
are left to the coordinator, not claimed passed here. Final diff inspection
confirms production source, benchmark runtime pin and protected acceptance are
unchanged; git diff --check passes. No normative architecture claim changed.
