# #1258 / #1270: complete independent equality populations

## Implementation and boundary

Reused the applicable runner, producer, shared gate and protocol implementation
from `abd0cd809f4364fa727f233d5fb9168b871e6d97`, also compared with
`80ce80fb43217c4fd9ee64468c7a2619d20859f3`. The protocol bytes and the three
implementation-file changes agree with those heads. Unrelated changes from those
heads were not applied. The architecture addition was applied without reverting
current-main flag documentation. Neither protected acceptance test was edited.

`graph_equality.CASES` is the authoritative geometry; all choice counts derive
from it. The short-context loader refuses incomplete or substituted populations,
invalid geometry, duplicate indices, invalid raw logprobs and digest conflicts.
The shared finish/verify gate checks represented population, bounded integral
nonboolean counters and independent role/pass identities. Eager membership can
vary; whole-batch signature membership, cross-case pooling and pooling across
eager outcomes remain unchanged. Text-keyed top-20 maps can have 1–21 keys;
recorded maps are still compared exactly. Long-context screens remain separate.
No replay-evidence expansion, consumer admission, identity seal, runtime pin or
serving-default change is included. This is receipt-contract regression evidence,
not compiled qualification, GPU numerical evidence or a new served measurement.

## Retention and retrieval

The four successful evidence-retention operations below have verified retained
stdout available through `read_test_artifact(operation=..., name="stdout",
offset=..., length=...)` (byte reads up to 32000). Each stdout contains the complete
pytest log, complete per-case JUnit XML and raw population JSON. Extract XML/JSON
between `RETAINED_RAW_BEGIN <name>` and `RETAINED_RAW_END <name>`; no reconstruction
from aggregate totals is necessary. `receipt` is separately retrievable and binds
provider/action identity, outcome and stdout digest. Temporary XML/JSON paths are
not claimed as archival locators: the retained stdout and its receipt are the
archive. Population source verification is distinct from provider/action binding;
the population JSON is not an action identity seal.

Before artifacts were supplied by the fresh research run and reused, not rerun
or reconstructed against historical heads. This session inspected their retained
stdout and population JSON; the independent reviewer inspected all four retained
operations, including XML. Both before runs had **48 intended failures / 58 passes**,
exit 1, no collection errors or skips, and zero tests with observed CUDA allocator
allocations (a floor metric, not proof of absence of all device activity). Regressions first
produce and verify complete valid evidence, then attack the population. Examples
include import-time runner failure and falsely attested partial, forged or invalid
matching-map evidence, rather than missing setup.

## Commands

The common three-file pytest invocation (before and after) is:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_receipt_population.py tests/test_graph_receipt.py tests/test_graph_attest_tp2.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Before research added `--junitxml="$TMPDIR/issue1258-before.xml"` and
`--surface-json="$TMPDIR/issue1258-before-surface.json"` and printed the raw files
into retained stdout. The original research wrapper's full shell spelling is not
reconstructed here. Its observable output is retained verbatim.

Exact passing-after command submitted with `run_tests`, once on the default x86
arm and once with `arm="gpu"`:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_receipt_population.py tests/test_graph_receipt.py tests/test_graph_attest_tp2.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest" --junitxml="$TMPDIR/issue1258-after.xml" --surface-json="$TMPDIR/issue1258-population.json"; status=$?; python -c 'import os,pathlib; print("ACTION_ENV", {k:v for k,v in os.environ.items() if any(s in k for s in ("ACTION", "SNAPSHOT"))}); [(print("RETAINED_RAW_BEGIN",n),print((pathlib.Path(os.environ["TMPDIR"])/n).read_text()),print("RETAINED_RAW_END",n)) for n in ("issue1258-after.xml","issue1258-population.json")]'; exit "$status"
```

These are admitted PrismaBuild actions, not nested launchers or local pytest
substitutions. No private interpreter, environment repair or install was used.
The combined selection includes all O1–O4 targeted cases and retained receipt/TP2
behavior tests. Both after runs passed **106 tests**, exit 0: x86 87.88s, GPU 17.30s.
Errors, failures, skips, xfails, xpasses and uncollected modules were all zero;
skip reasons and artifact skips were empty. Both used two workstealing workers.
The CPU arm reports torch 2.11.0+cpu, no CUDA. The GPU arm reports torch
2.11.0+cu130, one NVIDIA GB10. **Both report zero tests with observed CUDA
allocator allocations**; the population marks this as a floor metric, not proof
of absence of all device activity.

An initial request naming `arm="x86"` was refused without execution (the default
arm must be selected by omission). The first default-arm attempt then timed out
in setup with exit 124, before the command ran. That is unavailable infrastructure,
not a failing assertion or a pass. The unchanged retry passed as recorded above.
Its setup-only operation is `e7c1842ec8dedaa58e3901cdf76da19800801184885541e39ba633792ce8b66a`.

## Action and source bindings

All four population JSON records report source verification `verified`, identical
entry/final source hashes, measurement-span agreement true, both gw0/gw1 agreeing,
and no excluded metadata. These measurements bind source within each run; they
are not reconstructed historical agreement. After source snapshots differ between
arms because action snapshots are independently generated. Their Git objects are
not available in this checkout for a direct snapshot-to-working-tree comparison.
Only this evidence document was added after the passing runs; implementation and
architecture files were unchanged afterwards.

| Run | Provider/action identity | Snapshot | Source SHA256 |
| --- | --- | --- | --- |
| Before x86 | `cdf1ac7169ab5df302d8044292687c4efc46225ed5e374f1248ac7b2db6bb5b3` | `a2cb234eeb56acca42ab9a390340d322cf2175a5` | `d31431df6760adf100891e6fb39c8e48381cb33cc269fe76f74f0bdc0a64ce2c` |
| Before GPU | `521f925952518b9b4784e0c29e2b3190eb821613ec78c8065ad59f68842198ff` | `0a5448e12b9919af01ae78f1c00b9a4ea0d25db6` | `7209916c5037a26881700d310e057489dfb2407418a9998805d9660abfa085ea` |
| After x86 | `5a0738c1ccd6a573270572b09c33f04bc7ffe8c1802fb74c48c8cdb2226a661c` | `fb58ef1f756bcdb8f716137fb7dbbb1a11ae5892` | `1de5cd5ea698e015649b2f20a601efcaed8931593fba7b9e315398c8e48707aa` |
| After GPU | `fa777cfa58b595e4871f363250870e222716796cd07eefd3db1405015509acb8` | `588e6e0df836d712f3ec72fe2ed47480b9938910` | `2cb09c03d1c454501ce0a1b39a992aa00e6ce85fa11019e0194fa1b4100b02a4` |

### Failing-before retained evidence

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ee8ffc7c7c184842e6bb095528d11fdbed1e76b250096587d525c771c5f63432","size":443683}},"operation":"855f0bfcbaed1902ccdbbfde88e374b5c7ec9efb2dda747962aaacc81d7a2018","receipt":"8685eb78ab557c12128cafcf9b176f9740800e6230a0641c9ade03c9d0d52b69"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"865fdce57509c9d0c8d3af4b0283932cc4ae6adb31e0b3d1abd02af5107705db","size":500382}},"operation":"b2abebd45103cd121bab06bc74da5191b567c6ee95e325973a1bdc52d7e22fe0","receipt":"f12518f45c90395ef158b632b1b821805260e90e6f088d1bc65a62bbc69b0a78"}]

### Passing-after retained evidence

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6e3ab045cb2a3c8bf328e2f9582fbfc1b824b6bb3e7eece315aa8e69418cc8c7","size":19300}},"operation":"074f5e5d4f9812363f76f3d46e5ef854209c0af9e300f7ccfea45fcb2d1b7085","receipt":"fe9a7df1d3a41408e2c4f911362628d0215f480aa7d39aeb766f4d4d7a92452a"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"bb94a248b90fd2c42dd83165b36ed8be4ead9123f18db27d5e93686c04595366","size":19226}},"operation":"5baae58f551f96817dbb3cb1b3a6a722309faf3f11273b45d0ad9fdc2bf1264e","receipt":"4b048a5a0a5dbd1f2d4321ecf2092905dd36d847a439f458e46f4a4cbf7c70e3"}]

## Separate-session implementation review

A fresh read-only general-purpose subagent session inspected AGENTS.md, the final
implementation diff and new protocol owner, the prior heads, the membership
predicate, protected-test status and all four retained stdout/XML/population
artifacts. It found **no blocking implementation defect** and concluded that the
contract and both-arm regression evidence support the objectives. It did not edit
files or duplicate the tests. It confirmed the protocol SHA256
`061db6792190816add7458c44557932acf71115cdc0446b21ce3619b8a591dc6`
agrees with both prior heads and noted that the new protocol file must be included
in the delivered change. The reviewer independently recorded the same four source
bindings above. Its limitations were unavailable snapshot Git objects, separation
of action-environment reporting from population source verification, and zero tests
with observed CUDA allocator allocations rather than compiled qualification. A
second read-only session checked this final evidence document and architecture
addition against the retained receipts and population records, approving with a
minor request to qualify the allocation metric; that wording is corrected here.
Exact command spelling, absence of environment repair and unchanged implementation
after measurement are session-provenance statements, not facts independently
established by stdout. Workflow transitions and publication remain with TrustLadder.

Final `git diff --check` passed and the protected-test working diff is empty.
`python3 tools/impacted_tests.py --ref 9d418e8c770f84aaf1ce498e539f06ee06372da8...HEAD`
was run in the worktree (first attempt timed out at 20s; retry completed). It
reported `narrowed (2 changed, 417 tests)` for that committed comparison, with
outside-tree/unnamed-directory dependency limitations. This committed comparison
does not represent the uncommitted repair diff; no 417-file or full integration
run is claimed. The targeted three-file evidence above is the validation for this
repair; full merge-result qualification remains the coordinator's responsibility.

Historical disclosures are not relabelled by these fresh results: the original
historical before action `01688c4fb1e77f144e4a4dbec18b28df5847474d54e3e26d4c9c98efad77acad`
reported unknown source verification and false measurement-span agreement.
Historical unknown-source passing runs remain unknown-source; displayed matching
worker hashes do not reconstruct agreement. Prior attempts' inability to locate
durable raw per-case artifacts remains a historical limitation. The fresh retained
operations above, not those incomplete historical records, supply this repair's
individually retrievable before/after evidence.
