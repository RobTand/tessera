# W3 qualification: existing 1320 handoff remains PENDING-GPU

This is an incomplete-handoff record for the task concerning Tessera issue 1414,
not a qualification packet, approval, new campaign, or new blocker issue.
`qualification.json` is intentionally absent: no authentic owner-supplied packet
was verified. O1–O4 remain unmet. Native execution and qualified status remain
held; no native build, protected regression, or sweep was attempted.

## Evidence acquisition and limits

Inspected checkout: `030997ef99f81c6e222d61e9284414d0f0fea0dd`.
The protected `tests/test_w3_1320_qualification_packet.py` requires retained
record bytes and digest-bound tuple associations. It cannot authenticate
approvals, custody accessibility, chronology, numerical thresholds or independent
acceptance. It was not modified.

A bounded Markdown search of this checkout's documentation for `1320` returned
only incidental numeric/hash matches, not the owner handoff. A read-only research
pass over shared measurement records did not verify the requested packet;
some shared-storage searches timed out, so this is not an exhaustive absence
claim. The historical issue snapshot is not evidence of current owner delivery.
The available Strix Halo tester protocol explicitly identifies a different
protocol origin and does not establish equivalence to the required W3 matrix.

Read-only owner-record retrieval was attempted with `gh issue view` for Tessera
issues 1320, 1492, 1493 and 1484 (body/comments requested). All four invocations
failed before retrieval: `gh: Permission denied`, exit 126. No owner contact,
overlap agreement, current issue state or approval was established by that
attempt. The supplied titles for 1492 (native-eager), 1493 (compiled W3) and
1484 (qualification-owner delivery) are leads, not proof of six-case equivalence.
No issue was created or updated and no alternate campaign was started.

## Remaining delivery through the existing owners

The supplied issue discussion names **v6 Kernels Lead** as execution/evidence
owner and **v6 Review and Integration Lead** as independent reviewer. These are
attributed role names, not newly authenticated signatories. Continue through the
existing 1320 ownership chain:

- **O1:** locate the successful setup-qualification action and adoption note;
  obtain a fresh decision binding every actual source commit/content digest,
  native binary, exact image/toolchain, device, encoded input, geometry and
  launch identity. Retain authoritative custody manifests and independently
  verify actual readability. Obtain independent prerequisite review before
  missing native work. None of these records was verified here.
- **O2:** after that review, execute only the authorized protected regression
  first: rows=416, columns=512, q256=1024, m=256. The supplied estimate is about
  two GPU minutes, not a measured duration. Retain its actual native command,
  successful completion receipt, measured duration, population, numerical,
  replay, frozen-storage and no-redecode evidence, and review/pilot/sweep
  chronology before considering a sweep. No exact command was recovered.
- **O3:** obtain the authoritative W3 protocol, six distinct required case
  identities and acceptance thresholds from those owners. Establish overlap
  with existing delivery work using identities and protocol, not titles. Reuse
  matching retained results only after equivalence is established; retain any
  authorized missing results and genuine independent post-run acceptance.
  No case IDs, invocation, results or reviews are invented here.
- **O4:** independently authenticate authority, freshness, accessible custody,
  exact protocol agreement, chronology and scientific sufficiency. CPU JSON
  assertions cannot discharge these obligations or confer qualified status.

The eventual small `qualification.json` and supporting record references must
use repository-relative paths and SHA-256 digests. Large artifacts need not be
copied into this checkout. This note is not a substitute for that packet and
must not be referenced as a successful prerequisite or execution receipt.

## Current verification

Repository guards:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: **212 passed in 138.44s**, NO CUDA, 0 skips, 0 uncollected modules,
0 tests allocated on device. These are repository guards, not native coverage.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"747b642f1b630050dedef31a5a907955601198a9da70f4c634ba1a15ab25fb4a","size":1781}},"operation":"ef82753e7485e97620feaa72bdf07af5b09e2a27dd8f95351282aa94a8189211","receipt":"29e54fbb24f09b5957072dcaff0fecb2eb83e1a96490e8f7cbfd6d4ccc4dbf62"}]

Command (all three protected acceptance cases):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_w3_1320_qualification_packet.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: **3 failed in 12.62s**, all at `Missing owner-supplied Tessera 1320 W3
qualification packet`. Population: **NO CUDA**, 0 skipped tests, 0 uncollected
modules, 0 tests allocated on device. This is an expected evidence-gap failure,
not a test-harness defect or CUDA qualification. No GPU-arm check was requested:
these structural checks already establish missing evidence on CPU; spending GPU
time cannot supply the absent owner approvals or protocol.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c5884733b1ae7543793f60b481f1ded32e396bc6534435e0e4471aaa57a855ca","size":4616}},"operation":"192ee0978d52994107a321d2dd6c148640c3a8e73c7df59b7f16439606f81aa0","receipt":"6b00e6f433d6e07aa17d9704e0c98236ed813d3173d8f44134d68b8081793951"}]
