# Bound Checker CPU Replay

The current repository checker replayed the original #688 singleton through
PB without attaching a GPU or changing its original request, job, producer
identity, panel or samples. It created a distinct source-bound CPU job using
the current worker, executed the installed b40 contract preflight, and ran
the existing full external-panel validator before emitting the observation.

Actual action `cf0484b46363408ced3c50c3b1f39bd7e064735f3cc2d940126a164e5d0122c4`,
publication `1790970739.694331`, attempt `1`, finished rc0 on Sparky.
CAS receipt: `a205b6544942dc696976499ca1f84528dad6ef1b95309dffcfa753da643b51c6`.
Payload: `502ecad36268f4ed59d35ab053176ab3f785b3a4d2b7a0317297d0c33e9c6de4`,
7,237 bytes. The source snapshot selected commit
`d28f1dfef1c0187ab5744dcd7c65685168ded3f7` from parent
`c7f6d1aa1a1a57d31936271931d8b859823d94be`; its input is 12,945,026 bytes at
`718f4e907470eb580cff8d7468de3fd58b75ace75f4b5bbbecb201e6bad72d1d`.

The checker and current CPU worker executed from the sealed PB checkout
mounted read-only at `/checker`. The original producer remained a separate
read-only `/producer` mount. The immutable original image ran as uid/gid
1000:1000, with a 6 GiB memory cap, native threads one, no GPU flag, empty
CUDA visibility and the Docker shim's PB-assigned CPU affinity. One CPU and
6 GiB were admitted. The child returned preflight rc0 and did not initialize
CUDA. The original runtime image, source commit, installed RECORD and raw
contract were validated; no new device timing was taken.

The original panel SHA-256 remains
`fd5c2b4f24ffdae8058d28c39f38ada9f943eb8836d0433ecdcc584e36836129`, and request
SHA-256 remains `3d252977205a7938b641a0215ac59f185526ba5f400f65b7c7010f9acbbfa577`.
The measurement producer remains `8eb3c05174c292c5c1cf1e0d32f72802e6d2c8cf`.
The observation and actual current CPU job/phase/result are under
`/mnt/shared/tessera-measurements/pact-observation-bound-20261002-sol/r2/`.
The existing 30 samples, five warmups and median `0.07507199794054031` ms
describe only `dense 256x256 E4M3 R896 M512`, TP1, eager/resident, sm121.
Energy remains HOLD.

## Validation And Negative Results

PB `f4b48f6453211ca190e5d5f841dcddc42ff0419826426f0a3499c826efe10a4f`
ran `pytest -q -n 4 tests/test_native_shape_time_application.py
tests/test_native_timing_panel.py` on dl380g10: **102 passed, zero skips,
zero uncollected modules, zero CUDA allocations**. CAS receipt:
`1aaeeae7d0418c53e87a763205a264974af4e54c442027c18a42015ba9e610a9`.
The CPU interpreter was the separately provisioned SDK4/b40 environment,
Python 3.14.4 and CPU torch 2.11.0, with four CPUs, 6 GiB and native threads
one. This count does not cover the CUDA-gated surface.

Controls reject changed published-helper bytes, writable or escaping-symlink
helpers and metadata drift during an owned read. A historical immutable helper
whose held bytes equal the published owner passes. The original and replay
jobs retain independently bound worker/job identities and must agree on the
original request, producer and wire roles; wrong-worker and wrong-producer
controls refuse. A saved result still cannot create the private validation
issuer token.

Earlier actual PB outcomes are retained as bounded negative evidence:
`289ff4f1e93b` refused because an immutable historical helper path differed
from the current published path, although both files had the same authenticated
bytes. `76827b05abc3` exposed the output-directory uid mismatch; the launcher
now uses the existing uid/gid contract instead of changing shared permissions.
`a1d1c44642f9` passed actual installed preflight but refused the old full-result
equality between original and current CPU jobs. The validator now joins their
shared original inputs while retaining both distinct execution identities.
Original bytes were preserved throughout. Fixture failures from these changed
identities were corrected and are superseded by the 102-pass result.

PQ's independently source-pinned SDK4 consumer then converted, reloaded and
admitted exactly one TP1 row. Its final real trace action
`49e3447f8af8ebcc353790a9fe3ea2a80af6f405cdc6609d30b9b7230f6f57eb`
also refused TP2 and a fully forged panel/observation/sample/proof set with
matching hashes and no actual checker receipt. Root retains source acceptance
and merge authority. This closes the bounded handoff acceptance, not parent
#688, full PACT, quality, compiled, placement or served-latency qualification.

Explicit compile checks for the panel tool, worker, timing validator and
application controls passed through PB action
`37a61fe4f0df933044f8b07bb71cb0fa5e6ce9ab22fa3477a96cca2ff8972863`,
CAS receipt `cb09d4f7283e011e624e1849cb4a57529aa3ca8cdd1167b8f42b187df45ad33c`.
It reserved one CPU and 1 GiB on dl380g10. A prior submission stopped before
publication because the results document was added during snapshotting;
it produced no action or execution result and is superseded by this check.

## Root Review Rebase

The delivery branch was ordinarily rebased onto master
`5133f84625c4deb34fc5812b3132cb40611aac8d`, including MLA pass-buffer PR #854.
Only the architecture provenance insertion conflicted; both the MLA and
shape-checker blocks were retained. The three producer modules and both
domain test files have identical Git blobs before and after the rebase:

| File | Git blob |
| --- | --- |
| `src/tessera/serving/timing_panel.py` | `3e6ec647a89bd28474ea5b93c3f338f1c00686a0` |
| `tools/tessera_shape_time_panel.py` | `cb3b15fbf9e2f3f174dc59f7aa3f9bf168bb0497` |
| `tools/tessera_shape_time_worker.py` | `057e8db099da298e0a00726ad7fa48273394843b` |
| `tests/test_native_shape_time_application.py` | `b2887f0f4a572f272c2397d5bab4e86c16cdc2fc` |
| `tests/test_native_timing_panel.py` | `234ef09a9a6d3800dd267e1444c8600a508d92df` |

The prior 102-test and compile receipts above remain attributable to their
original sealed source. They were not rerun solely for this rebase. No new
installed-runtime or GPU qualification is asserted. PQ's independent checker
config still selects the original qualified `d28f1df...` snapshot with parent
`c7f6d1a...`; rebasing this delivery branch does not repoint that approval.

The existing `tools/refresh_issues.py` refreshed the actual issue snapshot.
PB action `ac0eddb39786c8711a039368d5f2f385918c52c4d2cef0bf1b1723add3282089`
ran `pytest -q -n 2 tests/test_issue_refs.py tests/test_refresh_issues.py` on
dl380g10: rc0, eight passed, zero skips or missing modules and no CUDA
allocations. Its CAS receipt is
`cccf7bb9cd741db9d62d5d94bc84b27c202060a64448d7f5d3b790f7fcc0eeb7`.
Published pbrun reserved two CPUs and 3 GiB with native threads one and CUDA
withheld, using `/home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python`.
Normative prose also corrects the stale b40/v42 spelling to b40/v45 and states
the separate current-worker replay job instead of pretending to reuse the
original worker job.

Master advanced during delivery to terminal-barrier merge
`0356c2d8e9585e271eda9c6387a3548c95de8b5e` (PR #861). A second ordinary
rebase retains that change too. Its only conflict was the issue snapshot,
resolved by rerunning the same GitHub-backed refresh owner. The three producer
and two domain-test blobs above remain identical; no prior receipt is
restamped onto this new source and no GPU or full domain suite was repeated.
