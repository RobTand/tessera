# Suite populations

One row per arm per `tools/merge_suite.py` run, and **every** arm gets a row:
an arm the run did not submit is written as `not submitted in this run` rather
than left out. A pass count means nothing without the device population it was
measured on, and a lone row is a result quoted without its counterpart --
exactly the misreading tessera#112 is about. So the rows of a run always name
both populations, even when only one was measured.

Rows above 2026-09-04T08:11 predate that rule and can be lone: a run submitted
with `--arm x86` wrote one row and said nothing about the GPU population. Read
a lone row there as "the other arm was not recorded", not as a whole result.

`master head?` is whether the commit under test was master's tip at submit
time. `yes` is a merge receipt; `no` is a branch's own run; `unknown` means no
master ref resolved in that checkout and the question was not answered.

`commit` is the tree that arm's own run reported measuring, which is not
always the tree the receipt was assembled against: the arms are separate
processes on separate boxes, and a queued arm can place after the checkout has
moved. `(assumed)` marks a row whose run predates that field, where the
receipt's own commit is the best available guess. PrismaBuild's parentless
snapshot commits also differ when only its verified action-specific closure
stamp differs. New populations retain that raw commit and independently hash
the effective source; the JSON receipt's
`commits_measured.effective_source.agree` distinguishes equivalent source from
different source, and is unknown for legacy or unverifiable populations.
`source <hash>` beside a row's snapshot commit names that verified source.
Pass counts alone do not establish a same-source merge check.

`exit` is the status the submitting process observed. `0 (pool)` is a status
this program did not watch and did not guess: the run was resumed, and the
number is the one PrismaBuild's own worker recorded for the action that wrote
that population. `not observed` is the remaining case -- a resumed row with no
single finished pool action behind it -- and there the failure count is still a
fact while a zero in it does not make the row green, because a suite can exit
non-zero after a clean summary.

`mode` is how that arm ran, and it changes what the row means as much as the
device does. The GPU arm is always `serial` -- its workers would share one
device and its CUDA venv has no xdist -- while the x86 arm runs `-n <cpus>`, so
two rows of one commit can differ by more than the box. On `d11dc01` the gpu
row is green and the x86 row is red at 5 failed, and those five are an
`-n`-only defect in the suite's own conftest, not a CUDA one. Match a pair by
`commit`, then read `mode` and the failures before attributing the difference
to the device. `--` is a row whose mode was not recorded: rows above
2026-09-04T10:30 predate the column, and a resumed row can only name a mode
when exactly one finished pool action wrote its population, since the mode is
read out of that action's own command.

`device` distinguishes three absences that are not the same thing. `not
submitted in this run` is an arm nobody asked for. `no population published`
is an arm that was submitted and returned nothing -- refused, never placed, or
dead before its summary. A device string is a measurement.

| measured (UTC) | commit | master head? | arm | mode | device | passed | failed | skipped | not collected | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -- | `e61974c0c40c` (assumed) | no | gpu | -- | no population published | -- | -- | -- | 0 | not observed |
| 2026-09-04T07:04:59Z | `e61974c0c40c` (assumed) | no | x86 | -- | torch 2.11.0+cpu reports no CUDA device | 1389 | 1 | 499 | 0 | not observed |
| 2026-09-04T07:40:15Z | `fbac91b496dd` (assumed) | no | x86 | -- | torch 2.11.0+cpu reports no CUDA device | 1398 | 0 | 499 | 0 | 0 |
| 2026-09-04T08:11:37Z | `dee1aa975212` | no | x86 | -- | torch 2.11.0+cpu reports no CUDA device | 1406 | 0 | 499 | 0 | 0 |
| 2026-09-04T09:13:34Z | `d11dc014f02a` | no | gpu | -- | torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10 | 1910 | 0 | 13 | 0 | 0 (pool) |
| -- | -- | -- | x86 | -- | not submitted in this run | -- | -- | -- | -- | -- |
| 2026-09-04T09:36:04Z | `d11dc014f02a` | no | x86 | -- | torch 2.11.0+cpu reports no CUDA device | 1406 | 5 | 499 | 0 | not observed |
| -- | -- | -- | gpu | -- | not submitted in this run | -- | -- | -- | -- | -- |
| 2026-09-04T09:24:38Z | `82f004772b8f` | no | x86 | -- | torch 2.11.0+cpu reports no CUDA device | 1536 | 5 | 503 | 0 | not observed |
| -- | -- | -- | gpu | -- | not submitted in this run | -- | -- | -- | -- | -- |
| 2026-09-04T14:34:06Z | `66f648017f44` | yes | gpu | serial | torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10 | 2121 | 0 | 13 | 0 | 0 (pool) |
| 2026-09-04T14:35:57Z | `66f648017f44` | yes | x86 | -n 8 | torch 2.11.0+cpu reports no CUDA device | 1604 | 0 | 517 | 0 | 0 (pool) |
| -- | `6072e572e749` (assumed) | no | gpu | serial | no population published | -- | -- | -- | 0 | -- |
| -- | `6072e572e749` (assumed) | no | x86 | -n 24 | no population published | -- | -- | -- | 0 | -- |
| 2026-09-04T22:25:35Z | `86aa47409242`<br>source `40f8d9ba0962` | unknown | gpu | serial | torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10 | 2535 | 0 | 12 | 0 | 0 |
| 2026-09-04T22:27:09Z | `08d1571d7598`<br>source `40f8d9ba0962` | unknown | x86 | -n 24 | torch 2.11.0+cpu reports no CUDA device | 2016 | 0 | 518 | 0 | 0 |


## Direct selected validation, 2026-09-05

Issue 198 and branch-cleanup checks, not a full release suite. Both arms are
aarch64 on the local GB10 host; no x86 or PrismaBuild arm was submitted.
Both publish `8c21637ae168` and verified source hash
`d45278d3b27b17b1da812193ac9e91623c7a8ebfd3e9b9fe8fe4e5012d05667d`.
[Full receipt, scope, commands and verbatim skip reasons](../measurements/branch-audit-validation-2026-09-05.md).

| arm | mode | device | selected files | passed | failed | skipped | xfailed | uncollected | exit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | -n 8, worksteal | torch 2.10.0+cpu, no CUDA | 126 | 1945 | 0 | 505 | 0 | 0 | 0 |
| GPU | -n 12, worksteal, strict-CUDA | torch 2.13.0+cu130, GB10 | 129 | 2485 | 0 | 5 | 1 | 0 | 0 |

GPU calls allocated in 472 tests; neither arm skipped for absent box artifacts.
The intentionally interrupted serial attempts are preserved separately and
are not success receipts.
