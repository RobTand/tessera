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
receipt's own commit is the best available guess. Rows of one run with two
commits are two measurements, not one merge receipt.

`exit` is the status the submitting process observed. `not observed` means the
receipt was assembled after the fact from what the run published (`--resume`):
the failure count in that row is still a fact, but a zero in it does not make
the row green, because a suite can exit non-zero after a clean summary.

`device` distinguishes three absences that are not the same thing. `not
submitted in this run` is an arm nobody asked for. `no population published`
is an arm that was submitted and returned nothing -- refused, never placed, or
dead before its summary. A device string is a measurement.

| measured (UTC) | commit | master head? | arm | device | passed | failed | skipped | not collected | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -- | `e61974c0c40c` (assumed) | no | gpu | no population published | -- | -- | -- | 0 | not observed |
| 2026-09-04T07:04:59Z | `e61974c0c40c` (assumed) | no | x86 | torch 2.11.0+cpu reports no CUDA device | 1389 | 1 | 499 | 0 | not observed |
| 2026-09-04T07:40:15Z | `fbac91b496dd` (assumed) | no | x86 | torch 2.11.0+cpu reports no CUDA device | 1398 | 0 | 499 | 0 | 0 |
| 2026-09-04T08:11:37Z | `dee1aa975212` | no | x86 | torch 2.11.0+cpu reports no CUDA device | 1406 | 0 | 499 | 0 | 0 |
