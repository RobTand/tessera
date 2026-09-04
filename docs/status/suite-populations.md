# Suite populations

One row per arm per `tools/merge_suite.py` run. The two arms of a run are
adjacent on purpose: a pass count means nothing without the device population
it was measured on, and this file exists so neither can be read without the
other (tessera#112).

`master head?` is whether the commit under test was master's tip at submit
time. `yes` is a merge receipt; `no` is a branch's own run; `unknown` means no
master ref resolved in that checkout and the question was not answered.

`exit` is the status the submitting process observed. `not observed` means the
receipt was assembled after the fact from what the run published (`--resume`):
the failure count in that row is still a fact, but a zero in it does not make
the row green, because a suite can exit non-zero after a clean summary.

| measured (UTC) | commit | master head? | arm | device | passed | failed | skipped | not collected | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-04T07:14:07Z | `e61974c0c40c` | no | gpu | no population published | -- | -- | -- | 0 | not observed |
| 2026-09-04T07:04:59Z | `e61974c0c40c` | no | x86 | torch 2.11.0+cpu reports no CUDA device | 1389 | 1 | 499 | 0 | not observed |
