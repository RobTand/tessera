# Original batch profiler #402 — retired diagnostic

Disposition on 2026-09-07: **retire the run-specific diagnostic and retain its
original outputs as bounded evidence**. Recommend closing [issue #402](https://github.com/RobTand/tessera/issues/402)
as **not planned**, not as a repaired profiler. No re-encode or replacement
profile of this batch is required to retire its unused harness. Root owns the
issue closure and merge. This record supersedes the earlier review's request
to repair an active harness; it does not change any historical output.

## Why it is no longer live work

The inspected Tessera master is `a29dbec6cf2f720f1cf91eda26cbbe934a737fcf`.
Neither `experiments/profile_original_campaign_batch.py` nor
`experiments/run_original_campaign_batch_profile.py` exists there. A tracked
text search for both names and their `encoder-profile-original382` root finds
no caller on that master. The original code commit
`f9c19e962960c648adbf3bad0604af3722c58a17` was reachable only through the local
`codex/original-wire-checkpoint` branch before this audit added recovery refs;
no fetched remote ref contained it, and origin has no branch of that name.
The local branch is clean at `c25281e148828938cb09f41594b9510baee04b1b` and
is retained unchanged.

Within that branch, the only reference to the profiling program is its paired
launcher at line 31. The launcher hard-codes `profile-01` and creates it with
`exist_ok=False` (lines 12–13); that directory already contains the historical
run. It is not a configurable production entry point. The current PB
`ready`/`claimed` records have no references to either script or its output
root. These checks establish the inspected live scope, not absence from every
possible external private script.

Merged [#407](https://github.com/RobTand/tessera/pull/407), merge commit
`e09d9a60e74039dac169bb6e5796c6ece41c671b`, delivered eight explicit original
checkpoint/generation scripts and their origin tests. It did **not** deliver
this profiling pair. The new full-reference path consumes the retained
campaign originals through the closed cached-unit exporter; it does not
re-encode the 32-expert diagnostic batch or consume its cProfile/Netdata
success flag. Therefore the diagnostic has no live consumer to repair under
the current campaign authorization.

## What the historical pass does and does not mean

PB `675ccae44cc9459b02cb15e64ba31e745a7213afd43dd113be603f050cf8df62`
executed snapshot `9f4a2574f379e51bafb3034e1dae8326e9430e00` on Sparky.
The original `environment.json` keeps `status: passed` unchanged. That status
is valid **only for the checked workload/wire correctness**: all 32 output
wires match their original hashes, batch size is 32, and the recorded source
and input identities agree. It does **not** certify profiler completeness,
complete numerical telemetry, bottleneck attribution, speedup or work/J.

`profile_original_campaign_batch.py:42–77` writes cProfile tables without a
coverage check; line 174 bases success on the correctness checks.
`run_original_campaign_batch_profile.py:50–52` checks capture status instead
of finite numerical telemetry coverage. Those unshipped lines are not fixed
by this disposition.

The joined run records 77.636932 seconds wall and 79.316089 process-CPU
seconds, including first-use work and profiler overhead. Its cProfile self
total is 25.537379 seconds; 21 malformed rows and absent enclosing roots make
complete attribution invalid. The 52.099553-second arithmetic gap remains
unexplained. Neither nested Triton profiling nor a particular interpreter bug
has been established as its cause. Do not turn the partial rows into a complete
operator ranking or a CPU/GPU/compilation split.

Power telemetry has null tails on both hosts. The recorded partial energy
covers 75.642654 of 77.636932 seconds on Sparky and 65.642654 seconds on
Sparklina. The framebuffer series is entirely null. No full-phase energy or
work/J result follows; recorded CUDA allocation/reservation counters remain
separate residency evidence. The existing review retains the detailed numeric
limitations without changing raw files.

## Replacement evidence has different scopes

- **Original-wire correctness:** #407's
  [full reference record](full-model-original-reference-2026-09-07.md)
  binds 2,142 original inner wires and 160 unchanged passthrough tensors to
  export proof SHA256
  `588d84dfdd48da1635c1b078fb3b60597c45a11987be9f975c31b19028585b1c`,
  then records the complete stopped generation proof SHA256
  `dbfb370e5286310a20999088b66010676a238166038d52ae5010ccb6e9f51d9f`.
  These are export/functionality proofs, **not encoder timing profiles**.
- **Current profiler execution:** PrismaQuant main
  `2b2715be6ec51410d948992a46711d6ab76937c2` includes
  `experiments/profile_command.py`, `experiments/pq_joint_profile_entry.py`
  and `tests/test_profile_command.py`. The sampler records its command's own
  child start/exit receipts and refuses profiler success without a successful
  matching child and nonempty stacks. It is wired into joint prepare/run.
  This is an execution-integrity improvement, not a declaration that every
  sample stream or Netdata interval has complete numerical coverage. Its
  `passed` field does not certify telemetry completeness.
- **Bounded reader measurements:** the
  [registered-grid record](2026-09-07-registered-grid-digest-cache.md) and
  [byte-unpack record](reader-byte-unpack-2026-09-07.md) retain actual
  fourteen-wire before/after checks, independent sampler/child exits,
  Torch CPU/CUDA profiles and both-host Netdata. The isolated decoder action
  `d2692d876acebbff144d7908d7ffc79d77bdd7f84696bf5b3297e7214a499a7e`
  has result SHA256
  `e65394019a0ae3c3f4db369d1d2cff72882ed59a99e059e2faf8ebd2cfc93d88`.
  Those measurements explicitly reject a per-arm work/J claim because power
  sampling is too coarse. They concern resident verification, **not the old
  32-expert encode batch**, and cannot fill that profile's missing interval.

A future request to measure encoder-batch performance needs a new explicit
workload and instrument-completeness contract. It must not silently reuse this
retired launcher or promote the old `passed` flag into a performance claim.
That is future scoped work, not an unfinished remeasurement promised here.

## Recovery and preserved artifacts

The original review and raw profiles remain at:

- `/mnt/shared/tessera-clean-runtime-20260907/encoder-profile-review-20260907/`
- `/mnt/shared/tessera-clean-runtime-20260907/encoder-profile-original382/profile-01/`

Two recovery refs now retain exact source history:

```text
refs/archive/issue-402-original-batch-profiler-20260907
  f9c19e962960c648adbf3bad0604af3722c58a17
refs/archive/issue-402-executed-snapshot-20260907
  9f4a2574f379e51bafb3034e1dae8326e9430e00
```

The self-contained `issue-402-recovery.bundle` in the review directory holds
both refs with complete history, verified by `git bundle verify`:
7,692,793 bytes, SHA256
`404f6c50f13a0de1c738fcdcb5fab0fdf216a558cb9843eaa650cb468cc6dbd8`.
The original PB input bundle is also retained in CAS at
`blobs/36/364a124e9f299dd7c3abb2ee9222ac4747c810412c640589fcd8759199bb49e5`.
Both historical script pairs match the saved executed copies byte for byte.

`disposition-402-20260907.json` in the review directory binds this disposition
to 94 rehashed raw/reference artifacts, including all 32 original and 32
profiled wires, both pstats/text pairs and all ten Netdata series. Its SHA256 is
`3562e78f4d0ab7d56d4187e084d2b9b7228a41fbccec4cf71feae3af462e02aa`.
The audit read existing data only. No original output or `review.json` was
rewritten, no production source was edited, and no test, benchmark, encoder or
GPU workload was run for this prose-only disposition.
