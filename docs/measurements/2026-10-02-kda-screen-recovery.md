# KDA screen recovery, 2026-10-02

The PTX convolution reference now matches the pinned stock prefill's actual
q/k/v output **and entire convolution state** in all 24 screen cases. The
overall numerical admission remains **failed**: the required `ex2-ftz`
mutant is invisible in the measured output/state. No timing screen, fused
kernel, serving override or serving window follows this result.

Scope: Tessera #814, related to #735. Base `4e5674d8b0c7b413dfdfe5829f261c92c6897748`;
the frozen screen implementation was recovered from `14c0df51f4` without
its vendored FlashKDA/build work. Original checkouts and artifacts were
preserved. No change to a serving contract, gate waiver, tolerance, route,
format or default is part of this repair.

## Cause and repair

At the serving image pin, `_causal_conv1d_fwd_kernel` sets logical
`state_len = KERNEL_WIDTH - 1`. Width four reads history from columns 0..2
and writes only columns 0..2. Extra physical cache columns are spare or
speculative state; the prefill does not consume them.

The old PTX reference instead read the physical cache's final three columns.
It therefore failed continued sequences with six-column state, while
fresh sequences and three-column state passed. It also left the state
unchanged, and the old harness compared only output. The corrected
reference reads and updates the logical prefix, preserves the physical
strides and extra columns, and updates short sequences using their actual
history, or leading positive zeros for fresh sequences.

Every case now compares actual q/k/v output and the entire state store as
raw BF16 patterns, including unused cache slots. No `allclose` or tolerance
gate is used. `bits_differing` in the JSON counts BF16 words with unequal
patterns, not individual changed bits.

Mutants now compare directly to the unmodified candidate. Previously a
common state-indexing error falsely counted as an `ex2-ftz` mutation
witness. Physical-tail read and write mutants cover the repaired state
semantics separately. All three original arithmetic mutants remain
required. Missing or invisible witnesses fail the action and stop later
parts before timing.

This is a convolution reference screen. It does **not** compare recurrent
output or final recurrent state. Any future fused recurrence or serving
implementation still owes those actual outputs and its own raw-bit gates.

## Numerical evidence

Image manifest:
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
Both Sparks independently resolve it to the same portable PB image reference:
`content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918`.
The manifest digest itself is not a `content:` digest; an initial submission
with that mistaken reference was refused before publication or execution.

The actual executing image reports PyTorch `2.13.0+cu130`, vLLM
`0.30.1rc1.dev336+gaf5b4857e.d20260929`, NVIDIA GB10.
The installed stock convolution source hashes to
`cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f`,
equal to the inspected image-source copy. The executed PTX source hashes to
`ac521b3588f88e596f54ea89fc2215ad65d5185b8eeed99a910cc702c5a6e427`.

| Evidence | Source parent | Actual outcome |
|---|---|---|
| Original numerical PB `be8289cfefdf334f73110f09b59079bf981041ffc4fc1ba731dc782acef370b8` | `d0b66e3b77c8a03bfa814da783b57a4c24de1a5a` | Worker rc0, but `served_bit_equal_all=false`; SD six-column continued-2048 has 36,452 differing words, max_abs 5.326171875. DS also fails. The zero exit was not a numerical pass. |
| Corrected numerical PB `2076a3782ab202f8c22d8d0b5484dc1f5f8ba0c155c1d76a6b4e163c386c24c9` | `d91f3f05fc06091ee8c65d6bd51192ec0e9c170c` | sparklina, action rc1; actual q/k/v and whole conv state exact in all 24 cases; required ex2-ftz witness absent. |

The corrected sealed snapshot is
`9f736399ce48f6c650876b606e6a6033f92add60`. Matrix: layouts SD/DS,
physical state lengths 3/6, and six cases per cell: varlen, short,
fresh-2048, continued-2048, signed zeros, and input magnitude scaled by 64.
Convolution inputs and weights are deterministic synthetic CUDA tensors;
the checkpoint path in metadata is not a claim that real model tensors
were read in this part.

| Mutant | Output words unequal to candidate | State words unequal to candidate | Observed |
|---|---:|---:|---|
| Two roundings per tap | 10,127 | 0 | yes |
| ex2-ftz | 0 | 0 | **no** |
| div.rn | 1,123 | 0 | yes |
| Physical-tail read | 455,332 | 49,107 | yes |
| Physical-tail write | 0 | 1,915,691 | yes |

These are aggregate counts across the 24 cases, counted once per mutant.
The state-write control demonstrates why output equality alone is
insufficient. The ex2 result establishes non-observation in this matrix,
not correctness for every possible input or permission to drop its gate.
A plausible explanation is that SiLU's `1 + exp2(...)` erases a subnormal
exponential on the large-positive-input side; that explanation is an
inference, not an additional measured witness. Astra decides whether a
separate raw intermediate observation is appropriate before any expansion.

Actual output:
`/mnt/shared/tessera-measurements/kda-recovery-20261002/numerical-prefix-v1/mhc_probe_numerics.json`,
SHA-256 `bc1b84c836432b492a6b3f7ef26ccd764f74372aedce645cff6e6364cf8b2e9d`.
Terminal record:
`/mnt/shared/prismabuild-fleet/pb-queue/failed/2076a3782ab202f8c22d8d0b5484dc1f5f8ba0c155c1d76a6b4e163c386c24c9.json`.
Its unsuccessful action correctly has **no success CAS receipt**.
Original numerical CAS claim/payload were checked at
`982ca3a63f7539a43e2eff1ea01dfc7157b76f49e47bc7f6f7c8053b8deb08d1`;
the receipt is the action's `cas/actions/v3/be/be8289cf...json`.

Numerical reproduction (submit from dl380g10, using an isolated checkout
at the stated source parent; these synthetic inputs need no bulk-read manifest):

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/codex-kda-recovery-20261002 \
  --gpu --cpus 4 --demand mem_gb=16 --anywhere --priority -10 \
  --timeout-s 900 --detach \
  --container-image content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918 \
  --env ORACLE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  --env MAX_JOBS=2 -- \
  bash experiments/mhc/mhc_probe.sh . \
    /mnt/shared/tessera-measurements/kda-recovery-20261002/numerical-prefix-v1 \
    --parts kdaptx --numerics-only
```

Native math threads are one, compile jobs two, and the wrapper retains
PB-assigned affinity. This was numerical admission, not measurement
admission. PB's scope recorded peak memory 4,579,889,152 bytes and CPU time
44.951997 seconds; full resource evidence is retained with the terminal
record. These resource observations are not a before/after performance claim.

## Harness regression evidence

All CPU rows were submitted from dl380g10 with two CPUs/four GiB, two xdist
workers, `--dist worksteal`, durations, and native threads one. Device
population: PyTorch `2.10.0+cpu`, no CUDA allocation; zero skips and zero
missing collection. These pass counts cover the harness policy, not CUDA.

| PB action | Source parent | Result |
|---|---|---|
| `46e22253aefe32c3d4e9f4f671df79e4128fe69e065489be72211e02dac48904` | `cbe6ba8b2a21d9bf49bb19413c209eeeeb0366dc` | RED: 4 failed, 1 passed; failed or invisible-mutant screens incorrectly return rc0. |
| `e15bcf5eb10ba025d7948a4cc4083ac6bd6866a38c8924e401336832096a6a36` | `d91f3f05fc06091ee8c65d6bd51192ec0e9c170c` | 5 passed after the exit-status fix. |
| `ba2ba47e7a2ce2153498ab1b7742820519fdc01cff6b0e93e0a1e9fcbba87bed` | `5b94c05ff1cd6390449e12d92e32e135587643f7` | RED: 1 failed, 5 passed; removing a required witness incorrectly passes. |
| `8e569d4a8e44cc02bcf99a568f5b2de5a2973bb30be712393f051da4935546e5` | `2eca0529c4ba7ab52cfdc14389513524484767a8` | RED: 2 failed, 5 passed; missing witness and timing invoked after numerical failure. |
| `91592819f83b1a2763901a461cb622e11b6a414a04367b58f28c9bb7708fef90` | `1e5b5a8db53aa6d7e5951ea605a25eb70600a443` | GREEN: 7 passed, compile checks passed, worker rc0. |

The final CPU row compiles `experiments/mhc/mhc_probe.py` and
`tests/test_kda_probe_gate.py` with `py_compile`, then runs the test file.
Its sealed snapshot is `572a96a1f8aad1b6bfe864563a24f2b944616278`;
local-result claim
`a75150e176c4e29bd4eaabc809fd7ae225667b3024044d4ce4d2d091fb3ade5b`
and payload hashes were checked. The receipt is
`/mnt/shared/prismabuild-fleet/cas/actions/v3/91/91592819f83b1a2763901a461cb622e11b6a414a04367b58f28c9bb7708fef90.json`.
The claim tool's attestation check remains explicitly unperformed because
it needs the full action manifest; this is not claimed as extra verification.

Later gate-only edits did not change the numerical body. The recovery
packet records source hash equality for all 11 relevant functions and
constants between the GPU-tested `d91f3f05fc` and the final tested code.
This reuses the unchanged numerical evidence rather than rebuilding or
rerunning it. The old stock-cubin/132-kernel/325,392-SASS rebuild evidence
at PB `46d6b23a` was preserved and not recomputed.

`tools/impacted_tests.py --ref 4e5674d8..HEAD --json` reports `narrowed` with
98 files due to experiment/document reachability. That list was sent to
Astra for the shared integration owner. No full-suite integration was
started by this executor.

## Timing disposition

Old timing action
`0b9c7c5011c69469ea86d23eff2e43300d616840a775c5b8f4d38a346210d79f`
failed with action rc2: the experiment parser rejected
`--placement 'exclusive GPU, ambient CPU'`. It produced no timing.
The repair is to remove that unsupported experiment argument and put the
resources and measurement semantics on the published PB submission.

If numerical admission is later resolved and Astra authorizes the screen,
the submission uses `--measurement --host-class gb10 --exclusive`, truthful
CPU/memory demand, and the portable immutable image reference above;
measurement does not use `--anywhere`. The smallest relevant existing
screen is `--parts kdafwd --kda-tokens 2048 --kda-heads 32 1` (both state
dtypes are already in the harness). Qualification still owes actual
class/platform bindings, in-process profiles and both Sparks' Netdata,
GPU power against the approximately 140 W envelope, and work per joule.
An ordinary numerical admission's timing is not an acceptable substitute.

Recovery packet on sparky:
`/home/rob/tmp/codex-campaign-takeover-20261002/kda/` contains
`pb-evidence.json`, `final-pbwait.json`, `numerical-source-binding.json`,
and `impacted-tests.json`. Recommend retaining this bounded negative result
and the fail-closed repair; do not build broad fusion from the old apparent
mutation success or infer prepare/recurrence-fusion value without timing.
