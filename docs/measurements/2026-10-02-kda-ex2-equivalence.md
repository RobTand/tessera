# KDA ex2 observables and v2 admission, 2026-10-02

Astra approved changing the experimental mutation contract to meaningful
observables after reviewing the retained [v1 negative result](2026-10-02-kda-screen-recovery.md).
The ex2-ftz mutation remains **required**: it must execute and change the
raw FP32 exponential, and preserve the actual denominator, FP32 SiLU and
BF16 output. It is not described as an output-changing mutant. FMA,
division, physical-tail read and physical-tail write remain required
output/state mutation controls. No tolerance, serving gate or default moved.

## Operation-order proof and scope

The pinned stock Triton source computes `acc / (1 + exp(-acc))`. The
inspected served SASS and the explicit-rounding reference evaluate:

```text
z   = mul.rn(sub.rn(+0, acc), float32(log2(e)))
exp = ex2.approx[.ftz].f32(z)
den = add.rn(exp, +1)
y   = div.full.f32(acc, den)
out = cvt.rn.bf16.f32(y)
```

For a positive subnormal FP32 exponential `0 < exp < 2^-126`, adding it
to 1 rounds to exactly 1: the next FP32 value above 1 is `1 + 2^-23`,
so the halfway increment is `2^-24`, over 100 binary orders larger.
Flushing that exponential to positive zero also yields exactly the same
denominator. Equal accumulator and denominator bits feed the same divide
and BF16 conversion; the exponential difference cannot reach either
output in this operation order.

NVIDIA defines the `.ftz` exponential variant as flushing subnormal inputs
and results; the approximation's general accuracy bound alone does not
prove exact equality for tiny inputs. See the pinned-toolkit
[PTX ISA 9.0 ex2 definition](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-ex2).

The stronger statement here is specific to the actual sm_121a binaries.
The stock SASS compares `z >= -126`; the normal-side route performs one
`MUFU.EX2`. Below -126 it halves z, performs `MUFU.EX2`, then squares to
retain a subnormal exponential. The FTZ route in the control performs the
direct `MUFU.EX2`; at `z >= -126` both routes feed the same operand to the
same instruction. Subnormal z lies in that common route. Four actual
tiny-input observations give exactly 1 in both modes. This is instruction
evidence at the image/device pin, not a universal formal proof about every
implementation permitted by an approximate PTX instruction.

Finite accumulator overflow when forming z can produce either infinity;
both routes then have the same exponential endpoint and downstream
operands. NaN/nonfinite accumulator scope is not admitted by this control.

The stock SASS file is
`/home/rob/tmp/claude-campaign-20260926/tmp/kda/conv-KFIH.sass`, SHA-256
`3e7f135cb291a5e9d759991a240928090b5a79352c079c5e890ea3a23e90fe00`.
Its relevant instruction offsets are `1450..1480` (z), `1490..1550`
(exponential), `1530/1580` (+1), and `1650` (BF16 conversion).

## Actual control and artifact bindings

The control has 29 finite FP32 accumulators: signed zeros, both signs of
subnormal and normal tiny values, ordinary values, values around the
exponential's -126 boundary and its underflow/overflow boundaries, and
extremal finite FP32 accumulators. One CUDA kernel stores both raw
exponentials **before** +1, both denominators, FP32 SiLU outputs and BF16
outputs. Both exponential instructions are volatile inline PTX and their
outputs are stored; the mutation is not optimized away.

| Observation | Actual result |
|---|---|
| Raw exponential words differing | 5 |
| Differing unmodified exponentials | Positive subnormals; FTZ result positive zero |
| Denominator words differing | 0 |
| FP32 SiLU words differing | 0 |
| BF16 output words differing | 0 |
| Subnormal exponent-input observations | 4; exactly 1 on both modes |

Initial proof action `85ac36e453a75725d6882bebae2da652847b5eab4fff48a6818f9e7103f28bd2`
ran on sparklina, worker rc0. The contract-bound control then ran as
`ec366aafa7facf620b82e059e65c9738f83c00d380f705d594b7deacdaffc77f`,
source parent `9b23161eb3`, also on sparklina, rc0, same observations.
It produces `schema=tessera.kda_ex2_control.v1`, and the gate recomputes
counts/classifications/equality from actual raw words instead of accepting
the supplied booleans. It checks each raw word's shape/type, finite
accumulators, current source/build flags, and actual nonempty compiled
module/SASS files against their recorded hashes. Missing, malformed,
vacuous, contradictory or stale evidence refuses.

Control output:
`/mnt/shared/tessera-measurements/kda-recovery-20261002/ex2-bound-v2/mhc_probe_numerics.json`,
SHA-256 `cd3373522c8840bcdad10ab16816c168c1eb71764d278f2ad4d18e28c8e559af`.
Current PTX source: `c9a07ebaf83b8400d44e94cea0776ec1bb28970f0e41f3ae2073c814f1cb6a1f`;
CUDA flags `[-O3]`. Actual compiled module:
`f973fddcffe7a29e2d71a8f927a6e255ccf9ca85e908d1e4e0668387d2ea2c8b`.
Actual dumped SASS: `80be4f79375056ab993a8385ade6e24627ae3a90dbeda04d6bf1d25991d71285`.
The module and SASS paths are inside the output directory above.

Image remains the immutable `5be13705` manifest, portable PB content
reference `a0b85c05...`, PyTorch `2.13.0+cu130`, vLLM `af5b4857e`, GB10,
driver `595.91.07`. The control was numerical admission, four CPUs/16 GiB,
two compile jobs and native threads one, PB affinity retained. Peak scope
memory 4,444,033,024 bytes; CPU 36.685904 seconds. These startup/resource
observations are not performance measurements.

Reproduction uses the numerical PB command in the v1 document with
`--parts kdaex2 --numerics-only` and a fresh output directory. No checkpoint
weights are read by this synthetic control.

## Reusing the 24 convolution cases

The 24 exact output/state cases were not rerun. PB CPU action
`36c46134fbb4601184c521ea7c219cce7791b7af380a5e4f428cac148bbff4a9`
extracted both actual module cubins and compared their `.text` sections
using the recovered existing `experiments/kda/cubin_cmp.py` section reader.
**All six convolution kernels, including every mutant, are byte-identical.**
The extra kernel is solely the new intermediate control. The old module
hash is `c07aa285e63617716e40faa74fafab6fdddbdd1d9d8042f9c8f01d3e4a46cbc1`;
the current module hash is above. The banked CUDA convolution source is
an exact prefix of the current CUDA source; input generator seed, cases,
dimensions, state construction and views are unchanged. This source
comparison is recorded at
`/home/rob/tmp/codex-campaign-takeover-20261002/kda/v2-numerical-bank-source-binding.json`.

The deterministic CPU audit composes the unchanged bank with the bound
control under `tessera.kda_conv_screen.v2`; it verifies device text,
installed stock source and the current control, and seals expected input
JSON digests into its PB command. It does not change the original failed
v1 record or claim that it was a successful v2 action.

## Causal policy checks

PB `a68a86c49868f7db601a196a4e7363ce7396034939a24eaa23f89473ea3405fb`
on `4e3724d253` is RED: 1 failed/7 passed. The original all-output-mutants
policy rejects a real intermediate difference whose downstream outputs
are equivalent. This is the explicitly reviewed policy change.

PB `70c62ea96820fac4644d6fd5662e8c2aefdf4897e638337cea32c9974df4f0af`
on `9b23161eb3` is GREEN: **21 passed**, compile checks passed, rc0.
Two xdist workers, worksteal, native threads one, CPU-only PyTorch
`2.10.0+cpu`; zero CUDA allocations, zero skips, zero missing collection.
Its CPU count is policy coverage, not CUDA coverage. Tests refuse absent
controls, malformed raw shapes/types, vacuous mutation, contradictory
counters, denominator/FP32/BF16 differences, nonfinite scope, changed
source, changed compiled/SASS artifacts and changed build flags.

PB endings, logs and CAS claim/receipt/payload checks were inspected for
the CPU policy, GPU control and byte-identity actions. The tool's full
worker-attestation check remains explicitly outside those claim checks.
Receipt roots are `cas/actions/v3/70/70c62ea9...json`,
`cas/actions/v3/ec/ec366aaf...json` and `cas/actions/v3/36/36c46134...json`
under `/mnt/shared/prismabuild-fleet/`. No full-suite integration was
launched. No stock timing or serving window is claimed by this packet;
the timing submission still awaits Astra's final evidence review.

The final digest-sealed composition is PB
`5f5a0fc439e7afb5f3db945f4c983443887822a8b81ff6db9c5f0037c409181f`,
source parent `42e59048e04da40ed46b9c5c6754b49aff37f708`, sealed snapshot
`10a93585bd18b8a14fc8a2f80bfab69ce0f86576`. It ran CPU-only inside the
pinned image on sparklina and exited zero, with `gate_passed=true`,
`errors=[]`, `stock_binding_ok=true`, and all six device-text sections
equal. Its command binds both expected JSON hashes above. Artifact:
`/mnt/shared/tessera-measurements/kda-recovery-20261002/sealed-admission-v2/kernel_identity.json`.
CAS claim `78edb312447b3f61d6555fd13fa9bfc638405e85317376066cb737ae8564c7dc`
and receipt/payload hashes were checked; the receipt is
`/mnt/shared/prismabuild-fleet/cas/actions/v3/5f/5f5a0fc439e7afb5f3db945f4c983443887822a8b81ff6db9c5f0037c409181f.json`.
This is an audited composition of unchanged numerical evidence and the new
required control, not another 24-case GPU run.
