# CHANNEL refit cancellation — 2026-09-06

Issue #360 was reproduced on pristine release source: 28 legacy-wire tests passed
and one failed on x86 CPU, while the ARM run passed. The differing E4M3 fixture
has identical alphabet and body bytes and one changed FP16 DIAG_SV word (row 12).
The remaining differences are dependent digests. The old ARM word is
1.7978515625; x86 writes 1.796875.

A fixed Torch random seed does not produce identical float32 input bytes on these
platforms. Supplying the exact ARM weight bytes to x86 did **not** change its
artifact, ruling that input difference out as the cause of this failure.

Instrumenting the fourth refit shows the same proposed word on both machines.
ARM's separately rounded quadratic losses compare equal; x86's difference is
-1.4901161193847656e-8. The factored difference on ARM is
-1.5745172277092934e-8 and a float64 reference is -1.575025447043965e-8:
the proposed word improves the stated objective. The fix factors the difference
instead of subtracting the two large rounded losses. No tolerance is introduced.

An exact-binary one-element witness reproduces the defect on CPU and CUDA:
`work = 1 + 2^-11 - 2^-16`, `unit = 1`, old stored word `1 + 2^-10`, global
scale 1. Both old guards hold the word; the nearest word 1 has strictly smaller
float64 squared error. PrismaBuild action `a589ebf822d6fec2187e6c90e1618596b04d6d2b04089db523136826e7b55113`
failed both regressions at `assert new_stored.item() == 1.0`. With the factored
guard, action `97baf29f6f01ce43410bcb6deae1117dc5aea7a067086c454a38c026e017db58`
passed both, with one CUDA-allocating test, no skips/uncollected modules, exit 0
and verified CAS payload. These are correctness checks, not throughput or served
quality measurements.

Attributable diagnostics are retained in
`/mnt/shared/tessera-runs/legacy-x86-diagnosis/` and terminal/CAS receipts in
`/mnt/shared/tessera-runs/issue-queue-receipts/`:

- x86 native input: `ec4a2ae755895b6054d48fb832dddff0ef3614ef9609f54f74c6c91557c2fe00`.
- ARM native input: `7ec2e29a27b05cd07de8df1cfaabfdb934cab7ec1545c69d4b7473392fd09ba5`.
- x86 with exact ARM input: `65400632f70b2266cee146ae4dda09bb32213a7a90795d6f664c214bf719e0b7`.
- x86 refit trace: `437e59d9a6afed4b5225bca74ce12e28d04a806ff6b8d51debf7c9fc3d1c9a80`.
- ARM refit trace: `4a5ba9fcb83258d4b7193da8deb358b639145f326c336b39791d55825e0ddf67`.
