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


## Byte and identity transition

The corrected guard reproduces the same eleven legacy-layout encodes on ARM
and x86. Ten remain byte-identical to the historical artifacts. The E4M3 case
changes only the one scale word above and its dependent digests; both corrected
outputs hash to `b8e8732b35f14715426ae5af87ec7b20af445149117181c4252b153f30f83c8d`.
The original blobs remain unchanged and are still checked for exact decoding
and reserialization. The corrected E4M3 reproduction is retained separately in
`tests/data/refit-cancellation/`. These legacy-layout encodes deliberately use
the historical explicit fixture stamp to isolate payload/writer behavior;
they are not shipping-identity attestations.

The existing 13 identity fixtures did not see the guard correction. The new
negative test demonstrated this before adding coverage: action
`78861d4cb05e2f84b96d3e33f96f924bd5736306af7408eafec0a43ac4c8caeb` failed at
`assert ei.encoder_fixture_id() != identity`, CPU ARM, no skips/uncollected.
A 16×256 seed-zero draw from the identity's portable Gaussian stream reaches
the boundary through the ordinary exporter. Its old ARM contribution is
`3ff89553f9f28f2d0de3dbc0a76c4d62e76da574b29bd24892065a77df277a85`;
its corrected contribution is
`48d4eb8c1ab0e1c54aca2bc181f896a5c0292df4d081f09f776abbe564665025`.
The old value is a one-time neutral baseline, never an advancing version.
Candidate seeds 20260904 and 7 did not distinguish the guards and were rejected;
repeating the historical row alone also did not distinguish them. Prototype
receipts `c5b7f34cf8c36820c0a2957390b203add0b36e61be1d0436c42fb498422f5dea`
and `c6cd04aeb1fcac609382919895bd945f9a7f09f217e345a5a92e5d75ef5b000d`
retain those negative results.

With the witness, both current CPU architectures report identity
`03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519`,
replacing `220ee0aaed5fd628f6fe92c02b08cbdf90b6e26b76313fa505a9b32fecbf973c`.
Actions `d6f3a7b0017459e50f1f62c51bc8929442ba60b704af58c97fa7e43387a16ffc`
(x86) and `ce48d7a091c9805a15ce9ccb17c11d2f5e6605b743777ff65a09566c27c51488`
(ARM) exited zero with identical, verified CAS outputs. The existing cache gate
therefore refuses units stamped by the old encoder under the corrected one;
existing artifacts remain readable and the wire schema/profile arguments do
not change. No saved checkpoint or published 0.1.0 release is rewritten.

Targeted validation includes identity mutation/refusal tests, CHANNEL behavior,
legacy read/rewrite/reproduction and the byte-baseline audit, with eight workers
per arm and native threads one. x86 CPU action
`e1413d57521f433c94fbb7e8eee016daa3611709134d55cf726cde3b37b65b54` passed
89 tests and skipped six (verbatim: `the kernel lane runs on CUDA` ×5;
`needs a CUDA device` ×1), with no uncollected modules. Strict-CUDA GB10 action
`95bb171f8e2b84c5b31f5402774c0987933981b62d507695bf931d8eb7c7c3a6` passed
all 95, with no skips/uncollected modules and six CUDA-allocating tests.
Both terminal exits were zero and CAS payloads verified. These validation
snapshots include the independently measured exact scalar-search optimization
from #364; the refit reproduction comparison above predates that optimization.


## Integration fixture follow-up

The combined GPU suite on integration commit `e080a6e39933` failed three
`test_encoded_unit_bytes_match_encoder_identity_baseline` cases at
`assert (len(blob), hashlib.sha256(blob).hexdigest()) == (size, digest)`.
Action `488bd21f8afefb39bd1b38f03f9caedc66fc303b6583d6cdec4b8d468ad7bdb2`
reproduced both LUT cases exactly under the old identity stamp: their new
hashes move only with the stamp. The E4M3 CHANNEL unit also moves payload.

Action `5b7315294aa8510f060a09331204c9321fa02374c540dca30b2d78e1ce17646a`
replayed the old rounded-loss guard and old stamp and recovered the original
E4M3 hash `26e006ff7e8ab53332ddd69c745b489ee42f9b116d409aeff7a21713a5c7bd3e`.
The corrected guard changes two stored scale words and seven body choices;
the artifact remains 21,192 bytes. The complete new hash is
`482e1ae55bac89f68513bfab0a3ce613d28180bfe9ff9d65ec38956a6be6724c`.
The synthetic unit's final plain SSE changes from 0.024550989270210266 to
0.024551386013627052. This is not a quality improvement claim: that fixture
uses the unweighted trellis default, for which refit/trellis alternation is
not guaranteed to descend the same objective (the refit docstring states the
conditions). The correction preserves improvements in each scale step.

Both scoped diagnostics exited zero and their CAS payloads were verified.
An earlier diagnostic, action `1462c33c6c347a9969c6997ca6cc209b5f5db157441da50d834f6e88d313b6e4`,
omitted `PYTHONPATH=src` and read an installed package; it is retained as
invalid evidence and superseded by the scoped replay above. The historical
layout-only hashes and original legacy artifacts remain untouched.

The corrected combined selection at `4e19ed5` passes all 48 selected files:
x86 CPU action `54926b2f28a247abf4cf74866a93d4d8c5e66212c1ea5fd54bdeb76e4cbb799f`
reports 1,237 passed / 159 skipped / zero uncollected, with 24 workers.
Strict-CUDA GB10 action `11d8fe4c703afba4ec75e3a0f77914048a161009214fe808ab5ce5734584d345`
reports 1,407 passed / four skipped / zero uncollected and 97 CUDA-allocating
tests, with 12 workers. The four skips are `e2m1-tcq-lut-release does not cut
4 ways along columns` ×2 and `e2m1-tcq-lut-release does not cut 8 ways along
columns` ×2. Both exits are zero, CAS payload hashes verified and effective
source identities agree. Native math/build threads were bounded to one per
worker. The complete CPU skip histogram, selected files and both populations
are in [the corrected receipt](issue-queue-corrected-2026-09-06.json);
[the ledger](../status/suite-populations.md) retains the original full red run
separately. No full green result is inferred from this targeted follow-up.

One off-task prose fix accompanies the queue report in separate commits:
`tools/merge_suite.py` now describes its legacy serial GPU setting accurately.
Scoped xdist is installed; shared device presence alone does not prevent
parallel test processes. No wrapper behavior or default changes.
