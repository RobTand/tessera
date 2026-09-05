# Historical encoder evidence scope (#198)

This is a single-unit weight-space comparison. It does not measure served
KL, establish a whole-model quality ordering, or re-export a checkpoint.
The existing serving receipts continue to describe their original bytes.
Nothing using this format/schema has been released; this corrects the
pre-release contract rather than providing a released-schema migration.

## Inputs and method

Historical artifact:
`gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook`, whose
`tessera_gridbook_manifest.json` records encoder commit
`8070ec6c4e0448826cda3f3f8d9401a125444e3b` (short spelling `8070ec6`), written
2026-09-02. Its `model.safetensors` is 846,726,118 bytes, SHA-256
`7f1193b1f014d957b7e7b78af897302223bd86a49b867a1b1b847ea9a18c68ec`.
The manifest counts 112 modules / 196 units and 440,401,920 quantized
parameters. Immutable BF16 regions are outside that denominator.

The comparison unit is `model.layers.0.mlp.down_proj`, sourced from
`/home/rob/models/Qwen3-0.6B/model.safetensors`. The current numerical encoder
is `1e58fdc72e8f6408ac79d76bb4a817f2cc9f58f4` (the branch changes evidence
metadata, not numerical encoding). `experiments/encoder_evidence_drift.py`
calls the dense export entry point `encode_linear_planes` with E4M3,
q256=1024, recipe defaults, no Hessian, verification enabled. Both arms
are read from their actual unit blobs by `read_unit_artifact`, then scored
against the same float32 source weight with a float64 sum of squared errors.
This is the no-Hessian weight-space objective; there is no calibration set,
sequence length, activation weighting or served runtime measurement here.

The JSON receipt beside this file records source tensor and payload hashes,
resolved historical scheme, encoder behavior identities, unit dimensions,
container/plane sizes, torch/CUDA versions and GPU model. Container overhead
is reported separately through whole-unit blob bpp; a header-minor change
must not be mistaken for a change in the packed plane budget.

## Result

The current encoder has slightly **lower** error on this unit. The ratio of
current to historical SSE is **0.9999818976425706** (0.00181024% lower).
This selects #198's evidence-scope correction, without changing the encoder
or the artifact.

| Quantity | Historical (`8070ec6`) | Current (`1e58fdc`) |
| --- | ---: | ---: |
| Weight SSE | 11.069609364171521 | 11.069408978146207 |
| Relative weight RMSE | 0.07148903433097972 | 0.07148838726802549 |
| Plane region bytes | 1,591,296 | 1,591,296 |
| Unit blob bytes | 1,592,031 | 1,592,059 |
| Header minor | 3 | 6 |

Exactly **63,952** plane-region bytes differ, first at **16,838** and last
at **1,591,290**, reproducing the mismatch reported in the issue. Plane
digests are unequal. The current blob's extra 28 bytes are envelope metadata,
not extra plane payload. The result is pinned in
[the JSON receipt](encoder-evidence-scope-2026-09-05.json).

## Reproduction

Executed directly in the existing pinned vLLM image on one NVIDIA GB10;
PrismaBuild was not used. The image has torch, safetensors and the encoder's
CUDA dependencies. No vLLM server was started. The host supplies the full
encoder commit because this image does not contain the Git CLI.

```sh
docker run --rm --gpus all --ipc=host \
  -v /home/rob:/home/rob:ro \
  -v /home/rob/tmp/tessera-198:/scratch \
  -v /home/rob/tessera-runs/issue-198:/results \
  -e PYTHONPATH=/home/rob/tessera-issue-198/src \
  -e TMPDIR=/scratch -e TRITON_CACHE_DIR=/scratch/triton \
  -e TORCH_EXTENSIONS_DIR=/scratch/extensions \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e MAX_JOBS=1 \
  -w /home/rob/tessera-issue-198 --entrypoint python3 \
  vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14 \
  experiments/encoder_evidence_drift.py \
  --checkpoint /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook \
  --source /home/rob/models/Qwen3-0.6B/model.safetensors \
  --encoder-commit 1e58fdc72e8f6408ac79d76bb4a817f2cc9f58f4 \
  --output /results/comparison.json
```

Raw output: `/home/rob/tessera-runs/issue-198/comparison.log` and
`comparison.json`. This is no timing claim: no before/after performance
profile, power comparison or served-speed result was taken.

## Contract correction

Contract v19 / lane-eligibility schema v8 adds required `evidence.artifact`.
The four dense E4M3 cells record the historical artifact and its encoder,
and the exact comparison commit/unit/metric and receipt. The remaining
cells carry null: their encoder reproduction was not measured by this run.
Missing scope, malformed commit IDs, nonportable artifact IDs, unsupported
metrics and inconsistent payload/error relations fail closed. An identical
payload must have equal error under the same source and decoder.

A weight-space screen does not contribute to `derive_evidence_grade`.
The historical plugin, window-GEMV and eager/compiled decode-KL receipts
retain their original claims and dates; their cells now explicitly attach
those claims to the historical artifact. Lower single-unit weight error
cannot prove that those historical KL values are conservative for a newly
encoded model. A future encoder is likewise not measured by this receipt:
its commit would need its own comparison.

## Regression evidence

Before the implementation, all 14 new cases in
`tests/test_evidence_artifact.py` failed. Representative failures:
`KeyError: 'artifact'`, `DID NOT RAISE <class 'ValueError'>` for a missing
scope, and `evidence carries unknown field(s) ['artifact']` for a valid
scope. Log: `/home/rob/tessera-runs/issue-198/pre-fix.log`. Population:
CPU, torch 2.10.0+cpu, 0 skips, 0 uncollected modules; no CUDA coverage.

## Follow-up after #199 merged: wire minor 7

Repeated the identical experiment after rebasing on
`331703661baaa67ed73900c3d9e99f300fdc8415` (PR #199 / issue #144).
Command as above with that `--encoder-commit` and
`--output /results/comparison-minor7.json`. Logs:
`/home/rob/tessera-runs/issue-198/comparison-minor7.log` and
`comparison-minor7.json`; committed
[minor-7 receipt](encoder-evidence-scope-minor7-2026-09-05.json).

The current **plane-region SHA-256 and SSE are identical** to the earlier
comparison: `402bb8eb6376b33329882b4a8b3d3d16cd9ef7807d05d8441fbc600bb937421a`
and **11.069408978146207**, respectively. The historical arm remains
**11.069609364171521** and the payload mismatch remains 63,952 bytes.
The current unit envelope is now minor **7**, **1,592,035 bytes**, with
encoder fixture ID
`220ee0aaed5fd628f6fe92c02b08cbdf90b6e26b76313fa505a9b32fecbf973c`.
The cells name this newer comparison commit. The original JSON remains as
the measured control before the wire revision; neither result is inferred
from #199's PR description.

The strict on-disk byte-identity xfail introduced by #199 remains an
intentional notice if a later encoder or replacement artifact restores
byte equality. #198 resolves the missing evidence scope and the unknown
weight-error direction; it does not promise historical byte reproduction.
