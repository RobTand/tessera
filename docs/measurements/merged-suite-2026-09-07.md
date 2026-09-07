# Merged Tessera suite — 2026-09-07

The merged source at `5a1cea00e0a3163930fb7d58aa3d3620a93b1a46`
passed both complete test populations through PrismaBuild. This includes
PRs #398, #406, #407, #409 and #410. It does not qualify an unmerged resource
recorder, a GLM serving route, or any new runtime performance claim.

| Population | Execution | Result | Missing modules | CUDA allocation coverage |
| --- | --- | --- | --- | --- |
| dl380g10, Torch 2.11.0 CPU | 24 xdist workers, worksteal | 3371 passed, 559 skipped, 0 failures/errors | 0 | 0, device unavailable |
| Sparky, Torch 2.13.0+cu130, NVIDIA GB10 | 8 xdist workers, worksteal, strict CUDA | 3941 passed, 7 skipped, 1 expected failure, 0 failures/errors | 0 | 494 tests; zero missing-artifact skips |

Both arms bounded OMP, MKL and OpenBLAS threads to one per process and
preserved PB affinity. The CPU reservation was 24 CPUs/40 GiB; GPU was
8 CPUs/56 GiB physical memory with a 40 GiB GPU subset cap. Actual PB peak
physical memory was 18,162,147,328 bytes on GPU, with no OOM and complete
scope cleanup. The GPU run's CPU time was 1707.80 seconds over 285.32 seconds
of action wall time. These are execution records, not isolated benchmarks.

The CPU skip histogram retains CUDA-dependent cases, 37 missing local-artifact
cases, three missing-vLLM cases and one missing-Transformers case. No test
module was omitted from collection. The GPU skips are two requiring two
CUDA devices, four unsupported E2M1 four/eight-way column cuts and one E2M1
reader-range case. The full histograms and expected failure count are retained
in the paired receipt. Neither population establishes multi-device coverage.

## Attributable records

The complete receipt is
`/mnt/shared/tessera-suite-receipts/merge-20260907-5a1cea00/receipt.json`,
SHA256 `bb307e1f0da5777855198a735927ecfd5217623d69a6194a9335e91cbeda3fe2`.
It contains both full surfaces, sealed commands, request hashes, snapshot
identities, attempt/log metadata, CAS evidence and resource telemetry.

| Arm | PB action | CAS receipt SHA256 | CAS output SHA256 |
| --- | --- | --- | --- |
| CPU | `ac0698c8dd363f052aa767ad3e62d478e8bcdad137c662bf81a6401713adf178` | `8beff5807e6c824ec055659b340ab1568d49eb10a9ef52d01412984795c27eb3` | `a73028d9c02b8eb0e3122993f01c2c6141990a6650fca1e59016049300351115` |
| GPU | `52b76d67447f8d2c918555e3324cc156ef41ce37689a8e0fc27c5fb037cf3687` | `f3ef9f058022ba6c8555ab19f737ddd82b6f095195b4920da4905b2e175e72b0` | `f4f896ed693a4967f89d58e3a9526779eca144db6b4bbad29bc646b89da66f04` |

PB's action-specific snapshot commits differ. All 24 CPU and eight GPU workers
verified the same effective source SHA256:
`e30af0061bf2a9d4ed5a95f2721bcc1bed9b52ae8bd655f20278badab986826c`.
Population publication hashes match the producing attempts' actual stdout.
Terminal and attempt exits, log bytes/hashes, source bundles, CAS receipts and
payloads were independently verified. The receipt was assembled from that
audit using the existing verdict and ledger writers. It explicitly records
`assembled_by: independent_pb_audit`; the legacy automatic resume parser does
not parse this Docker wrapper, and was not broadened to accept it.

## Environment repairs and retained failures

Three prior GPU attempts are retained under the same shared receipt directory,
with their own distinct surface names and immutable PB attempts:

- `168e94cc8a836eafa335fc927660079e30f19da37760d2b9a16453caf5470e17`:
  17 failures, 3924 passes. The older EUGR image lacked the required vLLM mapper
  API and the container mount hid the PB source-stamp location.
- `aca79f26b85f2f9ab702e3de1302eadeaaf02fc35221fbf35460dfede491c35e`:
  205 failures and four errors. The stock image needed git and CUDA wheel
  header links used by the existing plugin runner.
- `c43df2ae25d23cb51dd7464184d8be48bfd614a0b7aee824c889028b04c1add5`:
  156 failures and 90 errors. Dropping UID with `setpriv` retained `/root`
  as HOME, causing Triton/cache permissions failures.

The successful stock image was
`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
Bootstrap installed git in the ephemeral container, linked only missing CUDA
header names using the existing `tessera_plugin_run.sh` recipe, and used
`runuser` to select the actual user and HOME. Scoped pytest 8.4.2 and
pytest-xdist 3.8.0 were installed. Four source-identity, mapper and real-kernel
smokes passed before the full suite. The mounted source, model, served-log and
PrismaQuant reference roots and KL tool files are explicit in the sealed
command. No repository code changed to accommodate these environment repairs.
