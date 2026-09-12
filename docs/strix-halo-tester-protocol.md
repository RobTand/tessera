# Certify Tessera on Strix Halo (gfx1151)

This page tells a tester with an AMD Strix Halo machine how to produce the one
receipt Tessera needs from that hardware, and tells everyone else what that
receipt is allowed to say.

Tessera's RDNA3.5 lane is **Tessera-16** (`TESSERA_BF16_K1`, a WnA16 wire) at
rungs **896** and **1024**. That is the whole reference set. The harness
refuses an artifact at any other family or rung, because a number measured at
a rung nobody asked for cannot be joined to the table it would feed.

Nobody on the Tessera side owns a Strix Halo. Every gfx1151 fact in this
repository today is either a compile result or an expectation. Your receipt is
the first measurement.

The protocol and the harness come from #459; `docs/ARCHITECTURE.md` §4.5d
says where a receipt fits in the contract.

## Scope rule

Put this rule on every receipt. `tools/tessera_attest.py` writes it into the
receipt header; it is repeated here so you can read it before you run
anything.

| Receipt | Proves | Does not prove |
|---|---|---|
| `hipcc --offload-arch=gfx1151 compile of the port (wsl-gpu)` | the kernel source is valid for RDNA3.5; LDS/VGPR budgets per instantiation | anything about gfx1151 execution, numerics or speed |
| `gfx1201 execution on wsl-gpu` | the HIP code path (loader, shims, plan, window_decode+torch.mm prefill, BF16 route) executes and matches the CUDA/CPU reference bit-for-bit where the contract says it must | gfx1151 numerics or perf; any perf (no PMCs, WSL2, and wall-clock on a desktop RDNA4 says nothing about an APU) |
| `Strix Halo tester run` | gfx1151 device_qualified: correctness, KL-vs-BF16, decode/prefill tok/s, power | another platform's numbers; a rung outside the reference set |

Only the third row mints a `device_qualified` cell for gfx1151. The harness
enforces that: it reads the device's identity from torch's `gcnArchName`,
compares it with the platform the receipt claims, and withholds the
qualification when the two differ.

## Before you begin

You need:

* A Strix Point or Strix Halo machine (`gfx1150` or `gfx1151`) running
  **native Linux**. Do not certify from WSL2: `amdsmi` cannot initialise
  there, `rocm-smi --json` prints nothing, and a WSL2 number is a correctness
  receipt only.
* ROCm with `hipcc` on `PATH`, and a ROCm build of PyTorch that sees the GPU
  (`torch.cuda.is_available()` is `True` and
  `torch.cuda.get_device_properties(0).gcnArchName` reads `gfx1151`).
* A ROCm build of vLLM with the Tessera plugin installed, run from a
  container image you can name by digest. Record the digest: a
  `lane_eligibility` cell cannot be written without one.
* `amd-smi`, for package power only. If it is missing, the run still
  succeeds and the receipt records its absence.
* The reference artifact: a small dense model exported as Tessera-16 at rung
  896 or 1024. Ask the Tessera maintainers for it rather than building your
  own, so every tester measures the same bytes.

## Produce the receipt

Run the correctness half first. It needs no serve:

```bash
python tools/tessera_attest.py \
    --out receipt-gfx1151-correctness.json \
    --expect-platform gfx1151 \
    --build \
    --artifact /path/to/tessera16-q256-1024
```

`--expect-platform` makes the harness refuse before it does anything if the
device is not the one you meant to certify.

Then serve the artifact and produce the two receipts this harness ingests
rather than recomputes:

* the route census, from `tools/tessera_route_census.py` — it records what
  every served module actually executed;
* the served KL against a BF16 teacher on the canonical n=8 × 512 WikiText
  corpus, from the KL harness (`experiments/serve_and_dump_kl.sh` is the
  reference driver).

With the serve still running, run the full command:

```bash
python tools/tessera_attest.py \
    --out receipt-gfx1151.json \
    --expect-platform gfx1151 \
    --build \
    --artifact /path/to/tessera16-q256-1024 \
    --image docker.io/your-org/vllm-rocm@sha256:... \
    --census-receipt census.json \
    --kl-receipt kl.json \
    --endpoint http://127.0.0.1:8000 --model tessera16-reference \
    --execution-modes eager,compiled \
    --serve-mode streamed
```

Repeat it with `--serve-mode resident` and, if you can, at both rungs. The
harness samples package power at 1 Hz for the duration of the measured half.

Exit status 0 means a receipt was written. Exit status 2 means the harness
refused to write one at all, and the message says why. A receipt whose
`header.qualification` is `null` is a normal result: it records what was
proved and names the steps that did not run.

## What to send back

* every `receipt-*.json` the harness wrote;
* the census and KL receipt files those receipts name (the header carries
  their SHA-256, so the pair must travel together);
* the serve log, and the exact serve command you used.

Do not edit a receipt. If something is wrong in it, say so in your message
and send the file unchanged.

## What your numbers will be compared against

Everything in this section is **unmeasured**. These are the expectations the
design wrote down so your receipt has something to disagree with. None of
them may be quoted as a Tessera result.

* **Memory bandwidth:** unified LPDDR5X, 256-bit, product spec ~256 GB/s
  (unmeasured).
* **Compute:** 40 CUs on gfx1151, wave32, 64 KB LDS per workgroup (the LDS
  budget is the compiler's, confirmed on silicon only for gfx1201).
* **Decode ceilings**, from `tok/s <= bandwidth / bytes read per token` for a
  27B-class dense body (all unmeasured):

  | Wire | Body size | Ceiling |
  |---|---|---|
  | Tessera-16 at 4.0 bpp (rung 1024) | ~13.5 GB | ~19 tok/s |
  | Tessera-16 at 7.0 bpp (rung 1792) | ~23.6 GB | ~11 tok/s |
  | BF16 passthrough | ~54 GB | ~4.7 tok/s |

  A decode GEMV that reaches more than 70% of the ceiling is the target. The
  LDS-resident 32 KB bf16 table is the thing most likely to hold it below
  that.
* **Prefill** is compute-bound on bf16 WMMA through hipBLASLt, with the
  `window_decode` materialisation as a bandwidth-bound prologue amortised over
  the batch. Where the GEMV path stops winning is a measurement on your
  machine, not a constant.
* **Power:** the APU package envelope is shared with the CPU. Rank by tokens
  per joule, never by utilisation.

## Head-to-head against EXL3 — requires hardware we do not have

A comparison against EXL3 is the obvious question and it cannot be answered
today.

EXL3 has no ROCm build: its README lists ROCm support as a to-do, and its
quantisation kernels are inline PTX, so it does not run on your machine at
all. Splitting the comparison by what each half needs:

| Comparison | Where it could run | Status |
|---|---|---|
| KL-vs-BF16 at matched bytes (an artifact property, hardware-agnostic) | the EXL3 side needs an NVIDIA box | blocked on hardware and time; **requires hardware we do not have** on the AMD side |
| Decode tok/s, prefill tok/s, power on Strix Halo | Tessera-16 per this page; EXL3 cannot run there | **requires hardware we do not have**: there is no ROCm EXL3 to compare against |
| The same on gfx1201 | correctness receipts only | never a performance number |

The comparators that *can* run on your machine are llama.cpp GGUF at matched
bits per weight and Tessera's own BF16 lane. If you run one, say which
evaluator produced each number.

Any EXL3 KL figure also carries the caveat in RobTand/prismaquant#468:
first-window KL drift across matched scoring runs is unresolved, and EXL3's
own evaluator scores one 2048-token window rather than the n=8 × 512 corpus
Tessera reports.

## Reading the receipt

| Field | What it means |
|---|---|
| `header.platform` | the platform the receipt claims |
| `header.measured_platform` | the platform torch reported. If it differs from `header.platform`, no qualification is minted |
| `header.qualification` | `device_qualified`, `compile_only`, or `null` |
| `header.scope`, `header.scope_sentence` | which row of the scope rule this receipt occupies |
| `header.perf_claim` | whether a tok/s or power number from this run may be quoted at all |
| `sections."8.2"`, `sections."8.3"` | the protocol's own item numbering over the steps that ran |
| `problems` | why a claim was withheld |
| `limitations` | what the run could not establish, including absent power |
