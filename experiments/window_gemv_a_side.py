"""Which A sides the E4M3 window wire can express with today's code (#42).

The window GEMV reads a bf16 ``x`` (``csrc/window_gemv.cu:67``,
``kernel_window_gemv.py:42``) and the FP8 route quantises ``x`` per token
before ``torch._scaled_mm`` (``serving/fp8_route.py:272,277``).  One family
publishes one activation contract, so wiring the GEMV into that route needs an
A-side decision first.  This script establishes the two facts the decision
rests on, by running them rather than by reading:

1. The **same wire** decodes to a bf16 tile through the **existing** pure-torch
   window decoder -- one argument, ``code_map`` floating instead of uint8
   (``serving/window.py:209-212`` says so; this proves it).  So a W?A16 arm
   over the E4M3 grid needs no new decoder.
2. The bf16 tile holds the **same numbers** as the E4M3 tile the FP8 route
   decodes, exactly: bf16 is lossless over every legal E4M3 byte (3 mantissa
   bits, exponents inside bf16's).  So an A8-vs-A16 A/B over these bytes is a
   matched pair on the W side -- the only difference is the A side.

CPU only, no CUDA, no checkpoint: ``PYTHONPATH=src python
experiments/window_gemv_a_side.py``.
"""
import torch

from tessera.serving.window import prepare_window


def main() -> int:
    torch.manual_seed(0)
    steps, cols, window_bits, rate = 64, 32, 14, 4
    body = torch.randint(0, 1 << rate, (steps, cols), dtype=torch.uint8)
    rates = [rate] * cols
    table_codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8)

    # A grid's ``native`` map: code -> E4M3 byte, with the 0x7F/0xFF fold the
    # E4M3 grid applies (``fp8_route.py:153`` passes exactly this as uint8).
    native = torch.arange(256, dtype=torch.uint8)
    native[0x7F] = 0x7E
    native[0xFF] = 0xFE

    # A side today: uint8 code map -> a tile of E4M3 bytes for ``_scaled_mm``.
    fp8_tile = prepare_window(body, rates, window_bits, table_codes, "cpu",
                              code_map=native).decode()
    # A side under review: the same wire, a FLOATING code map of those bytes'
    # values -> a bf16 tile for the stock GEMM or the GEMV's value family.
    values = native.view(torch.float8_e4m3fn).float().to(torch.bfloat16)
    bf16_tile = prepare_window(body, rates, window_bits, table_codes, "cpu",
                               code_map=values).decode()

    same = torch.equal(fp8_tile.view(torch.float8_e4m3fn).to(torch.bfloat16), bf16_tile)
    legal = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    legal = legal[torch.isfinite(legal)]
    lossless = torch.equal(legal.to(torch.bfloat16).float(), legal)

    print(f"fp8 tile  {tuple(fp8_tile.shape)} {fp8_tile.dtype}")
    print(f"bf16 tile {tuple(bf16_tile.shape)} {bf16_tile.dtype}")
    print(f"same weight values: {same}")
    print(f"bf16 lossless over every legal E4M3 byte ({legal.numel()}): {lossless}")
    ok = bool(same and lossless and bf16_tile.dtype is torch.bfloat16)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
