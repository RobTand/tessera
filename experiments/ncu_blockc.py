"""ncu target: one warmed decode launch at a chosen ``block_c``, pricing lever 1.

Usage: ncu_blockc.py <rows> <cols> <block_c> [code|value]
`code` is the FP8 family's path -- the uint8 E4M3 table `decode_fp8_tile`
uses.  `value` is the float table, which is a different kernel: different
element size, different output dtype, different register pressure.
"""
import sys
sys.path.insert(0, "src")
import torch
from tessera import kernel_window as kw
from tessera.alphabet import E4M3_GRID

rows, cols, block_c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
which = sys.argv[4] if len(sys.argv) > 4 else "code"
rate = 4
torch.manual_seed(7)
body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
codes = torch.randint(0, 256, (1 << 14,), dtype=torch.uint8, device="cuda")
scale = torch.rand(rows, device="cuda") + 0.25
u = kw.prepare_window_unit(body, (rate,) * cols, 14, codes, E4M3_GRID, scale,
                           device="cuda")
table = u.code_table if which == "code" else u.value_table
print("table dtype", table.dtype, flush=True)
for _ in range(10):
    kw._decode_impl(u.plane_words, u.offsets, u.rates, u.initial, table,
                    u.rows, u.cols, u.window_bits, u.max_rate, block_c=block_c)
torch.cuda.synchronize()
print("done", rows, cols, block_c, which)
