"""ncu target: one warmed launch of decode or gemv at a chosen shape / M / split.

Usage: ncu_target2.py <kind> <rows> <cols> [M] [split]
`kind` is `decode` or `gemv`.  ncu serialises and times the kernel itself, so
this is the instrument to use when the box is carrying other work.
"""
import sys
sys.path.insert(0, "src")
import torch
from tessera import kernel_window as kw
from tessera.alphabet import E4M3_GRID

kind = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])
m = int(sys.argv[4]) if len(sys.argv) > 4 else 1
split = int(sys.argv[5]) if len(sys.argv) > 5 else None
rate = 4
torch.manual_seed(7)
body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
codes = torch.randint(0, 256, (1 << 14,), dtype=torch.uint8, device="cuda")
scale = torch.rand(rows, device="cuda") + 0.25
u = kw.prepare_window_unit(body, (rate,) * cols, 14, codes, E4M3_GRID, scale,
                           device="cuda")
x = torch.randn(m, cols, device="cuda", dtype=torch.bfloat16) * 0.1
for _ in range(10):
    if kind == "decode":
        u.decode()
    else:
        kw._gemv_impl(x, u.plane_words, u.offsets, u.rates, u.initial, u.value_table,
                      u.row_scale, u.rows, u.cols, u.window_bits, u.max_rate,
                      split=split)
torch.cuda.synchronize()
print("done", kind, rows, cols, m, split)
