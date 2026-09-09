"""The best form on a zero-row, positive-column input.

The reference accepts rows=0 with cols>0 and returns an empty states and
sse 0.0.  The best form launches _init_best, which stores back step 0 into
a back of shape [nmax, 0, LOW], and then _final_best at step = steps - 1 = -1.

Bounded on purpose: before the fix this poisons the CUDA context, so it runs
in an action of its own and measures nothing.

Measured on sparklina, PrismaBuild GPU, one action each:

    pre-fix   52cb04a22264   Triton Error [CUDA]: an illegal memory access
                             was encountered, raised from _init_best
    post-fix  e0872d56cd35   fused (0, 8) 0.0, the reference's answer
"""
import os
import traceback

os.environ.setdefault("TESSERA_WINDOW_BEST_FORM", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
from tessera.alphabet import E4M3_GRID
from tessera.encode import grid_vector_table, viterbi_window, window_table

L, R, COLS = 12, 3, 8
dev = "cuda"
vectors = grid_vector_table(E4M3_GRID, dev)
codes = window_table(E4M3_GRID, L, sigma=1.0, seed=0, device=dev)
table = vectors[codes.long()][:, :1].contiguous()
targets = torch.zeros(0, COLS, device=dev)

print("best form:", os.environ["TESSERA_WINDOW_BEST_FORM"])
print("targets", tuple(targets.shape), "table", tuple(table.shape))

ref_states, ref_sse = viterbi_window(targets, table, L, R, impl="reference")
print("reference:", tuple(ref_states.shape), ref_sse)

try:
    states, sse = viterbi_window(targets, table, L, R, impl="fused")
    torch.cuda.synchronize()
    print("fused:", tuple(states.shape), sse)
    print("RESULT: no fault raised")
except Exception:
    traceback.print_exc()
    print("RESULT: the fused best form faulted on a zero-row input")
