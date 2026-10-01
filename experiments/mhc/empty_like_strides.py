#!/usr/bin/env python3
"""Strides of ``torch.empty_like`` on a non-dense channel-last slice (device-independent; CPU is enough).

KDA's prefill conv (vLLM ``causal_conv1d_fn``) allocates its output with ``torch.empty_like(x)``
after ``x = x.to(conv_states.dtype)``. Run per q/k/v slice of the merged (3P, T) channel-last
input, ``x`` is a non-dense (P, T) view with strides (1, 3P). If ``empty_like`` keeps the layout
permutation, the output is dense token-major (strides (1, P)), so q/k/v reach FlashKDA dense and
its three ``.contiguous()`` copies become no-ops. Prints one JSON line.
"""
import json

import torch

P, T = 4096, 2048
x = torch.empty(T, 3 * P, dtype=torch.bfloat16).transpose(0, 1)
rows = []
for s in range(3):
    sl = x[s * P:(s + 1) * P]
    same = sl.to(torch.bfloat16)
    o = torch.empty_like(same)
    rows.append({"slice": s, "x_stride": list(sl.stride()), "to_same_dtype_is_self": same is sl,
                 "out_stride": list(o.stride()), "out_token_major_dense": o.transpose(0, 1).is_contiguous()})
print(json.dumps({"torch": torch.__version__, "P": P, "T": T, "rows": rows}))
