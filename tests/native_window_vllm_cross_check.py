"""Cross-check: native window MoE (folded BF16) vs vLLM's stock unquantized
fused_experts on identical weights and routing.

The research BF16 contract is the one designed to match stock unquantized
math: weights are bf16(values * row_scale), so vLLM's own Triton kernels are
a real oracle for the two-stage native adapter.  Run inside the vLLM image
(vLLM work is exempt from PrismaBuild; this is a bounded direct check).

Usage:
  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests \
    -v <worktree>:/work:ro -v <this file>:/control/cross.py:ro \
    --entrypoint python3 <image> /control/cross.py
"""
import json
import sys

import torch

sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/tests")

from tessera import native_window_moe as nwm          # noqa: E402
import test_window_gemm_grouped as tg                 # noqa: E402

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts   # noqa: E402


def folded(e):
    """The research contract: bf16(values * row_scale), exactly decode_folded."""
    w = e.values.float().cuda()[e.states.cuda()]
    return (w * e.scale[:, None]).bfloat16()


def main():
    torch.manual_seed(0)
    hid, inter, experts = 192, 96, 3       # vLLM's fused_experts output width == hidden width
    t, k = 16, 2
    seeds = [131, 132, 133]
    gu_stack = [tg.Expert(2 * inter, hid, (4,) * hid, s) for s in seeds]
    dn_stack = [tg.Expert(hid, inter, (4,) * inter, s + 10) for s in seeds]
    adapter = nwm.prepare_native_window_moe(
        [e.unit for e in gu_stack], [e.unit for e in dn_stack], arithmetic="folded")

    w1 = torch.stack([folded(e) for e in gu_stack]).contiguous()      # [E, 2I, H]
    w2 = torch.stack([folded(e) for e in dn_stack]).contiguous()      # [E, H, I]

    x = torch.randn(t, hid, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")

    report = {"device": torch.cuda.get_device_name(), "arms": []}
    ok = True
    for flag in (False, True):
        stock = fused_experts(x, w1, w2, rw, ids, activation=MoEActivation.SILU,
                              apply_router_weight_on_input=flag,
                              global_num_experts=experts)
        native = adapter(x, ids, rw, apply_router_weight_on_input=flag)
        if flag is False:
            print("shapes", tuple(x.shape), tuple(w1.shape), tuple(w2.shape),
                  tuple(stock.shape), tuple(native.shape))
        diff = (stock.float() - native.float()).abs()
        mag = stock.float().abs().max().clamp_min(1e-6)
        rel = float(diff.max() / mag)
        passed = rel < 2e-2
        ok = ok and passed
        report["arms"].append({
            "apply_router_weight_on_input": flag,
            "max_abs": float(diff.max()),
            "max_over_mag": rel,
            "passed": passed,
        })
    report["passed"] = ok
    print(json.dumps(report))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
