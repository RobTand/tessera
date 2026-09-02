"""An ncu target for the window-body GEMV: one synthetic unit, warmed, then N
cold launches of the kernel alone.

    ncu --kernel-name regex:window_gemv_kernel --launch-skip 5 --launch-count 1 \\
        --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats \\
        --section SpeedOfLight --section LaunchStats --section ComputeWorkloadAnalysis \\
        python experiments/ncu_window_gemv_target.py --rows 2560 --cols 9728 --M 1

No ``dram__*`` counters exist on GB10, so the DRAM fraction is not an ncu
number here; the receipt takes it from wall-clock over the measured streaming
read instead.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/rob/tmp/torch-ext-gemv")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_kernel_window_gemv import cold_units, parse_plan  # noqa: E402
from tessera import kernel_window_gemv as kg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2560)
    ap.add_argument("--cols", type=int, default=9728)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--launches", type=int, default=10)
    ap.add_argument("--ablation", type=int, default=0)
    args = ap.parse_args()
    plan = parse_plan(args.plan) or kg.default_plan(args.rows, args.cols, args.M)
    rot = cold_units(args.rows, args.cols, plan=plan, M=args.M)
    x = torch.randn(args.M, args.cols, dtype=torch.bfloat16, device="cuda")
    scratch = torch.zeros(args.M, args.rows, dtype=torch.float32, device="cuda")
    for _ in range(args.launches):
        kg.window_gemv(rot.next(), x, out=scratch, ablation=args.ablation)
    torch.cuda.synchronize()
    print("done", args.rows, args.cols, args.M, plan, "items", int(rot.items[0].items_for(args.M)[0].shape[0]))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
