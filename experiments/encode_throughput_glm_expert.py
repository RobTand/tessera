"""Encode throughput at a real GLM-5.3-Flash routed-expert shape.

The whole-model question this answers: GLM's routed experts are 311.65e9
parameters (288 experts x 43 MoE layers x 3 projections).  If the k=2 Viterbi
runs at X GB/hour on one box, the encode campaign is 156 GB / X hours -- and
whether that is an afternoon or a fortnight decides whether the encode stage
needs prismabuild's cross-box fan-out or can run inline.

Ranked by power against the envelope, not utilisation: on GB10
``gpu_utilization`` reads ~96% for a stalled kernel and a saturated one alike.
"""
import argparse, json, subprocess, time
import torch
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.encode import encode_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode


def power_w():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)    # moe_intermediate_size
    ap.add_argument("--cols", type=int, default=4096)    # hidden_size
    ap.add_argument("--rate", type=int, default=7)
    ap.add_argument("--arity", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    grid = tuple_grid(E2M1_GRID, a.arity)
    forests = {a.rate: build_forest(a.rate, grid=grid)}
    torch.manual_seed(0)
    w = (torch.randn(a.rows, a.cols, device="cuda") * 0.02).bfloat16()
    rates = (a.rate,) * a.cols
    cc = ConvCode(memory=6)

    # warm up kernels/compile so the timed runs measure the encoder
    encode_unit(w, forests, rates, cc, rotation=RotationState.NONE,
                with_diagonals=False, completion=0, group=32, half=16)
    torch.cuda.synchronize()

    times, powers = [], []
    for _ in range(a.repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        encode_unit(w, forests, rates, cc, rotation=RotationState.NONE,
                    with_diagonals=False, completion=0, group=32, half=16)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        powers.append(power_w())

    best = min(times)
    params = a.rows * a.cols
    pps = params / best
    GLM_ROUTED = 311_653_564_416
    hours = GLM_ROUTED / pps / 3600
    print(json.dumps({
        "shape": [a.rows, a.cols], "rate": a.rate, "arity": a.arity,
        "seconds_per_unit": round(best, 4),
        "median_seconds": round(sorted(times)[len(times)//2], 4),
        "params_per_second": round(pps),
        "Mparams_per_second": round(pps / 1e6, 2),
        "power_w": powers,
        "glm_routed_params": GLM_ROUTED,
        "glm_encode_hours_one_box": round(hours, 2),
        "glm_encode_hours_two_boxes": round(hours / 2, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
