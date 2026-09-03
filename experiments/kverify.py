"""One-hot probes isolate the decode from accumulation order: x = e_k makes the
GEMV return column k of W exactly, so any disagreement is a decode bug rather
than fp32 summation noise."""
import sys, time, subprocess, threading, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit, decode_codes_mixed, materialize_nvfp4
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.kernel import (pack_kernel_planes, build_value_lut, tessera_gemv_wide,
                            nvfp4_gemv_sliced, pack_nvfp4_column_major)
dev="cuda"; CC=ConvCode(memory=6); F={r:build_forest(r) for r in (1,2,3)}
torch.manual_seed(0)
rows, cols = 17408, 5120
W = (torch.randn(rows, cols, device=dev)*0.02).contiguous()
rates = bresenham_rate_schedule(root_from_q256(768), cols)
u = encode_unit(W, F, rates, CC, rotation=RotationState.NONE, with_diagonals=False, released_positions=0)
codes = decode_codes_mixed(u, F, CC)
_p, e4m3, gs = materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half)
sel, pnt = pack_kernel_planes(u.body_bits)
vlut = build_value_lut(F[3], CC, dev)
e4m3_t = e4m3.reshape(rows, cols//16).t().contiguous()
Wd = reconstruct_unit(u, F, CC).float()

exact = True
for k in (0, 1, 5, 6, 7, 17, 1023, 2048, 5119):
    x = torch.zeros(cols, device=dev); x[k] = 1.0
    y = tessera_gemv_wide(x, sel, pnt, vlut, e4m3_t, gs, rows, cols, lanes=128, split_k=32)
    same = torch.equal(y, Wd[:, k])
    exact &= same
    if not same:
        print(f"  column {k}: max|d| = {(y-Wd[:,k]).abs().max():.3e}  mismatches={(y!=Wd[:,k]).sum()}")
print(f"one-hot decode bit-exact on 9 columns (incl. the first 8, where the "
      f"history window straddles the pad): {exact}")

x = torch.randn(cols, device=dev); ref = Wd @ x
stop = threading.Event(); watts=[]
def poll():
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        try: watts.append(float(r))
        except: pass
def bench(fn, n=200):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    th = threading.Thread(target=poll); stop.clear(); watts.clear(); th.start()
    t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    stop.set(); th.join()
    return dt, (max(watts) if watts else 0.0)
tt, wt = bench(lambda: tessera_gemv_wide(x, sel, pnt, vlut, e4m3_t, gs, rows, cols, lanes=128, split_k=32))
packed_t = pack_nvfp4_column_major(codes)
tn, wn = bench(lambda: nvfp4_gemv_sliced(x, packed_t, e4m3_t, gs, rows, cols, block_n=512, split_k=128))
wb = rows*cols*(3/8+1/16); wbn = rows*cols*(0.5+1/16)
print(f"\n{'kernel':<16}{'us':>8}{'GB/s':>9}{'%peak':>8}{'peak W':>9}{'bpp':>7}")
print(f"{'tessera wide':<16}{tt*1e6:8.0f}{wb/tt/1e9:9.1f}{100*wb/tt/1e9/246:7.1f}%{wt:9.1f}{3.5:7.4f}")
print(f"{'nvfp4 matched':<16}{tn*1e6:8.0f}{wbn/tn/1e9:9.1f}{100*wbn/tn/1e9/246:7.1f}%{wn:9.1f}{4.5:7.4f}")
print(f"\ntessera is {tn/tt:.3f}x the speed of matched NVFP4 on {wb/wbn:.3f}x the resident bytes")
print(f"work per joule ratio: {(tn*wn)/(tt*wt):.3f}x" if wt and wn else "")
