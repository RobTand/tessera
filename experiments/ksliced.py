import sys, time, torch; sys.path.insert(0,"/home/rob/tessera/src")
from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit, decode_codes_mixed, materialize_nvfp4
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.kernel import (pack_kernel_planes, build_history_lut, tessera_gemv_sliced,
                            nvfp4_gemv_sliced, pack_nvfp4_column_major)

dev="cuda"; CC=ConvCode(memory=6); F={r:build_forest(r) for r in (1,2,3)}
torch.manual_seed(0)
rows, cols = 17408, 5120
W = (torch.randn(rows, cols, device=dev) * 0.02).contiguous()
rates = bresenham_rate_schedule(root_from_q256(768), cols)
u = encode_unit(W, F, rates, CC, rotation=RotationState.NONE, with_diagonals=False, released_positions=0)
codes = decode_codes_mixed(u, F, CC)
_p, e4m3, gs = materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half)
sel, pnt = pack_kernel_planes(u.body_bits)
lut = build_history_lut(F[3], CC, dev)
e4m3_t = e4m3.reshape(rows, cols//16).t().contiguous()
Wd = reconstruct_unit(u, F, CC).float()
x = torch.randn(cols, device=dev); ref = Wd @ x

def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n

wb = rows*cols*(3/8 + 1/16)
best=None
for BN in (128,256,512):
    for SK in (16,32,64,128,256):
        out = torch.zeros(rows, device=dev)
        f = lambda: tessera_gemv_sliced(x, sel, pnt, lut, e4m3_t, gs, rows, cols, block_n=BN, split_k=SK)
        y = f(); rel = ((y-ref).norm()/ref.norm()).item()
        if rel > 1e-4: print(f"  BN={BN} SK={SK} REL={rel:.2e} <-- wrong"); continue
        t = bench(f)
        if best is None or t < best[0]: best=(t,BN,SK,rel)
        print(f"  BN={BN:4d} SK={SK:3d}  {t*1e6:8.0f} us  {wb/t/1e9:7.1f} GB/s  {100*wb/t/1e9/246:5.1f}% peak  rel={rel:.1e}")
t,BN,SK,rel = best
packed_t = pack_nvfp4_column_major(codes)
bestn=None
for BN2 in (128,256,512):
    for SK2 in (16,32,64,128,256):
        f2 = lambda: nvfp4_gemv_sliced(x, packed_t, e4m3_t, gs, rows, cols, block_n=BN2, split_k=SK2)
        y2 = f2()
        if ((y2-ref).norm()/ref.norm()).item() > 1e-4: continue
        t2 = bench(f2)
        if bestn is None or t2 < bestn[0]: bestn=(t2,BN2,SK2)
tn,BN2,SK2 = bestn
wbn = rows*cols*(0.5 + 1/16)
print(f"\nmatched NVFP4 comparator: BN={BN2} SK={SK2}  {tn*1e6:.0f} us  {wbn/tn/1e9:.1f} GB/s = {100*wbn/tn/1e9/246:.1f}% peak")
print(f"  tessera vs nvfp4:  time {tn/t:.3f}x   resident bytes {wb/wbn:.3f}x  ({wb/(rows*cols)*8:.4f} vs {wbn/(rows*cols)*8:.4f} bpp)")
tb = bench(lambda: Wd.to(torch.bfloat16) @ x.to(torch.bfloat16))
print(f"\nBEST sliced: BN={BN} SK={SK}  {t*1e6:.0f} us  {wb/t/1e9:.1f} GB/s = {100*wb/t/1e9/246:.1f}% peak")
print(f"interleaved best was 1729 us (22.6 GB/s, 9.2%)  ->  {1729e-6/t:.2f}x")
print(f"torch bf16 GEMV {tb*1e6:.0f} us  ->  tessera {tb/t:.2f}x faster on {wb/(rows*cols*2):.3f}x the weight bytes")
