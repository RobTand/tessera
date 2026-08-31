import sys, time, itertools, torch; sys.path.insert(0,"/home/rob/tessera/src")
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
Wd = reconstruct_unit(u, F, CC).float(); x = torch.randn(cols, device=dev); ref = Wd @ x
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
wb = rows*cols*(3/8+1/16); wbn = rows*cols*(0.5+1/16)
best=None
for L, SK in itertools.product((32,64,128,256),(16,32,64,128)):
    if L*8 > rows: continue
    f = lambda: tessera_gemv_wide(x, sel, pnt, vlut, e4m3_t, gs, rows, cols, lanes=L, split_k=SK)
    try: y = f()
    except Exception as e: continue
    rel = ((y-ref).norm()/ref.norm()).item()
    if rel > 1e-4: 
        if best is None: print(f"  L={L} SK={SK} rel={rel:.2e} WRONG")
        continue
    t = bench(f)
    if best is None or t < best[0]: best=(t,L,SK,rel)
if best is None: raise SystemExit("no correct config")
t,L,SK,rel = best
packed_t = pack_nvfp4_column_major(codes)
bn=None
for BN2, SK2 in itertools.product((128,256,512),(32,64,128,256)):
    f2 = lambda: nvfp4_gemv_sliced(x, packed_t, e4m3_t, gs, rows, cols, block_n=BN2, split_k=SK2)
    if ((f2()-ref).norm()/ref.norm()).item() > 1e-4: continue
    t2 = bench(f2)
    if bn is None or t2 < bn[0]: bn=(t2,BN2,SK2)
tn = bn[0]
tb = bench(lambda: Wd.to(torch.bfloat16) @ x.to(torch.bfloat16))
print(f"tessera wide  L={L} SK={SK}   {t*1e6:7.0f} us  {wb/t/1e9:7.1f} GB/s  {100*wb/t/1e9/246:5.1f}% peak  rel={rel:.1e}")
print(f"nvfp4 matched                 {tn*1e6:7.0f} us  {wbn/tn/1e9:7.1f} GB/s  {100*wbn/tn/1e9/246:5.1f}% peak")
print(f"torch bf16 GEMV               {tb*1e6:7.0f} us  {rows*cols*2/tb/1e9:7.1f} GB/s")
print(f"\ntessera vs nvfp4:  time {tn/t:.3f}x   resident 3.5000 vs 4.5000 bpp (0.778x)")
print(f"prior sliced best was 470 us -> {470e-6/t:.2f}x")
