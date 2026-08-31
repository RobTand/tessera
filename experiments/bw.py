import sys, time, torch, subprocess; sys.path.insert(0,"/home/rob/tessera/src")
dev="cuda"
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
print("=== achievable read bandwidth on this box ===")
for mb in (64, 256, 1024):
    a = torch.empty(mb*1024*1024//4, dtype=torch.float32, device=dev).normal_()
    t = bench(lambda: a.sum())
    print(f"  reduce {mb:5d} MB: {t*1e6:8.0f} us -> {a.numel()*4/t/1e9:7.1f} GB/s")
    b = torch.empty_like(a)
    t = bench(lambda: b.copy_(a))
    print(f"  copy   {mb:5d} MB: {t*1e6:8.0f} us -> {a.numel()*8/t/1e9:7.1f} GB/s (r+w)")
    del a, b; torch.cuda.empty_cache()
print()
print(subprocess.run(["nvidia-smi","--query-gpu=power.draw,power.limit,clocks.sm,memory.used",
                      "--format=csv"],capture_output=True,text=True).stdout)
