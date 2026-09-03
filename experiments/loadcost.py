"""What does Tessera cost at SERVE time?  Three separate questions, and only
the first is free.  1) GEMM: already attested bit-identical NVFP4 bytes.
2) Load: trellis replay must run once per unit before the kernel sees anything.
3) VRAM: resident is 4.5 bpp either way.  This measures (2)."""
import sys, glob, time, torch
sys.path.insert(0, "/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.decode import decode_codes_mixed, materialize_nvfp4
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.grammar import bresenham_rate_schedule, root_from_q256

dev = "cuda"; CC = ConvCode(memory=6); F = {r: build_forest(r) for r in (1, 2, 3)}
files = sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))
W = None
for path in files[:4]:
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"):
                W = f.get_tensor(k).to(dev).float().contiguous()
if W is None: raise SystemExit("tensor not found")
# A load figure is a figure for a SHAPE.  This harness used to narrow any
# non-conforming width to a superblock multiple here, silently -- so every
# throughput number it ever published covered only conforming shapes, and the
# trailing partial superblock (the shape the layout fix in #22 is about) had
# never been measured at all.  A silent truncation reads as "covered
# everything" when it did not.  Measure the width we were given, say so, and
# let the encoder refuse if it must (#40).
TRUNCATE = "--truncate" in sys.argv
if W.shape[1] % 256 and TRUNCATE:
    W = W[:, : W.shape[1] // 256 * 256].contiguous()
rows, cols = W.shape
SUPERBLOCKS, PARTIAL = -(-cols // 256), cols % 256
SHAPE_NOTE = (f"{cols} columns = {SUPERBLOCKS} superblocks, the last holding "
              f"{PARTIAL} of 256" if PARTIAL else
              f"{cols} columns = {SUPERBLOCKS} whole superblocks")
if PARTIAL and TRUNCATE:
    SHAPE_NOTE += "  [--truncate: NARROWED, this is not the shipping shape]"
print(f"shape: {SHAPE_NOTE}")
rates = bresenham_rate_schedule(root_from_q256(768), cols)
u = encode_unit(W, F, rates, CC, rotation=RotationState.NONE,
                with_diagonals=False, released_positions=int(0.125 * W.numel()))

def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n

t_replay = timed(lambda: decode_codes_mixed(u, F, CC))
codes = decode_codes_mixed(u, F, CC)
t_pack = timed(lambda: materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half))
params = rows * cols
print(f"tensor {tuple(W.shape)}  {params/1e6:.1f}M params   rows(trellis steps)={rows}")
print(f"shape covered:   {SHAPE_NOTE}")
print(f"trellis replay   {t_replay*1e3:8.1f} ms   {params/t_replay/1e6:9.1f} M param/s")
print(f"nvfp4 pack       {t_pack*1e3:8.1f} ms   {params/t_pack/1e6:9.1f} M param/s")
tot = t_replay + t_pack
print(f"total decode     {tot*1e3:8.1f} ms   {params/tot/1e6:9.1f} M param/s")
for name, n in [("GLM-5.3-Flash body ~355B", 355e9), ("Qwen3.8-27B ~27B", 27e9)]:
    print(f"  extrapolated to {name}: {n/(params/tot)/60:8.1f} min of load-time decode")
