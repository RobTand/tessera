"""Does the (G, k) ladder hold in the FP8 band (4.5-8.5 bpp), on real tensors?

`tessera-8-and-the-payload-grid.md` measured this band on ONE tensor, before the
k-tuple lever existed, and its ladder had 1.0 bpp gaps.  This re-measures it on
five real Linears with the tuple rungs filled in, against the comparator that
matters up here: scalar FP8, not NVFP4.

Every arm replays its own body and asserts the decoder recovers the encoder's
codes.  Rate 9 taught the lesson: a widened rate produces PLAUSIBLE wrong
numbers, and a monotone-looking column is not evidence of correctness.
"""
import sys, glob, torch; sys.path.insert(0, "/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest, tuple_grid, lloyd_max_grid, E2M1_GRID, E4M3_GRID
from tessera.encode import encode_unit, _pack_scales, e2m1_value_table
from tessera.decode import reconstruct_unit, decode_codes
from tessera.trellis import ConvCode
from tessera.manifest import RotationState

dev = "cuda"; CC = ConvCode(memory=6)

def bpp(G, k):
    return (k * (G.bit_length() - 1) - 1) / k + 0.5

# name -> (grid, bpp, lane).  R is always the cap: Result 2 of the k-tuple doc
# showed k=2 below its cap is dominated, so there is nothing to sweep.
ARMS = {}
for G in (32, 64, 128, 256):
    ARMS[f"free{G} k=1"] = (lloyd_max_grid(G), bpp(G, 1), "kernel")
for G in (32, 64, 128):
    ARMS[f"free{G} k=2"] = (tuple_grid(lloyd_max_grid(G), 2), bpp(G, 2), "kernel")
# The stock lane cannot use a free grid; E4M3 is the grid it CAN materialise.
#
# R is NOT capped at 4.  bpp = R + 0.5, so TESSERA-8 reaches 2.5 and 3.5 bpp the
# moment you ask -- the 256-code grid sets the CAP, not the rate.  Below the cap
# the forest picks 2^(R+1) anchors OUT OF 256, so a 3.5 bpp TESSERA-8 body is a
# trellis over sixteen E4M3 values it CHOSE, where TESSERA-4 gets the sixteen
# E2M1 values it was dealt.  That selection freedom is the thing to measure.
for R in (2, 3, 4, 5, 6, 7):
    ARMS[f"E4M3 k=1 R={R}"] = (E4M3_GRID, R + 0.5, "stock")
# The reference points the sub-4.5 rungs have to beat.
ARMS["E2M1 k=1 R=3"] = (E2M1_GRID, 3.5, "stock")
ARMS["E2M1 k=2 R=7"] = (tuple_grid(E2M1_GRID, 2), 4.0, "stock")
ARMS["free16 k=1 R=3"] = (lloyd_max_grid(16), 3.5, "kernel")
ARMS["free16 k=2 R=7"] = (tuple_grid(lloyd_max_grid(16), 2), 4.0, "kernel")

FORESTS, RATES = {}, {}
for name, (grid, _, _) in ARMS.items():
    R = int(name.split("R=")[1]) if "R=" in name else grid.rate_cap
    RATES[name] = R
    FORESTS[name] = {R: build_forest(R, grid=grid)}

def rtn(W, values, peak):
    _, _, eff = _pack_scales(W, 32, 16, peak=peak)
    scale = torch.repeat_interleave(eff, 16).reshape(W.shape)
    t = (W / scale).unsqueeze(-1)
    return ((values[((t - values) ** 2).argmin(-1)] * scale - W).norm() / W.norm()).item()

E2M1_V = e2m1_value_table(dev)
E4M3_V = torch.tensor(E4M3_GRID.values, device=dev, dtype=torch.float32)

def measure(W):
    out = {}
    cols = W.shape[1]
    for name, (grid, _, _) in ARMS.items():
        R = RATES[name]
        u = encode_unit(W, FORESTS[name], (R,) * cols, CC, rotation=RotationState.NONE,
                        with_diagonals=False, completion=0)
        # The gate: the decoder must replay the encoder's own codes, exactly.
        replayed = decode_codes(u, FORESTS[name][R], CC, completion=0)
        assert torch.equal(replayed, u.codes), (
            f"{name}: replay disagrees with the encoder on "
            f"{(replayed != u.codes).sum().item()} of {u.codes.numel()} codes"
        )
        rec = reconstruct_unit(u, FORESTS[name], CC, completion=0).float()
        out[name] = ((rec - W).norm() / W.norm()).item()
    out["NVFP4 RTN"] = rtn(W, E2M1_V, 6.0)
    out["FP8 RTN (E4M3)"] = rtn(W, E4M3_V, 448.0)
    return out

WANT = ("gate_proj", "down_proj", "up_proj", "q_proj", "o_proj")
order = sorted(ARMS, key=lambda n: (ARMS[n][1], n)) + ["NVFP4 RTN", "FP8 RTN (E4M3)"]
BPP = dict({n: ARMS[n][1] for n in ARMS}, **{"NVFP4 RTN": 4.5, "FP8 RTN (E4M3)": 8.5})

rows = []
seen = set()
for path in sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
                             "snapshots/*/model*.safetensors"))[:3]:
    with safe_open(path, "pt") as f:
        for key in f.keys():
            role = next((w for w in WANT if key.endswith(w + ".weight")), None)
            if role is None or role in seen or ".layers.1." not in key:
                continue
            W = f.get_tensor(key)
            if W.ndim != 2 or W.shape[0] < 512 or W.shape[1] < 512:
                continue
            seen.add(role)
            W = W[:1024, :2048].to(dev).float().contiguous()
            res = measure(W)
            rows.append((role, res))
            print(f"  {role} done", flush=True)

print(f"\n{'arm':<20}{'bpp':>6}{'lane':>8}  " + "".join(f"{r:>11}" for r, _ in rows)
      + f"{'mean/FP8':>11}{'worst/FP8':>11}")
for name in order:
    ratios = [res[name] / res["FP8 RTN (E4M3)"] for _, res in rows]
    lane = ARMS[name][2] if name in ARMS else "-"
    print(f"{name:<20}{BPP[name]:>6.1f}{lane:>8}  "
          + "".join(f"{res[name]:>11.5f}" for _, res in rows)
          + f"{sum(ratios)/len(ratios):>11.3f}{max(ratios):>11.3f}")
