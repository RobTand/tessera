"""Two levers that cost no bits: the code's memory, and the codebook's shape.

``ConvCode.memory`` defaults to 6 and has never been moved.  It is close to
free: the state count is an ENCODER cost only -- ``TCQ.decode`` replays the
state machine one step per position whatever the memory is -- and it changes no
plane, no width and no container.  It is wire (two encoders at different memory
do not decode each other's streams) but it is wire the encoder profile id
already covers, so raising it is a re-encode, not a format change.

The second lever is the codebook.  ``tuple_grid(E2M1_GRID, 2)`` is a tensor
product: sixteen scalar levels crossed with themselves, which spends points on
the corners of a square where the weight density is a round blob.  A free
256-point 2D codebook measured 1.203x better than the product grid under
nearest-neighbour and 1.103x under the trellis; the shortfall is partly that
its partition was a crude greedy colouring while the grid gets ``coset``.

Both are swept here against a control that is not optional: the same Viterbi in
this file, driven by ``TCQ(...).subsets`` -- Tessera's own partition, taken from
Tessera's own code -- must reproduce ``encode_unit`` at memory 6.  Arm B's
number means nothing unless arm D lands on arm A.
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.manifest import RotationState
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd, viterbi

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
EXL3_K4 = 0.05653

#: Published maximum-free-distance rate-1/2 generators, the same Lin & Costello
#: table ``trellis._ODS_GENERATORS`` draws from, extended to the orders it stops
#: short of.  Listed here rather than edited into the source because raising the
#: default is a wire change and has to be earned by the numbers below first.
MORE = {7: (0o247, 0o371), 9: (0o1131, 0o1537), 10: (0o2473, 0o3217)}


def code_at(memory):
    return ConvCode(memory=memory, generators=MORE.get(memory))


def labels(subsets, anchors, size, device):
    """``TCQ.subsets`` gives positions into ``anchors``; the Viterbi wants a
    per-code label.  Going through the anchor list rather than assuming
    anchors == range(size) is what keeps this honest at rates below the cap."""
    lab = torch.full((size,), -1, dtype=torch.long, device=device)
    for u, group in enumerate(subsets):
        for position in group:
            lab[anchors[position]] = u
    assert int((lab < 0).sum()) == 0, "the partition did not cover the grid"
    return lab


def run(seq, cb, lab, code, chunk=None):
    """Viterbi in column blocks -- backpointers are (T, B, 2^memory)."""
    T, B, _ = seq.shape
    chunk = chunk or max(32, (1 << 29) // (T * code.states))
    return torch.cat([viterbi(seq[:, i:i + chunk], cb, lab, code)
                      for i in range(0, B, chunk)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--memories", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--out", default="experiments/results/tessera_memory_and_codebook.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, code_at(6)).subsets, forest.anchors, grid.size, "cuda")

    acc = {}
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            R, C = w.shape
            nrm = torch.linalg.norm(w)
            rel = lambda r: (torch.linalg.norm(w - r.float()) / nrm).item()
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            un = lambda q: (q.reshape(R // 32, 32, C) * s).reshape(R, C)
            seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
            back = lambda q: un(q.permute(0, 2, 1).reshape(R, C))

            u = encode_unit(w, {grid.rate_cap: forest}, (grid.rate_cap,) * C,
                            code=code_at(6), rotation=RotationState.NONE,
                            completion=0, group=32, half=16)
            acc.setdefault("A  encode_unit  E2M1x2/coset  m=6", []).append(
                rel(reconstruct_unit(u, {grid.rate_cap: forest}, code_at(6))))

            cb = lloyd(seq.reshape(-1, 2)[torch.randperm(R * C // 2)[:300000]], 256)
            lab = colour4(cb)
            for m in a.memories:
                c = code_at(m)
                acc.setdefault(f"D  this Viterbi  E2M1x2/coset  m={m}", []).append(
                    rel(back(run(seq, vals, coset, c))))
                acc.setdefault(f"B  free Lloyd    colour4      m={m}", []).append(
                    rel(back(run(seq, cb, lab, c))))
            print(f"  {layer:>3} {proj:<10} done")

    ref = sum(acc["A  encode_unit  E2M1x2/coset  m=6"]) / len(acc["A  encode_unit  E2M1x2/coset  m=6"])
    print(f"\n{'arm':<40}{'rel err':>10}{'vs A':>8}{'vs EXL3':>9}")
    for k in sorted(acc, key=lambda k: (k[0], sum(acc[k]))):
        m = sum(acc[k]) / len(acc[k])
        print(f"{k:<40}{m:>10.5f}{ref/m:>7.3f}x{m/EXL3_K4:>8.2f}x")
    print(f"{'EXL3 K=4 (reference, 4.0117 bpp)':<40}{EXL3_K4:>10.5f}"
          f"{ref/EXL3_K4:>7.3f}x{1.0:>8.2f}x")
    d6 = sum(acc["D  this Viterbi  E2M1x2/coset  m=6"]) / 4
    print(f"\ncontrol: D(m=6) {d6:.5f} vs A {ref:.5f} -> {abs(d6/ref-1)*100:.2f}% "
          f"{'OK' if abs(d6/ref-1) < 0.01 else '*** VITERBI DOES NOT MATCH ***'}")
    json.dump(acc, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}   (all arms 4.00 bpp: 3.5 payload + 0.5 scale)")


if __name__ == "__main__":
    main()
