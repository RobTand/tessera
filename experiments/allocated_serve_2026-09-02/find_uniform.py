import sys, json
from fractions import Fraction
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/rob/pq-wt/tessera-continuous")
sys.path.insert(0, "/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6/experiments")
from prismaquant.tessera_formats import artifact_bpp
from plan_from_layer_config import body_weights

shapes = body_weights(Path("/home/rob/models/Qwen3-0.6B"))
mult = Counter(shapes.values())
print("shape multiset:", dict(mult))
params = sum(r * c * n for (r, c), n in mult.items())
print("params", params)
targets = {"3.0": Fraction(1321320448, 440401920),
           "4.0": Fraction(1761722368, 440401920),
           "5.0": Fraction(2202124288, 440401920)}
for label, tgt in targets.items():
    best = None
    for q in range(256, 2049):
        try:
            bits = sum(Fraction(artifact_bpp("TESSERA_E4M3_K1", q, shape=(r, c))) * r * c * n
                       for (r, c), n in mult.items())
        except Exception:
            continue
        bpp = bits / params
        d = abs(bpp - tgt)
        if best is None or d < best[0]:
            best = (d, q, bpp, bits)
    d, q, bpp, bits = best
    print(f"target {label}: {float(tgt):.6f} bpp -> uniform q256={q} at {float(bpp):.6f} bpp "
          f"(delta {float(bpp - tgt):+.6f}), bits {bits}")
