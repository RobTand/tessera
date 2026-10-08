"""The vectorized fragment repack writes the same words as the numpy repack it replaced.

``_fragment_wire_numpy_58e59caf0.py`` is ``src/tessera/fragment_wire.py`` at 58e59caf0, kept
verbatim for this check.  Both repack the same random BODY planes, start states and offsets.
"""
import importlib.util
import itertools
import json
import os
import sys

import torch

from tessera import fragment_wire as new
from tessera.wire import pack_body

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tessera._fw_old", os.path.join(HERE, "_fragment_wire_numpy_58e59caf0.py"),
                                              submodule_search_locations=None)
old = importlib.util.module_from_spec(spec)
old.__package__ = "tessera"
sys.modules[spec.name] = old            # a dataclass resolves its module through sys.modules
spec.loader.exec_module(old)

cases = []
for group, steps, rows, state in itertools.product(("gate_up", "down"), ((3, 3), (4, 4), (4, 3, 4, 3), (3,) * 8 + (4,) * 8),
                                                   (128, 259, 1024), (False, True)):
    if group == "down" and len(steps) % 2:
        continue
    g = torch.Generator().manual_seed(rows + len(steps))
    rates = tuple(r for r in steps for _ in range(32))
    count = 2 if group == "gate_up" else 1
    bodies = [torch.stack([torch.randint(0, 1 << r, (rows,), generator=g) for r in rates], 1) for _ in range(count)]
    start = torch.randint(0, 1 << 14, (count, len(rates)), generator=g, dtype=torch.int32) if state else None
    planes = tuple(pack_body(b, rates) for b in bodies)
    a = old.repack_fragment(planes, rates, rows=rows, cols=len(rates), projection_group=group, start_state=start, word_offset=96)
    b = new.repack_fragment(planes, rates, rows=rows, cols=len(rates), projection_group=group, start_state=start, word_offset=96)
    same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in
               ("words", "perm", "rates", "history_offsets", "unit_offsets", "expert_offsets"))
    cases.append({"group": group, "steps": list(steps), "rows": rows, "start_state": state, "equal": same})
print(json.dumps(cases))
assert all(c["equal"] for c in cases), "the vectorized repack differs from the numpy repack"
print(f"{len(cases)} cases equal")
