"""The layer-0-only pair: the allocation on exactly the units it priced.

The whole-body arms differ in two things at once -- the allocation AND the
broadcast of a layer-0 answer to 28 depths.  This pair changes only the first:
the same seven Linears, once at the rungs the allocator chose and once at the
byte-matched uniform rung, everything else BF16.  Layer 0's seven units carry
15728640 params and the allocator charged 62918656 bits for them, i.e. the same
4.000260 bpp the whole body carries, so R1006 is the matched uniform here too.
"""
import json
import pathlib

src = json.loads(pathlib.Path('/home/rob/tmp/alloc-plans/plan_full_4.0_asalloc.json').read_text())
twin = {k: (v if v == "BF16" else {"grid": "E4M3", "q256": 1006})
        for k, v in src.items()}
out = pathlib.Path('/home/rob/tmp/alloc-plans/plan_l0_unif1006.json')
out.write_text(json.dumps(twin, indent=1))
print(f"wrote {out} ({len(twin)} units)")
