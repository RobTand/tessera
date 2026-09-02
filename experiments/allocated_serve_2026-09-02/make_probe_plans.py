"""Two mechanism arms that decompose the layer-0 separator.

The separator says the allocation loses 1.93x on the seven units it priced.
`l0_roles.py` says the allocator bought that trade on gate/up/v and paid for it
on down_proj, which it cut from 1006 to 749 q256 (3.93 -> 2.93 b/wt).  These two
arms move only down_proj and hold everything else, so between them they say
whether the served loss lives on that one cut.

  down749      = the uniform arm with down_proj alone cut to R749
  alloc-nodown = the allocation with down_proj alone restored to R1006

Bytes are deliberately NOT matched: these are mechanism arms, not controls.
"""
import json

BASE = '/home/rob/tmp/alloc-plans'
unif = json.load(open(f'{BASE}/plan_l0_unif1006.json'))
alloc = json.load(open(f'{BASE}/plan_full_4.0_asalloc.json'))
DOWN = 'model.layers.0.mlp.down_proj.weight'

unif[DOWN] = {'grid': 'E4M3', 'q256': 749}
json.dump(unif, open(f'{BASE}/plan_l0_down749.json', 'w'), indent=1)

alloc = {k: v for k, v in alloc.items()}
alloc[DOWN] = {'grid': 'E4M3', 'q256': 1006}
json.dump(alloc, open(f'{BASE}/plan_l0_allocnodown.json', 'w'), indent=1)
print('wrote plan_l0_down749.json and plan_l0_allocnodown.json')
