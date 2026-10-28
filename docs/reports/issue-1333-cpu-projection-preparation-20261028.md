# #1333 CPU projection preparation — PRI-684

This is a preparation slice, not source acceptance or serving qualification.
Exact implementation base: `17b3fdb396410698c9a142615345f046d2f94a5d`.

## Scope and owners

The new `tools/tessera_projection_cpu.py` harness consumes explicit JSON layer
fixtures. Scheme validation, wire parsing, shard planning and slicing stay
with their existing serving owners. The harness does not install another
constructor hook, infer placement from module names, or change any runtime
contract, decoder, recipe, pin, default or qualification rule.

The packet binds its normalized scheme and input layer fixture. A supplied
wire must match its declared SHA-256 and pass the production reader. Each
local role is decoded on CPU and compared exactly with the parent role's
declared slice; its shape, placement and decoded digest are recorded.
`weights_loaded` and `forward_executed` remain false even after that check:
these are not observed vLLM parameter loads or runtime route traces.

Fixtures cover the attention projections declared by the existing smoke-case
owner, for both families in that owner's fixture menu and for TP1 and both
TP2 ranks. They include KDA's complete replicated gate roles, row-parallel
outputs, replicated MLA low-rank inputs and DSA indexer projections. The
full-size KDA12576x4096 case reads configuration geometry through the existing
census preflight and allocates no full-size weight. That packet explicitly
does not claim wire parsing or decoded roles. No vision fixture is changed.

## Before/after evidence

Task-local interpreter: Python 3.14.4, torch 2.14.1+cpu. A private test venv
installs pytest-xdist and reads existing CPU torch dependencies without
modifying their environment. Local runs use `PYTHONPATH=src`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`-n 2 --dist worksteal --durations=10 --surface-json PATH`.

- Pre-feature run: all 73 initial cases failed. Representative failure:
  `tests/test_projection_cpu_packet.py:70: ImportError: cannot import name
  'tessera_projection_cpu' from 'tools'`. Receipt: `/tmp/pri684-before.log`.
- Packet binding regression, before adding scheme/declaration provenance:
  both CLI modes failed at `tests/test_projection_cpu_packet.py:180` with
  `KeyError: 'scheme'`. Receipt: `/tmp/pri684-binding-before.log`.
- Development run after implementation: 113 passed in 9.38 seconds, covering
  the new packet file plus `test_serving_projection_routes.py`,
  `test_projection_smoke.py` and `test_construction_preflight.py`.
  Population: **NO CUDA**, `torch 2.14.1+cpu reports no CUDA device`;
  0 skipped, no skip reasons, 0 uncollected modules, 0 device allocations.
  Receipt: `/tmp/pri684-target.surface.json`. This was a dirty-tree development
  run, not a verified commit population or a CUDA/merge gate. Exact-source
  final selector and population receipts are delivered on PRI-684 separately.

No wire writer, renderer, plane or layout behavior changed. No byte-baseline
claim or full-suite result is made.

## Remaining gates

GitHub #1241, #1245 and #1239 were inspected in this session; all remain OPEN
with `needs-decision`. This harness does not implement or assume their outcomes.
In particular, there is no T4 direct-consumer qualification. Existing fixture
rungs are not asserted to be admitted measured serving cells.

Actual selected-constructor parameter loads, route traces, direct-cache and
absorbed MLA BMM runtime evidence, eager/replay, G3 v2, matched-byte D41 through
M8192, power and work/J remain unmeasured here. No GPU, serve or native build
was launched. Independent review is assigned to Tactical Worker 31 by the
manager; this report is implementation evidence, not a review or approval.
