# Persistent producer terminal barrier (#855)

Status: source correction and CPU protocol regression only. Native CUDA
compilation, delayed-consumer GPU reproduction, numerical and termination
coverage remain pending. No performance or serving promotion is established.

The value/E4M3 and FP4 persistent kernels publish each chunk `q` with
`FULL(q % 2)`. Before reusing that stage they wait for `EMPTY(q % 2)`, which
the consumers release after consuming chunk `q-2`. The terminal publication
previously omitted that wait. After publishing its last real chunk `q-1`,
the producer has waited only for `EMPTY(q-3)`. A consumer delayed before
`FULL(q-2)` therefore permits the producer to arrive on the same FULL phase
twice without a reset. The producer-only claim barrier does not order it
against consumer progress. This violates the named-barrier reuse requirement
in [NVIDIA's PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

Both terminal branches now wait for the matching EMPTY when `gc >= 2`, before
writing the sentinel and arriving on FULL. Zero-work CTAs (`gc == 0`) and an
unused parity (`gc == 1`, a protocol control rather than an admitted item
shape) do not wait for a nonexistent consumer acknowledgment.

The descriptor and scale lifetimes were reviewed separately. Consumers copy
the descriptor into registers before releasing an item's first chunk. Dense
split items retain at least two chunks, and their epilogue uses register
accumulators without reading the shared scale slot. Routed and unsplit dense
items retain at least four chunks; the intervening item's producer EMPTY
waits establish that the preceding epilogue finished before its scale slot
is rewritten. FP4's epilogue reads register state and global ratios. The new
terminal wait changes neither item geometry nor these existing bounds.

`tests/test_routed_terminal_barrier.py` extracts both complete production
terminal branch bodies, compiles them as C++, and executes them against
named-barrier shims representing a consumer delayed before the final two
FULL phases. It detects duplicate arrival before reset, verifies the producer
blocks before sentinel publication, then releases the consumer acknowledgments
and verifies terminal publication. This is a source-executed protocol model,
not hardware emulation or a GPU result. It covers `gc=0,1,2,3,4,5,8,9` in
both paths, including odd/even parity and minimum dense split sizes.

PrismaBuild CPU actions (DL380, `pb-cpu` Python, no CUDA, 2 CPU / 4 GiB,
native threads 1, `pytest -n2 --dist worksteal --durations 5 -q -rA`):

| Source | Action | Actual result |
|---|---|---|
| Unfixed production `0d63b68f4b` plus regression | `98eca9503eb4884d17d9f26a2ffefd50c70b5dd18a7fde457ce58fcc98d9ee59` | 12 failed, 4 passed, zero skips |
| Both EMPTY waits added | `780629471d7f187871123c5acdc8f6e2936cbee1205d4ea2432d1a09a51f0b07` | 16 passed, zero skips |

The causal failure was `FULL reused before delayed consumer reset`; zero-work
and first-use controls already passed. Full CAS verification accepted the
green receipt `2002863be68779d7dbe48f85997667f549785d1dfc780e62c23d3c1f116835fd`.
Exact command manifests and terminal logs remain in the PB CAS/queue under
those action keys. CUDA source changes are separate from the earlier paired
layout screen, whose retained source overlay and ELF were not replaced.
