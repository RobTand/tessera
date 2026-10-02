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

## Standalone integration and private hardware witness

The standalone #855 branch was rebased onto merged MLA #854, master
`5133f84625c4deb34fc5812b3132cb40611aac8d`. Its production CUDA diff is only
the two EMPTY waits and their comments; CUDA SHA-256 is
`71fe64f23d304156faf95f30157321989a4195428f6cb073efccb924fc1247f4`.
The original PM branch's compiled artifacts remain source-bound context.

Private generated-source probes delayed consumers before FULL entry and
recorded one entry counter per consumer warp. Before the potentially illegal
terminal arrival, a producer-only barrier holds producers while thread zero
checks the necessary entry condition. This instrumentation detects an unsafe
arrival schedule; announcements alone do not prove barrier completion. It is
absent from shipping CUDA.

| Library | Unfixed private action | Fixed private action | Actual observation |
|---|---|---|---|
| E4M3-MMA | `5bf454eda16637d0090e9eb4c075f57f435dea6ff94d9a395493a6b5a3a50313` | `ce8755aef96f5169f26fcfd49d2b56308fbf342da459622a8ff2a095c871024c` | Old: phase violation gc4/warp0 and CUDA launch failure; fixed: 1 passed, zero skips |
| FP4 | `86fcc0d499a16b18e42dad9afee20985cdce387732ec0f5c18c9420b8fbf9259` | `999f50022e754185cb35c8736681c2f8d85ee9381ff4a679453aa90cfc22de61` | Old: phase violation gc4/warp0 and CUDA launch failure; fixed: 1 passed, zero skips |

Both old outer actions remain failed and have no success CAS receipt. Their
inner pytest exit was 1; the named violation appeared before CUDA reported
`unspecified launch failure`. The initial wrapper expected an illegal-instruction
message and rejected that driver wording. Root accepted the limited witness
after reading both immutable logs; no old GPU run was repeated.

Both fixed actions exited zero. Full CAS verification accepted receipts
`f7cca24af785f68ffc7f1a2e32682f64c43dd734011c8a45389f7dbaba76e29f`
and `5544844195bcdbc0128dc30a384899c310561fa6a863cc7187ac607567347e99`.
Mapped ELF/inode and source identities, counters, independent output oracles,
deterministic repeats and unchanged native artifacts passed. Exact owned Docker
labels were queried on each host after all four runs and returned no containers.

The durable private records are under
`/mnt/shared/astra-resume-20261002/t8_performance/terminal-855-protocol/`: the
original root GO, immutable negative observations, fixed verified receipts and
old/fixed container-cleanup records. This private proof does not qualify an
uninstrumented shipping binary, full serving, performance or energy.

## Canonical standalone native staging

The frozen common source is commit
`09cdb22f6cd4a51e7d480531fde82376c785f4f7`, including merged MLA #854 plus
#855. Its source manifest SHA-256 is
`926777b682fb57d144123fb96c5d525008333e162d75a55f22b7bbf8a0a787b5`.
The immutable image remains `5be13705...`. Build and test wrappers select the
consumer's source and extension paths, `/tessera/src` and `/ext`, while their
default paths remain unchanged. A pytest plugin binds the canonical package
before legacy test helpers alter sys.path, including in every xdist worker.

Initial canonical setup actions `bce46502...` and `adb8fee6...` failed before
compilation because the package requires adjacent `pyproject.toml` to resolve
its version. The same source commit's metadata is now mounted read-only at
`/tessera/pyproject.toml`, SHA-256
`736b317cda74962d73c35343e6def680a70873596f56d766ca4f5d7347489c85`;
no version fallback was introduced.

Four independent portable PB quanta then completed, each 1 CPU / 4 GiB /
native threads 1 / no GPU, with at most two in flight. Their complete receipts,
flags, objects, ELF hashes, native finalization records, resources and compressed
SASS are at
`/mnt/shared/astra-resume-20261002/t8_performance/terminal-855-shipping-71fe64f2/`.
Full CAS verification and independent artifact comparison passed. All 414
persistent specialization tails contain the conditional terminal EMPTY wait.

| Library | Actual CPU action | ELF SHA-256 |
|---|---|---|
| value | `b375c23a0d75288707f4662372f55cf7a0345d364d005a17ceb5048c1cb99f4b` | `97c5f4ca8a722721c114f8bca43eb04e755b9eacfafa3c6689411fdb229abbef` |
| E4M3-f16 | `24a5d905764ed4aba2531ae337ab5f4840a74c7e6c10578070c57fe4d1496fc8` | `dc2bc9e08752c013319c1c638798a8ea019e2fa7a830e523b7e380512a52099e` |
| E4M3-MMA | `050ee84e4bf9e735eaf20a52dd515ddc31a864ceb21c38178cdd5318554a7e7e` | `d68633d236bab421e574f6b5fc73bbcf3bf0a87141ad833690d4dba308f223c1` |
| FP4 | `41bc7a54e92084ecbd4aa915712b4742e908006854b833ccc8558bf6d0baec19` | `597cab7e707df53c77cf0e8406093ac14410964b03eecdf9565890ecc7de402b` |

Standalone CPU checks `dd7845e8...` passed 22 cases. Exact shipping selection
`5e994f24...` collected 43 cases. Counts from overlapping runs are not added
into a new suite result. The canonical shipping GPU result is recorded below.

## Canonical shipping GPU result

PrismaBuild action
`58da6a5427f806c048371deebf63643b10ee72f925154d0504a000ee2e771824`
completed on Sparklina with exit zero: **43 passed, zero skips**, using
`-n2 --dist worksteal --strict-cuda`, 2 CPU / 16 GiB aggregate / 8 GiB GPU
subset / one exclusive GB10 / native threads 1 per worker. This was one
600-second admitted attempt. Actual scope peak was 4,053,794,816 bytes;
the measured runtime is recorded in PB as execution evidence, not a speed claim.

The selection covered all four libraries at zero work, one item, idle CTAs,
minimum two-chunk splits, odd split parity and mixed odd/even item counts,
with exact item-plus-terminal counters and deterministic repeat bits. Existing
routed oracle and empty/repeated-expert controls plus graph replay passed.
Twenty-four per-worker files verified executable mappings from the hashed
ELF inode, six records per library, with source `/tessera/src` and extensions
`/ext`. Every frozen source file, the version metadata, native objects and
ELFs retained their bytes; native object/ELF timestamps did not change.

Full CAS verification accepted receipt
`4acbbdaa8f48f0839647627436a993a319f1ee7209f8d0d67d90e2b58228c456`.
The exact owned container label was queried on Sparklina afterward and
returned no containers. Durable JUnit, CUDA surface population, native identity
files, observation, verified CAS and cleanup evidence are at
`/mnt/shared/astra-resume-20261002/t8_performance/terminal-855-shipping-gpu-a4f96a5d/`.
`CANONICAL-HANDOFF.json` in the native bank binds the image, frozen common
source/version metadata, all four module recipes/objects/ELFs, build receipts
and this GPU receipt. Both serving ranks must mount that same existing bank.

The CPU source version was `09cdb22...`, and the GPU test wrapper was
`a4f96a5d...`; actual CUDA SHA-256 was `71fe64f2...` throughout. Later prose
edits do not relabel those sources. This qualifies the selected correctness,
termination, oracle, repeat and graph cases; full serving, throughput, quality
and energy remain separate gates. The impact selector narrowed this branch
to 306 tests; broader integration validation remains the coordinator's once
on the combined merge result.
