# Full-vocabulary BF16-teacher KL: prefill plus true decode (tessera#1172)

**Measured 2026-10-10.** Qwen3-0.6B BF16 teacher against an in-job FP8-RTN
student (per-tensor E4M3 round-trip, deterministic, no training) on the frozen
WikiText-2 test contract `corpus_qwen_n8_s512.json`. One input set, one mask,
one vocabulary for all four dumps: 8 chunks x 512 tokens, 4088 scored
positions, vocab 151936.

The serve surface cannot supply a 128k-vocabulary JSON dump, so the exact
number comes from the new in-runtime instrument
(`experiments/full_vocab_kl_dump.py`), which writes the file
`kl_tool.read_full_vocab_payload` reads. Served top-K bounds ride beside it
from the existing `kl_tool dump` / `compare` path. Quality and run parity
stay split: the exact number never claims to be served, and the served bound
never claims to be exact.

## 1. Result

Exact full-vocabulary KL (bound `exact (full vocabulary)`):

| regime | forwards | KL mean | KL p99 | KL max | top-1 agree | confident n | confident KL |
|---|---|---:|---:|---:|---:|---:|---:|
| prefill | 8 x 512-row | 0.01530802 | 0.10150094 | 1.25405822 | 92.71% | 1717 | 0.00844852 |
| decode (M=1) | 4088 x 1-row | 0.01544752 | 0.09833649 | 1.21349741 | 92.44% | 1713 | 0.00871236 |

Served top-K (k=1024) bounds on the same corpus:

| regime | served positions | KL lower | KL upper | tail mean | top-1 agree |
|---|---|---:|---:|---:|---:|
| prefill | 4088 | 0.01504591 | 2.38633416 | 0.02434578 | 92.69% |
| decode (stride-16) | 256 | 0.01329875 | 3.70332414 | 0.03752198 | 93.75% |

Cross-check: each served lower bound sits below its exact number (prefill
0.01504591 <= 0.01530802). The decode bounds cover different position sets
(256 strided served vs 4088 exact), so their gap is expected and labelled.

## 2. Same histories, same masks

All four exact payloads carry contract
`cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d`,
source `076d33efc4476dcc417a2bb249c0bc950bb54bbb471d73f69c15cef0010b53d0`,
tokenizer `76f13c8e6e553e35b09733ed5543274fdcd97285d3fcd7e1cccd4e0ad8089891`,
4088 positions and vocab 151936. The driver asserts this before it exits, and
`compare` refuses anything else. Decode advances teacher-forced on the
corpus's own next token, so histories never drift. Prefill scores one 512-row
forward per chunk; decode records one row forwarded per scored position.

## 3. IDs

- Input source: WikiText-2 test, held-out, source-bound above.
- Teacher: `/mnt/shared/tessera-runs/ig1172/inputs/Qwen3-0.6B-BF16`,
  `Qwen3ForCausalLM`, BF16, safetensors sha256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
  Tokenizer files match the corpus identity byte for byte.
- Student: `/mnt/shared/tessera-runs/ig1172/dumps/student_fp8rtn`,
  same arch, safetensors sha256
  `35f45a3d1637c8797e52d2668e896a4227d90a7b61ba848e55cf42c3550bafa5`.
  589485007 of 595984384 elements differ from the teacher. The student is a
  **baseline screen**: it prices no rate and promotes none.
- Payload shas: teacher prefill `0d77aa14d87d5aaf`, teacher decode
  `273ec4764b869b6c`, student prefill `9e57dae4908adcb5`, student decode
  `aecb435ecd094664` (first 16 hex each).

## 4. Smokes

Greedy smoke (`experiments/moe_greedy_smoke.py`, pair schema
`tessera.moe-greedy-smoke-pair/2`, reference `bf16_source`), 7 short raw
prompts plus one 2000-token long-context prompt (`PLONG`, WikiText-2 test
text, disjoint from no KL position by claim but a smoke, not a quality
measurement):

| prompt | student | teacher | shared |
|---|---|---|---|
| P0, P1, P3, P4 | recorded | recorded | yes |
| P2 | repetitive | repetitive | yes |
| P5 | repetitive | recorded | **no** |
| P6 | repetitive | repetitive | yes, identical |
| PLONG | recorded | recorded | yes |

Positive record: P0, P1, P3, P4, PLONG on both arms. P5 is an unshared cycle
(the student loops `Answer: D) Pacific Ocean`, the teacher rambles on): a
labelled divergence, not a pass. The long-context smoke passes on both arms.

Bare areas: the serve runs `--max-model-len 4096`, so the 32k native window
stays untested; the model is a base checkpoint, so only the raw-completion
interface is exercised and no chat-template form exists in the record.

## 5. Route census and exit status

Stock vLLM serves (`prismaquant/glm53-mia-sm121:487ecf187`, eager): no
tessera route executes, so there is no route histogram to print. The census
is the serve logs, the `/metrics` snapshots and the no-spec-decode gate, all
under `/mnt/shared/tessera-runs/ig1172/serve/`:

- `logs/serve_topk_{teacher,student}_{prefill,decode}.log` plus
  `.build.json` sidecars and `.metrics.txt` gate files.
- `logs/serve_smoke_{teacher_bf16,student_fp8rtn}.log`.
- `metrics_{teacher_bf16,student_fp8rtn}.txt`.
- `exit_status.txt`: all twelve steps exit 0 (four top-K dumps, two
  compares, prompts build, two smoke serves, smoke compare).

## 6. Device and receipts (strict-CUDA block)

- Dumps ran on sparklina/sparky GB10 (sm121, compute capability 12.1),
  `execution.device: cuda`, device name `NVIDIA GB10`, peak 2.35 GB
  (teacher) and 5.04 GB (student) from the CUDA allocator counter.
- Serves ran on sparklina, compute 12.1, 35.9 W mean GPU power.
- PrismaBuild CAS receipts: dump `cab050bd636e6a037ceee35ee977647e1d6d013a703c748101f938c288301a0a`,
  stage `74c15d2d006edc5b42296dc1969363f49d9e1248cf28cf91e78b0be1c`,
  prefill compare `da79c7d5e38350179ed5b32ff97ce22cd3ad60dd8da939b5a97acaf8d95dfe0f`,
  decode compare `911e2cf56e0f6965d3561e41107639c4ff8d560ac152147b15d5024f15f6ce8d`,
  serve `6129f61416bf3fdeba59427fe5d68f9980e4b7854e30b9d71a5019c4a4af42f2`,
  unit tests `d4a358628cd6d5816022b27b14bce7111020fc6d2d302635b7367f9acbe9ea77`.

## 7. What this does not claim

- The FP8-RTN student is a synthetic baseline. No rate is promoted and no
  cell grade changes; `test_no_cell_claims_full_vocabulary_kl` still holds.
- The served upper bounds (2.39 prefill, 3.70 decode) are slack screens: with
  2.3% mean tail mass outside top-1024 and 0.73 max, the DPI interval cannot
  read quality. Screens stay labelled as screens.
- A full-model MoE pair (GLM 642 GB / 162 GB) does not fit the 104 GB box
  cap; the instrument stands ready for one that does.
