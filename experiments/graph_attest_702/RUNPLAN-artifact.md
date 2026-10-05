# Run plan: the artifact-scope graph-equals-eager receipt (tessera#702, prismaquant#1586)

Not run. It needs a two-Spark window granted by the kernels lead. vLLM serving is exempt
from PrismaBuild, so the arms run directly, on sparky, inside that window.

## What it produces

A `tessera.graph_equals_eager.v2` receipt whose one graph arm attests the serve the
ship card publishes:

| Scope field | Value |
|---|---|
| image | `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705…` |
| model | the release artifact (default here: A8SE752MN's, `/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported`, 158 GiB). The issues lead names the card's artifact; the receipt binds its `config.json` sha256 |
| Tessera | the tree the card will pin. Run from a clean clone of the merged #930 master commit staged on `/mnt/shared`; the receipt binds its `src/` digest |
| compilation_config | `{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}` (release flags; the plugin pins capture sizes 2,4,6,8) |
| speculative_tokens | 1 (MTP, draft TP 2, triton MoE) |
| max_model_len / max_num_seqs | 8448 / 4 |
| tensor_parallel_size | 2 (rank 0 + API on sparklina, rank 1 headless on sparky; mp executor, `--nnodes 2`) |
| fabric | `socket` (both ranks' NCCL banners; `FABRIC=socket` on every arm) |

Other settings, as the release latency serve (pact/u4 `LAT_SERVE`, arm A8SE752MN):
- `fp8_ds_mla` KV, `--moe-backend triton`, resident, `TESSERA_FUSED_E4M3_MMA=e4m3`;
- FlashInfer autotune off, chunked prefill at 2048, no prefix caching, breakable graphs off, GLM53 NoPE off.

KV: 2 GiB per rank, the arm's c4 budget (`ARM_LAT_KV_BYTES`). Its MTP speed legs ran 384 MiB at c1. The equality set needs 4 resident requests, and one MTP request occupies a 4352-token block (about 0.47 GiB).

## Commands (on sparky, in the window)

```bash
# 1. stage the tree the card will pin (once; any box; a clean clone, never edited)
SHA=<merged #930 master commit>
git clone --no-checkout https://github.com/RobTand/tessera.git /mnt/shared/tessera-measurements/graph-attest-702/src/tessera-$SHA
git -C /mnt/shared/tessera-measurements/graph-attest-702/src/tessera-$SHA checkout --detach $SHA

# 2. dry run: checks inputs, prints both ranks' commands, starts nothing
export TS=/mnt/shared/tessera-measurements/graph-attest-702/src/tessera-$SHA
export ARTIFACT=/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported
export RECEIPTS=/mnt/shared/tessera-measurements/graph-attest-702/artifact-$SHA   # every plan arm sets FABRIC=socket
bash $TS/experiments/graph_attest_702/drive_tp2.sh $TS/experiments/graph_attest_702/plan-artifact.txt --dry-run

# 3. the window: aE1 (eager), aGR (graph), aE2 (eager); stops at the first failed arm
bash $TS/experiments/graph_attest_702/drive_tp2.sh $TS/experiments/graph_attest_702/plan-artifact.txt

# 4. the receipt (no GPU; any box)
echo '{}' > /tmp/no-pb.json
python3 $TS/experiments/graph_attest_702/receipt.py $RECEIPTS /tmp/no-pb.json \
  /mnt/shared/tessera-measurements/graph-attest-702/receipt-artifact-$SHA.json \
  --eager aE1,aE2 --graph aGR --commit $SHA \
  --not-measured "served KL against BF16 under graphs" --not-measured "graph-vs-eager speed"
```

`--dry-run` was exercised on the CPU (`tests/test_graph_attest_tp2.py`). It checks that:
- the two ranks serve one engine argv, apart from the rank flags;
- the eager and graph arms differ only in `--enforce-eager` versus `--compilation-config`.

## Time (derived, not measured at TP 2 with this harness)

| Step | Expected | Basis |
|---|---|---|
| load, per arm | 8 min | measured 7:07 + 27 s on the GLM body (u4.conf); `LOAD_DEADLINE_S` 1800 |
| first arm's cold JIT (Tessera extensions, Triton) | +5 to 10 min | `EXT=/home/rob/tmp/ga702-tp2-ext` is box-local and starts empty |
| graph capture, aGR | 1 to 3 min | two classes × sizes 2,4,6,8 × target + draft = 16 FULL graphs |
| equality set, two passes | 4 to 8 min | 20 cases of 24 to 32 tokens, 2 s pause per batched case |
| long single-block cases | 1 min | 4 cases, 2100 to 4000 tokens |
| teardown and copy | 1 min | |
| **per arm** | **15 to 22 min** | |
| **three arms** | **about 55 min, 80 min worst; ask for a 90-minute window** | |

## Disk and memory, per Spark

- **Memory** (MemTotal 121.6 GiB on each box):
  - Release MTP peaks were measured at 384 MiB KV: sparky 94.4 GiB, sparklina 96.1 GiB.
  - At 2 GiB KV the derived peaks are about 96 and 98 GiB. Two graph classes add graph-pool memory that has not been measured.
  - The preflight refuses unless MemAvailable ≥ `FLOOR_GIB` (16) + `EXPECT_PEAK_GIB` (98) = 114 GiB on **both** boxes. Today sparklina read 113.5 GiB with another container up; an idle box should clear it, but that is not guaranteed.
  - During the run, a 5-second watchdog removes both ranks if either box drops under 16 GiB.
  - If the preflight refuses at 114 GiB, do not lower the floor. The decision (kernels) is to shrink KV, which caps residency and changes scheduling, so the eager pool and graph arm must share whatever is chosen.
- **Disk:**
  - box-local `EXT` cache: about 3 to 6 GiB per box (sparky 134 GiB free, sparklina 155 GiB free);
  - box-local work directory with NCCL INFO logs: under 100 MB per arm;
  - receipts on `/mnt/shared`: about 30 MB per arm.
  - The artifact is read from `/mnt/shared` and not copied.

## What can still fail, and what each outcome means

- **A fourth cause at TP 2.**
  - The stub arms were TP 1. Under graphs vLLM may route the tensor-parallel all-reduce differently from eager (custom or symmetric-memory all-reduce inside a captured graph).
  - If aGR departs from eager while aE1 and aE2 agree, that is the finding. The receipt says `not_equal`, and the cause must be isolated: the next arm captures with the eager all-reduce path.
- **Eager not reproducing itself at TP 2.** If aE1 and aE2 disagree, the pool cannot judge anything; the cause is the fabric or NCCL reduction order.
  - The fabric is receipt scope (schema v2, dec-1005-003356-6ba2). Every plan arm sets `FABRIC=socket`: the Spark pair serves on sockets (RoCE `ibv_reg_mr` fails ENOMEM there; PrismaQuant `tools/gold_engine_options.py`), and PrismaQuant's quality evidence runs on sockets. `receipt.py` reads it from both ranks' NCCL banners (`Using network Socket`).
  - `arm_tp2.sh` takes no default fabric. An arm whose banners name another fabric than the one requested is refused (exit 5), and `receipt.py` refuses arms that disagree: one receipt is one fabric, never mixed. A card served on another fabric needs its own receipt; `verify` refuses a mismatch.
- **Load refusal on the artifact's export.** The artifact was exported at contract v44 (`a5f3b232`). No refusal is expected, since the plugin carries no contract-version gate, but aE1 is the load smoke.

## Follow-up, not in this plan: the index_topk boundary on the GPU

Review of #930 (#3): no GPU case puts a step's `max_seq_len` at exactly `index_topk`
(2048) and then `index_topk + 1` within one request. The CPU toy runner holds both
steps for the target and the MTP k=1 draft prefill, which replays on the target step's
record; draft decode steps (k >= 2) are refused by the plugin. A GPU case would be a
2040-token prompt generating 24 tokens: steps up to 2048 are in the equality class, and
later steps only in the screen, because eager does not reproduce itself above
`index_topk`. It needs a new case in `equal-508.py` and an arm of its own; it is not in
`plan-artifact.txt`.
