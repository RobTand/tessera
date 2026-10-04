# Run plan: the artifact-scope graph-equals-eager receipt (tessera#702, prismaquant#1586)

Not run. It needs a two-Spark window granted by the kernels lead. vLLM serving is exempt
from PrismaBuild, so the arms run directly, on sparky, inside that window.

## What it produces

A `tessera.graph_equals_eager.v1` receipt whose one graph arm attests the serve the
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
export RECEIPTS=/mnt/shared/tessera-measurements/graph-attest-702/artifact-$SHA FABRIC=socket
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
  - `FABRIC` is fixed per receipt and checked against both ranks' NCCL banners.
  - The default is sockets (`NCCL_IB_DISABLE=1`), the u4 default, chosen for repeatability; the release speed legs ran RoCE.
  - If the card's serve must run RoCE, run the whole plan with `FABRIC=roce`.
- **Load refusal on the artifact's export.** The artifact was exported at contract v44 (`a5f3b232`). No refusal is expected, since the plugin carries no contract-version gate, but aE1 is the load smoke.
