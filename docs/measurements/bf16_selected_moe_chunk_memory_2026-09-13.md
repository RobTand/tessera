# Selected BF16 fold scratch: bounded-chunk before/after

The first BF16 selected implementation called `decode` on every selected
expert, then converted the entire raw BF16 stack to FP32 for the fold.
`max_experts_per_chunk` bounded the window decoder but not the subsequent
temporary. A CPU regression passed five IDs with `max_experts_per_chunk=2` and
failed before the fix because one `decode` received all five; after the fix it
observed chunks `[2,2,1]` and all four selected BF16 tests passed. PrismaBuild
action `96ce4e6a08bdfa70c20860544b5451c8c028cf9d647560c09cd5391d9bd0bd30`
published receipt SHA-256
`ab8b67fd9e8ce63fab2178ae03f364ccefcede2e5a6608a51e7157d156fa6684`.

The same pinned stock-vLLM image and four-distinct-expert H256/N128/top-K2
fixture then ran one profiled native selected apply per arm on Sparklina,
serially. Both used `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
and identical harness SHA-256
`d2d8c43102363d3d4a10f159db9b248265ca13a2d0ca53fee7f8f6b8288b37c4`.
The only route-source difference was `bf16_route.py` SHA-256
`26bafcd3ae9e352286f643879cdd018e3d091b4e70860fce5f6aa1e7f93e098d`
before versus
`3b4038bb3db818136b3e555c99a6d5c05783fdcbf75aa25259629cfc11f3d2ea`
after. The before container ran
`2026-09-14T01:33:47.235661634Z`–`01:34:02.998897972Z`; the after ran
`01:34:09.674370682Z`–`01:34:25.397889855Z`. Both exited 0 and were removed.

| One selected apply, torch CUDA allocator | Before | After |
|---|---:|---:|
| Allocated immediately before | 45,762,560 B | 45,762,560 B |
| Peak allocated | 47,732,736 B | 47,469,568 B |
| Incremental peak | 1,970,176 B | 1,707,008 B |

The measured incremental-peak reduction is **263,168 B (13.36%) on this tiny
fixture**. The selected result and the stock `MoERunner` shared-expert sum
remain bit-exact to their full unquantized controls in both arms. The copied
[before](bf16_selected_moe_chunk_old_2026-09-13.json) and
[after](bf16_selected_moe_chunk_new_2026-09-13.json) result records have
SHA-256 `8b5ef74ecdaef0b0c1510e8f58825c5b1b28626fa338e3b1c631b4578694ff8e`
and `cb7bb8c1a99edfc4364d4cdc4a647a65253019b6c4d2c3b9b829bd5361713268`.

In-process `torch.profiler` Chrome traces, raw logs, and both-host Netdata
CPU/RAM/GPU-power series for the entire 50-second window are retained at
`/home/rob/dq-runs/glm-campaign-takeover-20260913/serving-format-closure/native/`.
The old/new trace SHA-256 values are
`f23a4abe36a75520e5d9b7db76b2579ad8dfbf9e8ff05b5fc80a6d15e43456bc`
and `2dc57cbd0993d59e0f51c798e7407abce0629fd9603d5633e5833bb77de4e605`;
they include memory events and the selected-decode record function. The
Sparklina whole-container power samples averaged 7.25 W before and 7.81 W
after, versus an approximately 140 W envelope; Sparky averaged 8.21 W and
14.13 W as unrelated background work changed. At this tiny size, profiler
traces include JIT setup and one-second Netdata power samples cannot resolve
useful energy per token. No speed, work-per-joule, full-model fit, or TP2
serving conclusion follows from this memory check.
