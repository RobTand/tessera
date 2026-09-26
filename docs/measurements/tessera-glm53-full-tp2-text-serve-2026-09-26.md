# Full GLM-5.3-Flash Tessera TP2 text serve, 2026-09-26

Tessera #626. The unchanged 120-shard export at
`/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported` reached
READY and returned three short deterministic chat responses with clean merged
Tessera `f790bcc1aaa039388f6218ae3ef70155a8a9e6fa`. This is a bounded
diagnostic of full-checkpoint loading and text generation, not a production
capacity or quality qualification. The earlier BODY-transfer and word-reversal
staging fixes, with their matched allocator before/after evidence, are recorded
in [the loader measurement](tessera-glm-window-loader-scratch-2026-09-26.md)
(PRs #628 and #629). The packaged runtime contract SHA-256 stayed
`04d5a20a33b932607be604fc78cc3559499a151be1471bdf66e9c8049d9d22e4`.

Both GB10 hosts used the same digest-qualified stock vLLM image
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`,
vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, Tessera's plugin,
`TESSERA_SERVE_MODE=resident`, custom NoPE attention, socket NCCL and TP2.
Rank 0/API ran on sparklina; rank 1 ran on sparky. The API bound only to
sparklina localhost and was reached through an owned loopback forward. The
trial used stock `--language-model-only`, `--enforce-eager`,
`--kv-cache-memory-bytes 536870912` per rank, `--max-model-len 1024`,
`--max-num-seqs 1` and `--max-num-batched-tokens 1024`. The stock text-only
setting excluded the vision tower; no image or video request was attempted.
vLLM reported 81.46 GiB model-loading memory per rank and actual KV admission
of 1,638 tokens, or 1.60 times the configured 1,024-token request length.

At 20:12:58–20:13:00 UTC both hosts passed the separate 24 GiB
MemAvailable READY gate: sparky 25,689,768 kB and sparklina 27,202,848 kB,
against 25,165,824 kB. The 16 GiB MemAvailable / PSI-full-20 watchdogs
remained armed. A health request returned HTTP 200. Three chat requests used
temperature 0, seed 0, `max_tokens=128` and
`chat_template_kwargs={"reasoning_effort":"low"}`:

| Prompt | Prompt + completion tokens | HTTP / finish | Returned content |
| --- | ---: | --- | --- |
| `Reply with only the number: 2 + 2 = ?` | 25 + 3 | 200 / stop | `</think>4` |
| Capital-of-France question | 26 + 3 | 200 / stop | `</think>Paris` |
| Why leaves are green | 21 + 33 | 200 / stop | A coherent one-sentence chlorophyll explanation, prefixed by `</think>` |

The visible `</think>` comes from the model/chat-template output. These three
latencies (1.895, 1.269 and 5.679 seconds) are observations, not a throughput
comparison. The exact prompts, response text, usage and latency are in
`/home/rob/tmp/astra-main-campaign-20260926/serve/logs/swap-halfkv/smoke.jsonl`.

On the same retained endpoint, the stock `/tokenize` route established exact
prompt-token counts for two more requests. A 512-token prompt with at most
eight output tokens returned HTTP 200, usage 512 + 3, `stop`, `</think>4`.
A 1,023-token prompt with one output token returned HTTP 200, usage 1,023 + 1,
`length`, `</think>`. The latter shows admission at the configured 1,024-token
total, not a meaningful answer at that length. Both requests passed 24 GiB
checks immediately before and after. After the 1,023 + 1 request, sparky had
25,206,396 kB, only 40,572 kB above the threshold. The ten-second process
sampler caught sparky at 25,138,304 kB at 20:19:31 UTC, **27,520 kB below**
the 24 GiB margin during that request. The 16 GiB watchdog did not trip.
No further request was sent. The raw request, `/tokenize` response, prompt
token-ID digest, usage and host checks are in
`serve/logs/swap-halfkv/boundary-smoke.jsonl` (SHA-256
`ab8adc6e4dca37dff148ce0061d165fc0bee93c44991b52f5d0d44ccd741d4a9`);
the host dip is in `sparky-proc.log`.

The owned loopback forward was stopped and both labeled containers were
removed at 20:19:55–57 UTC. Docker event records for both ranks show deliberate
SIGKILL from `docker rm -f`, exit 137 and destroy. Exit 137 here is controlled
teardown, not an OOM crash. The broker watcher released both named holds at
20:20:04 UTC. Postconditions recorded both containers absent, both NVIDIA
compute-process lists empty and the local forward closed. The raw teardown
receipts are `rank0-docker-events.jsonl`, `rank1-docker-events.jsonl` and
`/home/rob/tmp/astra-main-campaign-20260926/serve-gates.json`.

Raw rank logs, READY memory, Docker inspections, `/proc` memory/I/O samples
and py-spy snapshots are in
`/home/rob/tmp/astra-main-campaign-20260926/serve/logs/swap-halfkv/`.
Both-host Netdata windows are under `serve/logs/netdata-20260926T200415Z/`,
`...T201439Z/` and `...T201949Z/` there. The immediately preceding full
4 GiB KV and 1 GiB KV trials, including the 24 GiB gate failures, are in
`/home/rob/tmp/astra-main-campaign-20260926/serve/full-serve-capacity.md`.
No claim follows for continuously maintaining 24 GiB slack, 4 GiB KV,
long-context serving, image/video, CUDA graphs, MTP speculative generation,
served KL or throughput. This run changed no serving default or contract cell.
