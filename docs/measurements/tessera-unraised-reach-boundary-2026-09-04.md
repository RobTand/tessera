# Unraised CHANNEL reach boundary A/B — 2026-09-04

## Outcome

Keep round-to-nearest for unraised rows (issue #115, option 2). Directing the
32 residual row words upward costs no additional wire bytes, but it recuts 30
of 31 affected payloads and its reconstruction signal is mixed: relative
Frobenius error improves by roughly two parts per million while diagonal-H
error worsens by roughly seven to nine parts per million. When the refit itself
enforces reach, both arms already finish with zero source values beyond reach,
so the widened initial landing buys no additional final coverage.

These are weight-space screens. No served BF16-teacher KL was run, and no
served-quality claim is made.

## Fixed population and arms

- Source: Qwen3-0.6B revision
  `c1899de289a04d12100db370d81485cdf75e47ca`; model-safetensors etag
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
  config SHA-256
  `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
- Selection: every transformer-block 2-D Linear derived from the checkpoint
  index: 196 tensors, 344,064 rows, 440,401,920 source values.
- Diagonal H: all 196 tensors, receipt
  `/mnt/shared/tessera-runs/bf16/refs/h_diag.pt`, SHA-256
  `20baf460b1f6119a5f5a1add1dd4ab9d0cb1a7603ec4a171e19d685921bb1bf1`.
- Wire: E4M3, q256=1024, window L=14, seed 0, reach 384 grid units,
  `channel_sigma=94.18039850841602`, four scale refits.
- Arm A: current #87 behavior; only raised rows land upward, while unraised
  rows keep their nearest fp16 word. Arm B starts from A and steps only a
  residual below-floor unraised word to the next fp16 word. Every tensor ran
  A/B/A-repeat; all 31 A repeats were artifact-hash identical.
- Two separately dispatched configurations: the shipping refit with
  `refit_reach_floor=false`, then the protected counterfactual with it true.

The full screen found 31 tensors, 32 rows and 33 source values in the residual
interval. Arm A's initial clip-only relative errors were
`3.03278881119e-7` plain and `2.32832861598e-7` diagonal-H; arm B reduced both
to zero. Each affected scale moved exactly one fp16 word. Relative inflation
was 0.0616908% to 0.0971794%, mean 0.0832379%.

## Matched-byte results

| outcome | shipping: floor off | protected: floor on |
|---|---:|---:|
| container bytes, each arm | 37,336,770 | 37,336,770 |
| priced bytes, each arm | 37,314,560 | 37,314,560 |
| A/B tensors with a moved payload | 30 / 31 | 30 / 31 |
| final rows beyond reach, A -> B | 4,933 -> 4,927 | 0 -> 0 |
| final values beyond reach, A -> B | 5,042 -> 5,035 | 0 -> 0 |
| beyond reach and emitted at reach, A -> B | 4,002 -> 4,005 | 0 -> 0 |
| all emitted values at reach, A -> B | 13,653 -> 13,657 | 13,553 -> 13,545 |
| final row words moved | 6,630 | 6,029 |
| body codes moved | 380,104 | 369,613 |
| reconstructed values moved | 10,923,288 | 9,720,143 |
| relative Frobenius A -> B | 0.0702889307932 -> 0.0702888554818 | 0.0703308470621 -> 0.0703307260686 |
| relative Frobenius B/A | `0.999998928545` | `0.999998279652` |
| diagonal-H A -> B | 0.0810289111269 -> 0.0810294922227 | 0.0798600275363 -> 0.0798607666862 |
| diagonal-H B/A | `1.000007171463` | `1.000009255568` |
| tensors plain better / worse / equal | 17 / 12 / 2 | 20 / 10 / 1 |
| tensors H better / worse / equal | 13 / 17 / 1 | 17 / 13 / 1 |

The shipping arm modestly lowers the geometric beyond-reach count, but its
intersection with emitted table-reach values rises by three. The protected
arm is the mechanism check: the refit floor makes current A sufficient for the
stated final-reach property, while the widened initial word still causes a
large downstream recut. Neither result supplies a quality reason to move the
default.

## Exact affected rows and source values

The values below are the complete set. `word` is the stored fp16 bit pattern;
the raw JSON also records exact required/effective scale floats and represented
reach for every entry.

| tensor | row | column | source value | word A -> B | relative scale inflation |
|---|---:|---:|---:|---:|---:|
| `model.layers.10.mlp.down_proj` | 262 | 2587 | -0.1015625 | 0x3c55 -> 0x3c56 | 9.0169906616e-4 |
| `model.layers.10.mlp.up_proj` | 575 | 897 | 0.09423828125 | 0x3c05 -> 0x3c06 | 9.7179412842e-4 |
| `model.layers.11.mlp.down_proj` | 323 | 1762 | 0.1103515625 | 0x3cb5 -> 0x3cb6 | 8.2993507385e-4 |
| `model.layers.11.self_attn.q_proj` | 866 | 454 | 0.1396484375 | 0x3df5 -> 0x3df6 | 6.5577030182e-4 |
| `model.layers.11.self_attn.v_proj` | 565 | 1015 | -0.107421875 | 0x3c95 -> 0x3c96 | 8.5246562958e-4 |
| `model.layers.12.self_attn.o_proj` | 737 | 1847 | 0.0986328125 | 0x3c35 -> 0x3c36 | 9.2852115631e-4 |
| `model.layers.13.mlp.gate_proj` | 1701 | 803 | -0.203125 | 0x4055 -> 0x4056 | 9.0169906616e-4 |
| `model.layers.13.mlp.up_proj` | 961 | 236 | 0.1044921875 | 0x3c75 -> 0x3c76 | 8.7642669678e-4 |
| `model.layers.13.self_attn.v_proj` | 115 | 593 | -0.10009765625 | 0x3c45 -> 0x3c46 | 9.1493129730e-4 |
| `model.layers.15.self_attn.v_proj` | 419 | 808 | 0.119140625 | 0x3d15 -> 0x3d16 | 7.6866149902e-4 |
| `model.layers.16.mlp.gate_proj` | 2791 | 592 | -0.09716796875 | 0x3c25 -> 0x3c26 | 9.4246864319e-4 |
| `model.layers.17.self_attn.q_proj` | 1505 | 811 | 0.12353515625 | 0x3d45 -> 0x3d46 | 7.4124336243e-4 |
| `model.layers.19.mlp.gate_proj` | 2943 | 953 | -0.13671875 | 0x3dd5 -> 0x3dd6 | 6.6983699799e-4 |
| `model.layers.2.mlp.up_proj` | 2817 | 691 | -0.09423828125 | 0x3c05 -> 0x3c06 | 9.7179412842e-4 |
| `model.layers.20.self_attn.v_proj` | 591 | 904 | -0.12060546875 | 0x3d25 -> 0x3d26 | 7.5924396515e-4 |
| `model.layers.22.self_attn.k_proj` | 407 | 29 | 0.050048828125 | 0x3845 -> 0x3846 | 9.1493129730e-4 |
| `model.layers.23.mlp.up_proj` | 1834 | 140 | -0.13671875 | 0x3dd5 -> 0x3dd6 | 6.6983699799e-4 |
| `model.layers.23.self_attn.o_proj` | 807 | 1309 | -0.1103515625 | 0x3cb5 -> 0x3cb6 | 8.2993507385e-4 |
| `model.layers.24.mlp.down_proj` | 527 | 2490 | 0.1484375 | 0x3e55 -> 0x3e56 | 6.1690807343e-4 |
| `model.layers.25.self_attn.o_proj` | 654 | 650 | 0.119140625 | 0x3d15 -> 0x3d16 | 7.6866149902e-4 |
| `model.layers.26.mlp.down_proj` | 766 | 3049 | -0.130859375 | 0x3d95 -> 0x3d96 | 6.9975852966e-4 |
| `model.layers.27.self_attn.o_proj` | 552 | 1048 | -0.1015625 | 0x3c55 -> 0x3c56 | 9.0169906616e-4 |
| `model.layers.3.mlp.up_proj` | 2975 | 410 | 0.1015625 | 0x3c55 -> 0x3c56 | 9.0169906616e-4 |
| `model.layers.3.self_attn.q_proj` | 1929 | 100 | 0.11474609375 | 0x3ce5 -> 0x3ce6 | 7.9810619354e-4 |
| `model.layers.4.self_attn.o_proj` | 361 | 300 | 0.11474609375 | 0x3ce5 -> 0x3ce6 | 7.9810619354e-4 |
| `model.layers.5.self_attn.q_proj` | 1616 | 834 | 0.1044921875 | 0x3c75 -> 0x3c76 | 8.7642669678e-4 |
| `model.layers.6.self_attn.o_proj` | 536 | 592 | -0.1103515625 | 0x3cb5 -> 0x3cb6 | 8.2993507385e-4 |
| `model.layers.6.self_attn.q_proj` | 537 | 693 | -0.11474609375 | 0x3ce5 -> 0x3ce6 | 7.9810619354e-4 |
| `model.layers.7.self_attn.o_proj` | 976 | 894 | -0.107421875 | 0x3c95 -> 0x3c96 | 8.5246562958e-4 |
| `model.layers.8.self_attn.o_proj` | 428 | 1485 | -0.1044921875 | 0x3c75 -> 0x3c76 | 8.7642669678e-4 |
| `model.layers.9.mlp.down_proj` | 618 | 2750 | 0.1015625 | 0x3c55 -> 0x3c56 | 9.0169906616e-4 |
| `model.layers.9.mlp.down_proj` | 679 | 1968 | -0.10009765625 | 0x3c45 -> 0x3c46 | 9.1493129730e-4 |
| `model.layers.9.mlp.down_proj` | 679 | 2502 | -0.10009765625 | 0x3c45 -> 0x3c46 | 9.1493129730e-4 |

## Execution and power receipts

- Harness commit: `9106cc8` (`experiments/reach_boundary_ab.py`). Syntax and
  one-tensor CUDA smoke: PrismaBuild action `9ed1b5641bfe...`, CAS receipt
  `ccd4c0fc05e9ea9bf647890980e2aff50374876f4c1e96f4d648eeff136d647f`.
- Shipping run: action `273c1fb8ec8dca0a50b4ad6312796e4cfcb4a634332d8f8e9e69c6d9828fe8e0`,
  CAS receipt `0fed54b0058a0c159bc1ace3702290dd5667093677244fe16d5b26d38679ad19`,
  raw JSON `/mnt/shared/tessera-runs/ts115/shipping-9106cc8.json`.
- Protected run: action `93086d3a984fa85e958f7422d3350eb3033b25923f9e87bbb77553d5e026a937`,
  CAS receipt `6c0a67ba786b330cc8da34a613e92a9f6aaba3476124ae6f1404887b669fb1d5`,
  raw JSON `/mnt/shared/tessera-runs/ts115/protected-9106cc8.json`.
- Device: gx10-6b77, NVIDIA GB10, sm121, torch 2.11.0+cu130 / CUDA 13.0.
  PrismaBuild runtime generation
  `afb785e03ef8-1788534131-6fc34bbdfbc9`.
- Shipping power window: action `f470139af057...`, 193 seconds, 36.0 W
  median / 32.4 W mean / 57.0 W max; protected: action `ad9778ef091b...`,
  194 seconds, 42.5 W median / 36.95 W mean / 57.0 W max. Peak was 40.7%
  of the 140 W envelope in both; swap I/O was zero. Raw receipts are
  `/mnt/shared/tessera-runs/ts115/power-{shipping,protected}-9106cc8.json`.

Power covers each whole process (identity fixtures, census, A/B/A encodes and
JSON), not an arm-specific kernel window, so it is execution context rather
than an energy attribution. The behavior identity stayed
`49a718e25c916ccac1ac01c70047ed522dce59bd7fb751279846ff2313ba5182`
while 30 payloads moved; issue #116 separately repairs that gate blind spot.
