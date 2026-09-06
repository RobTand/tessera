# Tessera 0.1.0 release validation

Release: [`v0.1.0`](https://github.com/RobTand/tessera/releases/tag/v0.1.0), commit `b40b3265f83582a87c7e5c1f46d46409843b7a07`, tree `77eabe8618a248ee0fa5f39ef756b990706da59f`.

All agent tests, package builds and downloaded-package checks ran through PrismaBuild. The GitHub tag workflow separately built and published the PyPI distributions. No new quality or throughput benchmarks were run; the README compares attributable existing measurements.

[Full receipt, commands, CAS digests and original populations](https://github.com/RobTand/tessera/releases/download/v0.1.0/validation.json). [Independent source audit](https://github.com/RobTand/tessera/releases/download/v0.1.0/source-audit.json).

## Populations

| Run | Device / toolchain | Mode | Passed | Failed | Skipped | Xfailed | Uncollected | PB exit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Final full GPU, sparky | torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10 | -n 12, worksteal, strict-CUDA | 3581 | 0 | 10 | 1 | 0 | 0 |
| Final full CPU, sparklina | torch 2.11.0+cu130 reports no CUDA device | -n 12, worksteal | 3020 | 0 | 544 | 0 | 0 | 0 |
| Earlier full x86, dl380g10 | torch 2.11.0+cpu reports no CUDA device | -n 12, worksteal | 3012 | 1 | 548 | 0 | 0 | 1 |
| Final selected x86, 42 files | torch 2.11.0+cpu reports no CUDA device | -n 16, worksteal | 993 | 0 | 68 | 0 | 0 | 0 |

The final GPU run allocated on CUDA in **485 tests**, with no missing-box-artifact skips. Its 10 skips and one expected historical checkpoint reproduction failure remain visible below and in the receipt. The aarch64 CPU run had 544 skips, including CUDA and absent box artifacts; the x86 selected run had 68. Neither CPU population covers the CUDA surface.

The earlier x86 full suite is **red**: `test_ladder_wire.py:160` fails only the historical E4M3 byte-reproduction case. Pristine pre-README master `ba582d4` and the release tree both reproduce it in the targeted file: 28 passed, one failed, no skips or uncollected modules. Existing-artifact reader checks pass. [Tessera #360](https://github.com/RobTand/tessera/issues/360) retains the numerical/toolchain investigation. This earlier suite predates the smoke-origin fix and is not a same-source merge receipt for the final tag.

The smoke-source defect found by the first GPU run is fixed in [#359](https://github.com/RobTand/tessera/pull/359). The strengthened regression failed at `tests/test_moe_greedy_smoke_rule.py:345` before the fix; afterward all 16 tests in that file passed, four CPU workers, no skips or missing modules. The final GPU full suite and selected x86 suite include the fix.

## Source equivalence

The final full suites report snapshot commits `75701372d0ddbb763254c134991568967a5a357b` (GPU) and `2d20143782016f608524b1bb33dbfe0abdc98e96` (CPU). Tessera's reader recognizes only PrismaBuild snapshot v1, so its original v2 population hashes include action metadata. [#361](https://github.com/RobTand/tessera/issues/361) tracks that consumer compatibility issue.

The independent audit verified each sealed action, CAS bundle, closure digest, stamp contents and exact action fingerprint. It recomputed each population's full source digest from every snapshot blob, removed only the verified metadata entry, and compared every remaining path, mode and blob to the tagged tree. Both contain the same **1,017 source files**, normalized SHA-256 `67a690f5f5c0cf46efbec5ec5ed1a1ce27ab45e44faf0791b3aa2925b701578e`. The original populations are unchanged; `validation.json` records both their raw disagreement and the independently established effective-source agreement.

## Artifacts and execution limits

The PrismaBuild wheel/sdist build, isolated package-content/plugin check, and `twine check` passed. The tag workflow published PyPI 0.1.0; PrismaBuild downloaded both published distributions, verified their PyPI hashes, and repeated the wheel/sdist checks successfully. GitHub attachments and PyPI distributions have separate archive hashes because the publishing workflow rebuilt them. All **20 README URL destinations** returned HTTP 200 after the tag was pushed.

Two memory-limited attempts are retained as failures: the initial 24-worker CPU full suite hit its 48 GiB reservation (exit 137), and a four-worker selected run hit 10 GiB (exit 137). The successful selected x86 retry reserved 56 GiB for 16 workers. One build attempt lacked system `venv` support; the retry supplied scoped virtualenv tooling inside the admitted action. No failed or interrupted attempt is counted as a pass.

## Verbatim skip histograms

### Final full GPU, sparky

Uncollected modules: 0. CUDA-allocating tests: 485.

- 3: `could not import 'vllm': No module named 'vllm'`
- 2: `e2m1-tcq-lut-release does not cut 4 ways along columns`
- 2: `e2m1-tcq-lut-release does not cut 8 ways along columns`
- 2: `needs two CUDA devices`
- 1: `E2M1 publishes no reader range`

### Final full CPU, sparklina

Uncollected modules: 0. CUDA-allocating tests: 0.

- 88: `needs a CUDA device`
- 88: `the lane is a CUDA kernel`
- 84: `the encoder is a CUDA path`
- 52: `the fused window Viterbi is a CUDA path`
- 43: `encoder is a GPU job`
- 30: `the kernel lane is a CUDA path`
- 29: `the Viterbi is CUDA`
- 29: `the captured TCQ trellis is a CUDA path`
- 26: `the encoder is a GPU job`
- 20: `the fused window Viterbi is a CUDA path and needs triton`
- 14: `the Tessera encoder is a CUDA path`
- 8: `the fused window Viterbi is CUDA`
- 6: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 6: `needs CUDA`
- 5: `the kernel lane runs on CUDA`
- 3: `box artifact absent: the PrismaQuant checkout whose pricing this suite is pinned against -- /home/rob/prismaquant/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_DIR; documented default /home/rob/prismaquant)`
- 3: `could not import 'vllm': No module named 'vllm'`
- 2: `box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch -- /home/rob/pq-wt/tessera-continuous/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_WORKTREE; documented default /home/rob/pq-wt/tessera-continuous)`
- 2: `needs two CUDA devices`
- 1: `E2M1 publishes no reader range`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)`
- 1: `needs a CUDA device: the positive arm builds the extension`

### Earlier full x86, dl380g10

Uncollected modules: 0. CUDA-allocating tests: 0.

- 88: `needs a CUDA device`
- 82: `the encoder is a CUDA path`
- 76: `the lane is a CUDA kernel`
- 52: `the fused window Viterbi is a CUDA path`
- 43: `encoder is a GPU job`
- 30: `the kernel lane is a CUDA path`
- 29: `the Viterbi is CUDA`
- 29: `the captured TCQ trellis is a CUDA path`
- 23: `the encoder is a GPU job`
- 20: `the fused window Viterbi is a CUDA path and needs triton`
- 14: `the Tessera encoder is a CUDA path`
- 11: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook/model.safetensors is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 8: `the fused window Viterbi is CUDA`
- 6: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 6: `needs CUDA`
- 5: `box artifact absent: BF16 source checkpoints the encoder reads -- /home/rob/models/Qwen3-0.6B/model.safetensors is not on this box (set TESSERA_MODELS_DIR; documented default /home/rob/models)`
- 5: `the kernel lane runs on CUDA`
- 3: `box artifact absent: the PrismaQuant checkout whose pricing this suite is pinned against -- /home/rob/prismaquant/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_DIR; documented default /home/rob/prismaquant)`
- 3: `could not import 'vllm': No module named 'vllm'`
- 2: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-stock-twin/model.safetensors is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 2: `box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch -- /home/rob/pq-wt/tessera-continuous/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_WORKTREE; documented default /home/rob/pq-wt/tessera-continuous)`
- 2: `needs two CUDA devices`
- 1: `E2M1 publishes no reader range`
- 1: `box artifact absent: BF16 source checkpoints the encoder reads -- /home/rob/models/Qwen3-0.6B is not on this box (set TESSERA_MODELS_DIR; documented default /home/rob/models)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)`
- 1: `could not import 'transformers': No module named 'transformers'`
- 1: `needs a CUDA device: the positive arm builds the extension`

### Final selected x86, 42 files

Uncollected modules: 0. CUDA-allocating tests: 0.

- 43: `encoder is a GPU job`
- 8: `the encoder is a GPU job`
- 6: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 3: `box artifact absent: the PrismaQuant checkout whose pricing this suite is pinned against -- /home/rob/prismaquant/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_DIR; documented default /home/rob/prismaquant)`
- 2: `box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch -- /home/rob/pq-wt/tessera-continuous/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_WORKTREE; documented default /home/rob/pq-wt/tessera-continuous)`
- 1: `box artifact absent: BF16 source checkpoints the encoder reads -- /home/rob/models/Qwen3-0.6B is not on this box (set TESSERA_MODELS_DIR; documented default /home/rob/models)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)`
- 1: `box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)`
- 1: `needs a CUDA device: the positive arm builds the extension`
