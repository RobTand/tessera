# Eager sparse-MLA masked-tile qualification, 2026-10-02

Research opt-in only (Refs #812), using the shared stock-kernel override authority from #813. No production pin, default, artifact or ship gate changes. The target campaign remains2500tok/s atL8192 c1 TP2 MNBT2048; this receipt does not establish that target or a full-model speedup.

The operator retains stock MG's power-of-two Q scale rounding and visits nonempty64-key tiles in ascending canonical index order. It keeps tile0 even when all indices are masked and reproduces the stock positive-zero/FTZ effects across skipped gaps. A single shared tile list replaces the old prototype's long-lived per-thread64-bit mask. Stock conversion, cache residency, mixed batches, decode, Torch compilation and CUDA stream capture remain stock. The override requires pure prefill, the hash-bound image source,32heads/latent512/no RoPE, BF16 Q, page64/656-byte FP8 arbitrary-scale KV,2176 physical indices and FlashInfer's own resolved FP8/MG/direct plan. No SwapAB admission.

Environment: immutable image`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`, GB10 SM121, nvcc13.0.88, `-use_fast_math -O3`, native threads1,2reserved physical CPUs for paired measurements. The ABI's `declared_fast_math` records the authoritative loader flag list; it is not a compiler-verification predicate. A bounded nvcc preprocessing probe found no usable fast-math macro. Source, flags, nvcc binary/version, build.ninja, stock header and resulting library hashes are retained.

The original stock profile (`c3a8d99c429b8ed6a517f8016b0784b52963554328ef45ddabc199b69eb30496`) showed7.65ms at2048@8192,29.6%issue,0.41eligible warps/scheduler and31.7%barrier stalls. Canonical pooled masks were51.47%empty tiles for the first2048chunk,1.52%for later chunks and86.76%forL512. The hypothesis was to remove masked work, with little expected benefit later. The first compile (`af65a8e493740202c1641a983c06e012c6c3a0a7021565e44bff750bcda79800`) produced copy SASS identical to all3048stock instructions and control words; L0had3424instructions,168registers, no stack/local spills.

Actual bitwise gates through PB:

- `b3d1abe3e8a0409af36c32028954cea9aa81b43d0cacf1c7034a45c1ab9ee44a`: all-masked, tile boundaries, hole, signed zero and extreme query scales; both copy and L0output andLSE raw-bit exact.
- `ee57be7ec99d24f16cedf004ff0cefde76b3d61e41cddbae9948b6768caf0ba0`: all6representative shapes ×pooled/random causal indices,12rows exact for output andLSE.
- `daf40faf76c58c129f18736f75ac075edb30ee99985e491e3b6a4887a55012ea`: qualification-only drop-last-valid-tile mutation returned1and was detected. The unchanged copy remained exact. No mutation is selected by the serving install.

The default-off registration causal test failed before the feature with `explicit mask-skip flag did not register stock enum`. The known vLLM image's CPU runtime guard suite passed12tests (noCUDA,0skips/uncollected). A separate tiny actual vLLM GPU implementation test passed2tests onGB10:512@512 and2048@8192 eager native output matched stock bits, and CUDA capture executed stock without enteringL0. These vLLM tests ran directly under Rob's explicit vLLM exemption; no model or serving window launched. Shared contract/scanner tests through PB `384960cbe70fbb1b45a420c9c5aad7a708bbeaa1fe21d365e8da5d48abe605a8` passed145onCPU,0skips/uncollected.

Timing quanta use complete stock/L0 pairs inside each PB measurement action, with real GB10 reference facts sealed by the published client on sparky under dl380 orchestration. PB selects the worker and admits the measurement. Both arms share resident tensors, matched CUDA graphs for kernel qualification, ABBA order and before/after Torch profiles. The serving binding itself refuses graph prefill and requires actual eager/FULL_DECODE_ONLY served traces before a campaign claim. Earlier ordinary-exclusive rows are retained only as numeric/contextual evidence and are not timing qualification.

| Pooled call | Stock ms | L0ms | L0/stock |
|---|---:|---:|---:|
|2048@2048|7.6994|3.9796|0.51688|
|2048@4096|7.6696|7.6197|0.99349|
|2048@6144|7.6969|7.7078|1.00142|
|2048@8192|7.7693|7.7866|1.00222|
|2049@2049|7.7510|4.0787|0.52622|
|512@512|2.2321|0.41474|0.18581|

The random later-chunk cells regressed1.1%,3.1%and2.8%; that negative evidence is retained. This is a masked-work lever, not a general faster late-chunk kernel.

The aggregated pooledL8192four-chunk pair is PB`273a12b8831c4a7401f3cce3e46132fdfc05310ce813a085f6f423af38eaf272`, actually on sparklina. Eight steady arms lasted30.20–31.63seconds after startup/settling. Median CUDA time per four-chunk operator sequence was32.0852ms stock and27.9265msL0 (13.0%less kernel time). Clock drift from roughly2450to2327MHz was bracketed by ABBA; per-arm samples and actual before/after Torch traces are retained. This is an operator result, not a full-model prefill result.

Energy qualification is **HOLD**. Both-box Netdata raw series are retained, but aligned API groups extend beyond some requested windows and the final stock arm's coarse values disagree with the high-frequency sampler. No work/J ranking is accepted. Issue#819 fixes the shared collector's window provenance in a separate PR, using the recorded raw response without another GPU run; the sensor/cadence disagreement still needs resolution.

Actual VB1770 baseline evidence is native, not the legacy CUSTOM default: both banked Docker inspections set `TESSERA_RESEARCH_GLM53_NOPE=0`, omit an explicit attention-backend flag, and both logs resolve`FLASHINFER_MLA_SPARSE_SM120`. It has no speculative config,1GiB KV/rank, modeNONE/FULL_DECODE_ONLY, TP2/MNBT2048 and source608bbdf0d6909548ff7c6919e5cdb834c1fcef7c, contract SHA004b7d6e3909a81bcbb11dd929cbd223acb553e09e398627999792c0922be8fe. Its actual per-rankL8192profiles have44MG kernels (11per chunk), totaling347/355ms within4.6275sTTFT. The first-chunk delta therefore suggests roughly41ms of possible operator savings; that estimate is not a served delta. Later source d7c8b7b differs in other prefill/head hooks; old served TR3 cannot certify current-main source without a matched gate.

Artifact roots:

- `/mnt/shared/tessera-measurements/mla-mask-recovery-20261002/`: edge/served-shape/mutation gates; `measurement-v2`, `measurement-v3` paired cells; `energy-v1/l8192-pools` long arms, Torch traces, both-box and per-arm Netdata.
- `/home/rob/tmp/claude-campaign-20260926/tmp/sol-mla-recovery-20261002/`: exact PB receipt/source packets, direct vLLM test logs/tool identities, baseline native runtime/profile packet and preserved old prototype. Frozen qualification worktree4047abdba2773327a8ca83f5698a71c252955633 is separate from the serving implementation worktree.

Full served TR3 against both existing teachers, per-rank eager dispatch counts and actual post-timing CUDA events, plus the final combined merge surface, remain Astra's acceptance gates. Do not activate a production pin or infer a serving-window authorization from this receipt.
