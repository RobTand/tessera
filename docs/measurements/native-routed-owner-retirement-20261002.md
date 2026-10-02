# Native routed-owner retirement CPU controls — 2026-10-02

Issue: [Tessera #869](https://github.com/RobTand/tessera/issues/869).

The native constructor now owns immutable per-projection views retaining the
words, scales and initial state it executes, plus its composed native lookup,
run and descriptor tables and counters. The resident route replaces its compact
preparation owner only after native selection succeeds. Compact fallback and
caller-held compact bundles or slices stay intact. Retired views refuse compact
execution and recomposition. No artifact bytes, defaults or kernel arithmetic
changed.

The separate accounting correction charges actual declared backing storage
once, including selected byte/16-bit tables and counters. A slice holds its
whole allocation; an aliased BF16 table is charged once. External caller
references still keep their storage alive. Runtime owner bytes do not establish
CUDA allocator-reserved, host-resident or process-wide reclamation.

## Causal CPU evidence

All actions used PrismaBuild's portable CPU placement, `/home/rob/venvs/pb-cpu/bin/python`,
with native math threads bounded to one. The admitted worker was `dl380g10`,
PyTorch `2.11.0+cpu`. Every result below had zero skips, zero uncollected modules
and zero CUDA allocations. CUDA admission/toolchain are stubbed only in the
constructor ownership fixture; actual lookup/descriptor composition and Python
reference lifetime execute on CPU. These controls do not qualify CUDA arithmetic
or served performance.

| Control | PB action | Actual result |
| --- | --- | --- |
| Original ownership RED | `12ba1e88f9e3eba6fc0dd0c2b36846f52ecbebc91072cba16e4fc9394dd150cf` | 3 failed, 2 passed; unheld compact tensors survived, owner stayed original, compact execution did not refuse |
| Ownership GREEN | `22984151178709873f2a65f612f1515cb24d888cea7437cb2d43ebd2eb606678` | 9 passed in 3.76 s; receipt `d8a1d6e63ae7a72542a1f84b2dc76938ddb65d5c927a01cd3de355eb74b4e24f` |
| Recomposition RED | `e24c453029f177af40aa206a7a0028c6eefb9a59c82be5fb6da636cccd43b7e3` | 1 failed: returned a CPU-device refusal instead of identifying retired compact planes |
| Accounting RED | `c0209aec2ed60904859b46c88e47479de634e85c7b918b806352c56145fc39b0` | 4 failed, 8 passed; both FP8 kinds omitted counters, BF16 lookup aliases double-counted, retained slices undercharged |
| Targeted ownership/accounting/resident-pricing GREEN | `7b682f0412852e4502dc619b306c7bb79e49c324198ee85b956b3dfd3bdd80c1` | 45 passed in 67.34 s; receipt `a0774a3bc3fa7ac653e9e7bcaeb999d9ae6e628f582d79f89d996b44dcc42543` |

The subsequent eight-file route control, action
`8168d233a4939af40e35b1387a7d7b2f824cd6d73ea67cc10c8f1896ff560e18`,
passed 51 cases and caught one new guard regression at
`tests/test_routed_fused_window.py:543`: a lightweight CPU support screen did
not define `perm_all`, so direct field access raised before its established
device refusal. The guard now reads the explicit `None` retirement marker
without requiring fields the CPU screen never reads. Its targeted rerun with
all ownership/accounting controls passed 14 cases in 3.85 s, zero skips and zero
missing collection: action
`99fc9154af32277efb4d75b4ef0fd62e289c9a4ac350eb5c8d4c6183190a82df`,
receipt `8712d8a3fe7d836fa334949ed704f384d678b72d95a9f432b969f2a081c66ddb`.
The eight-file control skipped 492 cases: 443 `the lane is a CUDA kernel`,
33 `the native window MoE runs CUDA kernels`, 15 `the native method runs CUDA kernels`,
and one `could not import 'vllm': No module named 'vllm'`. It allocated on no
CUDA device and did not qualify those skipped cases.

The exact-base impacted selector used the published `pbsnapshot.py verify`
source verifier through PB and returned `narrowed`, with 369 files because
uncertain file readers and conftest scope conservatively broaden selection.
Action `ebfe280e753717a2ed5552350f8283a363f6191eb66394087a2889d6b3199629`,
receipt `c31f0cf26b42633ef5a5649475ac0f179ec427eaf4f8a9a1207df096a695b605`.
This list is retained for the coordinator's combined integration population;
the worker has not claimed that broader population passed.

The final action ran:

```text
python -m pytest -q -n 2 --dist worksteal --durations=5 \
  tests/test_native_routed_owner.py tests/test_resident_tensor_protocol.py \
  tests/test_export_routed_resident_pricing.py
```

The original ownership failures were at `tests/test_native_routed_owner.py:66`,
`:82` and `:110` in that RED snapshot; accounting failed at `:187` and `:201`
in its later RED snapshot. Line numbers are snapshot-specific. Failure records,
stdout and canonical passing CAS receipts are retained under
`/home/rob/tmp/astra-resume-20261002/t8_performance/native-owner-retire-sol/evidence/`.

GPU allocated/reserved memory, host profiles, native-use evidence and matched
prefill/decode outputs/timings remain unmeasured. The source-derived FP8
redundancy estimate is a candidate saving, not a measured allocator result.
Export-time preparation estimates, artifact fit records and fullserve memory
bounds are unchanged.
