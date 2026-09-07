# Full-census original-wire research reference

The first-model reference frames all 2,142 measured `TESSERA_E4M3_K1_R1024` originals: 30 dense tensors and 2,112 expert projections. It is a uniform research reference, not an allocator selection or production-default change. Every other tensor must remain bit-exact source precision. The canonical encoder and installed serving package remain unchanged.

`experiments/full_model_original_wire_checkpoint.py` verifies the merged anchor identity, census, per-unit envelope payloads and original blobs, then uses the strict `--cached-units` intake added in #403. It passes the existing 33.5 GB Hessian capture through `ActivationSource`; it does not create a second calibration cache. The output comparator parses every actual emitted serving blob and compares all 2,142 inner SHA-256 values with the originals, then compares every passthrough tensor bitwise.

The export is submitted as PB `c6de6cb4d34e7a41937c732fcb9839756cd681dd16f253f9d8bc42f4262defff`, eligible on either GB10 with four CPUs, 96 GiB aggregate memory and a 16 GiB GPU subset. The aggregate reservation covers the 33.5 GB deserialized Hessian, its file pages, source pages and output serialization. These are reservation estimates, not measured peaks. Outputs are rooted at `/mnt/shared/tessera-clean-runtime-20260907/full-model-original-r1024/`. At this entry, the action remains queued and there is no successful export claim.

The dependent finite generation proof uses the exact selected fixed-KV configuration `f5064609d62a3e61ef1d9bb87b2b62ea10b31b71759db3dce666543d7585233e`, the canonical installed package, and the immutable stock vLLM image. It requires the exported 38-owner roster, actual post-request prefill distinct-owner counts for both dense and routed modules, normal completion with final answer `blue`, exact loaded-source origins, the actual KV capacity assertions and all 4,967 unchanged core files. The public route histogram exposes distinct module counts per shape, not names; the proof records those counts honestly alongside the independently inspected loaded owner names. It uses trusted local observer RPC in a network-disabled container.

This harness establishes a bounded functional proof only when both jobs actually pass. It provides no statistical quality, throughput, concurrency, allocation or release-promotion evidence.

The export subsequently completed on Sparklina with actual exit zero and complete
PB scope cleanup. `export-proof.json` has SHA256
`588d84dfdd48da1635c1b078fb3b60597c45a11987be9f975c31b19028585b1c`:
all 2,142 original inner wire digests match and all 160 passthrough tensors are
bit-exact source values. The checkpoint has 38 runtime owners and 2,288 tensors.
Its `model.safetensors` is 5,127,618,426 bytes, SHA256
`3500698eda1019736bc941f2368ef9a5db9c732ad3c8d8319a92953f4d772063`.
The checkpoint and cached-unit bundle files were made read-only after the
independent hash audit.

Over 7,902,068,736 quantizable parameters, the manifest accounts for
3,994,185,728 inner wire bytes (4.043686 bpp) and 7,918,182,400 declared resident
weight bytes (8.016313 bpp). The 1,131,577,600 passthrough bytes are excluded
from both denominators. These are artifact/layout counts, not a measured
full-engine resource peak.

`pb-audit.json`, SHA256
`2c5574c81d5dd9d713e603442208fab286aae4090175f452e02d4ef69a3f7abb`,
independently rehashes every checkpoint file, all stdout-bound artifacts, the
CAS payload `eb980cb44d2b04c0c97be08e780312e95780617484b518d63d7943ba7f9df1ec`
and canonical receipt. It also joins the proof's complete unit roster/digests
with `originals.json` and checks actual exit/cleanup evidence.

The generation harness's final compilation passed PB
`14604d529cdaa4e0e739bdf3340d4d28c5f6df8eac5c25ede7ceb59665ac6cf6`
on Sparklina CPU with exit zero and rehashed empty-output CAS. The dependent
functional action is now
`a475cb59ec25778ae3098a5db3afbb3701e2a25f6ac53455a6607e1c08a1be93`,
using four CPUs and the prior bounded functional proof's 40 GiB aggregate /
32 GiB GPU reservation. It writes a fresh `functional-01` directory. At this
entry it is submitted; the export and compile passes do not assert generation
success.

The full-reference functional action subsequently passed on Sparklina with
engine/container exit zero and complete PB scope cleanup. Its
`functional-01/generation-proof.json` has SHA256
`dbfb370e5286310a20999088b66010676a238166038d52ae5010ccb6e9f51d9f`.
All twelve checks passed: the response stops normally after 79 output tokens,
closes its reasoning section and has final answer `blue`. The loaded roster
matches all 38 exported owners. Actual post-request prefill at M=15 reports
16 distinct dense and 22 routed owners; decode counters report 79 calls per
owner. Both package-origin observations match the canonical installed source
roster, all 4,967 stock core files are unchanged, and the actual selected KV
capacity passes. This is the complete-answer proof for the uniform original
reference; it does not establish an allocator solution or production promotion.

`functional-01/pb-audit.json`, SHA256
`9be5690a987fcb58d3e9bc2e456eddfbcc91dc54a4a2d4df5a366df94057d80a`,
rehashes all nine stdout-bound artifacts, the CAS payload
`c7356b27a2e877e3d4f6e7cd530422e933b74b8d53a2839d863e196c82a5d164`
and canonical receipt, joins the exact export/KV/plugin records, and verifies
actual exit and container/scope cleanup. The fresh functional cache logged
new-shape Triton compilation during the request; no latency or energy claim is
made from this finite serve. Separate warmed resource and timing measurements
must carry their own sealed generated binaries and evidence.

Delivery validation on current master with cached-dense export PR #403 merged
passed PB `570f43de99a9afba3bb845bb21197bc3e9e7a1b0101bc3774fdb6601e6f70d7a`:
all eight qualification scripts compile, and six package-origin/issue-reference
CPU tests pass with zero skips or missing modules. The source scripts match the
measured harness bytes. `delivery-cpu-audit.json` rehashes the terminal CAS
`cc71e51a1c37064462986606c7f5c24e303a6fa1338b3abd043f832f1a7b383a`
and canonical receipt and confirms exit zero and scope cleanup.
