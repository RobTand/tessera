# Full LFM MoE campaign: identities and completed preflights

## 1. Scope at 19:35 UTC

This is an **in-flight campaign record**, not a served result. At this point
one encoder partition is active and the second is queued behind the existing
Sparklina reservation. No full-model student load, census, or KL has completed.
No positive routed-MoE contract cell is claimed by this record.

The uncut source is `/mnt/shared/models/LFM2.5-8B-A1B-BF16`. Its 24 layers
contain 22 routed stacks of 32 experts, with three projection containers per
expert. The common explicit plan quantizes all routed stacks at E4M3
`q256=1024`, leaves dense weights BF16, and leaves routers on the exporter's
mandatory passthrough path. The population is checked against that plan and
the source, not accepted from a hardcoded expected count.

Both partitions use the immutable source commit
`5c56b7e5ad777136661cc854f2576772b7bf3274` and image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
The image is vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, not the existing
dense contract's vLLM 0.28.0 image. Issue #126 separately makes that scope
machine-readable before any MoE promotion.

Campaign root: `/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2`.
Its `plan.json` SHA256 is
`2cf926e1061b2ade55383e08eeabbc00d3cf5825e4f5defe437c64a626405504`.

| partition | PrismaBuild action | physical reservation | status at this section |
|---|---|---|---|
| 0/2 | `d7fd29d4f73f2ae3ca665a5c65ca4beba0b52da4b1c74982949f55b9347aa1ec` | Sparky, gpu=2, mem_gb=32, cpu=2 | active |
| 1/2 | `80fcb8d1cddb6059c795f5205860c430375728365ee3b11e24b7ea4a5d833817` | gx10-6b77, gpu=3, mem_gb=32, cpu=2 | queued, not yet claimed |

Each action additionally caps its container at 32 GiB memory with no additional
swap allowance, records power and host-memory telemetry every ten seconds,
and has an outer two-hour timeout. Guarded output namespaces stop deployed
PrismaBuild retries from repeating a failed encode. Both halves must retain
this exact source snapshot even while the final serving controller changes.
The earlier `astra-campaign-r1` attempt refused a router plan entry before
encoding; it is retained as a failed attempt, not reused or hidden.

The bounded one-expert profile and its actual CUDA memory observations are in
`tessera-lfm-encode-preflight-2026-09-04.md`. They are throughput and memory
measurements, not served quality evidence.

## 2. Reused teacher verified before the student serve

PrismaBuild action
`db05c6cd514f71ff48fb1ea4a33cef8fb517bfd2ac5267953637031fcf10ccce`
completed on dl380g10 with worker return code zero. It checked all four exact
file hashes below, then reran `experiments/kl_reference_usable.py` on the
teacher/corpus pair. Receipt CAS:
`eef10494c2b48f70ea659f52322c55f8a4768f2fa78793f1df954dafdee8bb4d`.
This was CPU-only reference validation, not a new GPU serve.

Teacher directory:
`/mnt/shared/tessera-runs/ts5/lfm25/teacher-eugr-0281rc1-r3`.

| file | SHA256 |
|---|---|
| `teacher_bf16.json.npz` | `a72b54e5f87026acca5aab748460e6bf47934a04693945e5c497558a55608248` |
| `teacher_bf16.build.json` | `b1048b955666f753de22a0ce6817dce843988867c8229fab983419ec5c0d20a5` |
| `teacher_bf16.meta.json` | `67c7dac89a439a485370367daec92d5677b5008b9700bd6cd4b81ab0eccf0158` |
| `/mnt/shared/tessera-runs/ts5/lfm25/teacher-gate/corpus_n8_s512.json` | `2fdd48eeab69109c6222ef2f857815d2b35d5422747815c495e0467712751d44` |

The unchanged reference gate reports 4,096 positions, returned K=1,025,
mean retained support mass 0.9904, 1,968 confident positions, next-token
top-1 accuracy 36.33%, true token present in support 100%, and median
true-token rank 2. It is usable as a reference. The original dump's build
sidecar is complete, eager, and names the same exact EUGR image above; its
metadata names the full BF16 source, the LFM tokenizer, prefill regime, and
the corpus contract. The student must match those axes.

## 3. Integrated sidecar preflight tests

The duplicate-shard-owner refusal and the explicit-plan population refusal
were integrated together at `82bf28a`, then tested through PrismaBuild action
`b3e8f8a1685e1c89b786be6ce952e60ccdf2b753a982dcf94ff503d860100773`.
Worker return code zero; test return code zero; `test_ts5_sidecar_check.py`
reports **9 passed in 1.04 seconds**. Receipt CAS:
`8881ee3e1c8d86f11f8e5125482d94f45267098a79d6b1828d21e2c86a7f03be`.

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Red-before-green receipts for the individual changes are recorded in
`moe-sidecar-duplicate-shards-2026-09-04.md` and
`tessera-explicit-plan-coverage-2026-09-04.md`. The eventual merger must run
from the integrated controller containing those guards; the older frozen
encoder outputs remain compatible. A loadable merged config is published
only after exact source/plan/code/runtime identity, complete source ownership,
shard hashes and tensor-header coverage, and explicit-plan obligations pass.

## 4. Half0 completed; half1 claimed without GPU overlap

Sparklina claimed action
`80fcb8d1cddb6059c795f5205860c430375728365ee3b11e24b7ea4a5d833817`
at 19:45:05 UTC after the #113 reservation released. Its PB snapshot is
`524d56f4827769265a109145103805d5af9e99f9`; half0's snapshot is
`96bad5276ff8785ad989886b7849688fd55b3af4`. These are independently generated
PB snapshot commits of the same frozen source tree, not two evolving encoder
branches. The checked merger compares their sealed code/source/plan/runtime
identities rather than assuming the short Git stamp is shared.

Half0 finished with worker return code zero and a published result:
`bc0016e36d5a139184239a590dc16a2b8d686df35febf80ae4d1cfab3a682772`.
The exporter reports 2,358 seconds, including its final write; PB reports
2,435 seconds for the whole invocation. It encoded all 1,056 projection
containers in its 11 planned stacks and wrote 1,158 output tensors. Its
partial checkpoint has no loadable `config.json`, by design.

| half0 quantity | measured value |
|---|---:|
| quantized parameters | 3,875,536,896 |
| wire bytes | 1,959,034,880 |
| wire bits per quantized parameter | 4.043898809523809 |
| container bytes | 1,959,821,590 |
| resident-mode prepared bytes | 3,883,466,752 |
| BF16 passthrough bytes | 953,739,904 |
| checkpoint bytes, including passthrough | 2,913,700,766 |

Independent CPU verification action
`396bb6ef48c969efd7100f59d6b5ed0f85a87e2f2b690a42e5f3bdd916c146fb`
read and hashed the completed shard, matched the manifest's sealed output
hash, and summarized its immutable telemetry log. Its successful worker
result is in CAS
`df2df4df7b43147e64fbe18e362e57c824d47f4cd1fe69405a0cdd270cf6ad16`.
The code identity sealed into the partition is
`a57ccfd37af16e82f99b172abda46c87652a855eb53f7cd78ac0326b4116b524`.

| half0 file | independently checked SHA256 |
|---|---|
| `model.safetensors` | `f3504c20e11188b0705556e5daee473788db66b02bd86083f0631fe6b16f9f82` |
| `tessera_serving_manifest.json` | `55d774d0cc512faa3ac46c310f043b783a742f02c2125ac834fd7513bb2a3102` |
| `tessera_part_config.json` | `dc1d84f140429a1102a0f02416b60758445e9dd24c1e7e3074fe0ea0f736a3d7` |
| `model.safetensors.index.json` | `ded483bd7d3b9dbfea37df9c25cc87210b99abc44111c2d9fec93dea032302fe` |

The 243 telemetry samples span 15:12:00–15:52:26 EDT, 2,426 seconds, with a
maximum sample gap of 11 seconds. Maximum sampled board power was 72.89 W;
minimum host MemAvailable was 111,953,124 KiB; swap use was 364 KiB at both
ends. Trapezoidal integration of `power.draw` over that interval gives
157,440.135 joules. This is gross sampled board energy including startup,
without idle-power subtraction, not a paired efficiency comparison. The
telemetry SHA256 is
`2b9c17b7cda875025b739168b2f2b6453ad15bd12157c29b472b7d25a5ade779`.

**Pool outcome caveat, not concealed as a clean client exit.** The verifier's
original worker returned zero after 7.4 seconds and published the result
above, but a live-claim/reaper race had already launched three guarded retries
within about 1.5 seconds. The submitting client first returned the retry's
`mkdir: File exists` failure. The authoritative `done/` record retains the
successful original `worker_detail.returncode=0`; both terminal records exist.
This is filed as PrismaBuild #36 with a read-only worker investigating; no
PrismaBuild code, queue record, or active encoder reservation was changed.
An earlier verifier attempt used the wrong partial-manifest key and failed
before producing a result; the corrected verifier explicitly reads
`export_partition.identity`, not the merged manifest's `export_identity`.

At this section's completion, half1 is still encoding. There is still no
full-model merge, student serve, or positive MoE contract promotion to report.

## 4. Fresh source-bound teacher, completed at 20:18 UTC

The earlier reference in §2 is retained as historical/usability evidence,
not used for the final student comparison. Review of its original action
`10813df3832a1f84c7b30370bb12891907b014eea47ef278d22114994c4baa53`
found a revision-string check but no pre/post check of the loaded source
weight bytes. A current dump hash cannot supply that missing provenance.

The replacement was produced by the tracked one-shot
`experiments/ts5_lfm_teacher_bound.py` at source `c2e7227`, through PB action
`ed0e0c4934d462cc4d503aef5ae82c046bda69b5c04c02e04046e35285474db3`.
It reserved Sparky's whole GPU (`gpu=2`, `mem_gb=64`, `cpu=2`), capped the
container at 64 GiB with no additional swap allowance, and ran under a
1,500-second outer timeout with 120 seconds of cleanup grace. The original
worker returned zero in 204 seconds. Receipt CAS:
`817af390900fac3d27fa27190c24eecc5e52ccfbcafa13abfda439a3d4198b3e`.
The actual parentless pool source snapshot was
`9a1cabcb30c6fc06b1567277e456155e37773065`.

`teacher-bound-r1/source-bound-result.json` records source identity before
and after serving, both exactly equal to the encoder's sealed source. The
model directory was explicitly mounted read-only. The exact EUGR image,
unchanged corpus, tokenizer, eager prefill execution, top-K 1,024 request and
4,096 scored positions are checked in the same action. The source model SHA
is `c9b9e3c4b3be50b576e6da8c02de1b4223614ffe131d812abf92bb84421f6217`.

| output under `teacher-bound-r1` | SHA256 |
|---|---|
| `teacher_bf16.json.npz` | `c5515a096078e059e8bf596f5284e017d64b365b512ad992da522da3b78838c2` |
| `teacher_bf16.build.json` | `a16a82eeb19d300bd9d1c4c51cade42f84e617ef34c8e5003c533688150c5c12` |
| `teacher_bf16.meta.json` | `f2797d6d63cb40121b7558112936e3345e3939b2fc3816552330589c66157e4c` |
| `serve_teacher_bf16.log` | `d29648c1f1b58189678eeaa5f3355189583d6d2e18498c0898182f5f7bd2bf72` |
| `teacher_reference_gate.json` | `19ac0a925912459105aff4c8cfe4c5939bb75a419c1acee78488cd23817036a2` |

The unchanged usability gate reads the new payload and reports 4,096
positions, 0.9903523325920105 mean support mass, 1,968 confident positions,
0.36328125 next-token top-1, true-token support 1.0 and median true-token
rank 2, with no refusals. These are reference-usability results, not a
student quality result. Cleanup records the exact container absent, no GPU
compute processes, and `safe_to_release=true` before the reservation ended.

The driver was also syntax-checked through CPU-only PB action
`b93997d6bb8dd237457f3a49464899ceb04944447bbb6496964f01c7f85c9e0d`,
return code zero; this is a syntax check, not a test-suite population.
The second encoder is still active at this point; no MoE cell is promoted.

## 5. Second half and checked assembly

The second frozen-source encoder completed successfully through PB action
`80fcb8d1cddb6059c795f5205860c430375728365ee3b11e24b7ea4a5d833817`:
the original worker returned zero in 2,454 seconds, with 2,367 seconds reported
by the exporter. It wrote 1,056 projection containers over 11 routed stacks
and 1,144 total tensors. Its actual source snapshot was
`524d56f4827769265a109145103805d5af9e99f9`; receipt CAS:
`db72e797fd3b8fcdef35aa71014c45483a467214a1c803e10908bcee55b3d891`.
No encoder source or plan changed between halves.

Checked assembly used the newer merger/plan-coverage gates at `c2e7227`,
without re-encoding any container. PB action
`0214babe365657ee7034e0a0e912f65bf77a314e1f431869d1bb0082970d9897`
returned zero in 69 seconds on dl380g10, CPU-only, using two CPUs and 8 GiB.
Its snapshot was `a971a4a678ab481d105e55c59abf9b5bdd19f9c2`; receipt CAS:
`137368784e974ad6f2c0ee6ea2446a1c78ebf4b3c4e17cac742203a8b67e971d`.
The resulting artifact has 22 routed stacks, 2,112 projection containers and
2,302 tensors. The explicit common-plan check passed; both copied destination
shards were independently hashed against their source parts.

The assembly seal is `merge-action-r1/artifact-seal.json`, SHA256
`b14af9c0a6dea5c146a30142eef59c7830338296959c2afd7caa94970081ca81`.
It records 7,751,073,792 quantized parameters, 3,918,069,760 wire bytes,
3,919,643,180 container bytes, 7,766,933,504 prepared resident expert bytes,
1,433,567,488 passthrough bytes and 5,353,487,052 safetensors checkpoint bytes.
These are distinct accounting surfaces, not interchangeable footprint claims.
The two shard SHA256 values are
`f3504c20e11188b0705556e5daee473788db66b02bd86083f0631fe6b16f9f82`
and `978056b050593faa8310022c111f87dc1bac31844d3fc081c2c085c7b570f1fa`.

## 6. Preserved failed full-model census attempts

Attempt one, PB
`cf2f43fe23de44de16e41350e783ceba5365eb6565bb4b6c485ad6cd2d30e2ca`,
failed before loading model weights: the copied safetensors shards retained
mode 0600, inaccessible to the root-squashed container identity. The failure
and producer correction are recorded in
`merged-shard-read-permissions-2026-09-04.md`. Operational correction action
`ecfd18aa9565944cb8e4a7c10af7971b2970158087ed4f601e750815b8253bb1`
added read bits only to the two exact assembled shards, yielding mode 0644;
full artifact content identity was unchanged. A CPU-only exact-image probe,
`8e26fc31d1a44137cbb4528d1cd84594d673fd4fa0e32ab82c93f64e595086e4`,
then read both headers successfully as container UID zero. The private
original encoder-part files retain their modes.

Attempt two, PB
`270b382c9fe6b23c9ef20fa7a16ae1d5dc483f4d3c77581e0184ec863569610d`,
passed that boundary but failed closed during construction: the producer
named LFM's dense `w1/w3` passthrough leaves, whereas the runtime constructs
one `w13` Linear. Neither attempt generated route or quality evidence. Each
retains its own `census-bound-rN` directory and a cleanup receipt with the
exact container absent, no GPU compute processes and `safe_to_release=true`.

The naming fix is recorded in `lfm-dense-passthrough-alias-2026-09-04.md`.
Its operational correction preserves `full-model` and the original seal,
creating `full-model-r3` with unchanged files hard-linked and a separate config.
PB action `96ea5542bfac99e8472ed37d5d44a840bd5aeae3ddc80c12c09a864efbfcd9af`
returned zero; receipt CAS:
`0b0dd13e3cb074f678b6759e69ce75c13cb4d13d290851ed886e13eaaaea24d1`.
Only four explicit dense leaf names become two fused names; unrelated
declarations, including the tied `lm_head`, are retained. Config length changes
from 29,553 to 29,476 bytes; its SHA256 changes from
`24d512c46d7fd631ebbb31a4bd84f6a8200da12d035082d606f5468da08f069b` to
`cbc35c8148399ce2c84e49016077a749d11ac3999110039a1d2024814ec0c795`.
Every weight/index/other auxiliary hash remains unchanged, and the original
artifact is rechecked unchanged after correction. The new seal at
`passthrough-correction-r3/artifact-seal.json` links the original seal and
actual correction snapshot `5a9f5f6ed6bba2fd22de29fcddfb33e18cbc571d`.
The same explicit-plan sidecar gate still finds all 22 stacks and 2,112 roles.

Two earlier correction-controller attempts refused before creating a model
directory because reconstructing the whole ignore set would drop the tied
head, which has no separate tensor. Their `passthrough-correction-r1/r2`
namespaces are retained. The final correction replaces only the observed
dense pair declarations. No original config or seal was overwritten.

At this section's completion, census attempt three is in flight against the
corrected model and its explicit new seal. No positive MoE cell is yet claimed.

## 7. Full-model route census, completed at 21:05 UTC

After the observer API correction documented in
`lfm-census-mapper-api-2026-09-04.md`, action
`7685fdfd8dc5c9aa92545123979d66804c60c10c21ef0e90750e4411e6a9b6cc`
completed on Sparky with original worker return code zero in 201.925 seconds.
The controller source was frozen `dcd18b1`; its actual PB source snapshot was
`a7bd2705062a83d953d0d8a68543913c68d1a7e5`. Receipt CAS:
`e0334201733d90533c26aaf85d2efd1de3fbe967ae1ade9d1ccf9a084b8f22f4`.

The actual result, preserved at `census-bound-r4/census.json`, has SHA256
`825157292db88bd3791d59d867743ddf8e37e68dd24c63e93cfc919d927a2028`.
It reports `verdict: served`, no problems, and all 22 declared expert stacks
in both phases: decode `M1:N3584:K2048`, prefill `M64:N3584:K2048`. Every record
names `vllm.fused_moe.modular_kernel:TRITON`, `torch_materialize_stock`,
`fp8_per_token_dynamic`, and resident mode. Execution is eager on sm_121,
using exactly the EUGR digest named in §1. The runtime versions are
vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, torch `2.13.0+cu130` and Python
3.12.3. The raw record retains the TRITON suffix; agreement normalizes only
the scheme-owned entry-point name.

The same action's collector verifies the common plan, 2,112 projection units,
exact owner bijection, every group/role rung, geometry and sidecar hashes.
Full artifact identity before and after matches the selected corrected seal,
whose SHA256 is `3208ca1e85ebcf2a512924789f1bacff3a22319cdbb67e81f4c3f36348dff634`.
The model is explicitly mounted read-only. Cleanup reports the exact container
absent, no GPU compute processes, measurement completed and safe to release.

An independent CPU-only replay, action
`7769fb00d89e6dc3db9528aba6276bfe70eca30888306b59f449a6929aa3ee2a`,
checked the original worker/snapshot, raw hash, selected seal/path, pre/post
binding and cleanup as well as all collector assertions. Receipt CAS:
`7c56ab96606689aac51a26b9d3c332f20ab0d4236f8b6ed6812ff577bc0c2b98`.
At capture time contract v15 honestly reports zero covered and 22 unattested
owners in each phase. A later contract replay must not rewrite that history.

## 8. Matched full-model student quality, completed at 21:10 UTC

Student action
`c5293606e22df3069cc5c8a4d3a7a1e622c544314c6ecea6d9fc6f38cb90c525`
used the same frozen controller, corrected artifact/seal, image and eager
execution as §7, and the fresh source-bound BF16 teacher from §4. Its actual
source snapshot was `6246a1ecdc54ab50bfedc28de75751d6eeaac8a7`.
The original worker returned zero in 285.542 seconds; receipt CAS:
`4243c59cbe304834a3f7e679cfcd454c94871573c19c3f11aa537759317220a8`.
Both metadata records agree on corpus/tokenizer, 4,096 prefill positions and
requested top-K 1,024. Alignment is checked with no problems. Actual returned
support can include the prompt token as a 1,025th entry; this is recorded,
not mistaken for a different requested K.

The comparison is a **top-1,024 intersection lower bound**, not full-vocabulary
KL. All-position mean lower bound is **0.08316135646812599**, with upper bound
1.1787049111897945 at the declared probability floor
`3.720075976020836e-44`. Top-1 agreement is **85.107421875%**. Teacher mass outside
the compared support averages 0.011342183808770109 and reaches
0.6442256699132436. Among 1,968 confident positions (48.046875%), the mean lower
bound is 0.04127552148969317 and upper bound 0.29772200627592027. The wide upper
bounds and the missing tail remain limitations; the legacy floor-substituted
score is not used as a bound or acceptance result.

| student output | SHA256 |
|---|---|
| `student_tessera.json.npz` | `282270b9c80c387b2a9c3e25e4b681af27f5abfbd2bb64857e915351baa2aa6f` |
| `student_tessera.meta.json` | `8a0a885845d698c23c1aa0e85b28bcd7eb37753876c61f9bc89cb3d18dd67b18` |
| `student_tessera.build.json` | `1045d4277f13c7e976fbc5e700e125410a05d68ad6048f1bcd68ee81d33bb1e2` |
| `serve_student.log` | `caac53bdc28a5ea3e39483b6cb208e32c899fbfd5279a415900a77bf8f2acd5c` |
| `kl_tessera_ts5lfm.json` | `e87351070f93345d2a13e806e23ab6aee54c6b5d29243fea6cf69004e370d078` |

The selected teacher binding hash is
`5d122c2587d61d1af8a6cf28e5d8996c719f2897cb2605821948b31f4db606a8`.
Every teacher output is rechecked against it before and after the student;
the quantized artifact also matches its seal before and after. The student
container and GPU processes are absent at verified cleanup.

The generic greedy smoke prompt produced repetitive `France is` text. That
observation is retained, not used as a quality verdict or attributed to
quantization without a matched BF16 prompt. **The matched prompt was run later
the same day** (`moe-evidence-debt-2026-09-04.md` §7): the BF16 source on this
same image returns the identical completion, character for character, so the
repetition belongs to the model and the prompt rather than to the quantization.
That control does not change this section's numbers or its refusal to call the
smoke a verdict. The measured comparison above is
prefill quality only; the decode census proves dispatch, not decode KL.
There is no general numeric student-KL cutoff in the current cell-promotion
contract, so successful comparison execution is not called a quality pass.

A bounded read-only source comparison against the exact EUGR image found no
concrete disagreement in gate/up order, stock FP8 conversion/quant config,
kernel arguments, sigmoid routing, normalization, scaling or expert-bias
handling. Inspection actions `bd068bb6e28a` and `1a9cf5051b74` were CPU-only;
their receipt CAS values are
`ce1399e7b9b3340cbeac7a6e972794188d946021f34c7a6ce5b06dda96b43091`
and `7fe42bcfe2be4c699bac59c3ab2eb904ad812d905f6eb4e5569ae6fcb9c04362`.
Static agreement does not prove output equality or assign a cause to the smoke.

## 9. Scoped contract promotion tests

The proposed v16 cells are only E4M3/q1024 routed MoE, sm_121, resident/eager,
the exact EUGR image, decode and batch. The raw r4 census is preserved under
`docs/measurements/census/lfm25-8b-a1b-served-r4.json` with one repository
trailing newline added; the test removes exactly that newline and checks its
original SHA256 before replay. The recorded v15 unattested result is untouched.

All 13 new measured-cell tests first failed in CPU action `1d0dcc90e27e` at
`tests/test_lfm_measured_cells.py:48`: `measured LFM routes have no matching
promoted cells` (`None is True`). Each scope/disagreement negative first checks
the same real positive control, so absent cells cannot make that matrix pass.
An earlier fixture-transfer attempt caught only the added trailing newline;
that failure is not the promotion's red proof.

After adding the exact measured cells, action
`759eb7313e9c864bcf2ceca46d8a802a54333cbcd35ae5ef9f3360defe49693d`
returned zero: 180 passed in 2.34 seconds across measured-cell replay, serving
contract, runtime scope, attested wire, census agreement/runtime scope and
the campaign collector. Population: serial CPU, torch 2.11.0+cpu, no CUDA
device, zero skipped and zero uncollected modules. Receipt CAS:
`c59c516c06effd80538a19c330f12f05bf2c389d527575110411a59b44959209`.
This does not replace the full artifact-bound `--require-attested` replay or
the coordinator's dual-population integration run.

## 10. Complete post-promotion replay and preserved scope

The full artifact/plan/sidecar collector was subsequently replayed with
`--require-attested` under CPU-only PrismaBuild action
`4510df8b36115685501166ba03a400eb21f31ae72bc1626d75656fce0f763ea8`.
The original worker returned zero; receipt CAS:
`fb4361f254a72f8ea67387a928a19af1f0621218c609cb8f5ad198211ed67763`.
Its output is `census-bound-r4/check-v16-promotion-r1.json` under the campaign
directory. The collector derives 22 required owners and 2,112 projection
containers from the sealed common plan and matching model sidecars. Both
decode and prefill have **22 modules, 22 covered, zero unattested and zero
unsupported records**. Each phase resolves only its corresponding exact-runtime
v16 routed-MoE cell. This is stronger than a top-level agreement boolean and
does not modify the original v15 census or rerun its GPU work.

Action `27f7ecb8aaa2f0024291792afa7e62369a8a00dc08c03b151eafda4137b9fbc4`
independently compared the v16 document against pre-promotion `424dad5` and
rehashed the complete corrected model against its original corrected seal.
It returned zero, proving all eight dense cells, all non-lane contract blocks
and prior changelog entries unchanged, and every current artifact file still
matching that seal. Receipt CAS:
`1c919a54c7a6bbe7c43124752a5aa35fa8a9fa6e5a5d571de9e645fb86b5885a`.
At `048325c`, the raw contract SHA256 was
`b2704acb6a2815baba4d014dde5ed1d1a126749512235d00043662bb259513ac`.

A separate prose-only correction, `af9d23b`, changes the TP note's stale
"both families" to "every declared family"; it changes no TP bound or gate.
Action `90f7fc2a3f0feab2850e139340f38bbaab16125f38e026d35b79d0c12ed3f13b`
verified that exact textual delta, then repeated the non-lane/dense/history
invariance checks with only that documented text normalized. Original worker
exit was zero; receipt CAS:
`47134ae73689afe5658048047683e03a54c5b6f8d83652a1e4bd15d1054af85a`.
The resulting raw contract SHA256 is
`75137c73bce8837713b427d977beb0eec280faccb39fd3225acf1b3bd00eb0b1`.
The collector's `current_contract_sha256` instead hashes canonical parsed
JSON; it is a different explicitly defined identity, not this raw-file hash.

**Verification-attribution correction to §7.** The independent reviewer
checked the original PB worker/snapshot identity by direct inspection.
Its CPU-only replay action `7769fb00d89e` checked the raw census hash,
selected seal/path, artifact pre/post identity, cleanup and collector
assertions, and printed the source snapshot. The earlier §7 sentence combined
those checks; it must not be read as saying the CPU command itself asserted
the original worker/snapshot equality. The verification and evidence remain
valid, but their mechanisms are distinct.

Independent review also corrected present-tense route docstrings and added a
dated supersession note to the historical MoE design document in `06e5ac9`.
Those are prose-only corrections to the now-measured scope; no frozen encoder
or serving snapshot, contract byte, default or quality threshold moved.

## 11. Selector-selected population and final assertion repair

The selector ran in the promotion checkout through CPU action
`b71e10f4a79c9de1f1e39d6d9bb35e180591553cf8168a0bfd8445a302e995d4`,
using the exact fetched base `424dad5f134f42e84e5f2ea8cd2ee1d19042c236`.
It reported `narrowed`, 52 files, via direct tree comparison because the PB
snapshot is parentless. Receipt CAS:
`ba9bfee7643e5776ab9dcf829b9fe9b33c4f127b4a212147ad1ff22398f151ab`.
Seven core files had the 180/0/0 CPU result in §9. The remaining 45 files
were submitted once as an explicit pytest argument list in action
`4ffaeea0a706b4ef0f88ba4e5996845971f2b17aedade2b268ca3796babd027b`,
on source snapshot `4d9c0e8d7a05dea7571dec9bd4f50e5180aa86ba`.
The already-reviewed census-test AST scoping fix `dcf8ee8` was included as
`16477b9` before this run; no old known test failure was intentionally rerun.

This action was **red: 647 passed, one failed, 301 skipped, zero uncollected**,
serial CPU with torch 2.11.0+cpu and no CUDA device. The deployed pool retried
its nonzero result three times; the final worker returned one in 208.057
seconds, with pytest reporting 206.16 seconds. Logs and the action's final
failure record remain in PB; no successful CAS result is claimed for it.
The population is `astra-lfm-promotion-selected-r1.surface.json` in the
parent `lfm25` directory, and earlier automatic-attempt populations were
preserved by the harness as `surface.superseded-*.json` siblings.

The sole failure was `tests/test_serving_moe_route.py:130`: expected regex
`every expert of a stack`, actual refusal
`m: group 'w13' carries 2 expert row(s) for 3 experts; every expert must have its own row of projection containers`.
The existing guard correctly refused the missing expert; the test retained
old explanatory wording after the projection-granularity prose correction.
Test-only commit `c98434d` matches the named group/row-count refusal instead
of its explanatory suffix. No production behavior, wire or gate changed.

Only that failing file was run against pristine master
`dbfce8bb72f2dd384a2cc723f3df434b4d3e7f47`: action
`c3733bc17a818cca0cc14011eb4fb8cd218b99881e13711e69f93c0540413246`,
six passed, zero skipped/uncollected, CPU-only, 65.33 seconds, original
worker exit zero; receipt CAS
`a43c16e1dc0b5001949a8f308f1911c70561616a2a2459d5631801bcd726eb42`.
The branch therefore owned this assertion drift rather than borrowing a
baseline excuse. After the fix, the complete affected file passed **9/0/0**,
CPU-only, in 66.16 seconds, through action
`9da0943d7e16081c563668b9c67d97efa95ce3f64b607ee27603b16aa5346266`
on snapshot `ee6b2a32ef3e926cb4ba154ca93f0cd2a9920de7`; receipt CAS
`ea6a03b0ff6d056b2f18835183a74a94cd8fa8e231f3b46140a4b2cb08e15ca1`.
Its population is `astra-lfm-moe-row-message-green-r1.surface.json`.
The 44 unaffected files were not recomputed, and these separate-source
targeted results are not described as one all-green merge-suite receipt.
The coordinator owns the dual-population full suite on the integrated tree.

The 45-file run's 301 skip reasons, verbatim:

| count | reason |
|---:|---|
| 81 | `the encoder is a CUDA path` |
| 79 | `needs a CUDA device` |
| 29 | `the Viterbi is CUDA` |
| 24 | `the kernel lane is a CUDA path` |
| 23 | `the lane is a CUDA kernel` |
| 15 | `the encoder is a GPU job` |
| 14 | `the Tessera encoder is a CUDA path` |
| 6 | `/home/rob/tessera-runs/compile-dispatch is not on this box` |
| 6 | `needs CUDA` |
| 5 | `the kernel lane runs on CUDA` |
| 5 | `the reach checkpoint is not on this box` |
| 4 | `Qwen3-0.6B is not on this box` |
| 2 | `no stock twin` |
| 1 | `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box` |
| 1 | `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box` |
| 1 | `E2M1 publishes no reader range` |
| 1 | `could not import 'vllm': No module named 'vllm'` |
| 1 | `the PrismaQuant tree or the allocation outputs are not on this box` |
| 1 | `the PrismaQuant tree with tessera_formats is not on this box` |
| 1 | `the shipped checkpoint is not here` |
| 1 | `the two surviving compile caches from 2026-09-02 are not on this box` |

Final independent read-only review found no blocking scope, provenance or
negative-matrix defect. Its prose and verification-attribution findings were
fixed in the separate commits recorded in §10. Contract raw SHA256 remains
`75137c73bce8837713b427d977beb0eec280faccb39fd3225acf1b3bd00eb0b1`.

**Note, 2026-09-04, after this campaign (#131, #133).** Contract v17 / lane-
eligibility schema v6 gives the two `routed_moe` cells the toolchain this
receipt records (`runtime.vllm 0.28.1rc1.dev397+gfd4a15126.d20260904`,
`runtime.torch 2.13.0+cu130`, the EUGR digest) and an `evidence` field: the
batch cell `kl_lower_bound` on §8's prefill top-1024 bound, the decode cell
`route_only` on §7's census, both with `smoke: repetitive` from the paragraph
above. The v16 SHA quoted in §10 and §11 is the file these sections were run
against; the v17 file's raw SHA256 is
`ba3a3c69027a11f7bf9ef570867c58d608d0865c3e6a17dfeea53ba43ce055e6`.
