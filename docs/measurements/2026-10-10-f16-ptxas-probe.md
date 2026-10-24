# Probe a newer ptxas for f16 prefetch hint retention (issue 1192)

Date: 2026-10-10. Parent: tessera#739. Deliverable: measurement. No kernel
change. No toolchain pin change.

## Question

The f16 library emits 96 `prefetch.global.L1` hints in PTX, and ptxas
13.0.88 drops all 96 before SASS. A newer ptxas that keeps the hints
would give the parent a cheap prefetch lever instead of the cp.async
ring. This record asks which ptxas versions the pool holds, and repeats
the f7d34fc probe on each one it finds.

## Method

The probe builds the fused routed kernel translation unit twice with
nvcc under the f16 library flags, then counts the hint in PTX and SASS:

```bash
IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
FLAGS="-O3 -lineinfo -std=c++17 -DTESSERA_ROUTED_FUSED_FP8=1 -DTESSERA_ROUTED_FUSED_MMA8=0 -gencode arch=compute_121,code=sm_121"
nvcc $FLAGS $INC --ptx base.cu -o base.ptx
nvcc $FLAGS $INC -c base.cu -o base.o
cuobjdump -sass base.o > base.sass
grep -ci prefetch base.ptx
grep -ci prefetch base.sass
```

`base.cu` is the master source. `new.cu` is the same file with the
one-line prefetch gate (`FAMILY_FP8 ? A_PREFETCH`) plus its comment.
`$INC` is the torch plus Python include paths from the image.

The baseline run is PB action f7d34fc on sparklina in that image. Its
inputs match this checkout byte for byte: `base.cu` sha256 `8a1536df`
equals `src/tessera/serving/csrc/routed_fused_window.cu` here, and
`new.cu` sha256 `d8e7daa2` equals the prefetch source blob. The sweep
for newer toolchains reads `nvcc --version` and `ptxas --version` on
both Sparks, in every CUDA-bearing image and on the hosts.

## Results

One ptxas version exists on the pool, so the table has one version.
Both arms come from the f7d34fc log, which this writer read from the
pool archive:

| ptxas | arm | PTX prefetch lines | SASS prefetch lines |
|---|---|---|---|
| 13.0.88 (V13.0.88) | base source | 0 | 0 |
| 13.0.88 (V13.0.88) | prefetch source | 96 | 0 |

The 96 PTX hints die in ptxas. Both SASS binaries hold no prefetch
instruction. The baseline repeats the known negative result.

## Sweep: no newer ptxas is available

| Location | ptxas | Evidence |
|---|---|---|
| sparklina host `/usr/local/cuda`, `cuda-13`, `cuda-13.0` | 13.0.88 | PB 0749fb3a |
| sparky host `/usr/local/cuda`, `cuda-13`, `cuda-13.0` | 13.0.88 | PB 14a01780 |
| sparky `/usr/local/cuda-13.4` | none: `doc`, `gds`, `targets` only, no `bin` | PB 4d6290ba |
| sparklina images: spark-vllm nightly-20260929, f8dbe1a0-kpooltail1, 0afec8d4, glm-tp2-diag nccl230 and nccl229, lfm25-teacher, stage6-encoder, glm53-mia-sm121, vllm-openai v0.30.0 | 13.0.88 | PB 0749fb3a, d2d8e6a2 |
| sparky images: spark-vllm nightly-20260929, f8dbe1a0-kpooltail1, a5424378-mtpmap1, 0afec8d4, pinned-0afec8d4, vllm-openai latest, glm53-mia-sm121 | 13.0.88 | PB 14a01780 |
| dl380g10 host | none: no CUDA toolkit; torch 2.11.0+cpu only | PB 5ed91cbd |

Two fresh repeats were submitted and withdrawn: PB 9555a3fe waited 9
hours READY while the pool deferred CPU-only work behind GPU load, and
PB f94b6ab0 waited 25 minutes with the same deferral. The pool admits
no CPU-only compile on the Sparks under current GPU load, so the
baseline evidence rests on the archived f7d34fc run above.

## Verdict

The prefetch lever cannot survive on a newer toolchain: no newer
ptxas exists on the pool to test it. k2, the cp.async ring, is the
only path. When a newer ptxas arrives, repeat the probe above before
any timing claim. A change to the repo toolchain pin needs a decision
from Rob.

## What a hint count is not

A hint count is not a speed result. A hint count is not a wait result.
This record makes no timing claim and no claim about C2 or C4. It did
not measure kernel time, stall or wait fractions, profiler counters,
other libraries, other architectures, or other toolchains.

## Clarification, 2026-10-11

The availability statements above apply only to the recorded toolchain sweep.
The two withdrawn repeats produced no result and establish no current admission rule.
The retained probe reports ptxas 13.0.88, with 96 PTX hints and zero SASS hints for the prefetch arm.
No newer-toolchain probe completed. Hint retention on a newer toolchain remains unknown.
A hint count does not establish a replacement for the cp.async ring.
This record supplies no speed, stall, wait, C2 or C4 result.

## Attribution correction, 2026-10-11

This correction supersedes two statements above:

- The Verdict claim that starts "The prefetch lever cannot survive" has no observed newer-toolchain result. Newer-toolchain retention remains unknown.
- The preceding clarification says the retained probe reports ptxas 13.0.88. The f7d34fc log prints the 96/0 counts, but no compiler version.

The 13.0.88 label comes from the sibling-image sweep and `docs/ARCHITECTURE.md`, not from the retained probe log.
No newer-toolchain probe completed. This correction adds no measurement or qualification.


## Decision-gate correction, 2026-10-24 (W1 / #1369)

Preflight plan: one recovered hint-off/on e4m3 compile/disassemble pair on
the configured GPU arm, sm_121 to match the historical target. This decides
static retention only, not execution or timing. The host interpreter lacks
torch and no compiler is selected there; the CPU arm must separately attest
its own inventory. The GPU image is needed for its configured compiler/include
evidence, not for running a kernel. Expected budget: at most 10 GPU-arm
minutes; no serving, device execution or timing sweep. Attempt this single
pair before considering any wider configured-toolkit comparison.

First GPU-arm submission failed before selection/compilation: shared sources
were not mounted in the arm. A second single-pair attempt uses the unchanged
checkout baseline plus a small recovered source diff transported in the snapshot;
verify both full hashes before compiling. This is byte recovery from inspected
retained sources, not a claim that an invented reconstruction emits 96 hints.
The retry has the same static-only question and <=10 minute budget.

### Preflight outcome and retained-evidence correction

The second snapshot/patch submission was refused with exit 125 at SETUP:
`ldconfig` was missing and requested outputs exceeded execution limits. Its
command did **not** run; it produced neither a compiler inventory nor a compile.
The first submission failed because the shared sources were not mounted.
Neither is a completed pair. No further GPU retry is authorized here. The x86
inventory reports torch 2.11+cpu, no CUDA_HOME or selected compiler via
PATH/PYTORCH_NVCC, and no selected nvcc, ptxas or cuobjdump. This supersedes the
preflight's assumption that a usable supported compiler would be available.

The recovered f7d34fc action log is CAS blob
`ea71df52217c2671f785c537a21f4ca50fd12cbc23af971bc88ff5373e024ae1`
(`cas/blobs/ea/…`, 1,293 bytes). It prints source hashes, PTX counts 0/96 and
literal SASS `prefetch` counts 0/0, but **no tool versions**. Literal spelling
counts do not recognize native cache-control instructions. The failed
`a28bed109503` result supplies no transferable retention or timing evidence.
The negative sweep and withdrawn/failed attempts above remain history, not a
current inventory or proof that no alternative exists.

Recovered source attribution (under the retained measurement root
`t8r-739-f16prefetch-20261010`; suffix
`src/tessera/serving/csrc/routed_fused_window.cu`):

| Side | Retained tree | Full SHA-256 |
|---|---|---|
| off | `src-master` | `8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f` |
| on | `src-f16pf` | `d8e7daa286a214a2640480ae6c7a460af41695f084e13e41482f2fb2d4d3622f` |

These are recovered bytes, not an invented hint-on reconstruction. However,
source recovery alone does not bind a disassembly to those inputs or attest
compiler, assembler, headers and flags. In particular, the **currently retained**
SASS files differ in length from the action's recorded 90,554,718/91,327,937
bytes. They must not be called the original f7d34fc disassemblies.

The local CPU-only analyzer rerun used the actual `sm_121` architecture and the
fixed mnemonic regex accepting underscores (the earlier parser rejected
`BAR.SYNC.DEFER_BLOCKING` and `F2FP…UNPACK_B`). Both retained files now parse
completely, with no errors:

| Retained file | Bytes | SHA-256 | Native prefetch hints | Other cache controls |
|---|---:|---|---:|---|
| `sass/master.sass` | 90,556,492 | `79e7e897916d6a977540edac507ae4eab558988d4355ea50b219cf318c79ceea` | 0 | 30 `CCTL.IVALL` |
| `sass/f16pf.sass` | 91,329,820 | `c14cc131544bc3a5f3ca9c64644412264eb8a77fbb3567d7df021f8257b7fed5` | 96 `CCTL.E.PF1` | 30 `CCTL.IVALL` |

For example, the on file contains `@!P1 CCTL.E.PF1 [R18+0x40];` at PC
`0x1930` in the routed kernel specialization
`ILb1ELi2ELb0ELb0ELi8ELb0ELi128ELb0ELb0E`. Both files also contain
`CCTL.IVALL;` (for example PC `0x6670` in specialization
`ILb1ELi2ELb1ELb1ELi7ELb1ELi64ELb0ELb0E`); invalidation is not prefetch.
These are raw disassembly observations, not a newly compiled or versioned pair.
They withdraw the claim that zero literal `prefetch` matches proves removal of
all 96 hints by ptxas. They establish no timing causation, speed, stall, wait,
C2 or C4 result. The recovered on-source comment explicitly says
“Its timing is unmeasured”; do not transfer instruction-family timings to this
16-bit E4M3 library or treat PR #1165 as a measured f16 speed result.

**Owner decision:** retain the raw hint evidence, but no fully versioned,
usable supported alternative has been established. The decision gate is
**inconclusive**, not permission to start cp.async and not proof that hints are
unusable. If a future supported retained pair confirms a usable hint alternative,
stop cp.async **before starting** and return to TrustLadder for replanning.
#1192 owns the toolchain/retention question; #1369 owns this evidence repair and
preflight. Do not open duplicate work. Parent #739 remains O2 native / O6 pending;
this static observation does not qualify or close the parent.

### Reproduction (not another GPU submission)

Run from the checkout; supply `RETAINED_ROOT`, `PROBE_OUT` (a fresh directory
outside the checkout), and the project's supported CUDA/toolchain configuration.
The script uses the serving selector, records selected tools and versions, and
never loads a cubin or runs a kernel. Do not infer a version from a sibling image.
The following recovers and hash-checks the patch pair before probing:

```bash
set -eu
: "${RETAINED_ROOT:?retained measurement root}" "${PROBE_OUT:?fresh external output root}"
mkdir -p "$PROBE_OUT/inputs"
SRC=src/tessera/serving/csrc/routed_fused_window.cu
cp "$RETAINED_ROOT/src-master/$SRC" "$PROBE_OUT/inputs/base.cu"
cp "$PROBE_OUT/inputs/base.cu" "$PROBE_OUT/inputs/new.cu"
patch "$PROBE_OUT/inputs/new.cu" < experiments/t4_code/f16_prefetch_on.patch
printf '%s  %s\n' \
  8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f "$PROBE_OUT/inputs/base.cu" \
  d8e7daa286a214a2640480ae6c7a460af41695f084e13e41482f2fb2d4d3622f "$PROBE_OUT/inputs/new.cu" | sha256sum -c -
PYTHONPATH=src python experiments/t4_code/f16_prefetch_probe.py \
  --off "$PROBE_OUT/inputs/base.cu" --off-identity recovered \
  --on "$PROBE_OUT/inputs/new.cu" --on-identity recovered \
  --architecture sm_121 --out "$PROBE_OUT/pair"
```

For raw retained SASS only, rerun the analyzer without invoking a compiler:

```bash
python - "$RETAINED_ROOT" <<'PY'
import collections, hashlib, json, pathlib, sys
sys.path.insert(0, "experiments/t4_code")
import sass_dump
for name in ("master.sass", "f16pf.sass"):
    p = pathlib.Path(sys.argv[1]) / "sass" / name
    raw = p.read_bytes()
    r = sass_dump.analyze_prefetch_sass(raw.decode(), architecture="sm_121")
    print(json.dumps(dict(file=name, bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(), status=r["status"],
        hint_count=r["hint_count"], errors=r["errors"],
        cache_controls=dict(collections.Counter(
            x["opcode"] for x in r["instructions"])))))
PY
```

Formal test evidence is to be appended separately; the raw rerun above is not a
compiler check or a test-suite result.

### TrustLadder-only historical PR handoff

Proposed PR #1165 scope/closure wording (handoff only, not published):

> Source-only experiment and negative evidence record for f16 prefetch; the
> optimization was reverted. This does not close #739 or establish C2/C4, wait
> reduction, serving qualification, or permission to proceed with cp.async.
> Literal SASS text counts did not establish hint removal; compiler attribution
> and newer-toolchain availability are limited as stated in the dated corrections.
> Retention evidence, if obtained under #1369/#1192, returns the decision gate for
> owner replanning without changing the runtime pin.

To complete the unresolved compiler alternative, repeat the bounded pair for
CUDA 13.0.88 and each newer ptxas actually accessible through the supported
configuration, recording `CUDA_HOME` or `CUDA_PATH` and any `PYTORCH_NVCC`
override separately. Run inventory first; use compiler-reported versions, not
root directory labels. Do not install a private toolkit. No additional toolkit
was discovered or compared in this run. The explicit PTX-to-ptxas pipeline
attests the assembler actually invoked rather than borrowing a sibling nvcc
version; it does not claim an unobserved implicit nvcc build is identical.

### Verification receipts and unavailable checks

Plain CPU check, after implementation:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_f16_prefetch_sass.py tests/test_f16_prefetch_probe.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed 231 passed, zero skips, zero device allocations. This covers the 16
protected analyzer cases, three additional probe/syntax controls, and both
mandatory guards; it is CPU/static evidence, not CUDA numerical qualification.
The guard snapshot was extended only with the supplied existing issue 1369;
no issue was filed. An unnamed-directory-read regression in the first probe
implementation was fixed by explicitly tracking generated artifact paths.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1d7333c52c4979919b3c2a35a79d7ef290c3e1af30e1c23030e5c87603f2999a","size":1870}},"operation":"a7e9a03e1a8158ca99300c28a31f28af57a1dd97cd92e8eabce52f5b311d933d","receipt":"39821e66931223c3e9df97b1b25e212438be2a0f78ab65fc52d709ca47b1d16f"}]

CPU inventory command was `PYTHONPATH=src python experiments/t4_code/f16_prefetch_probe.py --inventory-only --out "$TMPDIR/f16-inventory"`, followed by printing its manifest. Torch was 2.11.0+cpu; CUDA_HOME, CUDA_PATH and PYTORCH_NVCC were unset. No nvcc, ptxas or cuobjdump was selected; include discovery reported CUDA_HOME unset. Inventory exit zero means inventory completed, not compilation passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6c64b12f7798ba12bee8cce6ae59e9e7740bebb294ac18bc9324e3fb0262f847","size":2235}},"operation":"c4e40bee518b5e362b42762afe2a3a14f444d179751e07e8d435af3a74f87437","receipt":"02009a42569cc93a237b313e93a91ec72e8ed01e6f9915a3775ebe16e913afb1"}]

The first GPU-arm attempt used the exact shared recovered-source paths and
failed before compiler selection because those paths were not mounted there.
Its retained small archive contains the failure manifest, not compiled artifacts.

[Test evidence {"artifacts":{"outputs/f16-pair-evidence.tar.gz":{"digest":"8ffbd12ac7b6beba364879f685ff109f00f12abc51b8ac83a7330025946cb240","size":693},"stderr":{"digest":"4998c39b3aba17f95c592cc12bb9909f4deebe9c0170ee8e6ec809d3fa0dab85","size":160},"stdout":{"digest":"9db7f1fc7972ebdf70e7eefdae465077e29ff8dd51d600bb2fab79a377ea3882","size":983}},"operation":"589688ecb70c471992a527161db0f92624b2883d107f1a898828ed73b5d6acf8","receipt":"3e14967d0142919a98ed90affed45729b76c899998b989ff667ef00954e6a2e1"}]

The snapshot-patch retry was refused at SETUP (125): `ldconfig: command not
found`, then `ValueError: requested outputs exceed execution limits`.
COMMAND did not run; requested archive was missing. No successful compiler
inventory, compile, GPU execution or timing is claimed from that submission.
The probe now avoids duplicating whole parsed instruction streams in its
manifest; raw SASS and bounded hint context remain retained. No runner or
GPU environment modification was attempted.

[Test evidence {"artifacts":{"stderr":{"digest":"31cf213633c448b647d7047d7341b08b2adf4d09c6364989d6d39d3e5072c914","size":395},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"7113f5b2793f44223f3da078f853e8597835929b165b7ed9da5716441ed7854a","receipt":"1d86471121ee258f8a938b58d6d92938abb1f529994c1c4c7b0f265aa2ce7439"}]

An exact-checkout-base selector invocation, `python tools/impacted_tests.py
--ref 9131ad638f1f01781cc83acd59ce7a075e785074 --json`, failed before selection:
`fatal: bad object`. The runner snapshot again lacked that object; no impacted
selection pass is claimed. This is separate from the passing mandatory guard.
No historical-commit lookup was added to any test.

[Test evidence {"artifacts":{"stderr":{"digest":"fef12367ab306f4a6a5a061085fc9bee3efbd2ab88c8d3508585f52c8d28b201","size":117},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"2469375f35bef01ee139be0b0b06858ebf3df75c6e476cfb5d7f2cd713a34640","receipt":"1194b50fe34c560dbdefa9034955324bd49525a4a9a4da1e308844501f923885"}]

Final rerun of the same four-file command after documentation and test review:
231 passed, zero skips/uncollected modules, no CUDA and zero device allocations.
Removed a redundant new-test assertion that merely echoed the caller-supplied
source identity; actual source retention, diff, failure status and artifact
custody assertions remain. No existing test was removed or acceptance file edited.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"25a82aa410dc964e09f8f2127e84f388b89ae4522c4d21c65b830fac8454e921","size":1845}},"operation":"9f8829c9b616945c786f3d20096d95fc34b780fc4d9a9de100847c7fa36f4bdd","receipt":"dd950d4a5cebf8065d905bf7b1c4003fd1fc5e7e5c293371f8c0007031820eae"}]

## Review repair, 2026-10-24: malformed PCs and paired symbol boundaries

Review found two local evidence-generation defects despite the earlier passing
fixtures. Regressions added outside the protected acceptance file reproduced
both against the reviewed implementation at eb70fcf3f5ec5d7d11f693453310f257d949062d:
malformed-PC comments such as `/*0010 */` and bare known/unknown CCTL lines were
silently skipped after a valid EXIT, returning complete/zero. Also, anonymous
namespace normalization left the Itanium source-name length and could greedily
consume a following function length/name. Synthetic valid `base.cu`/`new.cu`
symbols therefore failed to pair. The pre-repair regression run observed six
failures and four passes, including all four new missing/malformed-PC examples
and two independent namespace-key expectations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"04d6df5a4dfefe4dc200ffa7eec8207edae979ad8b5b48e82ee7e442839d2783","size":8845}},"operation":"8027c9d6a2289573cb75644ec83b9d7c620ab61d072da5fd8347c33764edb76d","receipt":"7640bafe1e25a70a069ebcaa5d24795cf26d952eb89710b55d80f34a10f05eef"}]

The analyzer now rejects executable-looking CCTL leftovers after comment
stripping with inconclusive/null PC diagnostics. Genuine comment decoys and
encoding continuations remain ignored. Namespace normalization reads the whole
length-delimited root nvcc source-name component and replaces its length and
payload together, preserving the entire following specialization. Unsupported
or malformed namespace forms are left unchanged, not guessed; collision refusal
remains. Compiles continue to use the original source paths, so relative include
semantics are unchanged. The paired probe now fails a packet with any only-off
or only-on kernel, retaining its comparison diagnostics instead of reporting a
complete corresponding-kernel result.

Synthetic paired-output controls exercise the actual compile-side parsing,
manifest assembly and comparison for unequal source basenames, including a
hex-leading function name and a deliberately different template specialization.
They establish Python behavior only, not fresh compiler output, historical
retention, toolkit availability or adoption. Targeted analyzer/probe plus the
existing dump/compare consumer checks passed 103 tests on x86 (zero skips,
zero device allocations). No acceptance assertions or production settings were
changed; no test was removed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"29795ba0742497cdbf98f0502b1e36b0761c45181c74cd0aebcab7dde53aa08f","size":1652}},"operation":"27c8b9f7cf3ab8dd92e0531ff6feb9c42c1eca9bd291271c678fa484c8978d7d","receipt":"dd6449caa5911555313d2143267e017b676edede278b5646a9d450ab9e8d0510"}]

No new GPU-arm/compiler probe was needed to decide either deterministic Python
defect. The earlier raw artifact observations, unavailable version-attested
compilation and owner-facing inconclusive gate remain as stated above. This
repair does not turn synthetic pairing into a fresh supported-toolchain result,
change the runtime pin, or authorize cp.async implementation or qualification.

Final repair verification (plain command, native threads limited to one):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_f16_prefetch_sass.py tests/test_f16_prefetch_probe.py tests/test_glm53_shared_fold_check.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed 315 passed on x86, zero skips/uncollected modules and zero device
allocations; both mandatory guards passed. This verifies local Python/static
behavior and the existing dump/compare consumer, not CUDA compilation. Retained
stdout was read back to verify the result. No new GPU, timing or compiler
availability result is claimed. The prior exact-base selector unavailability
is not reclassified as a pass; no historical lookup was added to tests.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"76bf7061ba1e156d1406cd259ff166378aaf3ca782f3135eac03608919d4985f","size":1929}},"operation":"77e004d354b1ebb75d73d817cc2d40cc6ab62fe47e4e61bfa08ded30e6643fd6","receipt":"0160497321bb18cd55170822bf92f5ef4f922a8074f1e114a1695e88ffe2adb7"}]
