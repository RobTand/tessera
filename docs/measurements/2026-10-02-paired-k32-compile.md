# Paired-K32 routed E4M3 MMA compile screen, 2026-10-02

Refs #857. This is an opt-in source/CPU screen. No GPU result or measured
performance improvement exists. Terminal correctness #855 is prerequisite
and its separate EMPTY-before-terminal-FULL fix is retained.

The frozen design at `t8_performance/moe_architecture/PAIRED-K32-DESIGN.txt`
uses two paired word/history slots and two paired decoded slots, each holding
two ordinary K32 microtiles. Shared dynamic bytes are76,240 for gate/up and
59,664 for down. Every actual native allocation uses the same
`launch_smem_bytes` owner: the entry's live-device guard, exact instantiation's
`cudaFuncSetAttribute`, and its launch argument. The Python resource mirror
is compared to execution of this native C++ owner in CPU tests.

| Region | Gate/up bytes | Down bytes |
| --- | ---: | ---: |
| LUT |32768|16384|
| Four B microtiles |16384|16384|
| Four A microtiles |16384|16384|
| Row-scale slots |1024|1024|
| Descriptor slots |64|64|
| Claim |16|16|
| Reserved descriptor ring |384|192|
| Four history microtiles |1024|1024|
| Four word microtiles |8192|8192|
| Total |76240|59664|

The admitted source scope is E4M3 MMA, non-dense and unsplit routed MODE0/2,
one-run R4, exact slot8, BMT128, K%64=0, K>=192, and input rows>=512.
K128/K160/odd logical K32 count and small decode inputs fall back entirely.
The build flag defaults off and is consumed only by the existing E4M3 MMA
build owner. This parent source contains only legacy resident addressing;
no piece-major bytes or accessor are imported from a different live branch.

Producer pair q first stages its odd-K activation/map lookahead in the
original current/next registers. It waits for the one current copy group,
then256 producers synchronize. The next pair's words/history may now occupy
the retired pair q-1 slot. EMPTY[global_pair%2] acknowledges both previous
decoded microtiles before overwrite. Both microtiles are decoded in order,
then FULL is published once. Consumers run the original MI/G/parity MMA
sequence for u0 then u1 and release EMPTY once. The pair counter does not
reset across items. The final empty async group is retired before new LUT
copies. Three pairs are necessary: before producer item i+2 overwrites
row-scale slot i, pair2 of item i+1 must acknowledge its first pair, which the
consumer cannot release until item i's epilogue is complete.

CPU source-owner causal RED: PB `ea88a84a3b45c7620be6fb5bf420ccef1100b4177fee058079503fb3b275c4af`
ran25 cases against the unchanged parent plus new tests;25 expected setup
errors: `native paired scope owner is absent before this feature`.

CPU GREEN: PB `1261bff831be165015a0b15c2d281a64cafa12674a1615ec9035065c7cc85f76`,
dl380g10, CPU2/native1/memory3GiB, xdist2/worksteal:48 passed in6.91s,
zero skipped/uncollected/device allocations. Native C++ allocation/admission
and the actual paired producer branch are executed under a delayed-consumer
model. The forbidden two-pair item causally fails `item slot overwrite raced
last epilogue`. This model is not CUDA numerical or concurrency qualification.
The retained16 terminal-branch cases remain CPU protocol evidence only.

The source is narrowed in separate commits: extract the ordinary microdecode
and register advance without changing its body, then add paired scheduling.
Original and paired specializations must compile without new spill/local
traffic, and exact ELF/source/flags/resource/SASS are required before root may
authorize a finite real-wire GPU correctness action. Baseline and candidate
profiling, both-host Netdata, live shared-memory carveout and matched served
quality/decode remain required later. Energy remains HOLD while clocks are
unqualified. The historical121-register /57,552B dynamic gate/up profile is
context, not this candidate's resource or speed result.


Final CPU population on frozen source10f3d16d: PB
`5ac82fb983cbcb575af913b3c486a20532835b9064552e8112a9f2d336173a8c`,
dl380g10 CPU2/native1/memory3GiB, xdist2/worksteal,56 passed and423 skipped
in6.15s, zero uncollected/device allocations. All423 skip reasons are
`the lane is a CUDA kernel`. This includes original support predicates,
additional oversized-slot fallback controls and the CPU-only source tests.

Actual CPU native compile: PB
`6980a48f85a4e60a5abbad43d7232659f7d3e18714b5309a2750bf62ff167661`,
Sparklina, one CPU/native1/memory4GiB, CUDA hidden, one attempt, exit0.
The E4M3MMA library compiled in56.3s; whole action62.648s, CPU61.920s,
peak resident memory3,864,571,904B. Canonical receipt
`0bd1a58ba35db109c314dfc52d6bd74da17d80abcf970288953c45f5850a6135`,
ELF `bbbbb4d30815b64dd1bd65534d7a853bda0d77c948204fd1ff9aee917ab36f13`.

| Native specification | Original registers | Paired registers | Stack/local bytes | LDL/STL instructions |
| --- | ---: | ---: | ---: | ---: |
| R4 MODE0 BMT128 |121|120|0|0|
| R4 MODE2 BMT128 |122|124|0|0|

Both paired specifications are present in the same ELF as the original ones.
Both report1,024B compiled static shared memory, separately from the dynamic
76,240/59,664B launch requirement. Actual runtime allocation/carveout is not
measured. The compiler unrolls the original body differently (48 static QMMA
sites versus32 paired sites); those static counts are not work or speed
measurements. No FP16/BF16 conversion occurs between the first and last QMMA
site. Source comparison proves the per-microstep MMA body unchanged and
both terminal branches plus epilogue byte-identical to the parent. Real
numerical equality and execution order remain GPU acceptance work.

An additional wrapper finding was fixed separately: `bench_t8r.sh` dropped
the paired compile-choice environment, risking an unintended rebuild when
loading the retained binary. The causal CPU shell test (fake Docker, no GPU)
failed on explicit choice1 before forwarding was added: PB
`a1660204e185e7300124501eb3c6fc289778d5aa486a4daf700c7a164c33ea19`,
1 failed/1 passed (`[] != ['TESSERA_ROUTED_FUSED_PAIRED_K32=1']`).
The absent-choice control preserves the default-off behavior.


Wrapper GREEN: PB
`17b36906e6960cac62c68ba0bd186121dcffd7df7f10c1659acccca377e980e5`,
CPU2/native1/memory2GiB, xdist2/worksteal,5 passed in4.59s, zero
skipped/uncollected/device allocations. One earlier invocation refused while
snapshotting because this append-only measurement document changed; it
published no action and is not a test result. The completed GREEN above is
the actual fixed-wrapper action.

Final paired CPU controls: PB
`11b8b395ad8afb7696e87e0f8fad6cada02b099a9b0610154449ae87cd7b022f`,
CPU2/native1/memory3GiB, xdist2/worksteal,38 passed in5.71s, zero
skipped/uncollected/device allocations. Four intentionally unsafe variants of
the actual source loop causally refuse: missing copy wait, missing EMPTY,
reordered microstep and missing final async drain. The original loop passes.
These controls strengthen the CPU model; they do not turn it into GPU proof.
SASS confirms all producer barriers have256 participants and FULL/EMPTY have
512, in both paired roles.

## Closed-world numeric qualification preparation

The exact flag-off baseline compiled from10f3d16d through PB
`8244016432969f0485184effc7e89985769017d15d6a0cd2f7ffd686c2d9cfa2`,
Sparky CPU1/native1/memory4GiB, no CUDA device, exit0. Canonical receipt
`13750f6023bf23fc2873f80398de1baeff0bd9e4f03fbe154958eb82fed150d2`;
baseline ELF `b89ba2d62705abf0152fbc8bb6b338e7bd60f88199211b5f6d61effddfa0c705`.
The flag-on ELF remains unchanged. An initial misuse of `--snapshot-ref`
refused before publication: it carries branch refs; it does not select source.
The correction used a detached exact10f3d16d source checkout.

The existing T8R bench owner has a separate explicit numeric mode for exactly
A8SE L10 TP2rank0 balanced M1,512,2048, pinned readset/native FD, no graphs or
routing/timing overrides. Its normal historical counter gate remains fixed.
Raw gate/up, down-route and reduced-output words are retained and compared
across binaries. Repeated execution, independent fixed-order route sum and
actual full native kernel names/counts are checked. Both arms force the existing
wide knob to1 so M512 exercises the paired BMT128 scope; this does not change
serving defaults. M1 still uses the original scheduling by its row guard.

The existing independent materializing parser, TP sharder, `materialize_stock`
and routed fp64 oracle qualify token0's eight real experts, with the actual
full paired batch's prefix compared bitwise with teacher-forced composition.
Only this bounded prefix has the independent derived-bound proof. The old
oracle's emit-route pass field remains inapplicable to this direct operator
harness; the actual profile establishes native dispatch separately. No
full-model quality or full-batch fp64 proof follows. Its bounded reference
weights are prepared once per arm and reused across the three cases.

Synthetic encoded fixtures separately exercise downK128 fallback and the
minimum downK192 paired boundary atM512, with one persistent CTA forcing many
item transitions and expert changes. Existing fixture TP-history states,
public quantization and the original stage owner are reused. Odd K160/K224
cannot form this full fused adapter's intermediate geometry; their native
admission refusal remains a CPU-source control, not an invented serving test.

The actual PB a01cb090 sealer's CAS payload90eb8054 was recovered and both
archive hashes checked. The new readset preserves871 original artifact
ranges from the true outer-hashed ac4606e8 manifest, removes one unused
historical routing capture and adds two exact native binaries:873 ranges,
3,683,753,731 bytes. Original receipts are not resealed or restamped. The
older a1bca824 inventory with cached-inner hashes is excluded as negative
historical input evidence. Every new wire read still uses the existing public
pinned reader and independent cached-inner/canonical-outer validation.

CLI causal RED on original driver source10f3d16d: PB
`68758fe0a57566851687abbd058c37bf6782c18c50086c2cd797a6d5e71d6a5c`
refused the new exact numeric mode with `retained native artifact requires
the exact counter-only replay`. Final CPU harness/FD/wrapper controls:
PB `819e96949c30502125923f68fff9a4bcc9108ea1e8aa182a16add947c637b0b4`,
44 passed in2.28s, xdist2/native1/CPU2/memory3GiB, no skipped/uncollected or
CUDA allocations. An earlier attempt4bb5cd20 retained5 fixture failures
(CPU tensors/fence not explicitly mocked) and one missing published-SDK
collection error; the scoped CPU fixture and PYTHONPATH correction is included.
These are CPU qualification controls only. No numeric GPU action has run yet.

World-startup regression RED: PB
`c89a6e3e9cc8d8ff095be6181fbbe1445a06b92901c840cc22d437831aa6dc89`,
1 failed/2 passed: the exact driver assignment incorrectly initializes a
vLLM config/world in the closed direct-adapter numeric mode. The minimal fix
skips that initialization only for this mode. GREEN: PB
`d53a2b0598bd80868dc9e05a09986c0b201ffbfaf8a773dcf6c5274326b99041`,
35 passed in1.58s, CPU1/native1/memory2GiB, no skips/uncollected/CUDA
allocations. Both normal real-vLLM startup and stubbed legacy controls remain.
The actual builder passes TP_RANK0/TP_SIZE2 explicitly to
`_RankLocalPackedIntake`; its compact load/finish returns packed bundles and
`adapter()` directly, constructing no vLLM method/config/distributed world.
The exact registered stock FP8 quantizer is still executed. Root therefore
routes the GPU numeric run directly under the vLLM execution exemption;
these CPU preparation and control tests continue through PrismaBuild.

The approved direct-vLLM extension is at the existing StagedInputs and
NativeCallback owners. It uses the same sealed873-range manifest, held
nofollow regular original FDs, bounded pread, unchanged range SHA and
canonical-outer/cached-inner checks, six-field fstat before/after reads and
close, plus actual native FD pre/load/fence checks. There is no PB GPU action,
private queue, fake context, lease emulation, cache, decoder or whole-model
rehash. Every consumed range remains authenticated; unused model ranges are
not read. Source bindings now include the sealed input manifest itself.

Direct-owner RED: PB
`e73d2521bc0423099bfdabc4f52d8b30f117a23e769c7cd78d5c8c94c1a8cd02`,
7 failed because the direct-vLLM owner did not exist. Essential GREEN: PB
`daa406b752cd23ecc0f2b11d6f1b4d2e1c5cda4d62d427f79c933c9328b96d37`,
Sparky CPU2/native1/memory3GiB,101 passed in2.43s, no skips/uncollected or
device allocations; canonical receipt
`925703ae54707baeecde446a981241835f7b89481a87fdf55b34739e041adcce`.
Controls cover replacement, in-place mutation, truncated range, wrong digest,
symlink, native direct pre/load/fence ownership, closed CLI, wrapper memory/
thread/CID bindings, low-headroom refusal, active memory/PSI/time containment
and exact owner-label cleanup with foreign-CID/daemon-error refusal.

The direct container's host memory limit is16GiB with no additional swap,
CPU quota2/native1 and240s per arm inside a600s two-arm deadline. Existing
`bench_routed_load._host_pressure` supplies host UMA/PSI observations:
launch requires40GiB MemAvailable, active floor24GiB and PSI full avg10<20.
The8GiB GPU estimate is not a hard limit: GB10 CUDA unified allocations may
escape Docker host-memory accounting. Exact owned process group and CID/label
cleanup contains only this numeric arm, never a name match or peer service.
These are CPU preparation results. Actual GPU numerics and timing remain
pending root GO.

Actual direct GPU attempt1 on frozen731f5caf exited1 after23s, candidate
not started. Original flag-off library loaded through the held FD and its
pre/load/fence SHA remained b89ba2d6; all871 artifact reads authenticated,
and held original file identities remained stable. The numerical observer
failed before forward calls because the actual production adapter is frozen.
No output/profile/performance qualification follows from this run. Raw output
is retained at `paired-k32-direct-numeric-v1/`; no bank is replaced.

The observer now requires the exact production adapter type, temporarily
observes its class `_launch` with an exact-instance guard, forwards every
original call unchanged, and restores the class method in finally before
profiling or independent reference execution. Causal CPU RED against the
ACTUAL frozen production class: PB
`3abe115fee60b96039affd13128418bb255b4383fbcfdf199ddde6e6efe7b2f4`,
6 failed/4 passed (five frozen-owner cases plus lowercase Docker absence).
Focused GREEN: PB
`1c41d0a8c793e4a6305b5ed7aaa86fe0bb50e8bb65a86f37313f84f3999dcff4`,
50 passed in4.12s, CPU2/native1/memory3GiB, zero skips/uncollected/device
allocations; canonical6114ac6e. A second production-class instance verifies
the observation guard excludes foreign owners.
