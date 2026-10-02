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
