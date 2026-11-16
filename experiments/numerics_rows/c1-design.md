# Routed C1 work and exact arithmetic

This note is an implementation plan. It changes no pass rule. No speed-up is claimed before a profile and data comparison complete.

The signed A8S routed population contains 42 modules, each with 288 experts. TP2 needs both ranks and three roles. The campaign therefore has 72,576 logical expert-role-cuts. The item list keeps all 1,728 items of one module together before the next module. There are 252 module-role-rank passes if one process covers all 288 experts. These are two schedules for the same required case data.

## Baseline profile

Use IMG-N on a Spark, one core, memory=4 GiB, one native thread, Flux limit 30 minutes. Profile the existing Fraction path for L10, expert 0, gate_proj, TP2 rank 0. It includes peer decoding and the full M enumeration. It changes no scale population, input, rounding or search. The gate role is first; the profiler stops after its scale-index result and does not enter a second role. It produces no row-ready manifest.

A process timer at 1,500 seconds interrupts Python work and dumps cProfile data in a finally block. The outer container timeout is 1,740 seconds. A cap is recorded as CAPPED with generation_finished=false. Other exceptions remain failures. The output contains phase times for wire reads, decoding, separation, chunks, scale grouping and each make_plan call. It also contains evaluator call counts and time, each plan's N and number of completed reads, peak RSS, and IMG-N package availability. A grouping iterator timer includes the consumer's bitset construction. It does not materialize another array. Netdata records load and I/O on the participating hosts.

The profile tests these source-based hypotheses:

1. Grouping copies large integers while constructing masks over all experts. A TP2 gate/up row-scale array has N = 2 * 288 * 1024 = 589,824. The cost should appear in scale_grouping and joint_coverage, before expert-specific reads finish.
2. Fraction arithmetic dominates the per-read scale-value Cartesian product. evaluate and Fraction calls should then account for most of the remaining time.
3. The wrong-index scan dominates after grouping and evaluation. joint_coverage's own time should then remain large after subtracting grouping and evaluator time.

The known old sample's saved reader only locates the delay after separation and chunks. It does not measure these hypotheses. If the cap leaves a read or role incomplete, report that limitation and a lower bound, not a completed-item rate. Completed read counts permit an explicitly labelled extrapolation of the active plan, separate from measured totals.

## Module as work unit

Decode one expert and its peer at a time. Do not load all decoded expert tensors at once. Keep every expert's separation, chunk and scale-read results. Gate owns the joint gate/up scale plans as before. Up still owns its own separation and chunks. Expert 0 still owns the shared activation plans.

Build scale groups once for each exact coding descriptor, including N, base, low, digit, family, held ranges and held constants. Reuse them across experts only when all those fields match. Do not shrink the wrong-index domain to the current expert. Reuse evaluated results only for identical represented operands, coding values, rounding stages and measured D-table identity. Different expert weights still need their own output checks. Cache size must stay within the measured 4 GiB envelope; stream or discard entries when necessary.

The generator emits every logical item with a unique job name and also the 252 pass scopes. A feeder may run eight single-core slots per Spark, sixteen total. It finishes one complete module first. It starts no new CPU work when GPU rows need those cores. The generator submits nothing. Grouped-pass execution and manifest assembly stay disabled until exactness and memory checks pass.

For a measured item time t in seconds, serial work is 72,576*t seconds, and ideal wall time at sixteen slots is 1.26*t hours. For a measured module-role-rank pass time p, ideal wall time is 252*p/(16*3600) hours. Report one complete module's measured seconds after the fix; the profile's cap alone cannot supply that number. Include startup, decode, case writes and manifest serialization in that measurement.

## Exact arithmetic alternatives

Try module-level reuse before replacing arithmetic. If Fraction dominates, compare these two opt-in backends on the same work and data.

- GMP rationals. Check whether gmpy2 is already present in IMG-N. Use gmpy2.mpq with the same operation order and rounding functions if its integer and rational operations provide the required semantics. python-flint fmpq is an alternative if GMP is unavailable. If installation is needed, stage a pinned aarch64 cp312 wheel under the job inputs, record version and SHA-256, mount it read-only and refuse a hash mismatch. Install offline into job scratch. No network is used in IMG-N.
- Dyadic integers. Decode each finite represented float as a sign, nonnegative integer significand and power-of-two exponent. Keep zero's sign explicitly. Multiply significands and add exponents. At every existing FP32 or BF16 boundary, choose the normal or subnormal quantum from bit lengths, divide by that power of two with divmod, and increment on greater-than-half or a half with an odd quotient. Keep gradual underflow, signed zero and overflow. No intermediate is rounded through binary64. Use the unchanged measured D(gf) table after the same BF16 conversions and clamps. No exp or sigmoid approximation is permitted.

Both backends keep evaluate(case, wrong, values, table) and make_plan's result schema. The existing Fraction backend remains the default. Search order, the least collision tuple, coding range, held constants, input cases, separation and chunks stay fixed. Independent expected-bit tests cover even and odd ties, both signs, signed zero, subnormals, the subnormal/normal boundary, BF16 conversion and overflow. A difference stops the change and goes to the lead before the freeze.

## Acceptance

Compare all 14 dense modules and all 72 role-cuts against the retained cea02cb1/83b6c662 manifests. Verify their signed population and Q1 bindings first. Compare canonical serialized case data byte for byte, including separation recipes and masks, chunks, scale plans, read cases, reads covered and uncovered aliases. Exclude only job paths and measured timing, never numeric case data. Require every module, cut and declared role exactly once and generation_finished=true. Missing or unfinished references fail the comparison; do not silently omit them.

Also compare the module-pass path against one complete routed module generated by the old per-item path. Until that exists, compare a fixed sample through both paths: experts 0, 144 and 287; all three roles; both TP2 ranks. Expert 0 includes every activation plan. The sample is an arithmetic and scheduling check, not a full-module receipt. Only byte-equal data permits a backend or schedule to replace the old path. Compare performance with the same instrument and load evidence after correctness passes.

## Measured baseline and next experiment

The first profile, f2tKNb1eMNf at 618b4c19, ended at its 1,500-second cap with tool and container exits 0. Netdata collection reported no errors. make_plan spent 1,472.9 seconds on the row-scale array with N=589,824. Its final plans record counts 68 completed reads of 1,024; the phase record's zero was an entry-time snapshot, now corrected. Six groupings took about 14 seconds in total. joint_coverage used about 461 seconds in its own loop; 4,157,105 evaluate calls used about 973 seconds. No complete-item time was measured. Neither gmpy2 nor python-flint was present.

The first change is an opt-in bounded LRU around the unchanged Fraction evaluator. A cache instance owns one D table. Keys contain the epilogue selection, arity and binary64 bytes of every used numeric operand and selected scale value. Signed zeros stay distinct. Index and column labels do not enter the numeric calculation. Per-signature bitsets count distinct value tuples exactly even when the LRU evicts entries. Arithmetic, search order and the full wrong-index domain are unchanged. The next single-core job measures this same array with the same profiler and records hit/miss and distinct-tuple counts. Fraction without memoization stays the default while byte equality is checked. The module-pass restructuring follows this measurement; GMP and dyadic arithmetic wait.
