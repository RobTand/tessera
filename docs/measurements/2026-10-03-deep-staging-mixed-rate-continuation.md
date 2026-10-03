# Deep word staging and mixed-rate continuation

Refs #807 / PR #809, #793 / PR #794, parent performance gap #750. This is recovered evidence and a blocked future protocol, not a new GPU measurement or default/pin change. Original branches, source snapshots, native banks and protocols remain preserved. Evidence root `B=/mnt/shared/tessera-measurements/t8r-speed-20260929/l512-20260930`.

## #807: completed go/no-go is negative, not still queued

Frozen PR809 source `a873e49155b35bb9c0f611ff0282bc0901f918d0` stacks on D1 PR794 `03262cf3c7814ef796b123d80dbeb1b82b8f446a`. Env unset/3 leaves three stages; requested5/8 changes only single-run E4M3-instruction launches, with stage count clamped to shared capacity. Two-run launches retain the original descriptor-ring issue distance. No native arithmetic or wire change was intended; numerical equivalence is evidence below, not a substitute for it.

PB `0fc0bb3bb66a6fbfffe29064f3c0e6de288a78d7a1c375d891ee9a65c1ac1164` completed rc0 on Sparky. Receipt `cb77112d4be0fc73ed7591a86c092fad2427becebc28bb8da32a345c13b6c74b`; result `87484a2d309e5b3004f25327fcf9dee0b3850170fb8fa1703479d234b25bc794`; local claim `90cce87b80e5374de39894b7b2f4aa40e81b0e98675f1ae350d627541b631e71` was checked for payload SHA/bytes, receipt binding and self-consistency. MCP does not independently verify the full worker attestation.

`B/abl-lut-20261001T111516Z/rowC-ws/ab_summary.json`: A8S real TP2rank0,288 experts,top-k8, experts.R1024.L10; balanced M1/512/2048 plus eight recorded routes at each prefill M; forward/reverse19 cells. All19 cells have every-arm equality and no missing case. These are one R4 group, not mixed-rate/all-rung qualification. Ratios are candidate/reference kernel-profile times, not full-model throughput:

| Balanced M | ws5 forward / reverse | ws8 forward / reverse |
| ---: | ---: | ---: |
| 1 |1.0704 /0.9528|1.0434 /0.9971|
| 512 |1.0032 /0.9943|1.0081 /1.0040|
| 2048 |0.9965 /1.0089|1.0112 /1.0224|

Predeclared go criterion: approximately2% or better at M2048 in **both** passes. Neither candidate meets it on balanced M2048; none of the eight recorded M2048 cells meets it in both passes either. Recommendation: root explicitly records #807 as measured negative/not planned and leaves PR809 unpromoted/draft (or closes unmerged). This does not close #750's larger performance gap. Do not spend GPU work repeating this unchanged hypothesis.

CPU compile gate `312f56a0` was failed, not an overall passing row: one overly strict D1-versus-master SASS expectation refused32 single-rate kernels differing only by exact integer operand order. Its retained outputs show flag-off PR versus D1 equivalence and unchanged two-run/non-MMA8 instantiations; no spill claim is scoped to those compiled variants. Gate2 `15fc180b2adfb2e3d04af2ab0139ed0b62576fbfd59fb36585e24b05b6a2bc5a` completed rc0; receipt `41e19ce876e11b5a7b467b8fc4fd0b574408ffb0f0f4e2ae92861be7e7add4d2`, result `ba681e5172f40e6de8c081c3f39d979539d44485305071cfa98e3109309967b5`: actual measured prototype versus PR at ws5 and ws8 is137/137 SASS-equivalent, no missing kernels, no spills. Prototype source SHAs are ws5 `64bc6fa9aa742200c06f214f927078545169974971f5f5f0db6882f5309efb98`, ws8 `67d2056ef473f3bec6e082a281a615b6256bcc1c46106477810920ffd1e58d78`; D1 source `12fbf19e406f99775542eca21306688be27e364474cf398384d1ce168aed22ab`.

Further already-completed source qualification: PB `d57a6b9eac0872aed54fe1ce83092e1ea24f3fa6ff82f41bf8b563b558ff5b96` rc0, receipt `3ac18359a199e59fb44a8ae174174b6e0ca5f8a6a008adbaae752c80db2b3f96`, result `b8c73c12d82a5efc8584ca27cdf0a175cb85a82123a344da26fd1231badff08f`; rowD3b-routed records115 all-arm bitwise/no missing. PB `50fe2f426b1521d4bc2456e1ed185bde66f841f72a7ad94617ea5d670baf613f` rc0, receipt `e082182aa6ae148fef74ef3cf49f287384582403b24da7fa7b46569fc3615160`, result `93d665b8eca1e179f1026529d64c3114d502cd108319c03deacf71869ddd81cf`; rowD3b-dense records98 all-arm bitwise/no missing. These A8S source comparisons do not prove D1's two-rate release acceptance. No716-cell or served qualification is inferred from these counts.

## Recovered after-NCU counters, without another GPU run

PB `9bc86f9253219647e745c5909cf9c8c7295826e380121c6b39b41f4a050098c2` completed rc0; receipt `3c6dcb13520a33ad01c2e96405c70a2a80b6e4499b6ad0b4de269a1f01902c73`, result `417aa6e89688ec329290ce091273fc401724a9995a14ca5fc1da47019fea68ac`. Its apparent ncu.csv files contain only three profiler status lines; the real reports are `rowC-ncu/{fix,ws5,ws8}-ncu/t8r.ncu-rep`.

CPU-only PB `4584ab6bd5f497d9eb20a4998b19c38938c9c22d1e3fa32dbfaff009ba8f8da3` imported those existing reports, rc0, CPU1/native1/memory2GiB, CUDA hidden. Sparky placement was a real dependency: only its installed aarch64 NCU tool was available; DL had no /opt/nvidia SDK. Receipt `6d18327ca737e80d0fda1a8c55623e91d4dc87b744678abdcf907001d5fe1e23`, result `0d627bb466609e35c29f8c1b952d5543dee802867b71bd6e03260e60160b231f`. Input report SHAs fix `2405cb96a7b46cfec5aa40bf430cc0514a1a7695562f6d57ac42af9285f1d4a8`, ws5 `310ef37c92d2ac42e0f3d771592380bc10ce45a305cedcb4658061c0b14ef1fc`, ws8 `4bff1c523705bc30781d1e0e980ce2a3cb4b0dae5c954e3b556202dcaddd3d71`. Raw exports remain `/mnt/shared/astra-resume-20261002/t8_performance/deep-word-ncu-cpu-readout/`.

DL CPU1/memory1GiB/native1 PB `025a4809f4ee2b1a3cfc14bae8a8ca353b65e3ded86dcf8a66cb426e1b73be2d` extracted counters, rc0; receipt `de49c78caa82b6187f64998b80ab3855a18f48cc48fc02d9edf33837a477d6c2`, result `4a0f2b64a28ee64dc835b06a1bcf7595721c846faeb9217d75e008a0c7491053`. No tests/builds/new GPU profiles ran.

| Native M2048 BMT128 | three stages | five stages | eight stages |
| --- | ---: | ---: | ---: |
| MODE0 duration ms |7.977152|7.973536|8.036160|
| MODE2 duration ms |4.553888|4.610240|4.670336|
| MODE0 L1TEX shared-memory LSU wavefront throughput % peak elapsed |57.170900|57.182498|56.645574|
| MODE0 barrier stalls/issue-active ratio |5.392552|5.317478|5.593714|
| MODE0 long-scoreboard stalls/issue-active ratio |0.465702|0.462248|0.497495|
| MODE0 MIO throttle stalls/issue-active ratio |1.361138|1.299204|1.447979|

These single NCU launches locate continuing barrier/shared-pipe pressure; they do not identify which barrier owns all stall samples or price a realizable speedup. Deeper words do not remove the M2048 MODE0 term. Do not substitute the current PM R4 profiles for this source/shape. T4/T16 owners were informed of the negative scope; their default-off activation-prefetch mechanisms are different and receive no benefit claim from this experiment.
The exact throughput counter is
`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed`;
it is not an SM shared-cycle utilization measurement. Independent review
`INDEPENDENT-DEEP-STAGING-REVIEW.json`, SHA256
`762b9889db1a1213a32934081f4a03cc730f97b3d9175ad096a280cd6cdb23ee`,
PB `6b6e81bd`, authenticated the raw stop proof, source/SASS bridge and five
CAS records and accepted measured-negative/not-planned #807 disposition.
Root authorized closing #807 not planned and PR809 unmerged; #750 remains
open and #793 stays active. No fixed/speed/served/all-rung claim follows.

## #793 / PR794: source remedy exists, mixed proof does not

`STAGE_PREV = PREV_STAGED && !TWO` returns two-run history to the pre-#763 register path at five source sites while retaining the768B layout region, launch sizes and bank mapping. Single-run history staging is retained. This is a credible source remedy, not measured restoration. The current frozen `/mnt/shared/astra-793-intake-20261002/repo` contains `PREV_STAGED = FAMILY_MMA8` and no STAGE_PREV guard: it cannot inherit PR794's remedy by provenance prose.

Live PB named-row inspection confirms bce207ea,ebbc446e,91af0839,8cc4ebc5,97e2192f,6fd8cb0a all withdrawn with zero attempts. The preserved d1-bitwise and ab-stageprev roots contain source/native preparation, not terminal A/B summaries. Broad snapshot-parent census queries were truncated/timed out, so they are not evidence of global absence. No successful rekeyed mixed receipt was recovered from the named records, submission log, progress record or prepared roots. Keep #793 open as a mixed-rate release blocker; PR794 remains draft, no pin flip.

Historical master/fix E4M3 ELF hashes under `B/ab-stageprev-20261001T110925Z/ext-{master,fix}/tessera_routed_fused_mma_e4m3_sm_121_tessera_guarded_v1/` are `f43825996a8d32bc7598d80bbe095e536930719f9f5531138c3b96bdc7a7f5c3` / `e89af5412e4429bebd878d587ab8a4c3cf7f1287a34c46797b8ed48a246179c9`. Master/fix source SHAs `3a32d040668cc1fe5678c2088deb5afa8cd6f227a7920dfbd899c90863c03254` / `12fbf19e406f99775542eca21306688be27e364474cf398384d1ce168aed22ab`. These old banks predate separately qualified #855; do not launch them as current shipping proof without a root-frozen common terminal-safe source/native comparison.

### Smallest outstanding mixed decision protocol (BLOCKED, no submission)

Goal: qualify history-only source remedy, not retry deep words. Use the existing `ab_arms.sh` bench owner, exact-source native owner and matched held-input/residency transport; no new dispatcher or loader. Both arms must share the accepted #855 fix and current layout/ABI; vary only the five history-staging sites. Retain WS3 (unset default), legacy layout, image `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`, TP2rank0/top-k8/288 experts and original `pact-e4m3-accuracy-20260928/release-t8/exported` wire/routing identity. Bind both source/library SHA and actual loaded FD/hash/mtime before and after; refuse rebuild or readset mismatch. Current R4-only paired/PM output guards cannot substitute for mixed numeric qualification.

1. Numeric-only decision quantum: original routed R1024.L10 control, R1088.L11 (4+5) and R832.L42 (3+4), M1/2048 plus the identical recorded routing from the original #763 comparison; hash gate/up, down-route and reduced output, repeat per arm and run reverse. Dense/shared two-run groups must be enumerated from the original release manifest, not inferred from R4. This small quantum answers the initial mixed remedy question, not all-rung acceptance.
2. Only after numeric acceptance, exclusive timing quantum on those same cases, forward/reverse with per-event samples and separate exact MODE0/2 profiles. Preserve matched source, original geometry and residency, collect before/after native counters plus both-Spark timestamped CPU/power/clocks/temperature/memory/residency. Predeclare thermal/CPU drift handling; reject contaminated comparison rather than post-hoc dropping an arm. Original criterion: R4 within±2% both passes; R832/R1088 M2048 restore pre-#763 timing (roughly0.92–0.94 of staged master); decode/two-run dense restoration separately checked, not inferred.
3. Before release acceptance, original full mixed and single-rate coverage plus actual current allocation rungs across T4/T8/T16, boundaryM2049, decode M1–8, recorded routing, all role/adjacent-rate combinations required by the final allocation. Partition through PB; do not manually distribute hosts or treat uniform-only performance as all-rung proof. Served TR3 KL/decode identity/full-model throughput remains root-owned.

Budget for each initial quantum: published PB, `--exclusive`, GPU1, CPU2/native1, total UMA16GiB/GPU subset8GiB (raise only for a demonstrated resident population), finite600s owner deadline; immutable image dependency, preserved PB CPU affinity/container scope, current public input/helper generation. Reuse existing exact CID/token containment. No direct benchmark exemption. Inputs/readset/output namespace and terminal-safe arm source/native hashes must be frozen by the existing owner before a command is runnable. No root GPU GO exists.

Known acceptance-tool blockers: frozen `B/analysis/ab_stageprev_accept.py` formats actual string M keys with `:d` (line42); even if fixed it treats every non-R1024 label as mixed and accepts an empty union without a declared expected population. `ab_arms.sh` also derives observed keys from a union, so a case absent from every arm is invisible. Do not use those scripts' exit0 as complete coverage. The immutable old analysis bank is not edited here; the future protocol requires an explicit expected case/role/rate manifest and rejects empty/missing/duplicate cases. Fix that at the current owned acceptance seam with CPU RED/GREEN before root authorizes mixed measurement, rather than adding a second gate over a uniform subset.

## Root decisions

- #807/809: measured-negative/not-planned disposition warranted; no repeated unchanged D3 work and no promotion.
- #793/794: retain P1 mixed-rate blocker until terminal-safe original matched numerical/timing proof lands; source intent alone does not restore a release pin.
- Existing evidence is frozen and recoverable. The outstanding mixed protocol is blocked on exact terminal-safe comparison and fail-closed population acceptance ownership, not on a requested repeat of completed R4 work. #750 remains open.
