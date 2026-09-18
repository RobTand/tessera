# tessera#550 traced TP2/RoCE reproduction harness

One two-Spark TP2 vLLM serve of the 4-layer GLM stub over NCCL/RoCE (the `f0`
pair link), with bpftrace on both boxes recording the kernel side of every
`ibv_reg_mr`: `ib_uverbs_reg_mr` -> `mlx5_ib_reg_user_mr` -> `ib_umem_get`
(`pin_user_pages_fast`, long-term pin migration) -> `dma_map_sgtable` /
`iommu_dma_map_sg` / `alloc_iova` -> mlx5 mkey cache / `mlx5_core_create_mkey` /
UMR, plus every syscall that returns ENOMEM, the mlx5 debugfs `CREATE_MKEY`
failure counters before and after, the ranks' `/proc/PID/maps` at READY, a
memory monitor and the kernel journal for the window.

vLLM serves are exempt from PrismaBuild; the tracer needs `sudo -n` on both boxes.

    cd experiments/t550_reg_mr_trace
    TRACE=1 COLD=0 LOAD= GPU_UTIL=0.25 bash trial.sh t1          # plain serve
    LOAD=nfsread,compact LOAD_SECS=240 bash trial.sh t2          # under NFS-RDMA read + compaction load
    SERVE_EXTRA="--quantization fp8" MOE_TRITON=0 bash trial.sh t3  # the 2026-09-15 FP8 arm's shape

Outputs land in `out/<TAG>/` (`config.txt`, `engine-args.txt`, `rank{0,1}.log`,
`trace-{sparky,sparklina}.txt`, `journal-*.txt`, `mem-*.txt`, `maps-*.txt`) and one
line per trial in `out/summary.txt`. Any ERR line in a trace names the kernel
function that returned the error, its arguments and the calling stack.

Files: `trial.sh` (driver), `trace.bt` (38 probes), `load.sh` (NFS read /
compaction load), `memmon.sh` (2 s memory sampler), `evict.sh` (drop the stub's
page cache), `t550snap.sh` (dump `/proc/PID/{maps,smaps,status,limits}` when a
registration fails; install at `/home/rob/tmp/t550snap.sh` on both boxes).
