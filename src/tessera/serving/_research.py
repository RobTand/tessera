"""tessera#508 research instrumentation for the GLM53 NoPE backend.

Both knobs are read once at import and are unset in every production path:

* ``TESSERA_RESEARCH_GLM53_NOPE_SYNC=1`` turns Tessera-owned sites into device
  sync points. An asynchronous CUDA fault surfaces at the next sync, so a fault
  raised at a site happened between the previous sync and that site. Skipped
  while a stream is capturing (a sync is illegal in capture).
* ``TESSERA_RESEARCH_GLM53_NOPE_DUMP=<path>`` appends one JSON line per call
  with a digest of every tensor handed to :func:`dump`, so two repeats of one
  request can be diffed call by call (which input or output first differs).
  Each digest is a device sync; research only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

SYNC = bool(os.environ.get("TESSERA_RESEARCH_GLM53_NOPE_SYNC"))
DUMP = os.environ.get("TESSERA_RESEARCH_GLM53_NOPE_DUMP") or None
SLOTMAP_CHECK = bool(os.environ.get("TESSERA_RESEARCH_GLM53_NOPE_SLOTMAP_CHECK"))


def sync(site: str) -> None:
    if not SYNC:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 -- name the site, then re-raise
        raise RuntimeError(f"#508 sync site {site}: {exc}") from exc


def _digest(t: torch.Tensor) -> str:
    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def dump(site: str, **fields) -> None:
    if DUMP is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    rec: dict = {"t": time.time(), "site": site}
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            rec[key] = {"shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                        "sha": _digest(value)}
        else:
            rec[key] = value
    with open(DUMP, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def install_slot_mapping_check() -> None:
    """#508 bisect: validate the stock slot-mapping kernel's inputs before launch.

    ``TESSERA_RESEARCH_GLM53_NOPE_SLOTMAP_CHECK=1`` wraps
    ``vllm.v1.worker.gpu.block_table.BlockTables.compute_slot_mappings`` (the
    launch CUDA_LAUNCH_BLOCKING=1 named for the chunk-2 illegal memory access):
    a device sync first surfaces any earlier asynchronous fault at this site,
    the kernel's inputs are then checked on the host (idx_mapping range,
    query_start_loc monotone and within positions, positions within every
    group's block-table row, block_table_ptrs still equal to the live
    tensors), one summary line is logged per call, and a fault raised by the
    kernel itself is re-raised naming this site with the checked inputs.
    Research only; it patches a stock class and is never installed unless the
    variable is set.
    """
    import sys
    from vllm.v1.worker.gpu.block_table import BlockTables

    orig = BlockTables.compute_slot_mappings

    def checked(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None):
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"#508 slotmap check: fault pending BEFORE compute_slot_mappings: {exc}") from exc
        num_reqs = int(idx_mapping.shape[0])
        idx = idx_mapping.cpu().tolist()
        qsl = query_start_loc.cpu().tolist()
        n_actual = qsl[num_reqs] if num_reqs < len(qsl) else None
        problems = []
        if any(i < 0 or i >= self.max_num_reqs for i in idx):
            problems.append(f"idx_mapping out of [0,{self.max_num_reqs}): {idx}")
        if any(b < a for a, b in zip(qsl, qsl[1:])) or (qsl and qsl[-1] > positions.numel()):
            problems.append(f"query_start_loc not monotone or beyond positions ({positions.numel()}): {qsl}")
        pos_summary = None
        if n_actual is not None and 0 < n_actual <= positions.numel():
            pos = positions[:n_actual].cpu()
            pos_summary = (int(pos.min()), int(pos.max()))
            for g in range(self.num_kv_cache_groups):
                row_cap = int(self.block_tables[g].gpu.shape[1]); kbs = int(self.kernel_block_sizes[g])
                if pos_summary[0] < 0 or pos_summary[1] // kbs >= row_cap:
                    problems.append(f"positions {pos_summary} exceed group {g} block-table row (cap {row_cap} x kernel block {kbs})")
        live = [b.gpu.data_ptr() for b in self.block_tables]
        recorded = self.block_table_ptrs.cpu().tolist()
        if live != recorded:
            problems.append(f"block_table_ptrs stale: recorded {recorded} live {live}")
        summary = (f"#508 slotmap check: num_reqs={num_reqs} idx={idx} qsl={qsl[:num_reqs + 2]} "
                   f"positions[min,max]={pos_summary} num_tokens_padded={num_tokens_padded} "
                   f"slot_mappings={tuple(self.slot_mappings.shape)}")
        print(summary, file=sys.stderr, flush=True)
        if problems:
            raise RuntimeError("#508 slotmap check: BAD INPUTS: " + "; ".join(problems) + " | " + summary)
        result = orig(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out)
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"#508 slotmap check: kernel faulted with VALID inputs: {exc} | {summary}") from exc
        return result

    BlockTables.compute_slot_mappings = checked
    print("#508 slotmap check installed", file=sys.stderr, flush=True)
