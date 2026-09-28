"""tessera#508: the stock slot-mapping kernel's kpool-tail read, reproduced.

Runs inside the pinned vLLM image, on a GPU, ideally under the CUDA memcheck
tool with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` so every tensor is its own
device allocation and the tool can see a read past the block table (with the
caching allocator the 1 KiB table sits inside a 2 MiB segment, and a read a
few KiB past it is invisible unless it also leaves the segment).

The geometry is the 4-layer GLM-5.3-Flash stub's, from the serve's own
host-side check record (``fdo_chk2``, tessera#508 / PR #581): three KV-cache
groups, ``max_num_reqs`` 8, ``max_num_batched_tokens`` 2048; group 1 is the
kpool tail (``KpoolTailSpec``: block 4, kernel block 4, a 32-entry row that
``get_block_table_width(1, 4)`` pads to). Groups 0 and 2 stand in for the
MLA and the linear-attention state groups with rows that the positions used
here stay inside, so the only candidate out-of-bounds read is group 1's.

The chain is the V2 model runner's: ``BlockTables.gather_block_tables`` and
``compute_slot_mappings`` (``model_runner.py`` ``prepare_attn``), then the
kpool-tail metadata builder's ``compute_kpool_tail_slot_mapping``, which is
what the tail cache is actually written with.

  --tail-slot-mapping enabled   the pinned image's rule (model_runner.py:569
                                enables the generic mapping for every spec
                                that is not a CircularBufferSpec)
  --tail-slot-mapping disabled  upstream vLLM #57317 (70df48dc3d01): the tail
                                spec opts out of the generic mapping

It prints one JSON record: the tail slots actually written (after the
builder), the generic group-1 row, and whether both equal the expected
circular mapping ``own_block * 4 + position % 4``.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu.block_table import BlockTables

KPOOL = 4
MAX_NUM_REQS = 8
MAX_NUM_BATCHED_TOKENS = 2048
MAX_MODEL_LEN = 4096
# (block_size, kernel_block_size, KV blocks per request before alignment)
GROUPS = {
    0: (8704, 64, 1),     # MLA latent pages (attention block 8704, kernel 64)
    1: (KPOOL, KPOOL, 1),  # KpoolTailSpec: one 4-token circular block per request
    2: (8704, 8704, 1),   # linear-attention (KDA) state, one block per request
}
OWN_BLOCK = {0: [5], 1: [7], 2: [3]}
# Chunked prefill of the 3649-token prompt at max_num_batched_tokens 2048, then
# one decode step; the same steps the fdo_chk2 record logged.
STEPS = {"chunk1": (0, 2048), "chunk2": (2048, 3649), "decode": (3649, 3650)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-slot-mapping", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--steps", default="chunk1,chunk2,decode")
    args = parser.parse_args()
    device = torch.device("cuda")
    enabled = [True, args.tail_slot_mapping == "enabled", True]

    widths = []
    for g in range(3):
        block, kernel, blocks = GROUPS[g]
        align = None if g == 2 else 128  # model_runner.py: state groups skip token alignment
        widths.append(get_block_table_width(blocks, block, token_alignment=align))
    tables = BlockTables(
        block_sizes=[GROUPS[g][0] for g in range(3)],
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_blocks_per_group=widths,
        device=device,
        kernel_block_sizes=[GROUPS[g][1] for g in range(3)],
        slot_mapping_enabled=enabled,
    )
    tables.append_block_ids(0, tuple(OWN_BLOCK[g] for g in range(3)), overwrite=True)
    tables.apply_staged_writes()
    torch.cuda.synchronize()
    tail_table = tables.block_tables[1].gpu
    record = {
        "tail_slot_mapping": args.tail_slot_mapping,
        "slot_mapping_enabled": enabled,
        "tail_table_shape": list(tail_table.shape),
        "tail_table_bytes": tail_table.numel() * tail_table.element_size(),
        "tail_table_ptr": hex(tail_table.data_ptr()),
        "steps": {},
    }
    # Printed before any launch so a run the memcheck tool stops still names
    # the allocation its out-of-bounds addresses are measured from.
    print(json.dumps({"geometry": {
        "tail_table_ptr": record["tail_table_ptr"],
        "tail_table_end": hex(tail_table.data_ptr() + record["tail_table_bytes"]),
        "tail_table_bytes": record["tail_table_bytes"],
        "tail_row_entries": int(tail_table.shape[1]),
        "block_table_ptrs": [hex(b.gpu.data_ptr()) for b in tables.block_tables],
        "block_table_bytes": [b.gpu.numel() * b.gpu.element_size() for b in tables.block_tables],
    }}), file=sys.stderr, flush=True)
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    for name in args.steps.split(","):
        start, end = STEPS[name]
        n = end - start
        positions = torch.arange(start, end, dtype=torch.int64, device=device)
        query_start_loc = torch.tensor([0, n], dtype=torch.int32, device=device)
        inputs = tables.gather_block_tables(idx_mapping, num_reqs_padded=1)
        slot = tables.compute_slot_mappings(idx_mapping, query_start_loc, positions,
                                            num_tokens_padded=n)
        torch.cuda.synchronize()
        generic_tail = slot[1, :n].clone()
        written = compute_kpool_tail_slot_mapping(
            slot[1, :n], inputs[1], query_start_loc, positions, n, 1, KPOOL)
        torch.cuda.synchronize()
        expected = OWN_BLOCK[1][0] * KPOOL + positions % KPOOL
        max_index = (end - 1) // GROUPS[1][1]
        record["steps"][name] = {
            "positions": [start, end - 1],
            "generic_max_column": max_index,
            "generic_flat_index_max": max_index,  # request row 0
            "reads_past_row": max_index >= tail_table.shape[1],
            "reads_past_table": max_index >= tail_table.numel(),
            "generic_tail_all_pad": bool((generic_tail == PAD_SLOT_ID).all()),
            "generic_tail_equals_expected": bool((generic_tail == expected).all()),
            "written_tail_equals_expected": bool((written == expected).all()),
            "written_tail_head": written[:6].tolist(),
            "mla_slots_head": slot[0, :4].tolist(),
        }
    print(json.dumps(record, indent=1), flush=True)
    ok = all(s["written_tail_equals_expected"] for s in record["steps"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
