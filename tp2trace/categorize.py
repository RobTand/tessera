"""tessera#1185: categorize per-chunk key_averages tables into the four costs.

Reads one rank's hook output dir (chunk{i}.txt tables grouped by input
shape, steps.json) and prints ms per 2048-token prefill chunk for the four
costs tessera#735 names:

  mhc         mHC pre/post on the full [M, 4, 4096] stream, every rank
  kda_copy    KDA aten::copy_/fill_ on [1, 2048, 64, 128] and [1, 2048, 64, 64]
  mla_cat     MLA aten::cat of [2048, 64, 512] with zero-width [2048, 64, 0]
  mla_fill    MLA aten::masked_fill_ on [2048, 1, 64, 512]

Only Self CUDA time is summed, so parents never double-count children.
Usage: categorize.py --dir RANKDIR [--out ROWS.json]
Exit 2 on a malformed table or a missing prefill chunk.
"""

import argparse
import json
import re
import sys
from pathlib import Path

_M = 2048
_KDA_DIMS = frozenset({128, 64})
COSTS = ("mhc", "kda_copy", "mla_cat", "mla_fill")

_SHAPE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def parse_shapes(field):
    """All tensor shapes listed in a grouped table row field."""
    return [tuple(int(n) for n in m.group(1).split(","))
            for m in _SHAPE_RE.finditer(field)]


def _is_kda(shape):
    return len(shape) == 4 and shape[0] == 1 and shape[1] == _M \
        and shape[3] in _KDA_DIMS


def _is_mla_mask(shape):
    return len(shape) == 4 and shape[0] == _M and shape[1] == 1 \
        and shape[3] == 512


def _is_mhc_stream(shape):
    return len(shape) == 3 and shape[1] == 4 and shape[2] == 4096


def parse_table(text):
    """Rows of a key_averages text table: (name, shapes, self_cuda_us, calls).

    The grouped table carries a trailing Input Shapes column after
    # of Calls. The name field keeps its internal spacing: the trailing
    numeric cells plus the shapes cell are counted off from the right, so
    a grouped name with gaps never shifts a column. The Self CUDA column
    is found by its header; a CPU-only table (which lacks it) refuses by
    name instead of misreading another column. Time cells read like
    ``12.345ms``, ``678.9us`` or ``4.2s``.
    """
    lines = text.splitlines()
    try:
        head = next(i for i, l in enumerate(lines) if "Self CUDA" in l)
    except StopIteration:
        raise ValueError("table has no Self CUDA column")
    cols = [c.strip() for c in re.split(r"\s{2,}", lines[head].strip())
            if c.strip() != ""]
    if cols[0] != "Name" or "Self CUDA" not in cols \
            or "# of Calls" not in cols:
        raise ValueError(f"unexpected table header: {cols}")
    pos_cuda = cols.index("Self CUDA") - 1
    pos_calls = cols.index("# of Calls") - 1
    n_numeric = pos_calls + 1
    rows = []
    for line in lines[head + 2:]:
        if not line.strip() or set(line.strip()) <= set("- "):
            continue
        cells = [c for c in re.split(r"\s{2,}", line.strip()) if c != ""]
        if len(cells) <= n_numeric:
            continue
        shapes = ""
        if cells[-1].lstrip().startswith("["):
            shapes = cells.pop()
        numerics = cells[-n_numeric:]
        name = "  ".join(cells[:-n_numeric])
        m = re.match(r"([\d.]+)\s*(ms|us|s)$", numerics[pos_cuda])
        n = re.match(r"(\d+)$", numerics[pos_calls])
        if not m or not n:
            continue
        value, unit = float(m.group(1)), m.group(2)
        rows.append((name, shapes,
                     value * {"us": 1.0, "ms": 1e3, "s": 1e6}[unit],
                     int(n.group(1))))
    return rows


def classify(name, shapes=""):
    """Cost bucket for a grouped row, or None when it is none of the four.

    Shapes come from the row's Input Shapes cell (falling back to shapes
    embedded in the name). Head counts vary across layers (32 and 64 both
    appear), so KDA and MLA match the structural dims, not one head
    count. mHC covers both spellings the serve emits: the fused pre/post
    kernels by name, and aten ops on the [M, 4, 4096] stream by shape.
    """
    op = name.split()[0] if name.split() else ""
    seen = parse_shapes(shapes) + parse_shapes(name)
    if op in ("aten::copy_", "aten::copy", "aten::fill_",
               "aten::fill") and any(_is_kda(s) for s in seen):
        return "kda_copy"
    if op == "aten::cat" and (any(s == (2048, 64, 0) for s in seen) or (
            (2048, 64, 512) in seen and len(seen) >= 2)):
        return "mla_cat"
    if op in ("aten::masked_fill_", "aten::masked_fill") and (
            any(_is_mla_mask(s) for s in seen)):
        return "mla_fill"
    if "mhc_" in op or "hc_prenorm" in op:
        return "mhc"
    if any(_is_mhc_stream(s) for s in seen):
        return "mhc"
    return None


def chunk_ms(table_path):
    totals = {c: 0.0 for c in COSTS}
    detail = {c: [] for c in COSTS}
    seen = []
    for name, shapes, us, calls in parse_table(table_path.read_text()):
        bucket = classify(name, shapes)
        seen.append((us, name, calls, bucket))
        if bucket is None:
            continue
        totals[bucket] += us / 1e3
        detail[bucket].append(dict(op=name[:160], ms=us / 1e3, calls=calls))
    seen.sort(reverse=True)
    top = [dict(op=name[:160], ms=us / 1e3, calls=calls, bucket=bucket)
           for us, name, calls, bucket in seen[:15]]
    return totals, detail, top


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="rank hook output dir")
    ap.add_argument("--out", default=None, help="rows JSON path")
    args = ap.parse_args(argv)
    rankdir = Path(args.dir)
    steps = json.loads((rankdir / "steps.json").read_text())["steps"]
    prefill = [s for s in steps if s["tokens"] == 2048]
    if len(prefill) != 4:
        raise SystemExit(
            f"need four 2048-token chunks, steps say {[s['tokens'] for s in steps]}")
    chunks = []
    for step in prefill:
        table = rankdir / f"chunk{step['chunk']}.txt"
        totals, detail, top = chunk_ms(table)
        chunks.append(dict(chunk=step["chunk"], tokens=step["tokens"],
                           host_s=step["execute_model_host_s"], **totals,
                           detail=detail, top_rows=top))
    mean = {c: sum(ch[c] for ch in chunks) / len(chunks) for c in COSTS}
    result = dict(chunks=chunks, mean_per_chunk_ms=mean)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
