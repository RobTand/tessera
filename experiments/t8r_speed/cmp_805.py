"""Compare two repro_805.py runs cell by cell: the output hash of every shared cell must be equal.

usage: cmp_805.py A.jsonl B.jsonl.  A cell is (library, rows, cols, m, s, grid).  The verdict also
requires every cell of both arms to be free of NaN and of repeat mismatches, the two arms to hold
the same cells (so the same split at every (library, shape, m)), and each arm's launch at its own
split and the SM count to equal its public path.  Exit 0 only when all of that holds.
"""
import json
import sys


def load(path):
    cells, meta = {}, None
    for line in open(path):
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
            continue
        cells[(d["library"], d["rows"], d["cols"], d["m"], d["s"], d["grid"])] = d
    return meta, cells


def main():
    (ma, a), (mb, b) = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(set(a) & set(b))
    only = {"a": sorted(set(a) - set(b)), "b": sorted(set(b) - set(a))}
    differ = [k for k in shared if a[k]["sha256"] != b[k]["sha256"]]
    dirty = [(arm, k) for arm, cells in (("a", a), ("b", b)) for k, d in cells.items()
             if d["nan_launches"] or d["mismatch_launches"] or d.get("equals_public_api") is False]
    verdict = bool(shared) and not differ and not only["a"] and not only["b"] and not dirty
    print(json.dumps({"a": ma, "b": mb, "shared_cells": len(shared), "differ": differ[:50], "n_differ": len(differ),
                      "only_a": only["a"][:50], "only_b": only["b"][:50], "dirty": dirty[:50],
                      "bitwise_identical": verdict}, indent=1, default=str))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
