"""What does Tessera cost at SERVE time?  Three separate questions, and only
the first is free.  1) GEMM: already attested bit-identical NVFP4 bytes.
2) Load: trellis replay must run once per unit before the kernel sees anything.
3) VRAM: resident is 4.5 bpp either way.  This measures (2).

``--sweep`` measures the #44 matched triple in one process: the conforming
baseline plus the two extremes of trailing-partial fill, each a column prefix
of the same tensor.  The comparison is then a matched pair, not two runs at
different times (the LDLQ lesson on #13).  A ``cols % 256 == 1`` probe is an
odd width, and an odd width cannot pack nibble pairs -- ``decode.py`` refuses
-- so the pack line reports n/a there and the replay line is the measurement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.errors import GrammarError
from tessera.grammar import superblock_count

#: Columns per superblock on the load path.  Named once so the sweep, the
#: shape line and the release quota cannot disagree about it.
SUPERBLOCK = 256

#: Keep the old single-width behaviour, narrowed and labelled not shipping.
TRUNCATE_FLAG = "--truncate"
#: Measure the matched triple in one process instead of one width.
SWEEP_FLAG = "--sweep"


def sweep_widths(cols, superblock=SUPERBLOCK):
    """The #44 matched triple, derived from the tensor's own width.

    The full width first (the shipping shape, #40), then the largest
    conforming prefix (the baseline), then one column past a superblock
    multiple (a 1-column tail) and one column short of a multiple (a nearly
    full tail).  Entries below 1 column and duplicates fall away, so a narrow
    tensor measures fewer widths rather than a filler.  Every entry is a
    column prefix of the same tensor, which is what makes the sweep a matched
    pair.
    """
    if cols <= 0:
        raise GrammarError(f"sweep needs a positive column count, got {cols}")
    if superblock <= 0:
        raise GrammarError(f"superblock_columns must be positive: {superblock}")
    base = cols // superblock * superblock
    ordered = []
    for width in (cols, base, base - superblock + 1, base - 1):
        if width >= 1 and width not in ordered:
            ordered.append(width)
    return tuple(ordered)


def shape_note(cols, superblock=SUPERBLOCK, truncated=False):
    """The shape line printed beside every load figure.

    The block count comes through ``grammar.superblock_count``, the one
    authority (#22), so the line cannot floor what the layout ceilings.  The
    wording is unchanged from the #40 fix: a conforming width reads
    "20 whole superblocks", a partial one "the last holding 128 of 256".
    """
    blocks = superblock_count(cols, superblock)
    partial = cols % superblock
    note = (f"{cols} columns = {blocks} superblocks, the last holding "
            f"{partial} of {superblock}" if partial else
            f"{cols} columns = {blocks} whole superblocks")
    if partial and truncated:
        note += "  [--truncate: NARROWED, this is not the shipping shape]"
    return note


def pack_applies(cols):
    """Whether the NVFP4 pack half is defined at this width.

    NVFP4 packs 2 nibbles to a byte, so an odd column count cannot pack --
    ``decode.materialize_nvfp4`` refuses it.  That is the format, not a
    threshold, and every ``cols % 256 == 1`` probe is odd, so the sweep
    reports replay-only there instead of ending the matched pair early.
    """
    return cols % 2 == 0


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    truncate = TRUNCATE_FLAG in args
    sweep = SWEEP_FLAG in args
    if truncate and sweep:
        raise SystemExit(
            f"{TRUNCATE_FLAG} narrows to one width and {SWEEP_FLAG} measures "
            "the triple: they disagree about which widths to cover")

    import glob
    import time

    import torch
    from safetensors import safe_open

    from tessera.alphabet import build_forest
    from tessera.decode import decode_codes_mixed, materialize_nvfp4
    from tessera.encode import encode_unit
    from tessera.grammar import bresenham_rate_schedule, root_from_q256
    from tessera.manifest import RotationState
    from tessera.trellis import ConvCode

    dev = "cuda"
    code = ConvCode(memory=6)
    forests = {r: build_forest(r) for r in (1, 2, 3)}
    files = sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))
    W = None
    for path in files[:4]:
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k.endswith("layers.0.mlp.gate_proj.weight"):
                    W = f.get_tensor(k).to(dev).float().contiguous()
    if W is None: raise SystemExit("tensor not found")
    # A load figure is a figure for a SHAPE.  This harness used to narrow any
    # non-conforming width to a superblock multiple here, silently -- so every
    # throughput number it ever published covered only conforming shapes, and the
    # trailing partial superblock (the shape the layout fix in #22 is about) had
    # never been measured at all.  A silent truncation reads as "covered
    # everything" when it did not.  Measure the width we were given, say so, and
    # let the encoder refuse if it must (#40).
    if W.shape[1] % SUPERBLOCK and truncate:
        W = W[:, : W.shape[1] // SUPERBLOCK * SUPERBLOCK].contiguous()

    def timed(fn, n=3):
        fn(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    def report(W):
        rows, cols = W.shape
        note = shape_note(cols, truncated=truncate)
        print(f"shape: {note}", flush=True)
        rates = bresenham_rate_schedule(root_from_q256(768), cols)
        u = encode_unit(W, forests, rates, code, rotation=RotationState.NONE,
                        with_diagonals=False, released_positions=int(0.125 * W.numel()))
        t_replay = timed(lambda: decode_codes_mixed(u, forests, code))
        codes = decode_codes_mixed(u, forests, code)
        params = rows * cols
        print(f"tensor {tuple(W.shape)}  {params/1e6:.1f}M params   rows(trellis steps)={rows}")
        print(f"shape covered:   {note}")
        print(f"trellis replay   {t_replay*1e3:8.1f} ms   {params/t_replay/1e6:9.1f} M param/s")
        if pack_applies(cols):
            t_pack = timed(lambda: materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half))
            print(f"nvfp4 pack       {t_pack*1e3:8.1f} ms   {params/t_pack/1e6:9.1f} M param/s")
            tot = t_replay + t_pack
            print(f"total decode     {tot*1e3:8.1f} ms   {params/tot/1e6:9.1f} M param/s")
        else:
            print(f"nvfp4 pack            n/a   {cols} columns cannot pack 2 nibbles to a byte")
            print(f"total decode (replay only) {t_replay*1e3:8.1f} ms   {params/t_replay/1e6:9.1f} M param/s")
            tot = t_replay
        for name, n in [("GLM-5.3-Flash body ~355B", 355e9), ("Qwen3.8-27B ~27B", 27e9)]:
            print(f"  extrapolated to {name}: {n/(params/tot)/60:8.1f} min of load-time decode")

    if sweep:
        for width in sweep_widths(W.shape[1]):
            report(W[:, :width].contiguous())
    else:
        report(W)


if __name__ == "__main__":
    main()
