"""Write tessera#805's mutant: one source snapshot's kernel with the claim-to-read window widened.

usage: mutate_805.py <src-tree> [spin_cycles]

This patches ``<src-tree>/tessera/serving/csrc/routed_fused_window.cu`` in
place. The tree must be a snapshot, never a checkout. The patch adds a spin
of ``spin_cycles`` SM clocks (default 131072) in ``routed_fused_kernel``'s
consumers, between the first chunk's ``BAR_FULL`` sync and the descriptor
read, and only on the dense split launch (``DENSE && SPLIT``).

The spin gives the producers time to claim two items ahead. Where an item
and the next hold three chunks or more between them, the producers' per-chunk
``BAR_EMPTY`` wait still orders their slot write after this read, so the
output bits are unchanged. Where both hold one chunk, the read sees item
``j + 2``'s descriptor. The mutant must therefore corrupt at ``S = nk`` and
stay bitwise equal to master at ``S <= nk / 2``. That is how the repro shows
its check can see the race.
"""

import hashlib
import sys

ANCHOR = ("            int stage = gc & 1;\n"
          "            bar_sync(BAR_FULL0 + stage, THREADS);\n"
          "            const int e = desc[slot * 8 + 0];\n")
KERNEL = "routed_fused_kernel(const Params p) {"
FP4 = "namespace fp4 {"


def main():
    tree = sys.argv[1]
    spin = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    path = f"{tree}/tessera/serving/csrc/routed_fused_window.cu"
    text = open(path).read()
    start, fp4 = text.index(KERNEL), text.index(FP4)
    at = text.index(ANCHOR, start)
    if not start < at < fp4:
        raise SystemExit(f"anchor at {at} is outside routed_fused_kernel [{start}, {fp4})")
    if text.count(ANCHOR, start, fp4) != 1:
        raise SystemExit("the anchor is not unique inside routed_fused_kernel")
    sync_end = at + ANCHOR.index("            const int e")
    patch = ("            if constexpr (DENSE && SPLIT) {   // tessera#805 MUTANT (repro only)\n"
             "                const long long t805 = clock64();\n"
             f"                while (clock64() - t805 < {spin}LL) {{ }}\n"
             "            }\n")
    out = text[:sync_end] + patch + text[sync_end:]
    open(path, "w").write(out)
    print(f"mutant: {path} spin={spin} line={out[:sync_end].count(chr(10)) + 1} "
          f"sha256={hashlib.sha256(out.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
