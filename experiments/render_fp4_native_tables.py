"""Render the lever-battery JSONs as the markdown tables the measurement doc
carries, so no number in the doc is typed by hand."""
import json, sys

EXL3 = 0.05653
mean = lambda v: sum(v) / len(v)


def table(path, rows, base_key, title):
    d = json.load(open(path))
    arms, act = d["arms"], [x["served"] if isinstance(x, dict) else x for x in d["act"]]
    base = arms[base_key]
    print(f"\n{title}\n")
    print("| arm | bpp | weight-space | out-space weight leg | vs baseline | min–max | W4A4 (served A4) | vs EXL3@A4 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, key in rows:
        v = arms[key]
        ratio = [b["out"] / r["out"] for b, r in zip(base, v)]
        exl = mean([r["both_served"] / (EXL3 ** 2 + a ** 2) ** 0.5 for r, a in zip(v, act)])
        bpp = v[0]["bpp"]
        bpp_s = "—" if bpp != bpp else f"{bpp:.3f}"
        print(f"| {label} | {bpp_s} | {mean([r['wt'] for r in v]):.5f} | {mean([r['out'] for r in v]):.5f} "
              f"| {mean(ratio):.3f}× | {min(ratio):.3f}–{max(ratio):.3f} | {mean([r['both_served'] for r in v]):.5f} | {exl:.3f}× |")
    print(f"\nact leg (served quantiser, exact weights): {mean(act):.5f} over {len(act)} tensors; "
          f"EXL3@A4 projected {mean([(EXL3**2 + a**2)**0.5 for a in act]):.5f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "battery"
    if which == "battery":
        p = "experiments/results/tessera_fp4_native_levers.json"
        b = "P0 s6b h=1.00 (artifact)"
        table(p, [
            ("artifact plane (`_pack_scales`, floor mantissa)", b),
            ("headroom 0.97 (best of the sweep)", "P1 s6b h=0.97"),
            ("nearest mantissa, same words", "P1 s6b nearest-mantissa"),
            ("flat E4M3 per 16 (same 0.5 bpp)", "P2 e4m3 flat h=1.00"),
            ("E8M0 per 32 only (best threshold)", "P3 e8m0-only t=1.19"),
        ], b, "### P — the plane's rule and format")
        table(p, [
            ("artifact plane", b),
            ("LS refit ×1 → S6b", "R s6b LS plain -> s6b x1"),
            ("LS refit ×3 → S6b (**shipping default**)", "R s6b LS plain -> s6b x3"),
            ("LS refit ×5 → S6b", "R s6b LS plain -> s6b x5"),
            ("LS refit ×5 → fp32 (unrepresentable)", "R s6b LS plain -> fp32 x5"),
            ("H16-weighted LS ×3 → S6b, σ=0.1", "R s6b LS H16 s=0.1 -> s6b x3"),
            ("LS refit ×3 → flat E4M3", "R e4m3 LS plain -> e4m3 x3"),
            ("H16-weighted LS ×3 → flat E4M3, σ=0.1", "R e4m3 LS H16 s=0.1 -> e4m3 x3"),
            ("LS refit ×3 → E8M0 only", "R e8m0 LS plain -> e8m0 x3"),
        ], b, "### R — refitting the plane's values")
        table(p, [
            ("artifact plane", b),
            ("Wei L=2 on S6b (3.75 payload)", "M L=2 on s6b"),
            ("Wei L=2 on S6b + H16-LS ×1", "M L=2 on s6b + H16-LS x1"),
            ("Wei L=2 on E8M0-only (4.0 bpp)", "M L=2 on e8m0"),
        ], b, "### M — half the redundancy")
        table(p, [
            ("artifact plane", b),
            ("group-LDLQ (32×32 blocks) σ=1.0 ×2 + H16-LS, S6b", "F1 s6b group-LDLQ s=1.0 x2 + H16-LS"),
            ("group-LDLQ σ=1.0 ×2 + H16-LS, flat E4M3", "F1 e4m3 group-LDLQ s=1.0 x2 + H16-LS"),
            ("full-LDLQ σ=0.025 (EXL3's), S6b", "F2 s6b full-LDLQ s=0.025"),
            ("full-LDLQ σ=0.1, S6b", "F2 s6b full-LDLQ s=0.1"),
            ("full-LDLQ σ=1.0, S6b", "F2 s6b full-LDLQ s=1.0"),
            ("full-LDLQ σ=1.0, flat E4M3", "F2 e4m3 full-LDLQ s=1.0"),
        ], b, "### F — error feedback")
    elif which == "rank1":
        p = "experiments/results/tessera_rank1_plane_multidim.json"
        it = json.load(open(p))["args"]["iters"]
        b = f"S6b plane + LS x{it}  L=1"
        table(p, [
            (f"S6b plane + LS ×{it}, L=1 (the shipping plane)", b),
            (f"S6b plane + LS ×{it}, L=2", f"S6b plane + LS x{it}  L=2"),
            (f"S6b plane + LS ×{it}, L=4", f"S6b plane + LS x{it}  L=4"),
            (f"S6b plane + LS ×{it}, L=8", f"S6b plane + LS x{it}  L=8"),
            ("rank-1 field (row × 16-col block), amax start, L=1", "rank-1 field (amax start)  L=1"),
            (f"rank-1 field + LS ×{it}, L=1", f"rank-1 field + LS x{it}  L=1"),
            (f"rank-1 field + LS ×{it}, L=2", f"rank-1 field + LS x{it}  L=2"),
            (f"rank-1 field + LS ×{it}, L=4", f"rank-1 field + LS x{it}  L=4"),
            (f"rank-1 field + LS ×{it}, L=8", f"rank-1 field + LS x{it}  L=8"),
        ], b, "### Deleting the plane and spending its bits on L")
    elif which == "alt":
        p = "experiments/results/tessera_plane_alternatives.json"
        d = json.load(open(p))
        b = "S6b plane + LS x3  L=1"
        base = d["arms"][b]
        print("\n| arm | bpp | weight-space | out-space weight leg | vs S6b+LS | act leg | W4A4 (served) | W4A4 vs | vs EXL3@A4 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in d["arms"].items():
            ro = mean([bb["out"] / rr["out"] for bb, rr in zip(base, v)])
            rb = mean([bb["both_served"] / rr["both_served"] for bb, rr in zip(base, v)])
            exl = mean([rr["both_served"] / (EXL3 ** 2 + aa ** 2) ** 0.5 for rr, aa in zip(v, d["act"])])
            print(f"| {k} | {v[0]['bpp']:.3f} | {mean([x['wt'] for x in v]):.5f} | {mean([x['out'] for x in v]):.5f} "
                  f"| {ro:.3f}× | {mean([x['act'] for x in v]):.5f} | {mean([x['both_served'] for x in v]):.5f} | {rb:.3f}× | {exl:.3f}× |")
