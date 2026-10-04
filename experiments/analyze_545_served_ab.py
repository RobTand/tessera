#!/usr/bin/env python3
"""Reduce the #545 served A/B arm records into the comparison packet.

Reads every ``arm-*.json`` under the receipt tree, pairs before/after per
artifact, and writes ``analysis-<half>.json`` plus a printable summary: the
shared-convention per-phase summaries (real median, profiled rep excluded,
raw population retained), mean power against the 140 W envelope, tokens and
joules per window, and the route-record launch names that prove what each
arm actually dispatched.

SCOPE: the arms differ by the whole tessera runtime (v29-era pin vs v45 pin),
so every ratio here is a MATCHED RUNTIME comparison result.  Mechanism
attribution comes from the in-engine profiles and route records; nothing in
this packet may be reported as an isolated epilogue-only delta.

FAIL-CLOSED: any missing arm, fatal, missing profile proof, or mismatched
prompt digest marks the pair REFUSED and the tool exits nonzero.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from served_fused_ab_summary import phase_summary  # noqa: E402

ENVELOPE_W = 140.0
COMPARISON_SCOPE = ("matched runtime comparison (v29-era pin vs v45 pin); "
                    "not an isolated epilogue-only delta")


def load(out_dir: Path):
    arms = {}
    for f in sorted(out_dir.glob("arm-*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"unreadable {f}: {exc}", file=sys.stderr)
            arms.setdefault("unknown", {})[f.stem] = {"_unreadable": str(exc)}
            continue
        arms.setdefault(r.get("artifact_tag") or "unknown", {})[r.get("arm")] = r
    return arms


def phase_row(r, key):
    """Recompute the shared summary from the raw rows; refuse silently-absent."""
    ph = r.get(key) or {}
    rows = ph.get("reps") or []
    if not rows:
        return {"present": False, "reason": f"{key} has no raw reps"}
    profiled = ph.get("profiled_rep")
    s = phase_summary(rows, profiled)
    power = ph.get("power_window") or {}
    mean_w = power.get("mean_w")
    wall = s["wall_s_total_included"]
    gen = s["gen_tokens_included"]
    row = {
        "present": True,
        "n_total": s["n_total"],
        "n_included": s["n_included"],
        "excluded_profiled_rep": s["excluded_profiled_rep"],
        "engine_call_ms_median": s["engine_call_ms_median"],
        "wall_s_included": round(wall, 3),
        "gen_tokens": gen,
        "prompt_tokens": s["prompt_tokens_included"],
        "gen_tok_per_s": s["gen_tok_per_s_included"],
        "pre_tok_per_s": s["pre_tok_per_s_included"],
        "mean_w": round(mean_w, 1) if mean_w is not None else None,
        "envelope_frac": round(mean_w / ENVELOPE_W, 3) if mean_w else None,
        "joules_window": round(mean_w * (power.get("to", 0) - power.get("from", 0)), 1)
        if mean_w is not None and power.get("to") and power.get("from") else None,
        "power_window_covers_profiled_rep": power.get("covers_profiled_rep"),
        "in_engine_profile_files": len((ph.get("in_engine_profile") or {}).get("new_trace_files") or []),
        "timing_locus": s["timing_locus"],
    }
    if row["joules_window"] and gen:
        row["gen_tokens_per_kj_window"] = round(gen * 1000.0 / row["joules_window"], 1)
    return row


def route_names(r):
    pairs = {}
    for key in ("routes_after_warmup", "routes_final"):
        for mod, rec in (r.get(key) or {}).items():
            if isinstance(rec, dict) and (rec.get("symbol") or rec.get("decoder")):
                pairs.setdefault(f"{rec.get('symbol')} / {rec.get('decoder')}", set()).add(mod)
    return {k: len(v) for k, v in sorted(pairs.items())}


def identity_row(r):
    ident = r.get("identity") or {}
    return {
        "tessera_commit": ident.get("tessera_commit"),
        "contract_version": ident.get("contract_version"),
        "vllm": ident.get("vllm"),
        "torch": ident.get("torch"),
        "platform": ident.get("platform_token"),
        "image": ident.get("runtime_image_declared"),
        "fatal": bool(r.get("fatal")),
        "fatal_head": (r.get("fatal") or "").strip().splitlines()[-1][:200] if r.get("fatal") else None,
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else "/mnt/shared/tessera-runs/receipts/545-served-remeasure-20261004")
    any_refused = False
    for half in ("h1", "h2"):
        d = root / half
        if not d.is_dir():
            continue
        arms = load(d)
        packet = {"half": half, "comparison_scope": COMPARISON_SCOPE, "pairs": {}}
        lines = [f"## {half} — {COMPARISON_SCOPE}"]
        for tag, by_arm in sorted(arms.items()):
            pair = {"complete": True}
            digests = set()
            for arm in ("before", "after"):
                r = by_arm.get(arm)
                if r is None or "_unreadable" in (r or {}):
                    pair["complete"] = False
                    pair[arm] = {"present": False,
                                 "reason": "missing or unreadable arm record"}
                    continue
                wl = (r.get("workload") or {}).get("prompt_ids_sha256")
                if wl:
                    digests.add(wl)
                pair[arm] = {
                    "identity": identity_row(r),
                    "decode": phase_row(r, "phase_decode"),
                    "batch": phase_row(r, "phase_batch"),
                    "routes": route_names(r),
                    "load_s": round(r.get("load_s") or 0, 1),
                    "idle_floor_mean_w": (r.get("idle_floor") or {}).get("mean_w"),
                    "power_source": r.get("power_source"),
                    "prompt_ids_sha256": wl,
                    "engine_core_facts": r.get("engine_core_facts"),
                }
                if r.get("fatal"):
                    pair["complete"] = False
                for ph in ("decode", "batch"):
                    row = pair[arm][ph]
                    if not row.get("present") or not row.get("in_engine_profile_files"):
                        pair["complete"] = False
            if len(digests) > 1:
                pair["complete"] = False
                pair["prompt_digest_mismatch"] = sorted(digests)
            if not pair["complete"]:
                any_refused = True
            packet["pairs"][tag] = pair
            lines.append(f"\n### {tag} — {'COMPLETE' if pair['complete'] else 'REFUSED'}")
            for arm in ("before", "after"):
                a = pair.get(arm) or {}
                if not a.get("present", True) or "identity" not in a:
                    lines.append(f"- {arm}: {a.get('reason', 'MISSING')}")
                    continue
                ident = a["identity"]
                lines.append(f"- {arm}: commit={ident['tessera_commit']} "
                             f"contract=v{ident['contract_version']} "
                             f"vllm={ident['vllm']} fatal={ident['fatal']}")
                for ph in ("decode", "batch"):
                    p = a[ph]
                    if not p.get("present"):
                        lines.append(f"  - {ph}: ABSENT ({p.get('reason')})")
                        continue
                    lines.append(
                        f"  - {ph}: engine_call_ms_median={p['engine_call_ms_median']} "
                        f"(n={p['n_included']}/{p['n_total']}, profiled rep "
                        f"{p['excluded_profiled_rep']} excluded) "
                        f"wall={p['wall_s_included']}s gen_tok/s={p['gen_tok_per_s']} "
                        f"pre_tok/s={p['pre_tok_per_s']} mean_w={p['mean_w']} "
                        f"({p['envelope_frac']} of {ENVELOPE_W:.0f} W) "
                        f"J(window)={p['joules_window']} "
                        f"gen_tok/kJ(window)={p.get('gen_tokens_per_kj_window')} "
                        f"profile_files={p['in_engine_profile_files']}")
                lines.append(f"  - routes: {json.dumps(a['routes'])}")
        out = root / f"analysis-{half}.json"
        out.write_text(json.dumps(packet, indent=1))
        print("\n".join(lines))
        print(f"\nwrote {out}")
    if any_refused:
        print("\nREFUSED: at least one pair is incomplete (fail-closed)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
