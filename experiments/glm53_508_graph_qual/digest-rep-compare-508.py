"""tessera#508: place a repeat difference on the first module that differs.

For a DIGEST=1 serve that ran probe-rep-508.py (sequential batch-1 requests, 16
forwards each: one prefill and 15 decodes), split the last forwards of
$OUT/<arm>.dig.jsonl into one segment per request, in the probe's order
($OUT/<arm>.rep.summary.json), and compare:

  one arm:  every request with the first request of its length (same serve:
            does a repeat, after other requests used the engine, compute the
            same thing?)
  two arms: request i of arm A with request i of arm B (fresh serves)

For each compared forward it lists the slots whose digest differs, in the order
the modules ran, and names the origin: the first slot whose module output
differs while that module's input matched. Writes
$OUT/digrep-<A>[-vs-<B>].json.

  digest-rep-compare-508.py RECEIPTS_DIR ARM_A [ARM_B]
"""
import json
import pathlib
import sys

recs = pathlib.Path(sys.argv[1])
arm_a = sys.argv[2]
arm_b = sys.argv[3] if len(sys.argv) > 3 else None
FWD = 16


def segments(arm):
    order = json.loads((recs / f"{arm}.rep.summary.json").read_text())["order"]
    rows = [json.loads(line) for line in (recs / f"{arm}.dig.jsonl").read_text().splitlines() if line.strip()]
    need = FWD * len(order)
    assert len(rows) >= need, (arm, len(rows), need)
    rows = rows[-need:]
    segs = {}
    for i, req in enumerate(order):
        seg = rows[i * FWD:(i + 1) * FWD]
        n = req["prompt_tokens"]
        reals = [r["real"] for r in seg]
        assert reals[0] == n and all(x == 1 for x in reals[1:]), (arm, req["name"], reals)
        segs[req["name"]] = seg
    return order, segs


def diff(fa, fb):
    keys = [k for k in fa["digests"] if k in fb["digests"]]
    differ = [k for k in keys if fa["digests"][k] != fb["digests"][k]]
    origin = None
    for k in differ:
        base = k.split("#")[0]
        if base.endswith("|out"):
            twin = base[:-4] + "|in" + (k[len(base):])
            if twin in fa["digests"] and fa["digests"][twin] == fb["digests"].get(twin):
                origin = k
                break
    return dict(n_slots=len(keys), n_differ=len(differ), first=differ[0] if differ else None,
                origin=origin, differ=differ[:40])


order_a, segs_a = segments(arm_a)
pairs = []
if arm_b is None:
    first = {}
    for req in order_a:
        n = req["prompt_tokens"]
        if n in first:
            pairs.append((first[n], segs_a[first[n]], req["name"], segs_a[req["name"]]))
        else:
            first[n] = req["name"]
    tag = arm_a
else:
    order_b, segs_b = segments(arm_b)
    assert [r["name"] for r in order_a] == [r["name"] for r in order_b]
    pairs = [(r["name"], segs_a[r["name"]], r["name"], segs_b[r["name"]]) for r in order_a]
    tag = f"{arm_a}-vs-{arm_b}"
report = []
for name_a, sa, name_b, sb in pairs:
    steps = []
    for step, (fa, fb) in enumerate(zip(sa, sb)):
        d = diff(fa, fb)
        steps.append(dict(step=step, real=fa["real"], **d))
    first_bad = next((s for s in steps if s["n_differ"]), None)
    report.append(dict(a=name_a, b=name_b, first_differing_step=first_bad["step"] if first_bad else None,
                       origin=first_bad["origin"] if first_bad else None,
                       first_slot=first_bad["first"] if first_bad else None, steps=steps))
    if first_bad:
        print(f"{name_a} vs {name_b}: first differing forward {first_bad['step']} (real {first_bad['real']}): "
              f"{first_bad['n_differ']}/{first_bad['n_slots']} slots; first {first_bad['first']}; "
              f"origin {first_bad['origin']}", flush=True)
    else:
        print(f"{name_a} vs {name_b}: all {len(steps)} forwards identical", flush=True)
(recs / f"digrep-{tag}.json").write_text(json.dumps(report, indent=1))
