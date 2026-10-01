"""Per run table time penalty from a bench_geometry sweep (tessera#750).

For each launch (routed mode, dense shape) and M (and routing), each rung
q = 256r + {64, 128, 192} inside [r, r+1] is compared with
  * lerp: the linear interpolation of the whole rates 256r and 256(r+1);
  * top:  the whole rate 256(r+1).

The exclusion rule (per run table): exclude [r, r+1] when t(256(r+1)) < t(q)
beyond the spread for every rung q in the pair, at every M, on every launch.
spread = max of the two cells' |F - R| / mean.

Usage:
  python3 pair_time.py --json OUT.json bench_geometry_routed.json bench_geometry_dense.json

Prints one markdown table per launch: each cell is the range of the pair's
rungs' time against [r+1] (over rungs and routings), then the median time
against the lerp.
"""
import argparse
import collections
import json
import statistics


def load(path):
    d = json.load(open(path))
    out = collections.defaultdict(dict)   # launch -> {m cell: {q: (ms, spread)}}
    for g in d['groups'].values():
        if g.get('kind') not in ('routed', 'dense') or 'cells' not in g:
            continue
        launch = f"routed mode {g['mode']}" if g['kind'] == 'routed' else f"dense {g['shape']}"
        for mc, c in g['cells'].items():
            if 'ms' not in c:
                continue
            out[launch].setdefault(mc, {})[g['q256']] = (c['ms'], c.get('spread', 0.0))
    return out


def analyse(out):
    rows = []
    for launch, cells in sorted(out.items()):
        for mc, byq in cells.items():
            for r in range(1, 8):
                lo, hi = 256 * r, 256 * (r + 1)
                if lo not in byq or hi not in byq:
                    continue
                tlo, _ = byq[lo]
                thi, shi = byq[hi]
                for off in (64, 128, 192):
                    q = lo + off
                    if q not in byq:
                        continue
                    t, s = byq[q]
                    lerp = tlo + off / 256 * (thi - tlo)
                    spread = max(s, shi)
                    rows.append(dict(
                        launch=launch, m=mc, table=f"[{r},{r + 1}]", q=q, t=t, lerp=lerp,
                        vs_lerp=t / lerp - 1, vs_top=t / thi - 1, spread=spread,
                        top_faster=(t - thi) / t > spread,
                        pair_faster=(thi - t) / t > spread,
                        tlo=tlo, thi=thi))
    return rows


def verdicts(rows):
    v = {}
    for r in rows:
        x = v.setdefault(r['table'], {'n': 0, 'top_faster': 0, 'pair_faster': 0})
        x['n'] += 1
        x['top_faster'] += r['top_faster']
        x['pair_faster'] += r['pair_faster']
    for x in v.values():
        x['excluded_strict'] = x['top_faster'] == x['n']
        x['excluded_ties_count'] = x['pair_faster'] == 0
    return v


def pct(x):
    s = f"{100 * x:+.0f}"
    return '0' if s in ('+0', '-0') else s.replace('-', '−')


def tables(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['launch'], r['table'], int(r['m'].split(':')[0]))].append(r)
    ms = sorted({k[2] for k in g})
    lines = []
    for launch in sorted({k[0] for k in g}):
        lines += [f"#### {launch}", "", "| Table | " + " | ".join(f"M={m}" for m in ms) + " |",
                  "|---|" + "---:|" * len(ms)]
        for t in sorted({k[1] for k in g if k[0] == launch}):
            cells = []
            for m in ms:
                x = g.get((launch, t, m))
                if not x:
                    cells.append('-')
                    continue
                vt = [r['vs_top'] for r in x]
                vl = statistics.median(r['vs_lerp'] for r in x)
                cells.append(f"{pct(min(vt))}..{pct(max(vt))}% / {pct(vl)}%")
            lines.append(f"| `{t}` | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='output: every rung row and the per-table verdict')
    ap.add_argument('sweeps', nargs='+', help='bench_geometry JSON files')
    a = ap.parse_args()
    rows = []
    for p in a.sweeps:
        rows += analyse(load(p))
    v = verdicts(rows)
    json.dump({'rows': rows, 'verdict': v}, open(a.json, 'w'), indent=1)
    print(tables(rows))
    for k in sorted(v):
        print(k, v[k])


if __name__ == '__main__':
    main()
