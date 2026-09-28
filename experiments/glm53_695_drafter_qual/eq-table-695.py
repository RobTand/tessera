"""tessera#695: the #508 magnitude table for a pair of arms' equality-suite records.

  eq-table-695.py RECEIPTS ARM_A ARM_B [ARM_A2 ARM_B2 ...]

For each pair, over every case both arms ran and every choice in it, as the
tessera#508 record's section 3 table reports:

  completions differing      choices whose token ids or top-20 logprobs differ
  changed a generated token  choices whose generated token ids differ
  largest logprob change     over the tokens both top-20 lists name, at the
                             positions up to and including the first differing
                             generated token id (both runs condition on the
                             same tokens there; past it the lists answer
                             different contexts)
  largest logit change       the same positions, with the log-normalizer's
                             change removed: at one position every shared
                             token's logprob moves by its logit change minus
                             the normalizer's change, and the most common move
                             is the normalizer's alone.

It also lists the logit moves at each differing choice's FIRST differing
position (the smallest visible difference; on this stub eager's own are one
bf16 step, 2^-5 or 2^-4 nats), rounded to 1e-6.
"""
import collections
import json
import pathlib
import sys


def choices(recs, arm, case):
    res = json.loads((recs / f"{arm}.eq.{case}.json").read_text())
    return sorted(res["choices"], key=lambda c: c["index"])


def moves(xa, xb):
    """Logprob and logit moves over the tokens both top-20 lists name, at one position."""
    shared = set(xa) & set(xb)
    if not shared:
        return None, None, []
    d = {t: xb[t] - xa[t] for t in shared}
    norm = collections.Counter(round(v, 6) for v in d.values()).most_common(1)[0][0]
    logit = {t: v - norm for t, v in d.items()}
    return (max(abs(v) for v in d.values()), max(abs(v) for v in logit.values()),
            sorted(round(v, 6) for v in logit.values() if abs(v) > 1e-6))


def table(recs, a, b):
    sa = json.loads((recs / f"{a}.eq.summary.json").read_text())
    sb = json.loads((recs / f"{b}.eq.summary.json").read_text())
    total = differing = token_changed = 0
    lp_max = logit_max = 0.0
    first_moves, rows = collections.Counter(), []
    for case in sa:
        if case.startswith("_") or case not in sb or not isinstance(sa[case], dict):
            continue
        for ca, cb in zip(choices(recs, a, case), choices(recs, b, case)):
            total += 1
            ta, tb = ca["token_ids"], cb["token_ids"]
            pa, pb = ca["logprobs"]["top_logprobs"], cb["logprobs"]["top_logprobs"]
            if ta == tb and pa == pb:
                continue
            differing += 1
            first_tok = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            token_changed += first_tok is not None
            last = first_tok if first_tok is not None else min(len(pa), len(pb)) - 1
            first_top = next((i for i in range(last + 1) if pa[i] != pb[i]), None)
            row = dict(case=case, choice=ca["index"], first_top20_diff=first_top, first_token_diff=first_tok)
            for i in range(last + 1):
                if pa[i] == pb[i]:
                    continue
                lp, lg, mv = moves(pa[i], pb[i])
                if lp is None:
                    continue
                lp_max, logit_max = max(lp_max, lp), max(logit_max, lg)
                row["max_logprob_change"] = max(row.get("max_logprob_change", 0.0), lp)
                row["max_logit_change"] = max(row.get("max_logit_change", 0.0), lg)
                if i == first_top:
                    row["first_position_logit_moves"] = mv
                    for v in mv:
                        first_moves[abs(v)] += 1
            rows.append(row)
    return dict(a=a, b=b, admission=[sa.get("_admission"), sb.get("_admission")], completions=total,
                differing=differing, changed_a_token=token_changed, largest_logprob_change=lp_max,
                largest_logit_change=logit_max,
                first_position_logit_move_sizes={str(k): v for k, v in sorted(first_moves.items())},
                rows=rows)


def main():
    recs = pathlib.Path(sys.argv[1])
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    out = []
    for a, b in pairs:
        t = table(recs, a, b)
        out.append(t)
        print(f"{b} vs {a}: {t['differing']}/{t['completions']} differ, {t['changed_a_token']} changed a token, "
              f"largest logit change {t['largest_logit_change']:.5f}, largest logprob change "
              f"{t['largest_logprob_change']:.5f}, first-position logit moves {t['first_position_logit_move_sizes']}")
    name = "_".join(f"{b}-vs-{a}" for a, b in pairs)[:180]
    (recs / f"eqtable-{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
