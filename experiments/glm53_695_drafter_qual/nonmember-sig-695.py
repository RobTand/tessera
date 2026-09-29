"""tessera#695: the signature of an equality-suite non-member, against the eager outcomes.

  nonmember-sig-695.py [--out FILE] RECEIPTS RUN EAGER1,EAGER2,... CASE:CHOICE [CASE:CHOICE ...]

A completion is a non-member when its token ids and every top-20 logprob list
equal no eager outcome of the same case and choice (eq-member-508.py). This
reports how far it is from the nearest one, as the membership criterion's
annotation:

  same_token_ids       whether some eager outcome generated the same token ids
  nearest              the eager outcome (run, pass) that minimises the largest
                       top-20 logprob change, among the outcomes with the same
                       token ids when there are any, else among all of them
  positions_differing  positions whose top-20 lists differ from the nearest
                       outcome, up to and including the first differing token
                       (past it the two runs condition on different tokens)
  max_logprob_change   the largest change over the tokens both lists name at
                       those positions
  max_logit_change     the same with the log-normalizer's change removed (the
                       most common move at a position), as eq-table-695.py
                       reports it

EAGER names the serves of the pool the completion was judged a non-member
against (member-pools-695.py); against a larger set, a completion that one
excluded serve also produced reads as a distance of 0. Each serve's second
pass (<serve>-r2) is an outcome too when its records exist. Prints one line
per completion and writes FILE (default ./nonmember-sig-<RUN>.json).
"""
import collections
import json
import pathlib
import sys


def choice(recs, run, case, index):
    path = recs / f"{run}.eq.{case}.json"
    if not path.exists():
        return None
    for c in json.loads(path.read_text())["choices"]:
        if c["index"] == index:
            return c
    return None


def moves(xa, xb):
    """Largest logprob and logit change over the tokens both top-20 lists name."""
    shared = set(xa) & set(xb)
    if not shared:
        return None, None
    d = {t: xb[t] - xa[t] for t in shared}
    norm = collections.Counter(round(v, 6) for v in d.values()).most_common(1)[0][0]
    return max(abs(v) for v in d.values()), max(abs(v - norm) for v in d.values())


def distance(a, b):
    ta, tb = a["token_ids"], b["token_ids"]
    pa, pb = a["logprobs"]["top_logprobs"], b["logprobs"]["top_logprobs"]
    first_tok = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    last = first_tok if first_tok is not None else min(len(pa), len(pb)) - 1
    positions, lp_max, lg_max = 0, 0.0, 0.0
    for i in range(last + 1):
        if pa[i] == pb[i]:
            continue
        positions += 1
        lp, lg = moves(pa[i], pb[i])
        if lp is not None:
            lp_max, lg_max = max(lp_max, lp), max(lg_max, lg)
    return dict(same_token_ids=ta == tb, first_token_diff=first_tok, positions_differing=positions,
                max_logprob_change=lp_max, max_logit_change=lg_max)


def main():
    args = sys.argv[1:]
    dest = None
    if args[:1] == ["--out"]:
        dest, args = pathlib.Path(args[1]), args[2:]
    recs, run = pathlib.Path(args[0]), args[1]
    eager = [e for s in args[2].split(",") for e in (s, f"{s}-r2")]
    out = []
    for spec in args[3:]:
        case, index = spec.rsplit(":", 1)
        mine = choice(recs, run, case, int(index))
        if mine is None:
            raise SystemExit(f"{run}: no record for {case} choice {index}")
        found = []
        for e in eager:
            ref = choice(recs, e, case, int(index))
            if ref is not None:
                found.append(dict(outcome=e, **distance(ref, mine)))
        same = [f for f in found if f["same_token_ids"]]
        nearest = min(same or found, key=lambda f: f["max_logprob_change"])
        row = dict(case=case, choice=int(index), eager_outcomes=len(found), same_token_ids=bool(same),
                   nearest=nearest)
        out.append(row)
        print(f"{run} {case}:{index} same token ids as {len(same)} of {len(found)} eager outcomes; nearest "
              f"{nearest['outcome']}: {nearest['positions_differing']} positions differ, largest logprob change "
              f"{nearest['max_logprob_change']:.5f}, logit change {nearest['max_logit_change']:.5f}"
              + ("" if nearest["same_token_ids"] else f", first differing token at {nearest['first_token_diff']}"))
    (dest or pathlib.Path(f"nonmember-sig-{run}.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
