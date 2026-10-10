"""Audit a sharded population from its stored shard outputs and the pool's own records. No replay.

Usage: python3 -I population_audit.py DIR [--json OUT] [--expect-shards N]

Each shard output b-<i>.out holds the pytest text or, for an action the pool already held, a receipt line that
names a payload.  The terminal status comes from the pool's queue ending file for the shard's action key, which
this tool reads and never writes.  A shard is GREEN only when every one of these holds; any miss makes it
INCOMPLETE or RED, never zero:
  * an action key was found and the queue ending file for it exists and says status executed, returncode 0
  * exactly one pytest summary line was found, and it states passed
  * failed == 0 and errors == 0
  * the surface block states 0 modules not collected
Exit status 0 means every shard is GREEN and the shard count matches --expect-shards.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DONE = Path("/mnt/shared/prismabuild-fleet/pb-queue/done")
KEY = re.compile(r"\b([0-9a-f]{64})\b")
SUMMARY = re.compile(r"^=*\s*((?:\d+ [a-z]+(?: [a-z]+)?(?: \([^)]*\))?(?:, )?)+) in [\d.]+s")
TOKEN = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?|rerun|subtests passed)")


def receipt_lines(text):
    """Lines that are a pool receipt: a JSON object that names a payload."""
    for line in text.splitlines():
        if line.startswith("{") and "payload_path" in line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "payload_path" in d:
                yield d


def shard_text(path):
    text = path.read_text(errors="replace")
    for d in list(receipt_lines(text)):
        try:
            text += "\n" + Path(d["payload_path"]).read_text(errors="replace")
        except (OSError, KeyError):
            text += "\nPAYLOAD_UNREADABLE"
    return text


def action_key(text):
    for line in text.splitlines():
        if line.startswith("pbrun: queued "):
            m = KEY.search(line)
            if m:
                return m.group(1)
    for d in receipt_lines(text):
        key = (d.get("receipt") or {}).get("action_key")
        if key:
            return key
    return None


def terminal(key):
    if not key:
        return None, "no action key in the shard output"
    path = DONE / f"{key}.json"
    if not path.is_file():
        return None, "no queue ending file for the action"
    try:
        d = json.loads(path.read_text())
    except ValueError:
        return None, "queue ending file is not JSON"
    detail = d.get("detail") or {}
    return {"status": d.get("status"), "returncode": detail.get("returncode"), "host": detail.get("host") or
            (detail.get("resource_profile") or {}).get("host")}, None


def audit_shard(index, path):
    text = shard_text(path)
    out = {"shard": index, "file": str(path), "problems": []}
    key = action_key(text)
    out["action_key"] = key
    term, why = terminal(key)
    out["terminal"] = term
    if why:
        out["problems"].append(why)
    elif term["status"] != "executed" or term["returncode"] != 0:
        out["problems"].append(f"terminal status {term['status']!r} returncode {term['returncode']!r}")
    if "PAYLOAD_UNREADABLE" in text:
        out["problems"].append("the payload the receipt names is unreadable")
    # The stored output may already hold the payload text, so the same line can appear twice.
    # Distinct lines are what matter: two different summaries in one shard is a problem.
    lines = sorted({l.strip() for l in text.splitlines() if SUMMARY.match(l)})
    out["summary_lines"] = len(lines)
    counts = {}
    if len(lines) != 1:
        out["problems"].append(f"{len(lines)} distinct pytest summary lines, expected exactly 1")
    if lines:
        for n, name in TOKEN.findall(lines[-1]):
            counts[name.rstrip("s") if name.startswith("error") else name] = int(n)
        out["summary"] = lines[-1].strip()[:140]
    out["counts"] = counts
    if "passed" not in counts:
        out["problems"].append("the summary line states no passed count")
    for bad in ("failed", "error"):
        if counts.get(bad):
            out["problems"].append(f"{counts[bad]} {bad}")
    if re.search(r"^(FAILED|ERROR) ", text, re.M) and not (counts.get("failed") or counts.get("error")):
        out["problems"].append("FAILED or ERROR lines present but the summary counts none")
    unc = re.search(r"(\d+) test\(s\) skipped, (\d+) module\(s\) not collected", text)
    out["uncollected_modules"] = int(unc.group(2)) if unc else None
    if unc is None:
        out["problems"].append("no surface block: uncollected modules unknown")
    elif int(unc.group(2)):
        out["problems"].append(f"{unc.group(2)} modules not collected")
    dev = re.search(r"tessera surface: (CUDA[^\n]*|NO CUDA[^\n]*)", text)
    out["device"] = dev.group(1)[:100] if dev else None
    blocks, block, grab = [], None, False
    for l in text.splitlines():
        if "skip reasons, verbatim" in l:
            block, grab = [], True
            continue
        if grab:
            if l.startswith("tessera surface:") or not l.strip():
                blocks.append(block)
                grab = False
            else:
                block.append(l.strip())
    if grab:
        blocks.append(block)
    # The shard output and the pool payload carry the same pytest text, so the same block is read
    # twice. Keep one block per shard. Two different blocks are a problem.
    distinct = [list(b) for b in {tuple(b) for b in blocks}]
    if len(distinct) > 1:
        out["problems"].append(f"{len(distinct)} different skip-reason blocks, expected 1")
    block = blocks[0] if blocks else []
    reasons_total = sum(int(m.group(1)) for m in (re.match(r"(\d+)\s", row) for row in block) if m)
    if counts.get("skipped") is not None and reasons_total != counts["skipped"]:
        out["problems"].append(f"skip reasons add up to {reasons_total}, the summary says {counts['skipped']}")
    out["skip_reasons"] = block
    out["status"] = "GREEN" if not out["problems"] else ("RED" if any("failed" in p or "error" in p or "returncode" in p
                                                                    for p in out["problems"]) else "INCOMPLETE")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--expect-shards", type=int, default=12)
    a = ap.parse_args()
    d = Path(a.dir)
    found = {}
    for p in d.glob("b-*.out"):
        m = re.fullmatch(r"b-(\d+)\.out", p.name)
        if m:
            found[int(m.group(1))] = p
    rows = [audit_shard(i, found[i]) if i in found else
            {"shard": i, "status": "INCOMPLETE", "problems": ["no shard output file"], "action_key": None}
            for i in range(a.expect_shards)]
    extra = sorted(set(found) - set(range(a.expect_shards)))
    for r in rows:
        c = r.get("counts", {})
        print(f"shard {r['shard']:>2} {r['status']:<10} key={str(r.get('action_key'))[:12]} "
              f"rc={(r.get('terminal') or {}).get('returncode')} passed={c.get('passed')} failed={c.get('failed', 0)} "
              f"errors={c.get('error', 0)} skipped={c.get('skipped')} uncollected={r.get('uncollected_modules')} "
              f"device={str(r.get('device'))[:34]}")
        for p in r["problems"]:
            print("   PROBLEM:", p)
    green = all(r["status"] == "GREEN" for r in rows) and not extra
    tot = lambda k: sum((r.get("counts") or {}).get(k, 0) for r in rows)
    print(f"TOTAL passed={tot('passed')} failed={tot('failed')} errors={tot('error')} skipped={tot('skipped')} "
          f"shards={len(rows)} extra_outputs={extra} VERDICT={'GREEN' if green else 'NOT GREEN'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"dir": str(d), "verdict": "GREEN" if green else "NOT GREEN", "shards": rows}, indent=1))
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
