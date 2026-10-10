"""Compare the effective source of every action in a population. Replays no test.

Usage: python3 -I source_audit.py WORKTREE POPULATION_DIR EXPECT_PARENT OUT_JSON

WORKTREE         a Tessera checkout whose src/ holds tessera._dev.suite_source
POPULATION_DIR   a directory with audit.json from population_audit.py
EXPECT_PARENT    the commit every snapshot must name as its parent
OUT_JSON         where to write the report

For each action key in audit.json this tool:
  1. reads the sealed request from the pool's CAS and checks the action key;
  2. checks the snapshot bundle bytes against the digest the request seals;
  3. materializes the bundle in a new directory named the way pbsnapshot expects;
  4. calls tessera._dev.suite_source.measured_source with the same verifier
     merge_suite.py declares, given the request's own owner variable. That
     verifier checks the excluded closure metadata against the original sealed
     action. Its owner check matches by construction; its other checks do not.
It also measures EXPECT_PARENT in a clean clone with no verifier. Exit 0 only if
every action is verified, names EXPECT_PARENT as parent, and has the same hash
as that parent. Raw snapshot commits, bundle digests and excluded files stay in
the report.
"""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

WORKTREE, POP, EXPECT, OUT = sys.argv[1:5]
sys.path.insert(0, str(Path(WORKTREE, "src")))
from tessera._dev.suite_source import measured_source  # noqa: E402

POOL = Path("/mnt/shared/prismabuild-fleet")
VERIFIER = ["/usr/bin/python3", str(POOL / "repo/tools/pbsnapshot.py"), "verify"]


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def one_action(key, work):
    row = {"action_key": key, "problems": []}
    try:
        request = json.loads((POOL / "cas/requests" / key[:2] / f"{key}.json").read_bytes())
        snap = request["params"]["checkout_snapshot"]
        sealed = snap["input"]["sha256"]
        row.update(snapshot_commit=snap["commit"], parent=snap["parent"], bundle_sha256=sealed)
        if request["action_key"] != key:
            row["problems"].append("request names another action key")
        if snap["parent"] != EXPECT:
            row["problems"].append(f"parent is {snap['parent'][:12]}, expected {EXPECT[:12]}")
        bundle = POOL / "cas/blobs" / sealed[:2] / sealed
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != sealed:
            row["problems"].append("bundle bytes do not match the sealed digest")
        root = Path(work) / f"{key[:12]}.src" / "checkout"
        root.parent.mkdir()
        git("clone", "-q", "--no-hardlinks", str(bundle), str(root))
        git("checkout", "-q", "--detach", snap["commit"], cwd=root)
        owner = request["environment"]["variables"].get("PRISMABUILD_CONTAINER_OWNER")
        if not owner:
            row["problems"].append("the sealed request names no container owner")
        # The owner check compares this value with the request's own, so it
        # matches by construction; the key, closure, digest and snapshot checks
        # in the verifier do not.
        found = measured_source(root, verifier=[*VERIFIER, "--owner", owner or ""])
        row.update(verification=found["verification"], source_sha256=found["sha256"],
                   files_verified=found.get("files_verified"),
                   excluded_metadata=found["excluded_metadata"], reason=found.get("reason"))
        if found["verification"] != "verified":
            row["problems"].append("source identity not verified: " + str(found.get("reason")))
        for item in found["excluded_metadata"]:
            if item.get("action_key") != key:
                row["problems"].append("excluded file names another action")
        if not found["excluded_metadata"]:
            row["problems"].append("no excluded closure metadata was verified")
    except Exception as error:  # fail closed: any surprise is a problem, never a pass
        row["problems"].append(f"{type(error).__name__}: {error}")
    return row


with tempfile.TemporaryDirectory(prefix="source-audit-") as work:
    keys = [s["action_key"] for s in json.load(open(Path(POP, "audit.json")))["shards"]]
    rows = [one_action(k, work) for k in keys] if all(keys) else []
    parent_root = Path(work) / "parent"
    git("clone", "-q", "--no-hardlinks", str(Path(WORKTREE)), str(parent_root))
    git("checkout", "-q", "--detach", EXPECT, cwd=parent_root)
    parent = measured_source(parent_root, verifier=None)

problems = []
if not keys or not all(keys):
    problems.append("audit.json has a shard with no action key")
if parent["verification"] != "verified":
    problems.append("parent source not verified: " + str(parent.get("reason")))
hashes = {r.get("source_sha256") for r in rows}
for r in rows:
    if r.get("source_sha256") and r["source_sha256"] != parent["sha256"]:
        r["problems"].append("effective source differs from the parent's")
agree = len(rows) == len(keys) and len(hashes) == 1 and None not in hashes
verdict = "EQUIVALENT" if agree and not problems and not any(r["problems"] for r in rows) \
    and hashes == {parent["sha256"]} else "NOT PROVEN"
report = {"population": POP, "expect_parent": EXPECT, "verdict": verdict,
          "parent_measure": parent, "distinct_source_hashes": sorted(h or "none" for h in hashes),
          "problems": problems, "actions": rows}
Path(OUT).write_text(json.dumps(report, indent=1))
for r in rows:
    print(f"action {r['action_key'][:12]} snapshot={str(r.get('snapshot_commit'))[:9]} "
          f"parent={str(r.get('parent'))[:9]} verified={r.get('verification')} "
          f"source={str(r.get('source_sha256'))[:12]} files={r.get('files_verified')} "
          f"excluded={[e.get('path') for e in r.get('excluded_metadata', [])]}")
    for p in r["problems"]:
        print("   PROBLEM:", p)
print(f"parent {EXPECT[:9]} source={str(parent.get('sha256'))[:12]} files={parent.get('files_verified')}")
for p in problems:
    print("PROBLEM:", p)
print("VERDICT:", verdict)
sys.exit(0 if verdict == "EQUIVALENT" else 1)
