"""tessera#702: the equality script each legacy arm ran, read out of its own PrismaBuild snapshot.

  snapshot_script_hash.py MANIFEST OUT [--path experiments/glm53_508_graph_qual/equal-508.py]

Arms that predate arm.sh's recorded ``equal_script_sha256`` (the ba663a6b run) still name the
bytes they ran: PrismaBuild sealed each action's checkout as a git bundle in its CAS. For every
arm in MANIFEST (submit.py's arm -> action key), this reads the action's terminal record,
checks the bundle blob's sha256 against the record, fetches the bundle into a scratch
repository, and hashes PATH at the action's own snapshot commit. OUT is the JSON receipt.py's
``--legacy-script-provenance`` reads: ``{arm: {sha256, source}}``. Read-only on the queue and CAS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

QUEUE = pathlib.Path("/mnt/shared/prismabuild-fleet/pb-queue")
CAS = pathlib.Path("/mnt/shared/prismabuild-fleet/cas/blobs")


def terminal_record(key: str) -> dict:
    for state in ("done", "failed"):
        path = QUEUE / state / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
    raise SystemExit(f"{key}: no terminal record under {QUEUE}")


def script_hash(key: str, path: str, scratch: pathlib.Path) -> dict:
    snap = terminal_record(key).get("checkout_snapshot") or {}
    commit, blob_sha = snap.get("commit"), (snap.get("input") or {}).get("sha256")
    if not commit or not blob_sha:
        raise SystemExit(f"{key}: the record names no checkout snapshot")
    blob = CAS / blob_sha[:2] / blob_sha
    if hashlib.sha256(blob.read_bytes()).hexdigest() != blob_sha:
        raise SystemExit(f"{key}: CAS blob {blob} does not hash to its name")
    subprocess.run(["git", "-C", str(scratch), "fetch", "-q", str(blob), "+refs/*:refs/snap/*"],
                   check=True)
    data = subprocess.run(["git", "-C", str(scratch), "show", f"{commit}:{path}"],
                          check=True, capture_output=True).stdout
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "source": f"{path} at PrismaBuild snapshot {commit} of action {key} (CAS blob {blob_sha})"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--path", default="experiments/glm53_508_graph_qual/equal-508.py")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        scratch = pathlib.Path(tmp)
        subprocess.run(["git", "init", "-q", "--bare", str(scratch)], check=True)
        out = {arm: script_hash(v["action_key"], a.path, scratch) for arm, v in sorted(manifest.items())}
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    for arm, rec in out.items():
        print(arm, rec["sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
