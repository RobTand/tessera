"""PB-only, one-shot metadata correction of the sealed LFM campaign artifact.

Preserves the original artifact and seal. The new directory hard-links all
unchanged files and writes its own config. No weight tensor is rewritten.
"""
import copy
import contextlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback

sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd())]
from experiments.export_tessera_serving import ignored_modules
from tessera.serving_parts import source_identity, sha256_file

CAMPAIGN = Path("/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2")
ORIGINAL = CAMPAIGN / "full-model"
MODEL = CAMPAIGN / "full-model-r3"
OUT = CAMPAIGN / "passthrough-correction-r2"
SEAL = CAMPAIGN / "merge-action-r1/artifact-seal.json"


def main():
    original_seal = json.loads(SEAL.read_text())
    before = source_identity(ORIGINAL)
    assert before == original_seal["checkpoint_identity"]
    original_config = json.loads((ORIGINAL / "config.json").read_text())
    config = copy.deepcopy(original_config)
    ignored = set()
    for shard in before["files"]:
        with (ORIGINAL / shard).open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            assert length <= (ORIGINAL / shard).stat().st_size - 8
            header = json.loads(handle.read(length))
        for name, info in header.items():
            if name != "__metadata__":
                ignored.update(ignored_modules(name, info["shape"]))
    old_ignored = set(config["quantization_config"]["ignore"])
    added, removed = sorted(ignored - old_ignored), sorted(old_ignored - ignored)
    print(json.dumps({"ignore_added": added, "ignore_removed": removed}), flush=True)
    assert added and removed, "no passthrough correction was derived"
    # Only the observed dense gate/up naming defect may change here.
    assert all(name.endswith(".feed_forward.w13") for name in added)
    expected_removed = {name[:-len("w13")] + role for name in added for role in ("w1", "w3")}
    assert set(removed) == expected_removed
    assert all(name + ".weight" in before["tensors"] for name in removed)
    declared = {target for group in config["quantization_config"]["config_groups"].values()
                for target in group["targets"]}
    assert not declared & ignored
    config["quantization_config"]["ignore"] = sorted(ignored)
    check = copy.deepcopy(config)
    check["quantization_config"]["ignore"] = original_config["quantization_config"]["ignore"]
    assert check == original_config, "a field other than ignore changed"
    MODEL.mkdir()
    for path in ORIGINAL.iterdir():
        assert path.is_file() and not path.is_symlink()
        if path.name != "config.json":
            os.link(path, MODEL / path.name)
    with (MODEL / "config.json").open("x") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (OUT / "sidecar-check.log").open("x") as log:
        subprocess.run([sys.executable, "experiments/ts5_sidecar_check.py", str(MODEL),
                        "--plan-json", str(CAMPAIGN / "plan.json")], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=180)
    after = source_identity(MODEL)
    assert after["files"] == before["files"] and after["tensors"] == before["tensors"]
    assert {name: sha for name, sha in after["auxiliary_sha256"].items() if name != "config.json"} == {
        name: sha for name, sha in before["auxiliary_sha256"].items() if name != "config.json"}
    assert source_identity(ORIGINAL) == before, "the original artifact changed"
    correction = {
        "schema": "tessera.lfm-passthrough-correction/1",
        "original_checkpoint": str(ORIGINAL), "original_seal_sha256": sha256_file(SEAL),
        "source_snapshot": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ignore_added": added, "ignore_removed": removed, "weight_bytes_unchanged": True,
        "old_config_sha256": before["config_sha256"], "new_config_sha256": after["config_sha256"],
        "old_config_bytes": (ORIGINAL / "config.json").stat().st_size,
        "new_config_bytes": (MODEL / "config.json").stat().st_size,
    }
    seal = copy.deepcopy(original_seal)
    seal.update(checkpoint=str(MODEL), checkpoint_identity=after, metadata_correction=correction)
    with (OUT / "artifact-seal.json").open("x") as handle:
        json.dump(seal, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(correction, sort_keys=True), flush=True)


if __name__ == "__main__":
    OUT.mkdir()
    with (OUT / "action.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            main()
        except BaseException:
            traceback.print_exc()
            raise
