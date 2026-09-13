"""Install a Tessera source tree as the vLLM plugin inside a pinned image, without core edits.

The source-BF16 observers install a fixed archive into the stock
``vllm/vllm-openai`` image through ``_pb_native_moe_measure/per_job_install.py``.
A served artifact is observed in the image its serving lane pins, so this
installer takes the image reference, the frozen source tree and the core
manifest as arguments and records the same evidence: the launcher's image
identity, the vLLM core manifest unchanged before and after the install, the
installed plugin files and its entry point. ``upstream_commit`` is what the
installed vLLM reports about itself, and the observer plan compares it with
the selected configuration rather than trusting either alone.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

EVIDENCE_NAME = "per-job-runtime.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def files(root):
    return {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(Path(root).rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def source_tree_identity(tree):
    """Digest of the packaged source: ``src/`` and the build metadata, not receipts or tests."""
    tree = Path(tree)
    members = {}
    for name in ("pyproject.toml", "MANIFEST.in"):
        if (tree / name).is_file():
            members[name] = digest(tree / name)
    for path in sorted((tree / "src").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            members[str(path.relative_to(tree))] = digest(path)
    if not members or not any(name.startswith("src/") for name in members):
        raise ValueError("source tree carries no src/ package to install")
    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--base-reference", required=True,
                        help="pinned image reference the launcher resolved, e.g. repo@sha256:...")
    parser.add_argument("--launcher-image-id", required=True)
    parser.add_argument("--launcher-image-inspect", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path, required=True, help="frozen Tessera source tree (read-only)")
    parser.add_argument("--source-commit", required=True, help="commit the frozen tree was sealed from")
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--runtime-uid", type=int, default=1000)
    parser.add_argument("--runtime-gid", type=int, default=1000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    inspected = json.loads(args.launcher_image_inspect.read_text())
    if isinstance(inspected, list):
        assert len(inspected) == 1
        inspected = inspected[0]
    assert inspected["Id"] == args.launcher_image_id, "Launcher image ID must match actual image inspection"
    assert args.base_reference in inspected.get("RepoDigests", []), "Pinned base must appear in inspected RepoDigests"
    core_manifest_sha256 = digest(args.core_manifest)
    stock = json.loads(args.core_manifest.read_text())
    core = Path(importlib.util.find_spec("vllm").origin).parent
    assert files(core) == stock["files"], "Initial installed vLLM differs from the attested image manifest"
    evidence_path = args.evidence_dir / EVIDENCE_NAME
    assert not evidence_path.exists(), "Refuse to overwrite prior runtime evidence"
    source_sha256, source_members = source_tree_identity(args.source_tree)
    with tempfile.TemporaryDirectory(prefix="tessera-plugin-") as temp:
        staged = Path(temp) / "tessera"
        shutil.copytree(args.source_tree, staged, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                        "--no-cache-dir", "-q", str(staged)], check=True)
    assert files(core) == stock["files"], "Plugin installation changed vLLM core files"
    plugin = Path(importlib.util.find_spec("tessera").origin).parent
    entries = [{"name": e.name, "value": e.value} for e in importlib.metadata.entry_points(group="vllm.general_plugins")
               if e.name == "tessera"]
    assert entries == [{"name": "tessera", "value": "tessera.serving:register"}]
    import vllm._version as vllm_version
    record = {"launcher_image_inspect_sha256": digest(args.launcher_image_inspect), "launcher_image_inspect": inspected,
              "registry_base": args.base_reference, "launcher_declared_image_id": args.launcher_image_id,
              "identity_scope": "The launcher binds the image reference/ID; this installer verifies the complete vLLM file manifest before and after plugin installation.",
              "upstream_commit": vllm_version.__commit_id__, "upstream_commit_source": "vllm._version.__commit_id__",
              "core_manifest_sha256": core_manifest_sha256, "core_files_unchanged": len(stock["files"]),
              "plugin_source_commit": args.source_commit, "plugin_source_tree": str(args.source_tree),
              "plugin_source_sha256": source_sha256, "plugin_source_members": source_members,
              "plugin_files": files(plugin), "plugin_entrypoints": entries,
              "vllm_version": importlib.metadata.version("vllm"),
              "tessera_version": importlib.metadata.version("tessera-quant"),
              "affinity": sorted(os.sched_getaffinity(0))}
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(args.runtime_gid)
        os.setuid(args.runtime_uid)
    else:
        assert os.getuid() == args.runtime_uid and os.getgid() == args.runtime_gid
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"artifact": str(evidence_path), "sha256": digest(evidence_path),
                      "core_files_unchanged": record["core_files_unchanged"], "registry_base": args.base_reference,
                      "upstream_commit": record["upstream_commit"], "vllm_version": record["vllm_version"]}), flush=True)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
