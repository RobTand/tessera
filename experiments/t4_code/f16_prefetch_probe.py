"""CPU-only, one-pair e4m3 compile probe; never loads a cubin or runs a GPU.

Example (identities are caller assertions, not discoveries)::

    PYTHONPATH=src python experiments/t4_code/f16_prefetch_probe.py \
        --off before.cu --off-identity recovered \
        --on after.cu --on-identity reconstruction --out /tmp/prefetch-pair

Use --inventory-only --out DIR without sources to inventory even missing tools.
Each command is time bounded; no retries, installs, JIT loads or benchmarks.
PTX is produced by the selected nvcc, then assembled by an explicitly selected
CUDA_HOME/bin/ptxas, NOT an inferred implicit nvcc subprocess. Host compilers and
transitive system libraries are not fully attested. Header hashes come from
nvcc -M; missing dependencies make that coverage explicitly incomplete.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import sass_dump


PTX_HINT = re.compile(r"\bprefetch(?:\.[A-Za-z0-9_]+)*\s+[^;]*;", re.M)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": Path(path).stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Runner:
    def __init__(self, out, timeout, manifest):
        self.out, self.timeout, self.manifest = out, timeout, manifest
        self.artifacts = set()

    def run(self, name, command):
        record = {"name": name, "argv": list(map(str, command)),
                  "timeout_seconds": self.timeout, "exit_code": None}
        self.manifest["commands"].append(record)
        stdout, stderr = self.out / f"{name}.stdout", self.out / f"{name}.stderr"
        self.artifacts.update((stdout, stderr))
        record.update(stdout=stdout.name, stderr=stderr.name)
        with stdout.open("wb") as so, stderr.open("wb") as se:
            try:
                result = subprocess.run(record["argv"], stdout=so, stderr=se,
                                        timeout=self.timeout, check=False)
                record["exit_code"] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                record["error"] = str(exc)
        return record["exit_code"] == 0


def tool_record(name, selected, runner):
    resolved = shutil.which(str(selected)) if selected else None
    rec = {"selected": selected, "resolved": resolved, "available": False}
    if resolved:
        try:
            rec.update(realpath=str(Path(resolved).resolve()), **digest(resolved))
            rec["available"] = runner.run(f"{name}-version", [resolved, "--version"])
        except OSError as exc:
            rec["error"] = str(exc)
    else:
        rec["error"] = "not selected or executable unavailable"
    return rec


def configuration(args, runner, manifest):
    """Use exactly the serving selector; NVCC/PATH are not fallback selectors."""
    tools = {"nvcc": None, "ptxas": None, "cuobjdump": None}
    config = {"architecture": args.architecture, "library": "e4m3",
              "environment": {k: os.environ.get(k) for k in
                              ("CUDA_HOME", "CUDA_PATH", "PYTORCH_NVCC", "NVCC", "PATH")}}
    try:
        import torch
        from tessera.serving import ext

        home = ext._resolve_cuda_home(torch)
        tools["nvcc"] = ext._nvcc_for_build()
        if home:
            for name in ("ptxas", "cuobjdump"):
                tools[name] = str(Path(home) / "bin" / name)
        config.update(cuda_home=home, torch_version=torch.__version__,
                      abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))
        try:
            config["includes"] = sass_dump._includes()
        except Exception as exc:
            config["includes_error"] = str(exc)
    except Exception as exc:
        config["selection_error"] = str(exc)
    # -ptx emits virtual code. Preserve the owning API's flags except its
    # final real-code target, which must become a virtual-code target here.
    flags = sass_dump._flags("e4m3", args.architecture)
    flags = [f.replace("code=sm_", "code=compute_") for f in flags]
    config["ptx_flags"] = flags
    manifest["config"] = config
    manifest["tools"] = {name: tool_record(name, selected, runner)
                         for name, selected in tools.items()}
    return config


def header_inventory(runner, side, common, source):
    """Hash the dependency list, not just a guessed roster of CUDA headers."""
    if not runner.run(f"{side}-dependencies", [*common, "-M", str(source)]):
        return {"complete": False, "error": "nvcc -M failed", "files": {}}
    text = (runner.out / f"{side}-dependencies.stdout").read_text(errors="replace")
    files, errors = {}, []
    try:
        # nvcc emits Make dependencies, including escaped spaces/continuations.
        dependencies = shlex.split(text.replace("\\\n", " ").split(":", 1)[1])
        for entry in dependencies:
            path = Path(entry).absolute()
            try:
                files[str(path)] = digest(path)
            except OSError as exc:
                errors.append(str(exc))
    except (IndexError, ValueError) as exc:
        errors.append(str(exc))
    if not files:
        errors.append("empty dependency list")
    return {"complete": not errors, "errors": errors, "files": files}


def excerpts(text):
    """Per raw kernel, retain cache-control neighborhoods (or its opening)."""
    result, name, lines = {}, None, []

    def finish():
        if name is None:
            return
        hits = [i for i, line in enumerate(lines) if re.search(r"\bCCTL\w*(?:\.|\b)", line)]
        selected = sorted({j for i in hits for j in range(max(0, i - 3), min(len(lines), i + 4))})
        if not hits:
            selected = list(range(min(8, len(lines))))
        result[name] = {"cache_control_lines": len(hits),
                        "lines": [{"line": i + 1, "text": lines[i]} for i in selected]}

    for line in text.splitlines():
        match = sass_dump._FUNC.match(line)
        if match:
            finish()
            name, lines = match.group(1), []
        if name is not None:
            lines.append(line)
    finish()
    return result


def retain_source(side, source, identity, out):
    original = digest(source)
    retained = out / side / source.name
    retained.parent.mkdir()
    shutil.copyfile(source, retained)
    if digest(retained) != original:
        raise RuntimeError(f"{side}: source changed while copying")
    return {"source": str(source), "identity": identity,
            "identity_basis": "caller assertion; not independently recovered",
            "retained_source": str(retained.relative_to(out)), **original}


def compile_side(side, source, args, runner, manifest):
    config, tools = manifest["config"], manifest["tools"]
    rec = manifest["sides"][side]
    original = {k: rec[k] for k in ("sha256", "bytes")}
    retained = runner.out / rec["retained_source"]
    if digest(source) != original:
        raise RuntimeError(f"{side}: source changed before compilation")
    common = [tools["nvcc"]["resolved"], "--verbose", *config["ptx_flags"],
              *config["includes"], f"-D_GLIBCXX_USE_CXX11_ABI={config['abi']}",
              "-DTORCH_EXTENSION_NAME=f16_prefetch_probe"]
    # Compile the original path so relative includes retain their exact meaning;
    # its retained copy and post-build hashes detect changes during the probe.
    rec["headers"] = header_inventory(runner, side, common, source)
    ptx, cubin = runner.out / f"{side}.ptx", runner.out / f"{side}.cubin"
    runner.artifacts.update((ptx, cubin))
    if not runner.run(f"{side}-nvcc", [*common, "-ptx", str(source), "-o", str(ptx)]):
        raise RuntimeError(f"{side}: nvcc failed (see command logs)")
    ptx_text = ptx.read_text(errors="replace")
    # Strip comments so a source annotation cannot count as an emitted hint.
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", " ", ptx_text, flags=re.S)
    rec["ptx_hint_regex"] = PTX_HINT.pattern
    rec["ptx_hints"] = dict(collections.Counter(m.group().split()[0] for m in PTX_HINT.finditer(clean)))
    rec["ptx_hint_count"] = sum(rec["ptx_hints"].values())
    if not runner.run(f"{side}-ptxas", [tools["ptxas"]["resolved"], "-v",
                      f"-arch={args.architecture}", str(ptx), "-o", str(cubin)]):
        raise RuntimeError(f"{side}: explicit ptxas failed (see command logs)")
    rec["resources"] = sass_dump.parse_ptxas(
        (runner.out / f"{side}-ptxas.stderr").read_text(errors="replace"))
    if not runner.run(f"{side}-sass", [tools["cuobjdump"]["resolved"], "-sass", str(cubin)]):
        raise RuntimeError(f"{side}: cuobjdump failed (see command logs)")
    text = (runner.out / f"{side}-sass.stdout").read_text(errors="replace")
    rec["analyzer"] = sass_dump.analyze_prefetch_sass(text, architecture=args.architecture)
    rec["excerpts"] = excerpts(text)
    kernels = {}
    for name, seq in sass_dump.parse_sass(text).items():
        canonical = sass_dump._kernel_name(name)
        if canonical in kernels:
            raise RuntimeError(f"canonical kernel collision: {canonical}")
        kernels[canonical] = {"raw_name": name, "instructions": seq}
    rec["kernels"] = kernels
    if not kernels:
        raise RuntimeError(f"{side}: disassembly contains no kernels")
    rec["source_unchanged"] = digest(source) == original == digest(retained)
    if not rec["source_unchanged"]:
        raise RuntimeError(f"{side}: source changed during compilation")


def compare(sides):
    off, on = (sides[s]["kernels"] for s in ("off", "on"))
    result = {}
    for name in sorted(off.keys() | on.keys()):
        if name not in off or name not in on:
            status = "only_on" if name not in off else "only_off"
        else:
            a, b = off[name]["instructions"], on[name]["instructions"]
            status = ("identical" if a == b else "multiset" if
                      collections.Counter(map(sass_dump._canon, a)) ==
                      collections.Counter(map(sass_dump._canon, b)) else "differs")
        result[name] = {"status": status,
                        **{s: {"raw_name": sides[s]["kernels"][name]["raw_name"],
                               "excerpt": sides[s]["excerpts"].get(sides[s]["kernels"][name]["raw_name"])}
                           for s in ("off", "on") if name in sides[s]["kernels"]}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off", type=Path)
    parser.add_argument("--on", type=Path)
    for side in ("off", "on"):
        parser.add_argument(f"--{side}-identity", choices=("recovered", "reconstruction"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"sm_[0-9]+a?", args.architecture):
        parser.error("--architecture must be an sm_NNN or sm_NNNa token")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.inventory_only and not all((args.off, args.on, args.off_identity, args.on_identity)):
        parser.error("paired mode requires --off, --on and both --off-identity/--on-identity")
    args.out = args.out.absolute()
    # A fresh directory prevents stale evidence or accidental source overwrite.
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "mode": "inventory" if args.inventory_only else "paired",
                "pipeline": "nvcc --verbose -ptx -> explicit selected-toolkit ptxas -> selected-toolkit cuobjdump",
                "gpu_execution": False, "commands": [], "sides": {},
                "probe": {"path": str(Path(__file__).resolve()), **digest(__file__)},
                "sass_dump": {"path": str(Path(sass_dump.__file__).resolve()), **digest(sass_dump.__file__)}}
    runner = Runner(args.out, args.timeout_seconds, manifest)
    code = 0
    try:
        if not args.inventory_only:
            for side in ("off", "on"):
                manifest["sides"][side] = retain_source(
                    side, getattr(args, side).resolve(), getattr(args, f"{side}_identity"), args.out)
            before, after = (Path(manifest["sides"][s]["retained_source"]) for s in ("off", "on"))
            diff = difflib.unified_diff((args.out / before).read_text(errors="replace").splitlines(True),
                                        (args.out / after).read_text(errors="replace").splitlines(True),
                                        fromfile="off", tofile="on")
            (args.out / "sources.diff").write_text("".join(diff))
            runner.artifacts.update((args.out / before, args.out / after, args.out / "sources.diff"))
        config = configuration(args, runner, manifest)
        if not args.inventory_only:
            if not all(t["available"] for t in manifest["tools"].values()) or "includes" not in config:
                raise RuntimeError("selected compile toolchain/includes unavailable; see inventory")
            for side in ("off", "on"):
                compile_side(side, getattr(args, side).resolve(), args, runner, manifest)
            manifest["comparison"] = compare(manifest["sides"])
            # Revalidate inputs after BOTH compiles, including shared headers.
            for rec in manifest["sides"].values():
                for path, expected in {rec["source"]: {k: rec[k] for k in ("sha256", "bytes")},
                                       **rec["headers"]["files"]}.items():
                    if digest(path) != expected:
                        raise RuntimeError(f"input changed during paired probe: {path}")
            if any(s["analyzer"]["status"] != "complete" for s in manifest["sides"].values()):
                raise RuntimeError("SASS analysis inconclusive; no authoritative hint comparison")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        print(str(exc), file=sys.stderr)
        code = 1
    finally:
        # Instruction streams are retained verbatim in raw SASS; do not duplicate
        # hundreds of thousands of parsed instructions in the decision manifest.
        for rec in manifest["sides"].values():
            for kernel in rec.get("kernels", {}).values():
                kernel.pop("instructions", None)
        manifest["artifacts"] = {str(p.relative_to(args.out)): digest(p)
                                 for p in sorted(runner.artifacts) if p.is_file()}
        manifest["artifact_policy"] = "all output files except manifest.json (no recursive self-hash)"
        write_json(args.out / "manifest.json", manifest)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
