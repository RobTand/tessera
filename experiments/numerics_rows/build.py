"""Build cell libraries and probes, then save their binary evidence."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.numerics_rows.common import IMAGES, KERNEL, check_runtime, check_source, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", choices=IMAGES, required=True)
    args = p.parse_args()
    commit = check_source(args.source)
    runtime = check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    from tessera.routed_fused import LIBRARIES, _cflags
    from torch.utils.cpp_extension import load

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for key in IMAGES[args.image]["libraries"]:
        name = "tessera_routed_fused_e2m1" if key == "e2m1" else LIBRARIES[key][0]
        flags = _cflags("sm_121", key in ("e4m3", "e4m3mma"), key == "e4m3mma", key == "e2m1")
        build = args.out / name
        build.mkdir()
        lib = load(name=name, sources=[str(args.source / KERNEL)], build_directory=str(build),
                   extra_cuda_cflags=flags, verbose=True)
        probe = args.out / f"swiglu_{key}"
        probe_source = Path(__file__).with_name("swiglu_probe.cu")
        subprocess.run(["nvcc", *flags, str(probe_source), "-o", str(probe)], check=True)
        binaries = {}
        for label, path in (("library", Path(lib.__file__)), ("probe", probe)):
            dumps = {}
            for kind in ("sass", "elf"):
                dump = args.out / f"{key}.{label}.{kind}"
                with dump.open("w") as stream:
                    subprocess.run(["cuobjdump", f"-{kind}", str(path)], stdout=stream, check=True)
                dumps[kind] = {"path": str(dump), "sha256": sha256(dump)}
            binaries[label] = {"path": str(path), "sha256": sha256(path), **dumps}
        records.append({"key": key, "name": name, "cflags": flags, **binaries,
                        "probe_source_sha256": sha256(probe_source)})
        write_json(args.out / "build.json", {"commit": commit, "runtime": runtime, "libraries": records})


if __name__ == "__main__":
    main()
