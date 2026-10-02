"""Single closed arm of experiment842; existing Store/intake/build/profile owners."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zlib

from routed_lut.owner import (BASELINE_SOURCE, CANDIDATE_SOURCE, OWNERS,
                              arm_descriptor, verify_imports)


def publish(path, doc):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(doc, indent=2) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--phase", choices=["build", "check", "time", "profile"], required=True)
    ap.add_argument("--input-manifest")
    ap.add_argument("--descriptor")
    ap.add_argument("--artifact", default="/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported")
    args = ap.parse_args()
    if not os.environ.get("PRISMABUILD_ACTION_KEY"):
        raise ValueError("PB admission required")
    if args.artifact != "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported":
        raise ValueError("paired experiment requires actual A8SE artifact")
    if args.phase == "build" and args.arm != "B":
        raise ValueError("only the candidate may be built")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    import tessera
    from tessera import routed_fused as rf
    from tessera.serving import ext
    from tessera.serving.backend import platform_token
    verify_imports("/work/src", args.arm, tessera, rf, ext)
    token = platform_token(torch=torch)
    flags = rf._cflags(token, True, True)
    if args.arm == "B": flags += ["-DTESSERA_ROUTED_FUSED_LUT_READONLY=1"]
    meta = {"arm": args.arm, "phase": args.phase, "kernel_sha256":
            BASELINE_SOURCE if args.arm == "A" else CANDIDATE_SOURCE,
            "owners": OWNERS, "compile_flags": flags,
            "actual_python_root": str(Path(tessera.__file__).resolve().parent),
            "image": os.environ.get("ORACLE_IMAGE"),
            "action_key": os.environ["PRISMABUILD_ACTION_KEY"],
            "host": os.environ.get("HOST_NAME"), "start_unix": time.time(),
            "energy_status": "HOLD", "scope": "historical routing proxy, not VB1770 capture"}
    if args.phase == "build":
        original = rf._cflags
        def experimental_flags(token, fp8, mma8=False, fp4=False):
            result = original(token, fp8, mma8, fp4)
            if (fp8, mma8, fp4) == (True, True, False):
                result += ["-DTESSERA_ROUTED_FUSED_LUT_READONLY=1"]
            return result
        rf._cflags = experimental_flags
        try:
            library = rf._ext("e4m3mma")
            raw = Path(library.__file__).read_bytes()
            retained = out / "tessera_routed_fused_mma_e4m3.so"
            with retained.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            meta.update(binary_sha256=hashlib.sha256(raw).hexdigest(), binary_bytes=len(raw))
            verify_imports("/work/src", args.arm, tessera, rf, ext)
            publish(out / "build.json", meta)
            return 0
        finally: rf._cflags = original
    if not args.input_manifest or not args.descriptor:
        raise ValueError("closed paired readset/descriptor required")
    from pb_staged_store import StagedInputs, NativeCallback
    import bench_t8r as bench
    if bench.VLLM_STUBBED:
        raise ValueError("paired experiment refuses stubbed vLLM")
    reader = StagedInputs(args.input_manifest)
    native = None
    ctx = None
    try:
        descriptor = arm_descriptor(reader.read(args.descriptor), args.arm, flags)
        native = NativeCallback(reader, descriptor["binary"], rf,
                                out / "native-artifact",
                                expected_sha256=descriptor["binary_sha256"],
                                source_sha256=descriptor["kernel_sha256"])
        artifact = "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"
        module = "model.language_model.layers.10.mlp.experts"
        store = bench.Store(artifact, reader)
        reader.bind_roles(store.root, store.metadata("tessera_serving_manifest.json")["modules"][module]["roles"])
        ctx = bench._init_vllm_world1(str(out))
        with torch.inference_mode():
            fn, info, holder, bytes_for = bench.build_routed(store, module)
            native.bind(rf._ext("e4m3mma"))
            routing = "/mnt/shared/tessera-measurements/t8r-speed-20260929/prefill-routing-20260930/m2048/ids-414-000007.pt"
            ids = torch.load(io.BytesIO(reader.read(routing)), weights_only=True, map_location="cpu")["ids"]
            if ids.dtype != torch.int32 or tuple(ids.shape) != (2048, 8) or ids.min() < 0 or ids.max() >= 288:
                raise ValueError("historical routing geometry differs")
            torch.manual_seed(zlib.crc32(b"experts.R1024.L10:2048"))
            x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
            weights = torch.full((2048, 8), 1.0 / 8, dtype=torch.float32, device="cuda")
            ids = ids.to("cuda")
            meta["input_hashes"] = {name: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                                    for name, t in [("x", x), ("ids", ids), ("weights", weights)]}
            if meta["input_hashes"] != {
                "x": "03f453766116ebadd8978354bccb8a8eb81bc22471bcb306bc23b1b5d918f626",
                "ids": "65fb4b0e7355fa1fc2f1e79a29a6f44bf31d6109b05b4ce0ead0c2452d9d076a",
                "weights": "3ce58fa5a4bf3a9df60ce5937eadc4d569a503a01719de8c355f322f4304f214"}:
                raise ValueError("paired experiment inputs differ from retained real proxy")
            def call(): return fn(x, ids, weights)
            if args.phase == "check":
                cases = [("real", x, ids, weights)]
                cases += [(f"prefix-{n}", x[:n], ids[:n], weights[:n]) for n in (1, 127, 129, 2047)]
                cases.append(("single-expert-tail-129", x[:129], torch.full_like(ids[:129], int(ids[0, 0])), weights[:129]))
                witnesses = {}
                for name, a, i, w in cases:
                    y = fn(a, i, w); torch.cuda.synchronize()
                    if not torch.isfinite(y).all() or not torch.count_nonzero(y):
                        raise ValueError("vacuous or nonfinite output witness")
                    witnesses[name] = {"shape": list(y.shape), "sha256": hashlib.sha256(y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
                meta["output_witnesses"] = witnesses
            elif args.phase == "time":
                meta["wall"] = bench.summarize(bench.time_events(call, 10, 30))
                meta["torch_profile"] = bench.kernel_profile(call, full_names=True)
                sampler = bench.PowerSampler()
                meta["steady"] = sampler.sample_during(call, 30, capture_series=True)
                if meta["steady"].get("source") is None or meta["steady"]["seconds"] < 30:
                    raise ValueError("steady fast-power measurement missing")
            else:
                for _ in range(10): call()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                call(); torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            native.finish(torch.cuda.synchronize)
            verify_imports("/work/src", args.arm, tessera, rf, ext)
            meta.update(native=native.record, staged_reads=reader.reads, info=info, end_unix=time.time())
            publish(out / "arm.json", meta)
        return 0
    finally:
        try:
            if native: native.finish(torch.cuda.synchronize)
        finally:
            reader.close()
            if ctx: ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())
