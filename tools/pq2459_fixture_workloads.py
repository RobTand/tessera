#!/usr/bin/env python3
"""Execute the authorized eager scorer or fresh speed replay on real inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_panel_inputs(panel_path, arrays_root):
    import numpy as np
    raw = panel_path.read_bytes()
    panel = json.loads(raw)
    inputs = []
    receipts = []
    for row in panel["windows"]:
        path = arrays_root / Path(row["tokens_path"]).name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != row["tokens_bytes"] or digest != row["tokens_sha256"]:
            raise ValueError(f"sealed token input changed: {path}")
        tokens = np.load(io.BytesIO(data), allow_pickle=False)
        if tokens.shape != (panel["context_length"],) or tokens.dtype.kind not in "iu":
            raise ValueError(f"invalid token input geometry: {path}")
        if np.any(tokens < 0) or np.any(tokens >= panel["vocab_size"]):
            raise ValueError(f"token input exceeds the vocabulary: {path}")
        inputs.append(tokens.tolist())
        receipts.append({"path": str(path), "bytes": len(data), "sha256": digest})
    return panel, inputs, {"panel": str(panel_path), "panel_sha256": hashlib.sha256(raw).hexdigest(),
                           "arrays": receipts}


def speed_cases(profile):
    cases = [{"length": length, "concurrency": concurrency,
              "admissions": [concurrency], "label": f"L{length}-c{concurrency}"}
             for length, concurrency in product(profile["prompt_lengths"], profile["concurrency"])]
    # Four fresh L512 requests enter in stages. Two active decode sequences plus
    # two new prefills yield M1026. One decode plus three prefills yields M1537.
    # The first [2, 2] stage also exercises M1024 and later isolated M2 decode.
    cases += [{"length": 512, "concurrency": 4, "admissions": [2, 2], "label": "L512-c4-stagger-2-2"},
              {"length": 512, "concurrency": 4, "admissions": [1, 3], "label": "L512-c4-stagger-1-3"}]
    return cases


def engine_kwargs(packet, profile, model, topology):
    allowed = ("max_num_seqs", "max_model_len", "max_num_batched_tokens",
               "enable_chunked_prefill", "enable_prefix_caching")
    return {"model": str(model), "enforce_eager": True, "dtype": "bfloat16",
            "language_model_only": True, "trust_remote_code": True,
            "kv_cache_dtype": "fp8_ds_mla", "moe_backend": "triton",
            "kernel_config": {"enable_flashinfer_autotune": False},
            **{name: profile[name] for name in allowed}, **topology}

def require_fixture_runtime(packet):
    import torch
    import vllm
    from tessera.serving.runtime_image import declared_reference
    declaration = declared_reference(packet["runtime"]["image"])
    if str(torch.__version__) != packet["runtime"]["torch"] or vllm.__version__ != packet["runtime"]["vllm"]:
        raise ValueError("runtime versions are outside the fixture authorization")
    return declaration



def trace_from_worker(worker):
    from tessera.serving.telemetry import route_trace_snapshot
    return route_trace_snapshot()


def speed_replay(args, packet, inputs, profile, topology):
    from vllm import LLM, SamplingParams
    import tessera.serving
    from tessera.serving.telemetry import ROUTE_TRACE_ENV
    os.environ["TESSERA_SERVE_MODE"] = "resident"
    os.environ[ROUTE_TRACE_ENV] = str(args.output.with_suffix(".route.json"))
    # This runner observes route traces but does not qualify them. PQ #2459 owns
    # source, rank, image, route, geometry, and numerical qualification verdicts.
    runtime_declaration = require_fixture_runtime(packet)
    llm = LLM(**engine_kwargs(packet, profile, args.model, topology))
    if llm.llm_engine.vllm_config.speculative_config is not None:
        raise ValueError("speculative decode is outside the fixture authorization")
    flattened = [token for window in inputs for token in window]
    required = 4 * max(profile["prompt_lengths"])
    if len(flattened) < required:
        raise ValueError("the sealed token input cannot fill four distinct prompts")
    engine = llm.llm_engine
    trials = []
    for index, case in enumerate(speed_cases(profile)):
        active = 0
        outputs = {}
        prompt_counts = {}
        for admission in case["admissions"]:
            for _ in range(admission):
                request = f"fixture-{index}-{active}"
                start = active * max(profile["prompt_lengths"])
                tokens = flattened[start:start + case["length"]]
                engine.add_request(request, {"prompt_token_ids": tokens},
                                   SamplingParams(max_tokens=32, temperature=0.0,
                                                  ignore_eos=True, detokenize=False))
                prompt_counts[request] = len(tokens)
                active += 1
            for output in engine.step():
                outputs[output.request_id] = output
        while engine.has_unfinished_requests():
            for output in engine.step():
                outputs[output.request_id] = output
        if active != case["concurrency"] or set(outputs) != set(prompt_counts):
            raise ValueError("fresh replay lacks a declared request")
        if any(not output.finished or len(output.outputs[0].token_ids) != 32
               for output in outputs.values()):
            raise ValueError("fresh eager replay lacks complete decoded outputs")
        trials.append({**case, "prompt_tokens": prompt_counts,
                       "generated_token_ids": {name: output.outputs[0].token_ids
                                               for name, output in outputs.items()}})
    traces = llm.collective_rpc(trace_from_worker)
    return {"profile": args.profile, "engine_kwargs": engine_kwargs(packet, profile, args.model, topology),
            "trials": trials, "route_traces": traces, "runtime_declaration": runtime_declaration,
            "qualified_cells": 0,
            "qualification_owner": "prismaquant#2459"}


def scorer_command(args, packet, topology):
    if args.prismaquant_source is None:
        raise ValueError("tr3_batch requires the PrismaQuant source package")
    script = args.prismaquant_source / "experiments/measure_glm_tr3_vllm.py"
    if not script.is_file():
        raise ValueError("the source package lacks the source-owned TR3 scorer")
    command = [sys.executable, str(script), "--model", str(args.model),
               "--candidate-digest-cache", str(args.output.with_suffix(".checkpoint-digests.json")),
               "--panel", str(args.panel), "--arrays-root", str(args.arrays_root),
               "--teacher", str(args.teacher), "--teacher-sha256", args.teacher_sha256,
               "--serve-image", packet["runtime"]["image"], "--output", str(args.output),
               "--execution-mode", "eager", "--logits-layout", "vllm_v2_chunk1024",
               "--kv-cache-dtype", "fp8_ds_mla", "--expected-kv-cache-dtype", "fp8_ds_mla",
               "--kernel-config", '{"enable_flashinfer_autotune":false}',
               "--moe-backend", "triton", "--qualify-then-score", str(args.output.with_suffix(".hook.json"))]
    for name, value in topology.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    return command


def main(argv=None):
    os.environ["TESSERA_SERVE_MODE"] = "resident"
    from tools.pq2459_source import add_source_arguments, activate_source
    source_parser = argparse.ArgumentParser(add_help=False)
    add_source_arguments(source_parser)
    source_args, _unknown = source_parser.parse_known_args(argv)
    source_info = activate_source(source_args.source_archive, source_args.source_archive_sha256)
    from tessera.serving.topology import add_topology_arguments, topology_kwargs
    parser = argparse.ArgumentParser(description=__doc__, parents=[source_parser])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("tr3_batch", "speed_batch", "speed_decode"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--arrays-root", required=True, type=Path)
    parser.add_argument("--scorer-bundle", type=Path)
    parser.add_argument("--scorer-bundle-sha256")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-preflight", action="store_true")
    add_topology_arguments(parser)
    args = parser.parse_args(argv)
    topology = topology_kwargs(args)
    if topology["tensor_parallel_size"] != 2:
        parser.error("the authorized fixtures require TP2")
    packet = json.loads(args.packet.read_bytes())
    if not any(Path(row["path"]).resolve() == args.model.resolve()
               for row in packet["artifacts"].values()):
        parser.error("model is not a published additive fixture")
    profile = packet["profiles"][args.profile]
    panel, inputs, token_receipts = read_panel_inputs(args.panel, args.arrays_root)
    if args.profile == "tr3_batch":
        if args.teacher is None or args.teacher_sha256 is None:
            parser.error("tr3_batch requires the sealed teacher and its digest")
        if args.scorer_bundle is None or args.scorer_bundle_sha256 is None:
            parser.error("tr3_batch requires the immutable scorer Git bundle")
        from tools.pq2459_scorer_source import prepare_scorer_source
        scorer_source = prepare_scorer_source(args.scorer_bundle, args.scorer_bundle_sha256)
        args.prismaquant_source = Path(scorer_source["root"])
        if hashlib.sha256(args.teacher.read_bytes()).hexdigest() != args.teacher_sha256:
            parser.error("teacher manifest bytes changed")
        command = scorer_command(args, packet, topology)
    else:
        command = None
    if args.cpu_preflight:
        import torch
        modules = [importlib.import_module(name) for name in
                   ("numpy", "safetensors", "tessera.export_serving", "tessera.serving.scheme",
                    "tessera.serving.source_identity", "tessera.serving.topology")]
        if command is not None:
            # Exercise the real scorer parser and its imports. No engine starts.
            subprocess.run([command[0], command[1], "--help"], check=True)
        print(json.dumps({"profile": args.profile, "token_inputs": token_receipts,
                          "dependencies": [module.__file__ for module in modules],
                          "engine_kwargs": engine_kwargs(packet, profile, args.model, topology),
                          "cases": speed_cases(profile) if command is None else command,
                          "qualified_cells": 0,
                          "cuda_initialized": torch.cuda.is_initialized()}, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output exists; retain the previous evidence")
    if command is not None:
        require_fixture_runtime(packet)
        os.environ["TESSERA_SERVE_MODE"] = "resident"
        return subprocess.run(command, check=False).returncode
    result = speed_replay(args, packet, inputs, profile, topology)
    result["token_inputs"] = token_receipts
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
