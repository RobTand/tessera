"""Seeded OFF/OFF restart control and conditional single-lever investigation.

This is correctness-only evidence, not the existing 33-output ship comparison,
performance qualification, token-ID equality, or teacher-forced quality.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

from managed_window import Refused, atomic_json

MODE = "investigate-eager-determinism-2048"
NAMES = ("control_off_first", "control_off_restart", "control_decode_once", "control_kda_split", "control_piece_major")
FLAGS = ("TESSERA_E4M3_DECODE_ONCE", "TESSERA_GLM53_KDA_CONV_SPLIT", "TESSERA_ROUTED_PIECE_MAJOR")
VALUES = (("0", "off", "0"), ("0", "off", "0"), ("1", "off", "0"), ("0", "on", "0"), ("0", "off", "1"))
PROTOCOL = dict(schema="tessera.eager_determinism_protocol.v1", lens=[2048], conc=[1],
                trials=10, warmup=1, output=128, temperature=0.0, ignore_eos=True,
                server_seed=0, request_seed_base=0, stream=True,
                stream_options=dict(include_usage=True, continuous_usage_stats=True),
                max_batched=4096, mtp_tokens=1, draft_tensor_parallel_size=2)


def prompt_population(prompts):
    try:
        rows = prompts["prompts"]["2048"]["1"][:11]
        if prompts["warmup"] != 1 or len(rows) != 11:
            raise ValueError("requires one warmup plus ten timed trials")
        if any(len(row) != 1 or len(row[0]) != 2048 or
               any(type(token) is not int or token < 0 for token in row[0]) for row in rows):
            raise ValueError("requires L2048 token-ID prompts at concurrency one")
        return [row[0] for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused(f"seeded control prompt population differs: {exc}") from exc


def stock_instrument():
    from eager_benchmark import CLIENT
    spec = importlib.util.spec_from_file_location("seeded_control_stock_generation", CLIENT / "u4_speed_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_population(result, prompts, prompt_sha256):
    """Validate and retain every full completion, seed, usage and trial identity."""
    from eager_benchmark import MODEL
    from seeded_control_client import SCHEMA
    population = prompt_population(prompts)
    if (result.get("schema") != SCHEMA or result.get("protocol") != PROTOCOL or
            result.get("prompts_sha256") != prompt_sha256 or result.get("model") != MODEL or
            len(result.get("requests", [])) != 11):
        raise Refused("seeded control output protocol/model/input population differs")
    outputs = {}
    instrument = stock_instrument()
    try:
        for trial, (record, prompt) in enumerate(zip(result["requests"], population)):
            request_id = f"L2048-c1/trial{trial}/slot0"
            seed = PROTOCOL["request_seed_base"] + trial
            expected_payload = dict(model=MODEL, prompt=prompt, max_tokens=128, temperature=0.0,
                                    seed=seed, ignore_eos=True, stream=True,
                                    stream_options=PROTOCOL["stream_options"])
            if (record.get("trial") != trial or record.get("slot") != 0 or
                    record.get("warmup") is not (trial == 0) or record.get("request_id") != request_id or
                    record.get("request_seed") != seed or record.get("request_payload") != expected_payload):
                raise ValueError("request seed/payload/trial identity differs")
            text, finish = instrument.generation_value(record, prompt, 128)
            value = dict(text=text, finish_reason=finish, usage=record["usage"],
                         prompt_sha256=record["prompt_sha256"], request_seed=seed)
            # Preserve optional IDs/log probabilities if the stock API emitted them;
            # no new request option or inferred token/logit equality is introduced.
            optional = [{key: choice[key] for key in ("token_ids", "logprobs") if choice.get(key) is not None}
                        for choice in record["generation"]["choices"]]
            if any(optional):
                value["optional_api_outputs"] = [row for row in optional if row]
            value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                       separators=(",", ":")).encode()).hexdigest()
            outputs[request_id] = value
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise Refused(f"seeded control incomplete or unbound generated output: {exc}") from exc
    return dict(schema="tessera.eager_determinism_outputs.v1", protocol=PROTOCOL,
                prompts_sha256=prompt_sha256, model=MODEL, requests_per_arm=11, outputs=outputs,
                equality="Full decoded UTF-8 text, length finish and usage for every warmup/timed request; optional API outputs retained",
                not_claimed="Effective internal seeds, unreturned token IDs/logits, quality, speed or adoption")


def compare(reference, candidate):
    if ({key: value for key, value in reference.items() if key != "outputs"} !=
            {key: value for key, value in candidate.items() if key != "outputs"}):
        raise Refused("seeded control paired protocol/seeds/model/input scope differs")
    if set(reference["outputs"]) != set(candidate["outputs"]) or len(reference["outputs"]) != 11:
        raise Refused("seeded control paired full output population differs")
    differences = [key for key in reference["outputs"] if reference["outputs"][key] != candidate["outputs"][key]]
    return dict(schema="tessera.eager_determinism_comparison.v1", passed=not differences,
                requests_per_arm=11, differing_requests=differences,
                equality=reference["equality"], not_claimed=reference["not_claimed"])


def stored_outputs(root, name):
    """Bind normalized evidence to its own full raw request/output records."""
    from eager_benchmark import sha
    directory = root / "arms" / name
    binding = json.loads((directory / "invocation.json").read_bytes())
    prompts_path = Path(binding["source_bindings"]["prompts"])
    outputs = output_population(json.loads((directory / "control.json").read_bytes()),
                                json.loads(prompts_path.read_bytes()), sha(prompts_path))
    if outputs != json.loads((directory / "output-hashes.json").read_bytes()):
        raise Refused("seeded control normalized outputs differ from their own raw records")
    return outputs


def require_deterministic_off(root):
    """Read real output records anew; a boolean label cannot authorize bisection."""
    reference = stored_outputs(root, NAMES[0])
    restart = stored_outputs(root, NAMES[1])
    comparison = compare(reference, restart)
    if not comparison["passed"]:
        raise Refused("SERVING NONDETERMINISM: seeded OFF/OFF full outputs differ; no lever bisection")
    return reference


def probes(adapter, arm, peer):
    from eager_benchmark import BASE, MODEL, sha
    name, config = arm["arm"], adapter.config
    if name in NAMES[2:]:
        require_deterministic_off(adapter.rdv)
    out = adapter.rdv / "arms" / name
    out.mkdir(parents=True, exist_ok=True)
    invocation = uuid.uuid4().hex
    argv = [sys.executable, str(Path(__file__).with_name("seeded_control_client.py")),
            "--base-url", BASE, "--model", MODEL, "--prompts", config["prompts"],
            "--out", str(out / "control.json"), "--server-seed", "0", "--request-seed-base", "0"]
    binding = dict(schema="tessera.eager_determinism_invocation.v1", invocation=invocation,
                   protocol=PROTOCOL, source_bindings=config, arm=arm, identity=adapter.identity,
                   peer=peer, client_argv=argv, server_seed_scope="Both rank command lines explicitly request --seed 0; internal RNG state is not observed",
                   restart_scope="Each arm starts fresh rank containers after both previous ranks acknowledge owned cleanup")
    atomic_json(out / "invocation.json", binding)
    with (out / "client.log").open("w") as stream:
        adapter.command(argv, stdout=stream, tick=adapter.tick, limit=adapter.envelope.remaining())
    prompt_path = Path(config["prompts"])
    outputs = output_population(json.loads((out / "control.json").read_bytes()),
                                json.loads(prompt_path.read_bytes()), sha(prompt_path))
    atomic_json(out / "output-hashes.json", outputs)
    if name != NAMES[0]:
        reference = stored_outputs(adapter.rdv, NAMES[0])
        comparison = compare(reference, outputs)
        comparison["classification"] = ("DETERMINISTIC OFF CONTROL" if comparison["passed"] else
                                         "SERVING NONDETERMINISM") if name == NAMES[1] else (
                                         "SINGLE LEVER MATCH" if comparison["passed"] else "SINGLE LEVER OUTPUT DIFFERENCE")
        atomic_json(out / "output-comparison.json", comparison)
        if name == NAMES[1] and not comparison["passed"]:
            raise Refused("SERVING NONDETERMINISM: seeded OFF/OFF full outputs differ; no lever bisection")
        if name == NAMES[-1]:
            singles = {candidate: json.loads((adapter.rdv / "arms" / candidate / "output-comparison.json").read_bytes())
                       for candidate in NAMES[2:]}
            atomic_json(adapter.rdv / "investigation-summary.json", dict(
                schema="tessera.eager_determinism_summary.v1", off_deterministic=True, singles=singles,
                conclusion=("All single levers match OFF; the earlier combined failure remains an unexplained interaction candidate"
                            if all(row["passed"] for row in singles.values()) else
                            "At least one single lever changes outputs; root cause is not established"),
                adoption=False, performance_claim=False))
    binding["control_sha256"] = sha(out / "control.json")
    atomic_json(out / "invocation.json", binding)
    return dict(invocation=invocation, correctness_only=True)


def input_preflight(config):
    """D38 on the real driver: actual input geometry, imports and bounded reads."""
    from eager_benchmark import PANEL
    from seeded_control_client import parser
    prompts_path = PANEL / "prompts.json"
    prompts = prompt_population(json.loads(prompts_path.read_bytes()))
    stock_instrument()  # imports the actual frozen generation validator
    parser().parse_args(["--base-url", "http://127.0.0.1:8142", "--model", "glm53-artifact",
                         "--prompts", str(prompts_path), "--out", "/tmp/control-dry.json",
                         "--server-seed", "0", "--request-seed-base", "0"])
    manifest = json.loads(Path(config["artifact_manifest"]).read_bytes())
    artifact = Path(config["artifact"])
    names = [entry["name"] for entry in manifest]
    if len(names) != 128 or len(set(names)) != 128 or names != sorted(names):
        raise Refused("seeded control artifact manifest roster differs")
    reads = []
    for entry in manifest:
        path = artifact / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise Refused(f"seeded control artifact byte length differs: {entry['name']}")
        with path.open("rb") as stream:
            sample = stream.read(64)
        reads.append(dict(name=entry["name"], sample_bytes=len(sample)))
    model_config = json.loads((artifact / "config.json").read_bytes())
    contract = json.loads((Path(config["ts"]) / "src/tessera/serving/runtime_contract.json").read_bytes())
    return dict(protocol=PROTOCOL, requests_per_arm=len(prompts), input_reads=reads,
                model_type=model_config.get("model_type"), runtime_contract_schema=contract.get("schema"),
                scope="CPU arguments/imports/input shapes and bounded real reads only; no GPU or served parity")
