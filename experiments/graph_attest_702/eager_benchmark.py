"""Named A8S pairs and unchanged October 5 served instruments.

This is an adapter for the existing admitted rank lifecycle, not a launcher.
The stock profile instrument's SSH is read-only trace inspection, never a rank.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

from managed_window import Refused, atomic_json
SHARED_ROOT = Path("/mnt/shared")
PQ_PIN_COMMIT = "e36e60b77b3d2ab0c5265272515958a0cb67d32b"

MODE = "window4-eager-2048-4096"
SHIP_MODE = "ship-eager-4096-8192"
GRAPH_SHIP_MODE = "ship-graph-2048"
EAGER_LEVER_MODE = "ship-eager-levers-4096"
LEVER_VALUES = {"TESSERA_E4M3_DECODE_ONCE": ("0", "1"),
                "TESSERA_GLM53_KDA_CONV_SPLIT": ("off", "on"),
                "TESSERA_ROUTED_PIECE_MAJOR": ("0", "1")}
PAIRS = {MODE: [("eager2048", "2048"), ("eager4096", "4096")],
         SHIP_MODE: [("eager4096", "4096"), ("eager8192", "8192")],
         GRAPH_SHIP_MODE: [("graph2048_off", "2048"), ("graph2048_on", "2048")],
         EAGER_LEVER_MODE: [("eager4096_off", "4096"), ("eager4096_on", "4096")]}

RUNTIME_COMMIT = "2dbac1910c88254d9c6391f02a34c4b07e516803"
CONTRACT_SHA = "47f180efaf97faa5c411df5d48f9da7dff4b9c9fc0c3ddbf9f815bcd4d0aed78"
ARTIFACT_SHA = "45407d43e09381b73197498d7f37c848c167c03a83d415c68e63c615d99eb840"
CLIENT = Path("/mnt/shared/tessera-runs/exl3-preflight/stock-client-source-profile-receipt-v2-20261005")
PANEL = Path("/mnt/shared/tessera-runs/exl3-preflight/rdv-validated-9ccccd9945214c309b28bb0ddb0c860a/inputs")
SOURCE_IDENTITY_SHA = "af4ec4bd8947e9d7398b1fd7099ff383b4ecfc3c40a1e111e7a2008d80e391f3"
TIMING_SHA = "f55657078191f51a22040ab936b6dbf29679e0c11d73bfefc408f3d9aeb9824b"
PROFILE_SHA = "6ae3087f117d9e0341846e499ba19eeb21e10d1e2b6e672e5ffd6ed5f644905f"
PROMPTS_SHA = "8cd21019b03ffe1875704441878acf3bbd8546177036e2734ab5cfa85f1a4cb8"
MANIFEST_SHA = "7410e55b8696c566cf47a98ddc394537c5fcadfed559c91ff0c3526c135d7cea"
POWER_SHA = "e56e704d671bd0e629ddfbf9374de122009cc454f59f011371c59119aef3bb7c"
MODEL = "glm53-artifact"
BASE = "http://10.100.96.2:8142"

def pair_refusal(mode):
    if mode == MODE:
        return "Window4 is exactly eager2048/2048 then eager4096/4096"
    if mode == SHIP_MODE:
        return "Ship window is exactly eager4096/4096 then eager8192/8192"
    if mode == GRAPH_SHIP_MODE:
        return "Ship graph window is exactly graph2048_off/2048 then graph2048_on/2048"
    if mode == EAGER_LEVER_MODE:
        return "Ship eager lever window is exactly eager4096_off/4096 then eager4096_on/4096"
    return "unknown WINDOW_MODE; no inferred benchmark scope"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(compute):
    """The actual side of a run-identity seal (D32): no digest in dev mode.

    A digest over existing data computed only to satisfy an identity comparison
    is sealing; dev mode stamps ``NOT_COMPUTED`` instead of computing one.
    Certified mode (``PRISMAQUANT_DEV_MODE=0``) computes and compares for real.
    """
    from tessera.dev_mode import NOT_COMPUTED, dev_mode_enabled
    return NOT_COMPUTED if dev_mode_enabled() else compute()


def bindings(env, artifact):
    # Function-local import: the simulated rank child (rank_window -> tp2_recipe
    # -> this module) runs without the repo's src on sys.path, and never seals.
    from tessera.dev_mode import NOT_COMPUTED, dev_mode_enabled, seal_check
    # Run-identity seals go through tessera.dev_mode.seal_check (D32): certified
    # mode refuses with the original message, dev mode stamps and continues with
    # the stored bindings returned below. Damage to owned data -- the manifest
    # structure/roster, the loaded file population and lengths, metadata bytes
    # against the manifest's own digests, and source_identity members against
    # their recorded hashes -- is integrity, not sealing, and refuses in both
    # modes.
    seal_check("source commit", RUNTIME_COMMIT, env.get("SOURCE_COMMIT"), where="Window4 runtime",
               refusal=lambda: Refused("Window4 requires the qualified public runtime 2dbac191, not its producer"))
    contract = Path(env["TS"]) / "src/tessera/serving/runtime_contract.json"
    seal_check("runtime contract sha256", CONTRACT_SHA, _identity(lambda: sha(contract)),
               where="Window4 runtime contract",
               refusal=lambda: Refused("Window4 raw runtime contract differs from v56/47f180ef"))
    seal_check("PQ pin commit", PQ_PIN_COMMIT, env.get("PQ_PIN_COMMIT"), where="Window4 PQ pin",
               refusal=lambda: Refused("Window4 requires the exact approved corrected PQ e36e60b7 pin commit"))
    power = Path(__file__).resolve().parents[1] / "box_power_window.py"
    seal_check("power program sha256", POWER_SHA, _identity(lambda: sha(power)),
               where="Window4 power instrument",
               refusal=lambda: Refused("Window4 existing power instrument bytes changed"))
    manifest_path = Path(env.get("ARTIFACT_MANIFEST", ""))
    if not manifest_path.is_file() or not manifest_path.is_relative_to(SHARED_ROOT):
        raise Refused("Window4 requires the supplied complete A8S manifest on shared storage")
    entries = json.loads(manifest_path.read_bytes())
    # The historical ARTIFACT_SHA is the artifact run's identity seal, not this
    # manifest's integrity; the roster/length/metadata checks below own that.
    seal_check("artifact manifest sha256", ARTIFACT_SHA,
               _identity(lambda: hashlib.sha256(json.dumps(
                   entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode() + b"\n").hexdigest()),
               where="Window4 artifact manifest",
               refusal=lambda: Refused("Window4 complete A8S content manifest digest differs"))
    names = [entry["name"] for entry in entries]
    if len(entries) != 128 or names != sorted(names) or len(set(names)) != 128:
        raise Refused("Window4 complete A8S manifest roster differs")
    actual = sorted(str(p.relative_to(artifact)) for p in artifact.rglob("*") if p.is_file())
    if actual != names:
        raise Refused("Window4 loaded A8S file population differs from authenticated manifest")
    for entry in entries:
        path = artifact / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise Refused(f"Window4 A8S byte length changed: {entry['name']}")
        # Campaign's authenticated full-body audit is retained, not a new 175GB rehash.
        if not entry["name"].endswith(".safetensors") and sha(path) != entry["sha256"]:
            raise Refused(f"Window4 A8S metadata changed: {entry['name']}")
    instruments = {CLIENT / "source_identity.json": SOURCE_IDENTITY_SHA,
                   CLIENT / "u4_speed_client.py": TIMING_SHA, CLIENT / "comparison_inputs.py": PROFILE_SHA,
                   PANEL / "prompts.json": PROMPTS_SHA, PANEL / "manifest-decode.json": MANIFEST_SHA}
    for path, digest in instruments.items():
        seal_check("October 5 instrument/input sha256", digest, _identity(lambda path=path: sha(path)),
                   where=str(path),
                   refusal=lambda path=path: Refused(f"Window4 October 5 instrument/input bytes differ: {path}"))
    for name, digest in json.loads((CLIENT / "source_identity.json").read_bytes())["files"].items():
        if sha(CLIENT / name) != digest:
            raise Refused(f"Window4 EXL3 source identity member changed: {name}")
    return dict(window_mode=env.get("WINDOW_MODE", MODE), artifact_manifest=str(manifest_path), artifact_content_sha256=ARTIFACT_SHA,
                artifact_bytes=sum(entry["bytes"] for entry in entries), artifact_files=len(entries),
                artifact_authentication="Campaign full-body hashes retained; current roster/lengths and metadata checked",
                client_source=str(CLIENT), client_identity_sha256=SOURCE_IDENTITY_SHA,
                timing_program_sha256=TIMING_SHA, profile_program_sha256=PROFILE_SHA,
                prompts=str(PANEL / "prompts.json"), prompts_sha256=PROMPTS_SHA,
                profile_manifest=str(PANEL / "manifest-decode.json"), profile_manifest_sha256=MANIFEST_SHA,
                profile_dir=str(Path(env["RECEIPTS"]).parent / "profiles"),
                runtime_contract_sha256=CONTRACT_SHA, pq_pin_commit=PQ_PIN_COMMIT, power_program_sha256=POWER_SHA)


def require_timing(result):
    for length in (512, 2048, 8192):
        cell = result["cells"][f"host-L{length}-c1"]
        if cell.get("skipped") or not cell.get("complete") or len(cell["trials"]) != 10:
            raise Refused(f"Window4 incomplete exact timing population at L{length}")
        for trial, row in enumerate(cell["trials"], start=1):
            if row["trial"] != trial or len(row["requests"]) != 1:
                raise Refused("Window4 timing trial/concurrency differs")
            request = row["requests"][0]
            usage = request.get("usage") or {}
            if (request.get("error") or usage.get("prompt_tokens") != length or usage.get("completion_tokens") != 128
                    or not request["generation"]["done"] or request["completion_tokens"] != 128):
                raise Refused("Window4 timing request counts/stream incomplete")

def output_hashes(result, prompts, prompt_sha256):
    """All 33 deterministic decoded-text/finish outputs, not token IDs or KL."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eager_generation_instrument", CLIENT / "u4_speed_client.py")
    instrument = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instrument)
    geometry = dict(lens=[512, 2048, 8192], conc=[1], trials=10, warmup=1,
                    output=128, temperature=0.0, ignore_eos=True)
    labels = [f"host-L{length}-c1" for length in geometry["lens"]]
    if (result.get("prompts_sha256") != prompt_sha256 or result.get("model") != MODEL
            or result.get("labels", {}).get("mode") != "eager"
            or result.get("labels", {}).get("fabric") != "socket"
            or any(result.get("config", {}).get(key) != value for key, value in geometry.items())
            or set(result.get("cells", {})) != set(labels) or prompts.get("warmup") != 1):
        raise Refused("Ship eager lever output population/model/input geometry differs")
    hashes = {}
    try:
        for length, label in zip(geometry["lens"], labels):
            cell = result["cells"][label]
            if cell.get("skipped") is not False or cell.get("complete") is not True:
                raise ValueError("incomplete output cell")
            entries = [cell["warmup"]] + cell["trials"]
            if len(entries) != 11:
                raise ValueError("missing or extra output trials")
            for trial, entry in enumerate(entries):
                if entry["trial"] != trial or len(entry["requests"]) != 1:
                    raise ValueError("output trial/slot population differs")
                prompt = prompts["prompts"][str(length)]["1"][trial][0]
                if len(prompt) != length:
                    raise ValueError("output prompt length differs")
                value = instrument.generation_value(entry["requests"][0], prompt, 128)
                hashes[f"{label}/trial{trial}/slot0"] = hashlib.sha256(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise Refused(f"Ship eager lever incomplete or unbound generated output: {exc}") from exc
    return dict(schema="tessera.eager_lever_outputs.v1", prompts_sha256=prompt_sha256,
                requests_per_arm=33, geometry=geometry, outputs=hashes,
                equality="decoded UTF-8 text and length finish; warmup plus every timed request",
                not_claimed="token-ID/logit equality, KL or quality qualification")


def compare_output_hashes(reference, candidate):
    """Compare an entire validated same-input output population; never a subset."""
    return dict(schema="tessera.eager_lever_output_comparison.v1", passed=reference == candidate,
                requests_per_arm=33, equality=reference["equality"], not_claimed=reference["not_claimed"])



def probes(adapter, arm, peer):
    config, name = adapter.config, arm["arm"]
    out = adapter.rdv / "arms" / name
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(config["profile_dir"]) / name
    invocation = uuid.uuid4().hex
    graph_ship = config.get("window_mode") == GRAPH_SHIP_MODE
    eager_levers = config.get("window_mode") == EAGER_LEVER_MODE
    binding = dict(schema=("tessera.ship_graph_invocation.v1" if graph_ship else
                           "tessera.eager_lever_invocation.v1" if eager_levers else "tessera.window4_eager_invocation.v1"), invocation=invocation,
                   source_bindings=config, arm=arm, identity=adapter.identity, peer=peer,
                   client_host="sparky", base_url=BASE, target_model=MODEL,
                   differences_from_EXL3=["endpoint :8142", "model glm53-artifact", "eager/socket/public-runtime labels",
                                          "fresh output namespace", "A8S runtime, MTP1 and unchanged 2GiB KV"],
                   timing_started_unix=time.time(), profile_timing_samples=False)
    if graph_ship or eager_levers:
        binding["differences_from_EXL3"][2] = ("graph" if graph_ship else "eager") + "/socket/public-runtime labels and explicit per-arm lever env"
        binding["lever_env"] = arm["lever_env"]
    atomic_json(out / "invocation.json", binding)
    argv = [sys.executable, str(CLIENT / "u4_speed_client.py"), "--base-url", BASE, "--model", MODEL,
            "--prompts", config["prompts"], "--out", str(out / "timing.json"),
            "--lens", "512", "2048", "8192", "--conc", "1", "--trials", "10", "--output", "128",
            "--label-mode", "graph" if graph_ship else "eager", "--label-fabric", "socket", "--label-server", "T8-A8S-2dbac191-" + name,
            "--events", str(out / "events.jsonl")]
    binding["timing_argv"] = argv
    with (out / "client.log").open("w") as stream:
        adapter.command(argv, stdout=stream, tick=adapter.tick, limit=adapter.envelope.remaining())
    binding["timing_finished_unix"] = time.time()
    timing = json.loads((out / "timing.json").read_bytes())
    require_timing(timing)
    if eager_levers:
        prompt_path = Path(config["prompts"])
        outputs = output_hashes(timing, json.loads(prompt_path.read_bytes()), sha(prompt_path))
        atomic_json(out / "output-hashes.json", outputs)
        binding["output_hashes_sha256"] = sha(out / "output-hashes.json")
        if name == PAIRS[EAGER_LEVER_MODE][1][0]:
            reference = adapter.rdv / "arms" / PAIRS[EAGER_LEVER_MODE][0][0] / "output-hashes.json"
            comparison = compare_output_hashes(json.loads(reference.read_bytes()), outputs)
            atomic_json(out / "output-comparison.json", comparison)
            if not comparison["passed"]:
                raise Refused("Ship eager lever OFF/ON decoded output/finish hashes differ")
    atomic_json(out / "invocation.json", binding)
    # Execute the frozen instrument's own declared order, not a second roster.
    import importlib.util
    spec = importlib.util.spec_from_file_location("window4_profile_instrument", CLIENT / "comparison_inputs.py")
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    manifest, _, _, _ = profiles.load_manifest(config["profile_manifest"])
    for cell in profiles.declared_cells(manifest):
        kind, length = cell["kind"], cell["L"]
        argv = [sys.executable, str(CLIENT / "comparison_inputs.py"), "--manifest", config["profile_manifest"],
                "profile", "--base-url", BASE, "--length", str(length), "--kind", kind,
                "--events", str(out / "events.jsonl"), "--directory", str(profile_dir), "--invocation", invocation]
        with (out / "client.log").open("a") as stream:
            adapter.command(argv, stdout=stream, tick=adapter.tick, limit=adapter.envelope.remaining())
    binding["profile_finished_unix"] = time.time()
    # One-second requested cadence. The reused sampler retains actual groups and coverage;
    # missing power never becomes an energy claim or a utilization-based substitute.
    power = Path(__file__).resolve().parents[1] / "box_power_window.py"
    for host in ("sparklina", "sparky"):
        after, before = math.floor(binding["timing_started_unix"]), math.ceil(binding["timing_finished_unix"])
        window = ":".join(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) for t in (after, before))
        argv = [sys.executable, str(power), "--host", host, "--label", name + "-timing-with-warmups",
                "--window", window, "--points", str(before - after), "--out", str(out / (host + "-power.json"))]
        with (out / "client.log").open("a") as stream:
            adapter.command(argv, stdout=stream, tick=adapter.tick, limit=adapter.envelope.remaining())
    binding.update(timing_sha256=sha(out / "timing.json"), events_sha256=sha(out / "events.jsonl"),
                   energy_scope="Entire timing episode including three excluded warmups; profile windows excluded. Raw Netdata coverage governs any work/J claim.")
    atomic_json(out / "invocation.json", binding)
    return dict(invocation=invocation, events=str(out / "events.jsonl"), profile_dir=str(profile_dir))
