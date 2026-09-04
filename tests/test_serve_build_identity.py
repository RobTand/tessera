"""Which compiled build served an arm, recorded so a later comparison can refuse.

WHY THIS FILE EXISTS (issue #30).  ``docs/measurements/serving-compile-
divergence-2026-09-02.md`` measured the thing these tests defend: a compiled
vLLM artifact *replayed* is bit-identical (0.000000 / 100%), and the same graph
*rebuilt* is not (0.017117 / 95.65%, 120 of 196 autotuned Triton kernels picking
a different ``XBLOCK``/``num_warps``).  So a KL difference between two arms is a
statement about the weights only if both arms served the same build -- and until
now nothing recorded which build served.

The trap this file is built around: **the AOT key does not identify the build.**
Both of those builds live under the same ``torch_aot_compile/15957ad9...`` key,
because vLLM keys the cache by its *inputs* (``cache_key_factors.json`` is
byte-identical between them).  Stamping the key alone would look like
provenance and certify a rebuild as a replay, which is worse than stamping
nothing.  What identifies the build is the *content* of that cache slot: the
autotune choices inductor wrote into it.  ``test_the_fingerprint_is_the_cache_
content_not_the_key`` is that property, and
``test_the_two_surviving_caches_are_told_apart`` is the same property against
the two real caches the receipt was written from.

The determinism knob (``TORCHINDUCTOR_DETERMINISTIC``) is exercised here, not
merely named: that the env var is the live inductor switch on the installed
torch, that it enters the fingerprint so a campaign cannot silently mix
deterministic and non-deterministic arms, and that setting it on a serve which
*replayed* a warm cache does not make the arm deterministic -- inductor never
ran, so the flag decided nothing.  Whether two builds under the flag actually
agree needs a live serve and is not claimed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from tessera.serving.build_identity import (
    BuildIdentityError,
    SCHEMA,
    build_identity,
    compare,
    deterministic_effective,
    read_cache_root,
    read_serve_log,
    require_deterministic_build,
    require_distinct_build,
    require_same_build,
    require_same_dispatch,
)
from tessera.serving.runtime_image import pinned_reference

ROOT = Path(__file__).resolve().parent.parent
# Read, never copied: the serve-image pin lives in runtime_contract.json and a
# second literal here would pass while the pin was wrong (issue #100).
PIN = pinned_reference()
PIN_DIGEST = PIN.split("@", 1)[1]
AOT_KEY = "15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec"
BACKBONE_KEY = "573bb6beb1"

# Real lines, copied from the serve logs the receipt was written from
# (/home/rob/tessera-runs/tsplugin/serve_qwen_tessera_k2-resident-graph-fresh.log
# and census_mixed_streamed_compiled.log).
_ENGINE = ("INFO 09-02 19:49:30 [core.py:122] Initializing a V1 LLM engine "
           "(v0.28.0) with config: model='/models/qwen', enforce_eager=False, "
           "quantization=tessera\n")

REPLAY_LOG = _ENGINE + (
    "INFO 09-02 19:49:46 [caching.py:343] reconstructed serializable fn from "
    "standalone compile artifacts. num_artifacts=29 num_submods=29\n"
    "INFO 09-02 19:49:46 [decorators.py:311] Directly load AOT compilation from "
    f"path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/{AOT_KEY}/rank_0_0/model\n"
    "INFO 09-02 19:49:46 [monitor.py:53] torch.compile took 1.36 s in total\n")

BUILD_LOG = _ENGINE + (
    "WARNING 09-02 18:13:27 [decorators.py:321] Compiling model again due to a load "
    "failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/"
    f"{AOT_KEY}/rank_0_0/model, reason: Source code has changed since the last "
    "compilation. Recompiling the model.\n"
    "INFO 09-02 18:13:35 [backends.py:1094] Using cache directory: "
    f"/root/.cache/vllm/torch_compile_cache/{BACKBONE_KEY}/rank_0_0/backbone for "
    "vLLM's torch.compile\n"
    "INFO 09-02 18:13:35 [backends.py:1155] Dynamo bytecode transform time: 8.04 s\n"
    "INFO 09-02 18:13:42 [decorators.py:708] saved AOT compiled function to "
    f"/root/.cache/vllm/torch_compile_cache/torch_aot_compile/{AOT_KEY}/rank_0_0/model\n")

EAGER_LOG = _ENGINE.replace("enforce_eager=False", "enforce_eager=True") + (
    "INFO 09-02 13:53:11 [gpu_model_runner.py:3220] Model loading took 1.42 GiB\n"
    "INFO 09-02 13:53:40 [api_server.py:1611] Starting vLLM API server 0 on "
    "http://0.0.0.0:8000\n")


def _cache(tmp_path: Path, name: str, *, xblock: int, time_ms: float,
           graph: bytes = b"# computation graph\n") -> Path:
    """A minimal vLLM compile-cache root: one AOT slot, one backbone slot."""
    root = tmp_path / name
    aot = root / "torch_compile_cache" / "torch_aot_compile" / AOT_KEY / "inductor_cache" / "q7"
    aot.mkdir(parents=True)
    (aot / "ctriton_poi_fused_add.best_config").write_text(json.dumps({
        "XBLOCK": xblock, "num_warps": 4, "num_stages": 1,
        "time_taken_ms": time_ms, "triton_cache_hash": "deadbeef"}))
    bb = root / "torch_compile_cache" / BACKBONE_KEY / "rank_0_0" / "backbone"
    bb.mkdir(parents=True)
    (bb / "computation_graph.py").write_bytes(graph)
    # Byte-identical between the two real builds; that is the whole problem.
    (bb / "cache_key_factors.json").write_text('{"vllm": "0.28.0"}')
    return root


def _stamp(log_text: str, tmp_path: Path, name: str, **kw) -> dict:
    log = tmp_path / f"{name}.log"
    log.write_text(log_text)
    return build_identity(serve_log=log, **kw)


# --------------------------------------------------------------- the log ---

def test_the_log_says_whether_the_serve_replayed_a_build_or_made_one():
    replay = read_serve_log(REPLAY_LOG)
    assert replay["aot_keys"] == [AOT_KEY]
    assert replay["aot_keys_loaded"] == [AOT_KEY] and replay["aot_keys_saved"] == []
    assert replay["fresh_compiles"] == 0
    assert replay["reload_failures"] == []
    assert replay["compiled_forward"] is True
    assert replay["vllm_version"] == "0.28.0"

    built = read_serve_log(BUILD_LOG)
    assert built["aot_keys"] == [AOT_KEY] and built["aot_keys_saved"] == [AOT_KEY]
    assert built["backbone_keys"] == [BACKBONE_KEY]
    assert built["fresh_compiles"] == 1
    assert built["reload_failures"] and "Source code has changed" in built["reload_failures"][0]


def test_an_eager_serve_records_no_compiled_build_and_is_still_complete(tmp_path):
    rec = _stamp(EAGER_LOG, tmp_path, "eager")
    assert rec["schema"] == SCHEMA
    assert rec["identity"]["compiled_forward"] is False
    # Nothing was compiled, so "which build served" has an answer: none.
    assert rec["complete"] is True
    require_same_build(rec, _stamp(EAGER_LOG, tmp_path, "eager2"), why="two eager serves")


# ------------------------------------------------- the fingerprint itself ---

def test_the_fingerprint_is_the_cache_content_not_the_key(tmp_path):
    """Two builds under ONE AOT key must not stamp as one build.

    This is the measured case: byte-identical ``cache_key_factors.json``, same
    key, 120/196 kernels retuned, logits 0.017117 apart.
    """
    a = _stamp(BUILD_LOG, tmp_path, "a", cache_root=_cache(tmp_path, "ca", xblock=1024, time_ms=0.51))
    b = _stamp(BUILD_LOG, tmp_path, "b", cache_root=_cache(tmp_path, "cb", xblock=512, time_ms=0.49,
                                                           graph=b"# other graph\n"))
    assert a["identity"]["aot"][AOT_KEY]["autotune_digest"] != \
        b["identity"]["aot"][AOT_KEY]["autotune_digest"]
    assert a["build_fingerprint"] != b["build_fingerprint"]
    verdict = compare(a, b)
    assert verdict["same_build"] is False
    assert verdict["differs"] == ["aot"]
    with pytest.raises(BuildIdentityError, match="different compiled build"):
        require_same_build(a, b, why="the K2 A/B")
    require_distinct_build(a, b, why="the rebuild arm")


def test_the_serve_that_built_it_and_the_serve_that_replayed_it_are_one_build(tmp_path):
    """The row the stamp exists to certify: build, then replay, same cache.

    vLLM logs ``Using cache directory: .../backbone`` only when it compiles --
    measured on the two real arms, the rebuild log has one such line and the
    replay log none.  So the backbone slot is provenance: fingerprinting it
    would have handed one build two fingerprints and refused this pair, which
    is exactly backwards.
    """
    root = _cache(tmp_path, "one", xblock=1024, time_ms=0.51)
    built = _stamp(BUILD_LOG, tmp_path, "built", cache_root=root)
    replayed = _stamp(REPLAY_LOG, tmp_path, "replayed", cache_root=root)
    assert built["provenance"]["fresh_compiles"] == 1
    assert replayed["provenance"]["fresh_compiles"] == 0
    assert built["provenance"]["backbone"] != replayed["provenance"]["backbone"]
    assert built["build_fingerprint"] == replayed["build_fingerprint"]
    require_same_build(built, replayed, why="one build, served twice")


def test_the_same_build_read_twice_is_the_same_build(tmp_path):
    root = _cache(tmp_path, "shared", xblock=1024, time_ms=0.51)
    a = _stamp(REPLAY_LOG, tmp_path, "first", cache_root=root)
    b = _stamp(REPLAY_LOG, tmp_path, "second", cache_root=root)
    assert a["build_fingerprint"] == b["build_fingerprint"]
    # The provenance differs (two logs, two timestamps) and must not be fingerprinted.
    assert a["provenance"]["serve_log"] != b["provenance"]["serve_log"]
    require_same_build(a, b, why="one build served twice")
    with pytest.raises(BuildIdentityError, match="same compiled build"):
        require_distinct_build(a, b, why="a rebuild that did not rebuild")


def test_a_re_benchmarked_but_unchanged_kernel_is_not_a_different_build(tmp_path):
    """``time_taken_ms`` is the measurement, not the choice.

    74 of the 196 real records differed only in their timing.  Fingerprinting
    those would cry wolf on every stamp and the check would be turned off.
    """
    a = _stamp(REPLAY_LOG, tmp_path, "a", cache_root=_cache(tmp_path, "ca", xblock=1024, time_ms=0.51))
    b = _stamp(REPLAY_LOG, tmp_path, "b", cache_root=_cache(tmp_path, "cb", xblock=1024, time_ms=0.83))
    assert a["build_fingerprint"] == b["build_fingerprint"]


# ------------------------------------------------ a partial stamp refuses ---

def test_a_log_only_stamp_of_a_compiled_serve_cannot_certify_anything(tmp_path):
    """The AOT key alone is exactly the false security this exists to remove."""
    a = _stamp(REPLAY_LOG, tmp_path, "a")
    b = _stamp(REPLAY_LOG, tmp_path, "b")
    assert a["complete"] is False
    assert a["identity"]["aot"][AOT_KEY]["autotune_digest"] is None
    # Same key, same fingerprint -- and still refused, because the key is not the build.
    assert a["build_fingerprint"] == b["build_fingerprint"]
    assert compare(a, b)["incomplete"] == ["a", "b"]
    with pytest.raises(BuildIdentityError, match="incomplete"):
        require_same_build(a, b, why="two log-only stamps")
    with pytest.raises(BuildIdentityError, match="incomplete"):
        require_distinct_build(a, b, why="two log-only stamps")


def test_a_cache_root_missing_the_slot_the_log_names_is_incomplete(tmp_path):
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)
    other = tmp_path / "empty"
    (other / "torch_compile_cache").mkdir(parents=True)
    assert _stamp(REPLAY_LOG, tmp_path, "a", cache_root=root)["complete"] is True
    rec = _stamp(REPLAY_LOG, tmp_path, "b", cache_root=other)
    assert rec["complete"] is False
    assert rec["identity"]["aot"][AOT_KEY]["present"] is False


# ------------------------------------------------- the determinism knob ----

def test_the_determinism_env_var_is_the_live_inductor_knob():
    """Not "the flag exists": the installed torch's config actually flips.

    ``torch/_inductor/config.py`` reads ``TORCHINDUCTOR_DETERMINISTIC`` at
    import, so this has to be two subprocesses.  If a torch bump renames or
    drops the knob, this fails instead of the campaign silently stamping a flag
    that decides nothing.
    """
    pytest.importorskip("torch")
    prog = "import torch._inductor.config as c; print(int(bool(c.deterministic)))"

    def _run(value: str | None) -> str:
        env = dict(os.environ, TMPDIR="/home/rob/tmp", CUDA_VISIBLE_DEVICES="")
        env.pop("TORCHINDUCTOR_DETERMINISTIC", None)
        if value is not None:
            env["TORCHINDUCTOR_DETERMINISTIC"] = value
        out = subprocess.run([sys.executable, "-c", prog], env=env,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()

    assert _run(None) == "0"
    assert _run("1") == "1"


def test_the_determinism_flag_is_part_of_the_build_identity(tmp_path):
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)
    off = _stamp(BUILD_LOG, tmp_path, "off", cache_root=root, deterministic=False)
    on = _stamp(BUILD_LOG, tmp_path, "on", cache_root=root, deterministic=True)
    assert off["build_fingerprint"] != on["build_fingerprint"]
    assert compare(off, on)["differs"] == ["inductor_deterministic"]
    with pytest.raises(BuildIdentityError, match="inductor_deterministic"):
        require_same_build(off, on, why="a campaign that mixed the two")


def test_the_flag_on_a_replayed_build_did_not_make_the_build_deterministic(tmp_path):
    """The knob's silent no-op: a warm cache replays, inductor never runs.

    vLLM's AOT key is over its own inputs, so a warm ``$VLLM_CACHE`` hands back
    the build that was autotuned WITHOUT the flag while the arm stamps
    ``inductor_deterministic: true``.  ``fresh_compiles`` is the tell.
    """
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)
    replayed = _stamp(REPLAY_LOG, tmp_path, "replay", cache_root=root, deterministic=True)
    built = _stamp(BUILD_LOG, tmp_path, "built", cache_root=root, deterministic=True)

    assert replayed["provenance"]["fresh_compiles"] == 0
    assert deterministic_effective(replayed) is False
    assert deterministic_effective(built) is True
    with pytest.raises(BuildIdentityError, match="replayed"):
        require_deterministic_build(replayed)
    require_deterministic_build(built)

    off = _stamp(BUILD_LOG, tmp_path, "off", cache_root=root, deterministic=False)
    assert deterministic_effective(off) is False
    with pytest.raises(BuildIdentityError, match="TORCHINDUCTOR_DETERMINISTIC"):
        require_deterministic_build(off)


# ------------------------------------------------------------- the wiring ---

def test_the_cli_writes_the_sidecar_a_later_comparison_reads(tmp_path):
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)
    log = tmp_path / "serve.log"
    log.write_text(REPLAY_LOG)
    out = tmp_path / "arm.build.json"
    env = dict(os.environ, TMPDIR="/home/rob/tmp", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [sys.executable, "-m", "tessera.serving.build_identity", "stamp",
         "--log", str(log), "--out", str(out), "--cache-root", str(root),
         "--image", PIN, "--image-digest", PIN_DIGEST,
         "--image-local-id", "sha256:" + "89" * 32,
         "--serve-mode", "resident",
         "--eager", "0", "--deterministic", "0",
         "--artifact-path", "/models/qwen3-0.6b-tessera-k2"],
        env=env, check=True, capture_output=True, text=True)
    rec = json.loads(out.read_text())
    assert rec["schema"] == SCHEMA and rec["complete"] is True
    assert rec["identity"]["serve_mode"] == "resident"
    assert rec["identity"]["image"] == PIN
    # WHAT RAN, not what was asked for (issue #100).  The resolved manifest
    # digest is what the fingerprint is over; the local docker id is provenance
    # only, because the two GB10s report different ids for identical bytes.
    assert rec["identity"]["image_digest"] == PIN_DIGEST
    assert rec["provenance"]["image_local_id"] == "sha256:" + "89" * 32
    assert rec["provenance"]["artifact_path"] == "/models/qwen3-0.6b-tessera-k2"
    require_same_build(rec, _stamp(REPLAY_LOG, tmp_path, "again", cache_root=root,
                                   image=PIN, image_digest=PIN_DIGEST,
                                   image_local_id="sha256:" + "61" * 32,
                                   serve_mode="resident", eager=False),
                       why="the CLI and the API stamp one build alike")


def test_the_cli_refuses_a_cross_regime_pair_and_passes_a_pinned_one(tmp_path):
    """The A/B rule reachable from a shell, not only from Python.

    ``require_same_dispatch`` existed as a library function with no way to
    call it from a campaign script, which calls ``compare --require`` and
    nothing else.  ``same`` is the wrong check for this: it is about the
    compiled *build*, so it refuses the pinned pair too -- the pair whose
    served KL is 0.000000 (docs/measurements/serving-compile-dispatch-2026-09-03.md
    section 3).  A rule a gate cannot read is a note, so the gate reads it here.
    """
    root = _cache(tmp_path, "one", xblock=8, time_ms=1.0)
    kw = dict(cache_root=root, image=PIN, image_digest=PIN_DIGEST,
              serve_mode="resident")
    arms = {
        "eager": _stamp(EAGER_DISPATCH_LOG, tmp_path, "e", eager=True, **kw),
        "compiled": _stamp(COMPILED_DISPATCH_LOG, tmp_path, "c", eager=False, **kw),
        "pinned": _stamp(PINNED_DISPATCH_LOG, tmp_path, "p", eager=False, **kw),
    }
    paths = {}
    for name, rec in arms.items():
        paths[name] = tmp_path / f"{name}.build.json"
        paths[name].write_text(json.dumps(rec, indent=1) + "\n")

    env = dict(os.environ, TMPDIR="/home/rob/tmp", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT / "src"))

    def _compare(a, b, require):
        return subprocess.run(
            [sys.executable, "-m", "tessera.serving.build_identity", "compare",
             str(paths[a]), str(paths[b]), "--require", require],
            env=env, capture_output=True, text=True)

    crossed = _compare("eager", "compiled", "same-dispatch")
    assert crossed.returncode == 4, crossed.stdout + crossed.stderr
    assert "different implementations" in crossed.stderr

    # The pinned compiled arm ran the eager arm's implementations, so the
    # dispatch check passes -- while the build check, correctly, does not.
    assert _compare("eager", "pinned", "same-dispatch").returncode == 0
    assert _compare("eager", "pinned", "same").returncode == 4


def test_the_shell_helper_forwards_the_knob_and_stamps(tmp_path):
    """The serve wrappers' own code path, run without docker.

    ``experiments/build_identity.sh`` is what the wrappers source; testing the
    functions is testing the wrappers' behaviour, not a grep for a string.
    """
    helper = ROOT / "experiments" / "build_identity.sh"
    log = tmp_path / "serve.log"
    log.write_text(REPLAY_LOG)
    out = tmp_path / "arm.build.json"
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)

    def _sh(script: str, **extra) -> str:
        env = dict(os.environ, TMPDIR="/home/rob/tmp", CUDA_VISIBLE_DEVICES="", **extra)
        r = subprocess.run(["bash", "-euo", "pipefail", "-c",
                            f'source "{helper}"\n{script}'],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    assert _sh("build_identity_docker_env") == ""
    assert _sh("build_identity_docker_env", TESSERA_SERVE_DETERMINISTIC="1") == \
        "-e TORCHINDUCTOR_DETERMINISTIC=1"

    _sh(f'build_identity_stamp "{log}" "{out}" "{root}"')
    rec = json.loads(out.read_text())
    assert rec["complete"] is True
    assert rec["identity"]["inductor_deterministic"] is False

    out2 = tmp_path / "arm2.build.json"
    _sh(f'build_identity_stamp "{log}" "{out2}" "{root}"', TESSERA_SERVE_DETERMINISTIC="1")
    assert json.loads(out2.read_text())["identity"]["inductor_deterministic"] is True


@pytest.mark.parametrize("wrapper", ["serve_and_dump_kl.sh", "tessera_plugin_served.sh"])
def test_every_serve_wrapper_stamps_its_dump(wrapper):
    """A wiring check, and only that: it says the call is there, not that it ran."""
    text = (ROOT / "experiments" / wrapper).read_text()
    assert "build_identity.sh" in text, f"{wrapper} does not source the stamper"
    assert "build_identity_stamp" in text, f"{wrapper} does not stamp its dump"


# ------------------------------------------------------ the measured case ---

_CACHES = Path("/home/rob/tessera-runs/tsplugin")


@pytest.mark.skipif(
    not (_CACHES / "vllm-cache-fresh" / "torch_compile_cache" / "torch_aot_compile"
         / AOT_KEY).is_dir(),
    reason="the two surviving compile caches from 2026-09-02 are not on this box")
def test_the_two_surviving_caches_are_told_apart():
    """The receipt's own two builds, one key, different fingerprints.

    ``docs/measurements/serving-compile-divergence-2026-09-02.md`` §3: 196
    autotune records in both, 120 with a different tuning choice, and the two
    serves' logits 0.017117 apart.  If this mechanism could not separate these
    two directories it would be decoration.
    """
    a = read_cache_root(_CACHES / "vllm-cache", [AOT_KEY], [])
    b = read_cache_root(_CACHES / "vllm-cache-fresh", [AOT_KEY], [])
    assert a["aot"][AOT_KEY]["autotune_records"] == 196
    assert b["aot"][AOT_KEY]["autotune_records"] == 196
    assert a["aot"][AOT_KEY]["autotune_digest"] != b["aot"][AOT_KEY]["autotune_digest"]


# --------------------------------------------------- the reading side ------

def _divergence_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serving_compile_divergence",
        ROOT / "experiments" / "serving_compile_divergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_reporter_refuses_a_rebuild_row_whose_arms_replayed_one_build(tmp_path,
                                                                          monkeypatch):
    """The row that exists to measure a rebuild must not accept a replay.

    ``build_vs_build`` holds one rebuild row and one replay row.  Swapped, the
    replay's 0.000000 reads as "rebuilds are reproducible" -- the exact
    inversion the receipt was written to prevent.
    """
    scd = _divergence_module()
    monkeypatch.setattr(scd, "DUMPS", tmp_path)
    root = _cache(tmp_path, "c", xblock=1024, time_ms=0.51)
    for name in ("arm_a", "arm_b"):
        (tmp_path / f"{name}.build.json").write_text(json.dumps(
            _stamp(BUILD_LOG, tmp_path, name, cache_root=root)))

    label = "plugin K2 resident compiled: chain build vs fresh build"
    assert scd.BUILD_EXPECTATION[label] == "distinct"
    problems: list[str] = []
    check = scd._build_check(label, "arm_a", "arm_b", problems)
    assert check["status"] == "REFUSED"
    assert problems and "same compiled build" in problems[0]

    # The replay row, correctly labelled, passes on the same two sidecars.
    problems = []
    replay_label = "plugin K2 resident compiled: build replayed by a second serve"
    assert scd._build_check(replay_label, "arm_a", "arm_b", problems)["status"] == "same_build"
    assert problems == []


def test_an_unstamped_row_is_still_compared_but_never_silently(tmp_path, monkeypatch):
    """Every dump on disk predates the stamp; the archive must stay readable."""
    scd = _divergence_module()
    monkeypatch.setattr(scd, "DUMPS", tmp_path)
    problems: list[str] = []
    check = scd._build_check(
        "plugin K2 resident compiled: build replayed by a second serve",
        "old_a", "old_b", problems)
    assert check["status"] == "unstamped"
    assert len(problems) == 1 and "not stamped" in problems[0]
    # A row that makes no build claim reports the gap without a problem entry.
    problems = []
    assert scd._build_check("plugin K2 resident", "old_a", "old_b",
                            problems)["status"] == "unstamped"
    assert problems == []


# ------------------------------------------------ which program ran (#16) ---
#
# The eager-vs-compiled gap this repo measured (0.0269 on the FP8 route, 0.2445
# on the NVFP4 route) is not the compiler reassociating a sum: vLLM 0.28 runs
# different implementations of the same math depending on whether it compiles,
# and prints which ones it resolved in its own startup line.  These fixtures are
# that line, abridged from the two serves that produced the receipt's stock-twin
# row: /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log:12 (eager)
# and serve_qwen_stock_tessera-k2-graph.log:12 (compiled).

_CFG = ("INFO 09-02 08:01:13 [core.py:122] Initializing a V1 LLM engine (v0.28.0) "
        "with config: model='/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4', "
        "quantization=compressed-tensors, enforce_eager={eager}, "
        "compilation_config={{'mode': <CompilationMode.{mode}: {lvl}>, "
        "'custom_ops': [{ops}], 'pass_config': {{'fuse_norm_quant': False}}}}, "
        "kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig("
        "rms_norm=[{ir}], fused_add_rms_norm=[{ir}]), enable_flashinfer_autotune=True)\n")

EAGER_DISPATCH_LOG = _CFG.format(eager="True", mode="NONE", lvl=0, ops="'all'",
                                 ir="'vllm_c', 'native'")
COMPILED_DISPATCH_LOG = _CFG.format(
    eager="False", mode="VLLM_COMPILE", lvl=3, ops="'none'", ir="'native'") + (
    "INFO 09-02 08:01:33 [decorators.py:708] saved AOT compiled function to "
    f"/root/.cache/vllm/torch_compile_cache/torch_aot_compile/{AOT_KEY}/rank_0_0/model\n")
# The arm this issue's fix makes possible: compiled, with the dispatch pinned
# back to the kernels the eager arm ran (--kernel-config ir_op_priority,
# --compilation-config custom_ops).
#
# TWO config lines, because that is what a pinned arm actually logs.  vLLM
# prints the request the CLI made and then the config it resolved, and only
# the second says what ran: on the real compiled-both arm the first line reads
# ``ir_op_priority=['vllm_c']`` and the second ``['vllm_c', 'native']``,
# identical to the eager arm's single line -- which is why its served KL
# against eager is exactly 0.000000 at 100.00% top-1.  The one-line fixture
# this replaces asserted the resolved values on the *first* line, a shape no
# serve produces, and so it certified a gate that refused three pairs the
# serve says agree.  See docs/measurements/serving-compile-dispatch-2026-09-03.md
# and /home/rob/tessera-runs/compile-dispatch/.
PINNED_DISPATCH_LOG = (
    _CFG.format(eager="False", mode="VLLM_COMPILE", lvl=3, ops="'all'",
                ir="'vllm_c'")
    + _CFG.format(eager="False", mode="VLLM_COMPILE", lvl=3, ops="'all'",
                  ir="'vllm_c', 'native'")
    + "INFO 09-02 08:01:33 [decorators.py:708] saved AOT compiled function to "
    f"/root/.cache/vllm/torch_compile_cache/torch_aot_compile/{AOT_KEY}/rank_0_0/model\n")


def test_the_log_says_which_implementations_the_runtime_resolved():
    eager = read_serve_log(EAGER_DISPATCH_LOG)["dispatch"]
    compiled = read_serve_log(COMPILED_DISPATCH_LOG)["dispatch"]
    assert eager == {"custom_ops": ["all"],
                     "ir_op_priority": {"rms_norm": ["vllm_c", "native"],
                                        "fused_add_rms_norm": ["vllm_c", "native"]}}
    assert compiled == {"custom_ops": ["none"],
                        "ir_op_priority": {"rms_norm": ["native"],
                                           "fused_add_rms_norm": ["native"]}}


def test_an_eager_arm_and_a_compiled_arm_did_not_run_the_same_program(tmp_path):
    """The refusal the 2026-09-02 pair should have hit before it was compared."""
    root = _cache(tmp_path, "one", xblock=8, time_ms=1.0)
    a = _stamp(EAGER_DISPATCH_LOG, tmp_path, "eager", cache_root=root, eager=True)
    b = _stamp(COMPILED_DISPATCH_LOG, tmp_path, "compiled", cache_root=root, eager=False)
    verdict = compare(a, b)
    assert verdict["dispatch_known"] is True
    assert verdict["same_dispatch"] is False
    assert "dispatch" in verdict["differs"]
    with pytest.raises(BuildIdentityError, match="different implementations"):
        require_same_dispatch(a, b, why="eager vs compiled KL")


def test_pinning_the_dispatch_makes_a_compiled_arm_comparable_to_an_eager_one(tmp_path):
    """Same implementations, still different builds -- the two checks are orthogonal."""
    root = _cache(tmp_path, "one", xblock=8, time_ms=1.0)
    eager = _stamp(EAGER_DISPATCH_LOG, tmp_path, "eager", cache_root=root, eager=True)
    pinned = _stamp(PINNED_DISPATCH_LOG, tmp_path, "pinned", cache_root=root, eager=False)
    require_same_dispatch(eager, pinned, why="pinned-dispatch A/B")
    assert compare(eager, pinned)["same_build"] is False


def test_the_resolved_config_line_wins_over_the_requested_one():
    """A pinned arm logs its ask and then vLLM's answer; only the answer ran.

    The first line is the CLI request.  Reading it made ``require_same_dispatch``
    compare an ask against a resolution, and the gate then answered "different
    implementations" for arms whose served logits were bit-identical.
    """

    record = read_serve_log(PINNED_DISPATCH_LOG)
    assert record["dispatch"] == read_serve_log(EAGER_DISPATCH_LOG)["dispatch"]
    assert record["dispatch_requested"] == {
        "custom_ops": ["all"],
        "ir_op_priority": {"rms_norm": ["vllm_c"],
                           "fused_add_rms_norm": ["vllm_c"]}}
    # An arm that was never pinned logs one line and has nothing to disagree
    # with -- the field must stay None rather than echo the resolution.
    assert read_serve_log(EAGER_DISPATCH_LOG)["dispatch_requested"] is None


#: The seven arms of the 2026-09-03 dispatch campaign and the served KL each
#: measured against the eager arm.  ``True`` means the serve says the two arms
#: ran the same program -- 0.000000 KL at 100.00% top-1 over 4088 positions --
#: and so the gate must PASS the pair; ``False`` means it moved 0.244-0.249 KL
#: and ~30% of top-1, and the gate must REFUSE it.  This is the ground truth
#: the gate exists to reproduce, and the fixtures above cannot check it.
_SERVED_AGAINST_EAGER = {
    "compiled-both": True,             # KL 0.0,       top-1 100.00%
    "compiled-both-noauto": True,      # KL 0.0,       top-1 100.00%
    "compiled-eagerbackend": True,     # KL 0.0,       top-1 100.00%
    "compiled": False,                 # KL 0.24730,   top-1  70.43%
    "compiled-ir": False,              # KL 0.24393,   top-1  69.45%
    "compiled-ops": False,             # KL 0.24892,   top-1  70.06%
}
_DISPATCH_RUN = Path("/home/rob/tessera-runs/compile-dispatch")


@pytest.mark.parametrize("arm,agrees", sorted(_SERVED_AGAINST_EAGER.items()))
def test_the_gate_answers_what_the_serve_measured(arm, agrees):
    """Every arm of the campaign, gate verdict against served ground truth.

    The gate's whole claim is that it can tell "these two arms ran the same
    program" from the serve log alone.  That claim is checkable, because six
    of these pairs were actually served and their KL is on disk -- so this
    compares the gate's answer with the runtime's, arm by arm, rather than
    with a fixture written to agree with it.

    It is what the one-line ``PINNED_DISPATCH_LOG`` could not check: with the
    first config line read, three of these six answers were wrong, and every
    fixture-based test still passed.
    """

    eager_log = _DISPATCH_RUN / "serve_qwen_dispatch_eager.log"
    arm_log = _DISPATCH_RUN / f"serve_qwen_dispatch_{arm}.log"
    if not (eager_log.is_file() and arm_log.is_file()):
        pytest.skip(f"{_DISPATCH_RUN} is not on this box")
    a = read_serve_log(eager_log.read_text(errors="replace"))["dispatch"]
    b = read_serve_log(arm_log.read_text(errors="replace"))["dispatch"]
    assert a is not None and b is not None
    assert (a == b) is agrees, (
        f"eager vs {arm}: the log says same_dispatch={a == b}, the serve "
        f"says {agrees} (a={a}, b={b})")


def test_a_log_that_does_not_record_the_dispatch_certifies_nothing(tmp_path):
    """Absence of the field is not agreement -- an old log must refuse, not pass."""
    root = _cache(tmp_path, "one", xblock=8, time_ms=1.0)
    a = _stamp(BUILD_LOG, tmp_path, "old_a", cache_root=root, eager=False)
    b = _stamp(EAGER_DISPATCH_LOG, tmp_path, "new_b", cache_root=root, eager=True)
    assert a["identity"]["dispatch"] is None
    assert compare(a, b)["same_dispatch"] is False
    with pytest.raises(BuildIdentityError, match="does not record the dispatch"):
        require_same_dispatch(a, b, why="mixed-vintage A/B")


def test_the_dispatch_is_part_of_the_build_identity(tmp_path):
    """It decides the arithmetic, so it belongs in the fingerprint, not beside it."""
    root = _cache(tmp_path, "one", xblock=8, time_ms=1.0)
    a = _stamp(COMPILED_DISPATCH_LOG, tmp_path, "default", cache_root=root, eager=False)
    b = _stamp(PINNED_DISPATCH_LOG, tmp_path, "pinned", cache_root=root, eager=False)
    assert a["build_fingerprint"] != b["build_fingerprint"]


@pytest.mark.parametrize("log,expected", [
    ("/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log",
     {"custom_ops": ["all"],
      "ir_op_priority": {"rms_norm": ["vllm_c", "native"],
                         "fused_add_rms_norm": ["vllm_c", "native"]}}),
    ("/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log",
     {"custom_ops": ["none"],
      "ir_op_priority": {"rms_norm": ["native"],
                         "fused_add_rms_norm": ["native"]}}),
])
def test_the_real_serve_logs_of_the_measured_pair(log, expected):
    """Against the two logs the 0.2473 stock-twin row was measured from."""
    path = Path(log)
    if not path.is_file():
        pytest.skip(f"{log} is not on this box")
    assert read_serve_log(path.read_text(errors="replace"))["dispatch"] == expected


# ---------------------------------------------------------------- the gate --
# A gate that refuses on a match (#16)

_WRAPPER = Path(__file__).resolve().parents[1] / "experiments" / "serve_and_dump_kl.sh"


def test_a_matching_pattern_returns_zero_only_when_the_producer_is_not_piped():
    """``producer | grep -q`` returns NON-zero on a match under ``pipefail``.

    grep -q exits at the first hit, the producer takes SIGPIPE and dies 141, and
    that is the status ``set -o pipefail`` propagates.  This is not a hypothesis:
    it is what ``serve_and_dump_kl.sh`` did to every arm of the dispatch ladder
    on 2026-09-03, refusing each serve whose log said exactly what the arm had
    asked for.  ``yes`` is used as the producer because it is guaranteed to
    still be writing when grep leaves; a short producer may finish first and
    hide the bug, which is how it survived review.
    """
    piped = subprocess.run(
        ["bash", "-c", "set -euo pipefail; yes match 2>/dev/null | grep -Eq match"],
        capture_output=True)
    assert piped.returncode != 0, (
        "the piped form is expected to FAIL on a match; if it now succeeds the "
        "premise of the fix below has changed and the fix should be re-argued")

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "serve.log"
        log.write_text("match\n" * 100000)
        filed = subprocess.run(
            ["bash", "-c", f"set -euo pipefail; grep -Eq match {log}"],
            capture_output=True)
    assert filed.returncode == 0


def test_the_serve_wrapper_greps_its_log_file_rather_than_a_pipe():
    """The wrapper's own gate must use the form that survives a match."""
    body = _WRAPPER.read_text()
    assert "TESSERA_KL_REQUIRE_IN_LOG" in body
    assert 'docker logs "$NAME" 2>&1 | grep' not in body, (
        "the require gate is piping docker logs into grep again; under pipefail "
        "that refuses precisely the arms whose log matches")
    assert 'grep -Eq "$TESSERA_KL_REQUIRE_IN_LOG" "$LOG"' in body


def test_the_serve_wrapper_keeps_entrypoint_and_image_command_distinct():
    """A CLI-style image needs ``vllm serve``, not only an entrypoint override."""
    body = _WRAPPER.read_text()
    assert ('"$IMAGE" \\\n'
            '  ${TESSERA_KL_IMAGE_COMMAND:-} \\\n'
            '  "$MODEL"') in body, (
        "the image command must be an explicit token after the image; Docker options "
        "such as --entrypoint belong before it and cannot spell `vllm serve`")


def test_a_half_parsed_dispatch_line_is_not_a_known_dispatch() -> None:
    """Absence must not read as agreement -- the thing the field exists for.

    `custom_ops` and `ir_op_priority` are printed on the same startup line, and
    vLLM 0.28 flips them together.  Returning a record when only one of them
    matched left the other key `None` while `dispatch` itself was no longer
    `None`, so the record claimed the dispatch was *known* and two half-parsed
    arms compared equal on the half that was missing.  When the halves really
    differ that is worth 30% of the top-1 predictions
    (`docs/measurements/serving-compile-dispatch-2026-09-03.md` section 3).
    """

    ops_only = "... enforce_eager=False, 'custom_ops': ['all'], other=1)"
    ir_only = ("... ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c'], "
               "fused_add_rms_norm=['vllm_c']))")
    both = ("... 'custom_ops': ['all'], "
            "ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c'], "
            "fused_add_rms_norm=['vllm_c']))")

    from tessera.serving import build_identity as module

    assert module._read_dispatch(ops_only) is None
    assert module._read_dispatch(ir_only) is None

    read = module._read_dispatch(both)
    assert read is not None
    assert read["custom_ops"] == ["all"]
    assert read["ir_op_priority"]["rms_norm"] == ["vllm_c"]
