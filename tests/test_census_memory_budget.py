"""The memory the census engine is allowed to spend, and the cap it fills.

THE DEFECT THIS PINS.  ``tools/tessera_route_census.py`` named only
``gpu_memory_utilization``, a fraction of the DEVICE's total memory, and vLLM
sizes the KV cache to fill whatever that fraction leaves after weights and
activation.  On a serve whose torch allocator is capped below that fraction --
which is every bounded campaign serve -- the KV fill is what reaches the cap.
Measured on sparklina, ``denseA8A16-layers0-1-20260917``: the a4cap plugin
verified the requested 24.0 GiB fraction exactly, vLLM then computed 17.26 GiB
of KV from a 121.63 GiB device, and the model's own attention kernel OOM'd
23.30 GiB into the 24.00 GiB cap (attempt ``-20260917T055226Z``).  The served
campaign's own launcher already carried the fix -- a bound on the KV cache
itself -- and the census tool had no way to say it.

THE FAIL-BEFORE.  On the pre-change tree ``memory_budget_kwargs`` does not
exist and ``--kv-cache-memory-bytes`` is not an argument, so these tests fail
at attribute lookup and at parse.

WHY A HELPER.  The kwargs are the rule the engine actually reads, so the test
reads the same function the tool's ``LLM(...)`` call passes.  Asserting on the
call site's text would pin the spelling of a line rather than the value.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.runtime_image import container_env, resolve  # noqa: E402

IMAGE = "example/runtime@sha256:" + "1" * 64
# 24 GiB of a 121.63 GiB device: the fraction this run's launcher resolved from
# the box, not a round number.
SPARKLINA_FRACTION = 0.1973
ONE_GIB = 1073741824


def _tool():
    spec = importlib.util.spec_from_file_location(
        "census_memory_under_test", ROOT / "tools" / "tessera_route_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared_env():
    """What the launcher exports: resolved from docker's own RepoDigests."""
    return container_env(resolve(IMAGE, inspector=lambda _ref: {
        "present": True, "local_id": "sha256:" + "ab" * 32,
        "repo_digests": [IMAGE]}))


def _args(*argv):
    return _tool().parse_args(
        ["/model", "/out.json", "--runtime-image", IMAGE,
         "--gpu-memory-utilization", str(SPARKLINA_FRACTION), *argv],
        env=_declared_env())


def test_the_default_command_line_builds_the_engine_it_always_did():
    """An unbounded census must reach the engine with the kwargs it always had."""
    assert _tool().memory_budget_kwargs(_args()) == {
        "gpu_memory_utilization": SPARKLINA_FRACTION}


def test_a_kv_bound_reaches_the_engine_inside_the_fraction():
    """The flag the served launcher carries must survive to the engine kwargs."""
    kwargs = _tool().memory_budget_kwargs(
        _args("--kv-cache-memory-bytes", str(ONE_GIB)))
    assert kwargs == {"gpu_memory_utilization": SPARKLINA_FRACTION,
                      "kv_cache_memory_bytes": ONE_GIB}
    # The fraction is NOT widened to make room: 1 GiB of KV sits inside the
    # 24 GiB the campaign declared, it does not replace the declaration.
    assert kwargs["gpu_memory_utilization"] == SPARKLINA_FRACTION


def test_a_negative_bound_is_a_usage_error_before_any_load(capsys):
    """Refused by name at parse time: a census is two 85-160 s model loads.

    The message is asserted, not only the exit: ``argparse`` exits 2 for an
    unrecognised argument too, so on the pre-change tree this test would have
    passed for the opposite reason -- the flag did not exist.
    """
    with pytest.raises(SystemExit) as excinfo:
        _args("--kv-cache-memory-bytes", "-1")
    assert excinfo.value.code == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_the_budget_is_published_in_the_receipt():
    """What the engine was allowed to spend is a fact about the receipt."""
    source = (ROOT / "tools" / "tessera_route_census.py").read_text()
    assert '"memory_budget": memory_budget_kwargs(args),' in source


def test_the_default_command_line_leaves_the_concurrency_limit_alone():
    """An uncapped census must not acquire a scheduler limit it never asked for."""
    assert _tool().scheduler_kwargs(_args()) == {}


def test_a_concurrency_limit_reaches_the_engine_and_is_published():
    """A capped KV budget caps the Mamba block cache; capture must fit inside it.

    Sparklina at 1 GiB of KV held 123 Mamba blocks, and compiled mode refused at
    vLLM's default 256 before capturing anything, so the limit has to be a value
    the census can set rather than one it inherits.
    """
    args = _args("--max-num-seqs", "8")
    assert _tool().scheduler_kwargs(args) == {"max_num_seqs": 8}
    source = (ROOT / "tools" / "tessera_route_census.py").read_text()
    assert '"scheduler": scheduler_kwargs(args),' in source


def test_a_negative_concurrency_limit_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _args("--max-num-seqs", "-1")
    assert excinfo.value.code == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_batched_token_cap_reaches_engine_with_served_memory_scope():
    args = _args("--max-num-seqs", "1", "--max-num-batched-tokens", "1024")
    assert _tool().scheduler_kwargs(args) == {
        "max_num_seqs": 1, "max_num_batched_tokens": 1024}


def test_negative_batched_token_cap_refuses_before_model_load(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _args("--max-num-batched-tokens", "-1")
    assert excinfo.value.code == 2
    assert "--max-num-batched-tokens must be >= 0" in capsys.readouterr().err


def _tessera(decode_decoder="native_window_gemm", prefill_decoder="native_window_gemm"):
    return {"decode": {"m0": {"decoder": decode_decoder}, "m1": {"decoder": decode_decoder}},
            "prefill": {"m0": {"decoder": prefill_decoder},
                        "m1": {"decoder": prefill_decoder}}}


def test_decoder_coverage_says_so_when_nothing_was_required():
    """A receipt must not leave 'nothing was required' to be inferred."""
    block, problems = _tool().required_decoder_coverage(_tessera(), [])
    assert block["required"] == []
    assert problems == []


def test_the_native_decoder_on_every_module_passes_and_is_counted():
    block, problems = _tool().required_decoder_coverage(
        _tessera(), ["native_window_gemm"])
    assert problems == []
    assert block["phases"]["decode"]["decoders"] == {"native_window_gemm": 2}
    assert block["phases"]["decode"]["modules"] == 2


def test_a_fallback_on_one_module_is_refused_not_averaged_away():
    """The green-receipt-with-a-fallback case this argument exists to refuse."""
    records = _tessera()
    records["decode"]["m1"]["decoder"] = "torch_window"
    _, problems = _tool().required_decoder_coverage(records, ["native_window_gemm"])
    assert len(problems) == 1
    assert "do not report a required decoder" in problems[0]


def test_a_required_decoder_that_took_no_module_is_refused():
    """The other side of the same refusal: a named decoder nobody took."""
    _, problems = _tool().required_decoder_coverage(
        _tessera(decode_decoder="torch_window"), ["native_window_gemm"])
    assert any("no module reports required decoder" in p for p in problems)


def test_decoder_coverage_is_published_in_the_receipt():
    source = (ROOT / "tools" / "tessera_route_census.py").read_text()
    assert '"decoder_coverage": decoder_coverage,' in source
