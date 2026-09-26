"""The engine backends a census may name, and the receipt that records them.

THE DEFECT THIS PINS (issue #618).  ``tools/tessera_route_census.py`` built its
engine from the memory budget, the scheduler limit and the topology alone.
GLM-5.3-Flash on sm_121 serves only through the opt-in GLM53 NoPE attention
backend, which registers as ``CUSTOM`` and accepts only the ``fp8_ds_mla`` KV
cache dtype, and the serving image has no environment variable for either.  So
a census could not load the checkpoint at all, and no GLM serving-image cell
could be earned.

THE FAIL-BEFORE.  On the pre-change tree ``engine_backend_kwargs`` does not
exist and none of the arguments parse, so these tests fail at attribute lookup
and at parse.

WHY A HELPER.  The kwargs are what the engine actually reads, so the test reads
the same function the tool's ``LLM(...)`` call passes.
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
TOOL = ROOT / "tools" / "tessera_route_census.py"


def _tool():
    spec = importlib.util.spec_from_file_location("census_backends_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared_env():
    """What the launcher exports: resolved from docker's own RepoDigests."""
    return container_env(resolve(IMAGE, inspector=lambda _ref: {
        "present": True, "local_id": "sha256:" + "ab" * 32,
        "repo_digests": [IMAGE]}))


def _args(*argv):
    return _tool().parse_args(["/model", "/out.json", "--runtime-image", IMAGE, *argv],
                              env=_declared_env())


def test_the_default_command_line_adds_no_backend():
    """A census written before these arguments must build the engine it always did."""
    assert _tool().engine_backend_kwargs(_args()) == {}


def test_the_glm_serve_flags_reach_the_engine_unchanged():
    """The flags a GLM-5.3 serve on sm_121 carries, as vLLM's own engine fields."""
    args = _args("--attention-backend", "CUSTOM", "--kv-cache-dtype", "fp8_ds_mla",
                 "--moe-backend", "triton",
                 "--kernel-config", '{"enable_flashinfer_autotune": false}',
                 "--trust-remote-code")
    assert _tool().engine_backend_kwargs(args) == {
        "attention_backend": "CUSTOM",
        "kv_cache_dtype": "fp8_ds_mla",
        "moe_backend": "triton",
        "kernel_config": {"enable_flashinfer_autotune": False},
        "trust_remote_code": True,
    }


@pytest.mark.parametrize("value, message", [
    ("{not json", "--kernel-config is not JSON"),
    ("[1, 2]", "--kernel-config must be a JSON object"),
])
def test_a_malformed_kernel_config_is_a_usage_error_before_any_load(capsys, value, message):
    """Refused by name at parse time, not as an engine error mid-load.

    The message is asserted, not only the exit: argparse exits 2 for an
    unrecognised argument too, so on the pre-change tree this would have
    passed for the opposite reason.
    """
    with pytest.raises(SystemExit) as excinfo:
        _args("--kernel-config", value)
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_the_backends_and_the_nope_switch_are_published_in_the_receipt():
    """A cell earned under a named backend serves only under it; the receipt says which."""
    source = TOOL.read_text()
    assert '"engine_backends": engine_backend_kwargs(args),' in source
    assert "**engine_backend_kwargs(args)," in source
    assert '"TESSERA_RESEARCH_GLM53_NOPE": os.environ.get("TESSERA_RESEARCH_GLM53_NOPE")' in source
