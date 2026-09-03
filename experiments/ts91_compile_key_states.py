#!/usr/bin/env python3
"""Issue #91: does vLLM's compile-cache key see which lane a streamed serve took?

Computes, with the pinned runtime's OWN key function, the AOT cache key for the
two states of one streamed serve -- every module on the window-GEMV lane, and
every module on the torch-window fallback -- and prints them side by side.  It
loads no model and needs no GPU: the key is a function of the config, the env
compile factors and the forward's qualname (``compilation/decorators.py``
:537-552), and only the config differs between the two states.

Run it inside the serving image, once per source tree::

    experiments/tessera_plugin_run.sh -- "python3 experiments/ts91_compile_key_states.py"

On a tree where the lane is not declared the two keys are EQUAL, which is the
bug: vLLM hands the second serve the first's compiled forward, having verified
only that the files the FIRST run traced are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm.compilation.caching import aot_compile_hash_factors          # noqa: E402
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config  # noqa: E402

from tessera.serving import compile_identity as ci                     # noqa: E402

GEMV_OP = "tessera::fp8_streamed_apply"
GEMM_OP = "torch._scaled_mm"
#: Stand-ins for one checkpoint's modules; only WHICH lane each takes varies.
MODULES = [f"model.layers.{i}.{p}"
           for i in range(28)
           for p in ("self_attn.qkv_proj", "self_attn.o_proj",
                     "mlp.gate_up_proj", "mlp.down_proj")]


def key_for(lane_of):
    """The AOT cache key for a streamed serve whose modules take ``lane_of``."""
    if hasattr(ci, "reset_for_tests"):
        ci.reset_for_tests()
    # The device is named rather than inferred, so this runs on a box with no
    # visible GPU: nothing here allocates, and ``DeviceConfig.compute_hash`` is
    # empty anyway (``startup_plan.py`` says so), so it is not a key input.
    config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    with set_current_vllm_config(config):
        ci.declare_compile_identity(serve_mode="streamed")
    note = getattr(ci, "note_traced_dispatch", None)
    for name in MODULES:                      # at weight load, per module
        if note is not None:
            note(name, lane_of(name))
    factors = aot_compile_hash_factors(config)
    # decorators.py appends the forward's qualname hash, one constant across
    # these states; the key is the sha256 of the factor list's repr.
    return (hashlib.sha256(str(factors + ["<one model, one forward>"]).encode()).hexdigest(),
            config.additional_config.get("tessera"))


def main(out_path=None):
    states = {
        "every module on the window-GEMV lane": lambda name: GEMV_OP,
        "every module on the torch-window fallback": lambda name: GEMM_OP,
        "mixed: the odd layers refused": lambda name: (
            GEMM_OP if int(name.split(".")[2]) % 2 else GEMV_OP),
        "mixed: the even layers refused": lambda name: (
            GEMV_OP if int(name.split(".")[2]) % 2 else GEMM_OP),
    }
    keys, records = {}, {}
    for label, lane_of in states.items():
        key, record = key_for(lane_of)
        keys[label] = key
        records[label] = record
        print(f"{key[:16]}  {label}")
        print(f"                  additional_config['tessera'] = {json.dumps(record)}")
    distinct = len(set(keys.values()))
    print(f"\n{distinct} distinct AOT keys over {len(keys)} states")
    if distinct == 1:
        print("VERDICT: the key does not see the lane -- one cache slot for every "
              "state (issue #91)")
    elif distinct == len(keys):
        print("VERDICT: every state keys its own compiled forward")
    else:
        print("VERDICT: some states still share a key")
    # Re-running one state must land on the same key, or the cache never hits.
    again, _ = key_for(lambda name: GEMV_OP)
    same = again == keys["every module on the window-GEMV lane"]
    print(f"one state, computed twice: {'same key' if same else 'DIFFERENT KEYS'}")
    if out_path:
        import vllm
        Path(out_path).write_text(json.dumps({
            "schema": "tessera.serving.compile_key_states/1",
            "vllm": vllm.__version__,
            "keys": keys,
            "records": records,
            "distinct_keys": distinct,
            "states": len(keys),
            "one_state_twice_is_one_key": same,
        }, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out_path}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
