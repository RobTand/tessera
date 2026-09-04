#!/usr/bin/env python3
"""Census of which Linears the pinned runtime ROUTES THROUGH A QUANT CONFIG.

``tessera_route_census.py`` observes what a served checkpoint *executes*.  This
observes something the route census structurally cannot reach: whether the
plugin is asked about a module **at all**.

``LinearBase.__init__`` (vLLM 0.28, ``model_executor/layers/linear.py:258``)
takes ``UnquantizedLinearMethod()`` in the ``quant_config is None`` branch
*without calling* ``quant_config.get_quant_method``.  A model implementation
that builds a projection with ``quant_config=None`` -- GLM-5.3-Flash does this
for every MLA projection, for the whole KDA layer and for the indexer's
``wk_weights_proj`` -- therefore takes vLLM's own BF16 method and no plugin can
refuse, warn, or even see the prefix.  An exporter that writes a wire there
deletes the ``<module>.weight`` the runtime wants and puts bytes in its place
that nothing decodes.

So the producer needs the fact "this prefix is never offered to a quant config"
BEFORE it encodes.  Principle 14 says that fact is derived from the runtime,
never asserted beside it, and there is no runtime table that publishes it -- so
this tool MAKES one, by construction rather than by reading:

1. Build the model exactly as the loader does (``initialize_model`` under
   ``set_current_vllm_config``), on the ``meta`` device so no weights are read
   and no memory is allocated.  Construction does not depend on weight values.
2. Install a PROBE quant config in the ``VllmConfig``.  It is a real
   ``QuantizationConfig`` whose ``get_quant_method`` records every ``(prefix,
   layer class)`` it is asked about and returns vLLM's own unquantized method.
   A quant config must be present or every Linear trivially gets ``None``.
3. Walk ``named_modules()`` afterwards and record every ``LinearBase``: its
   class, whether ``layer.quant_config is None``, and whether the probe was
   asked about its prefix.  The two agree by construction; recording both means
   a future vLLM that changes the branch shows up as a disagreement rather than
   as silence.

The receipt is the input to the exporter's construction gate and to the
``construction`` block of ``tessera/serving/runtime_contract.json``; it is
stamped with the image, the vLLM version and the model's own architecture and
layer-type lists, because the answer is a property of that triple and nothing
else.

usage (inside the pinned serving image)::

    tessera_construction_census.py <model-or-config-dir> <out.json> \
        [--device meta] [--max-model-len 512]

The directory needs only ``config.json`` (plus whatever the tokenizer loader
wants); no weights are read.  Run it once per (architecture, image).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import platform
import re
import subprocess
import sys
import time

#: A decoder-layer index in a vLLM module prefix.  The census records the exact
#: prefix AND a normalised form, because a 4-layer cut of a 92-layer model
#: builds the same module NAMES and the producer must be able to join the two.
#: The layer indices each normalised prefix was seen at travel with it, so a
#: reader can tell "seen on every layer" from "seen on one".
LAYER_INDEX = re.compile(r"(?<=\.layers\.)(\d+)(?=\.)")

#: Any purely numeric path segment.  A repeated block is a repeated block
#: whether the model spells its stack ``layers.N`` or ``blocks.N`` (the vision
#: tower does), and a census that normalised only the first spelling published
#: 300 near-identical rows for the tower.
NUMERIC_SEGMENT = re.compile(r"(?<=\.)\d+(?=\.|$)")


def normalise(prefix: str) -> str:
    return NUMERIC_SEGMENT.sub("*", prefix)


def _json_safe(value):
    """A mapper field as JSON.

    The refused fields (``orig_to_new_regex``, ``orig_to_new_renaming``) do not
    need to round-trip -- the producer refuses on their PRESENCE
    (``contract._require_replayable_mapper``), so a lossy rendering of one
    cannot mislead a gate.  What matters is that a non-empty field is never
    dropped on the way into the receipt.
    """
    if isinstance(value, dict):
        return {_json_key(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, re.Pattern):
        return {"regex": value.pattern, "flags": int(value.flags)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    return repr(value)


def _json_key(key) -> str:
    return key.pattern if isinstance(key, re.Pattern) else str(key)


def _mapper_field_names(unstacked) -> list:
    """Every field the runtime's ``WeightsMapper`` declares, not the four we know.

    A hardcoded roster is what made this receipt lossy: it listed substr /
    prefix / suffix / stacked, so a model class declaring ``orig_to_new_regex``
    or ``orig_to_new_renaming`` produced a receipt that OMITTED the rule, and a
    producer reading that receipt would compute a name as though the rule were
    not there.  Reading ``dataclasses.fields`` means a field vLLM adds tomorrow
    lands in the receipt today, where the producer's refusal can see it.
    """
    if dataclasses.is_dataclass(unstacked):
        return [f.name for f in dataclasses.fields(unstacked)]
    return [name for name in vars(type(unstacked)) if name.startswith("orig_to_new_")]


def _weights_mapper_table(model_class) -> "dict | None":
    """The rename table vLLM hands a quant config, as data.

    ``configure_quant_config`` hands ``quant_config.apply_vllm_mapper`` the
    model's name-only mapper (``get_rename_mapper`` in the pinned EUGR build,
    ``get_unstacked_mapper`` in earlier builds) for a class that is not
    ``SupportsQuant``, so a producer writing ``config_groups`` in
    the CHECKPOINT's namespace has to apply the same table to know which vLLM
    module it named.  Publishing it here means the producer reads it rather
    than reproducing it.

    Every non-empty field is recorded, including the ones the producer cannot
    replay: the producer's job is to REFUSE on those, and it can only do that
    if the receipt says they are there.
    """
    mapper = getattr(model_class, "hf_to_vllm_mapper", None)
    if mapper is None:
        return None
    from tessera.serving.weights_mapper import module_name_mapper
    unstacked = module_name_mapper(mapper)
    table = {}
    for field in _mapper_field_names(unstacked):
        value = getattr(unstacked, field, None)
        if value:
            table[field] = _json_safe(value)
    return table


def _probe_config_class():
    """A ``QuantizationConfig`` that records what it is asked about.

    It must be a real one: ``configure_quant_config`` hands it the model
    class's ``hf_to_vllm_mapper``/``packed_modules_mapping``, and ``LinearBase``
    raises when ``get_quant_method`` returns ``None``.  It returns vLLM's own
    ``UnquantizedLinearMethod`` for a Linear and ``None`` for everything else,
    which is the fallback every non-Linear layer already handles.
    """
    import torch
    from vllm.model_executor.layers.linear import (LinearBase,
                                                   UnquantizedLinearMethod)
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)

    class ProbeConfig(QuantizationConfig):
        """Records every prefix vLLM offers a quant config, and answers BF16."""

        def __init__(self) -> None:
            super().__init__()
            self.asked: list[tuple[str, str]] = []

        @classmethod
        def get_name(cls) -> str:
            return "tessera_construction_probe"

        @classmethod
        def get_supported_act_dtypes(cls) -> list:
            return [torch.bfloat16, torch.float16, torch.float32]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            return []

        @classmethod
        def from_config(cls, config) -> "ProbeConfig":
            return cls()

        def get_quant_method(self, layer, prefix: str):
            self.asked.append((prefix, type(layer).__name__))
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

    return ProbeConfig


def build_model(model_path: str, device: str, max_model_len: int):
    """Construct the model the way the loader does, on ``device``."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    set_default_torch_dtype = _set_default_torch_dtype()

    engine_args = EngineArgs(
        model=model_path, load_format="dummy", enforce_eager=True,
        max_model_len=max_model_len, trust_remote_code=True,
        # A census is about module NAMES, and TP would cut them per rank.
        tensor_parallel_size=1,
    )
    vllm_config = engine_args.create_engine_config()
    probe = _probe_config_class()()
    vllm_config.quant_config = probe
    # ``initialize_model_parallel`` reads the CURRENT config, so the whole
    # bring-up sits inside the context the loader itself uses.
    with set_current_vllm_config(vllm_config, check_compile=False):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}", backend="gloo")
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with torch.device(device):
                model = initialize_model(vllm_config=vllm_config)
    return model, probe, vllm_config


def _set_default_torch_dtype():
    """vLLM moved this helper between 0.28 point builds; find it, do not pin it."""
    try:
        from vllm.utils.torch_utils import set_default_torch_dtype
    except ImportError:
        from vllm.model_executor.model_loader.weight_utils import (  # type: ignore
            set_default_torch_dtype)
    return set_default_torch_dtype


def _free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def census(model, probe) -> dict:
    from vllm.model_executor.layers.linear import LinearBase

    asked = {prefix for prefix, _cls in probe.asked}
    rows: dict[str, dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        # ``LinearBase`` stores its own prefix; prefer it, because a module can
        # be built under a name its parent does not hold it at.
        prefix = getattr(module, "prefix", "") or name
        key = normalise(prefix)
        row = rows.setdefault(key, {
            "prefix_pattern": key,
            "class": type(module).__name__,
            "quant_method": type(module.quant_method).__name__,
            "quant_config_is_none": module.quant_config is None,
            "offered_to_quant_config": prefix in asked,
            "layers": set(),
            "examples": [],
        })
        match = LAYER_INDEX.search(prefix)
        if match:
            row["layers"].add(int(match.group(0)))
        if len(row["examples"]) < 2:
            row["examples"].append(prefix)
        # A pattern whose members disagree is a real fact, not a bug in the
        # census: record the disagreement rather than letting the last one win.
        for field, value in (("quant_config_is_none", module.quant_config is None),
                             ("offered_to_quant_config", prefix in asked),
                             ("quant_method", type(module.quant_method).__name__)):
            if row[field] != value:
                row.setdefault("disagreements", {}).setdefault(field, []).append(prefix)
    for row in rows.values():
        row["layers"] = sorted(row["layers"])
    # Every prefix the probe WAS asked about that is not a LinearBase -- the
    # MoE modules, the LM head -- so a reader can tell "not a Linear" from
    # "never offered".
    non_linear = sorted({(normalise(p), c) for p, c in probe.asked
                         if normalise(p) not in rows})
    return {
        "linears": [rows[k] for k in sorted(rows)],
        "offered_non_linear": [{"prefix_pattern": p, "class": c} for p, c in non_linear],
    }


def _supports_quant(model_class) -> bool:
    """Whether vLLM skips ``configure_quant_config`` for this class.

    A ``SupportsQuant`` model is handed no mapper and no packed mapping, so a
    producer must NOT apply the tables above for it.  Publishing the flag
    beside them is what keeps the two facts from being read apart.
    """
    try:
        from vllm.model_executor.models.interfaces import SupportsQuant
    except Exception:  # noqa: BLE001
        return False
    return issubclass(model_class, SupportsQuant)


def runtime_stamp() -> dict:
    import torch
    import vllm
    stamp = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "image": os.environ.get("TESSERA_CENSUS_IMAGE", "unstamped"),
        "image_id": os.environ.get("TESSERA_CENSUS_IMAGE_ID", "unstamped"),
    }
    try:
        stamp["vllm_file"] = vllm.__file__
    except Exception:  # noqa: BLE001
        pass
    return stamp


def model_stamp(vllm_config, model_path: str) -> dict:
    hf = vllm_config.model_config.hf_config
    text = getattr(hf, "text_config", hf)
    def get(name):
        value = getattr(text, name, None)
        return list(value) if isinstance(value, (list, tuple)) else value
    return {
        "path": model_path,
        "architectures": list(getattr(hf, "architectures", []) or []),
        "model_type": getattr(hf, "model_type", None),
        "num_hidden_layers": get("num_hidden_layers"),
        "layer_types": get("layer_types"),
        "mlp_layer_types": get("mlp_layer_types"),
        "first_k_dense_replace": get("first_k_dense_replace"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model or config-only directory (no weights are read)")
    ap.add_argument("out", help="receipt JSON to write")
    ap.add_argument("--device", default="meta",
                    help="construction device; meta allocates nothing (default)")
    ap.add_argument("--max-model-len", type=int, default=512)
    args = ap.parse_args()

    started = time.time()
    model, probe, vllm_config = build_model(args.model, args.device, args.max_model_len)
    body = census(model, probe)
    model_class = type(model)
    receipt = {
        "schema": "tessera.construction-census.v1",
        "taken": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
        "runtime": runtime_stamp(),
        "model": model_stamp(vllm_config, args.model),
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "construction_device": args.device,
        # The two tables vLLM hands a quant config so it can match a fused
        # module -- published here because a producer that must name vLLM's
        # module and not the checkpoint's leaf needs exactly them.
        "packed_modules_mapping": getattr(model_class, "packed_modules_mapping", None),
        "hf_to_vllm_mapper_unstacked": _weights_mapper_table(model_class),
        "supports_quant": _supports_quant(model_class),
        **body,
    }
    counts = collections.Counter(
        "never_offered" if not row["offered_to_quant_config"] else "offered"
        for row in receipt["linears"])
    receipt["summary"] = {
        "linear_patterns": len(receipt["linears"]),
        "offered": counts.get("offered", 0),
        "never_offered": counts.get("never_offered", 0),
    }
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=False)
    print(json.dumps(receipt["summary"], indent=1))
    for row in receipt["linears"]:
        if not row["offered_to_quant_config"]:
            print(f"  NEVER OFFERED  {row['prefix_pattern']}  "
                  f"({row['class']}, layers {row['layers']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
