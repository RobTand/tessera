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
            self.asked_layers = {}

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
            self.asked_layers[prefix] = layer
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

    return ProbeConfig


def build_model(model_path: str, device: str, max_model_len: int, *, quant_config=None):
    """Construct through the stock loader, with a probe or explicit quant config."""
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
        enable_prefix_caching=False,
        # A census is about module NAMES, and TP would cut them per rank.
        tensor_parallel_size=1,
    )
    vllm_config = engine_args.create_engine_config()
    probe = _probe_config_class()() if quant_config is None else quant_config
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


def _output_sizes(module) -> list[int]:
    """The output partition list the runtime built this Linear with."""
    sizes = getattr(module, "output_sizes", None)
    if sizes is None:
        sizes = [module.output_size]
    return [int(s) for s in sizes]


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
            # The runtime's output partition list -- ``ColumnParallelLinear.
            # output_sizes`` (a merged/QKV Linear's per-member sizes; one
            # entry for a plain column cut) and ``[output_size]`` for a
            # Linear that has no such list (row-parallel, replicated).  It is
            # what ``create_weights`` later receives as
            # ``output_partition_sizes`` and what an exporter must declare
            # its roles against (tessera#377).  Global sizes: the census
            # builds at tp=1.
            "output_sizes": _output_sizes(module),
            "input_sizes": set(),
            "output_size": int(module.output_size),
            "replicated_shard_ids": sorted(int(i) for i in getattr(module, "replicated_shard_ids", ())),
            "instances": [],
            "layers": set(),
            "examples": [],
        })
        row["input_sizes"].add(int(module.input_size))
        match = LAYER_INDEX.search(prefix)
        if match:
            row["layers"].add(int(match.group(0)))
        if len(row["examples"]) < 2:
            row["examples"].append(prefix)
        row["instances"].append({
            "prefix": prefix, "module_name": name,
            "class": type(module).__name__,
            "input_size": int(module.input_size),
            "output_size": int(module.output_size),
            "output_sizes": _output_sizes(module),
            "replicated_shard_ids": sorted(int(i) for i in getattr(module, "replicated_shard_ids", ())),
            "quant_config_is_none": module.quant_config is None,
            "offered_to_quant_config": prefix in asked,
            "class_module": type(module).__module__,
            "quant_method_class": type(module.quant_method).__module__ + "." + type(module.quant_method).__qualname__,
            "tp_size": getattr(module, "tp_size", None),
            "tp_rank": getattr(module, "tp_rank", None),
            "input_size_per_partition": getattr(module, "input_size_per_partition", None),
            "output_size_per_partition": getattr(module, "output_size_per_partition", None),
            "parameter_shapes": {p: list(v.shape) for p, v in module.named_parameters(recurse=False)},
            "parameter_dtypes": {p: str(v.dtype) for p, v in module.named_parameters(recurse=False)},
        })
        # A pattern whose members disagree is a real fact, not a bug in the
        # census: record the disagreement rather than letting the last one win.
        for field, value in (("quant_config_is_none", module.quant_config is None),
                             ("offered_to_quant_config", prefix in asked),
                             ("quant_method", type(module.quant_method).__name__),
                             ("output_sizes", _output_sizes(module)),
                             ("class", type(module).__name__),
                             ("output_size", int(module.output_size)),
                             ("replicated_shard_ids", sorted(int(i) for i in getattr(module, "replicated_shard_ids", ())))):
            if row[field] != value:
                row.setdefault("disagreements", {}).setdefault(field, []).append(prefix)
    for row in rows.values():
        row["layers"] = sorted(row["layers"])
        row["input_sizes"] = sorted(row["input_sizes"])
    # Every prefix the probe WAS asked about that is not a LinearBase -- the
    # MoE modules, the LM head -- so a reader can tell "not a Linear" from
    # "never offered".
    non_linear = sorted({(normalise(p), c) for p, c in probe.asked
                         if normalise(p) not in rows})
    return {
        "linears": [rows[k] for k in sorted(rows)],
        "offered_non_linear": [{"prefix_pattern": p, "class": c, "instances": [
            {"prefix": prefix, "class_module": type(layer).__module__,
             "parameter_shapes": {name: list(value.shape) for name, value in layer.named_parameters(recurse=False)},
             "parameter_dtypes": {name: str(value.dtype) for name, value in layer.named_parameters(recurse=False)}}
            for prefix, layer in probe.asked_layers.items()
            if normalise(prefix) == p and type(layer).__name__ == c]} for p, c in non_linear],
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


def runtime_stamp(runtime_image: str) -> dict:
    """The runtime this receipt is scoped to, from the launcher's declaration.

    The image is a join key: ``construction_entry`` scopes every eligibility
    verdict to it and ``test_serving_construction`` compares it to the pin.
    So it is read the way the route census reads it (issue #132) -- from the
    ``TESSERA_CENSUS_RUNTIME_IMAGE*`` pair ``experiments/tessera_plugin_run.sh``
    exports after resolving docker's RepoDigests -- and checked against the
    image this run was asked for.  An operator-typed string is refused; the
    previous ``TESSERA_CENSUS_IMAGE`` env was exactly that, and a receipt
    taken without it said ``unstamped``, which the tree test then refused.
    """
    import torch
    import vllm
    # Deliberately local: this file's top level stays stdlib-only so the tree
    # test can load it beside any interpreter (see test_serving_construction).
    from tessera.serving.runtime_image import RuntimeImageError, declared_reference
    try:
        declaration = declared_reference(runtime_image)
    except RuntimeImageError as exc:
        raise SystemExit(f"--runtime-image {runtime_image}: {exc}") from None
    record = declaration["record"]
    stamp = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "image": declaration["image"],
        "image_id": record.get("local_id") or record.get("resolved_digest") or "unstamped",
        "image_declaration": declaration,
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

def preflight(model_path: str, runtime_image: str) -> dict:
    """Read actual configuration shapes and producer imports without construction."""
    import hashlib
    from pathlib import Path
    import torch
    from tessera.serving import dense_ownership, scheme, weights_mapper

    path = Path(model_path) / "config.json"
    raw = path.read_bytes()
    config = json.loads(raw)
    text = config.get("text_config", config)
    if not config.get("architectures"):
        raise ValueError("The model config must declare its architecture")
    layers = text.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0:
        raise ValueError("The model config must declare a positive layer count")
    for field in ("layer_types", "mlp_layer_types", "indexer_types"):
        if field in text and len(text[field]) != layers:
            raise ValueError(f"{field} does not match the model layer count")
    names = ("hidden_size", "intermediate_size", "moe_intermediate_size",
             "q_lora_rank", "kv_lora_rank", "num_attention_heads",
             "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
             "index_n_heads", "index_head_dim", "n_routed_experts")
    shapes = {name: text[name] for name in names if name in text}
    if any(type(value) is not int or value < 0 for value in shapes.values()):
        raise ValueError("The model dimensions must be nonnegative integers")
    if type(text.get("hidden_size")) is not int or text["hidden_size"] <= 0:
        raise ValueError("The hidden size must be positive")
    linear = text.get("linear_attn_config", {})
    heads = text.get("linear_num_heads", linear.get("num_heads"))
    head_dim = text.get("linear_head_dim", linear.get("head_dim"))
    geometry = []
    if heads is not None or head_dim is not None:
        if type(heads) is not int or type(head_dim) is not int or heads <= 0 or head_dim <= 0:
            raise ValueError("The KDA head count and head dimension must be positive")
        global_roles = [heads * head_dim] * 3 + [heads, head_dim, head_dim]
        for world in (1, 2):
            if heads % world:
                raise ValueError("The KDA head count must divide the requested tensor-parallel size")
            local = [r if index in (4, 5) else r // world for index, r in enumerate(global_roles)]
            geometry.append({"tp_size": world, "columns": text["hidden_size"],
                             "global_output_sizes": global_roles, "local_output_sizes": local,
                             "global_rows": sum(global_roles), "local_rows": sum(local),
                             "replicated_shard_ids": [4, 5]})
    vision = config.get("vision_config", {})
    if torch.cuda.is_initialized():
        raise RuntimeError("The portable preflight must not initialize CUDA")
    return {"schema": "tessera.construction-preflight.v1", "status": "preflight-only",
            "construction_performed": False, "model_construction_performed": False,
            "cpu_linear_construction_performed": False, "runtime_image_requested": runtime_image,
            "config": {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "architectures": config["architectures"], "num_hidden_layers": layers,
            "layer_types": text.get("layer_types"), "mlp_layer_types": text.get("mlp_layer_types"),
            "text_dimensions": shapes, "vision_dimensions": {name: vision[name] for name in
                ("hidden_size", "intermediate_size", "out_hidden_size", "projection_intermediate_size", "num_heads", "depth") if name in vision},
            "KDA_geometry": geometry, "producer_imports": [module.__name__ for module in
                (dense_ownership, scheme, weights_mapper)], "cuda_initialized": False}

def cpu_linear_construction(*, selected=False) -> dict:
    """Construct actual vLLM classes on CPU; do not substitute runtime objects."""
    import torch
    import vllm
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.platforms import current_platform
    from tools.tessera_projection_smoke import CASES, _construct

    if not current_platform.is_cpu():
        raise RuntimeError("The Linear construction check needs the real CPU platform")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=1))
    records = []
    with set_current_vllm_config(config, check_compile=False):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}", backend="gloo")
        initialize_model_parallel(1, 1)
        with torch.device("cpu"):
            for prefix, kind, columns, roles in CASES:
                row = {"prefix": prefix, "kind": kind, "columns": columns, "roles": roles}
                layer = _construct(row, 1, 0)
                expected = (sum(size for _, size in roles), columns)
                if tuple(layer.weight.shape) != expected or layer.weight.dtype != torch.bfloat16:
                    raise AssertionError(f"{prefix}: actual Linear weight shape or dtype differs")
                if layer.weight.device.type != "cpu":
                    raise AssertionError(f"{prefix}: actual Linear weights are not on CPU")
                if prefix.startswith("visual.") and tuple(layer.bias.shape) != (expected[0],):
                    raise AssertionError(f"{prefix}: actual vision bias shape differs")
                records.append({"prefix": prefix, "class": type(layer).__name__,
                                "class_module": type(layer).__module__, "weight_shape": list(expected),
                                "output_sizes": _output_sizes(layer),
                                "quant_method": type(layer.quant_method).__name__,
                                "replicated_shard_ids": sorted(getattr(layer, "replicated_shard_ids", ()))})
    if selected:
        from tessera.serving.projection_routes import install
        from tessera.serving.weights_mapper import module_name_mapper
        from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration
        mapper = module_name_mapper(Glm5NextForConditionalGeneration.hf_to_vllm_mapper)
        install()
        with set_current_vllm_config(config, check_compile=False), torch.device("cpu"):
            for case, record in zip(CASES, records):
                prefix, kind, columns, roles = case
                row = {"prefix": prefix, "kind": kind, "columns": columns, "roles": roles}
                config.quant_config = None
                control = _construct(row, 1, 0)
                if (type(control.quant_method).__name__, control.prefix, list(control.weight.shape)) != (
                        record["quant_method"], prefix, record["weight_shape"]):
                    raise AssertionError(f"{prefix}: the install hook changed stock construction")
                record["selected"] = []
                for family in ("TESSERA_FP8", "TESSERA_BF16"):
                    scheme = {"family": family, "grid": "E4M3" if family == "TESSERA_FP8" else "BF16",
                              "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                              "rows": sum(size for _, size in roles), "columns": columns,
                              "roles": [list(role) for role in roles], "wire_bytes": 1}
                    checkpoint = _checkpoint_name(prefix, mapper)
                    quant = _observed_config_class()(
                        {"cpu": {"targets": [checkpoint], "scheme": scheme}}, (), {"tp_agnostic": True})
                    quant.apply_vllm_mapper(mapper)
                    config.quant_config = quant
                    layer = _construct(row, 1, 0)
                    if (prefix, type(layer).__name__) not in quant.asked:
                        raise AssertionError(f"{prefix}: the real constructor did not call the selected config")
                    if tuple(layer.wire_bytes.shape) != (1,) or layer.wire_bytes.dtype != torch.uint8:
                        raise AssertionError(f"{prefix}: the selected constructor created an invalid wire parameter")
                    record["selected"].append({"family": family, "calls": list(quant.asked),
                                               "method": type(layer.quant_method).__name__,
                                               "output_sizes": _output_sizes(layer)})
    if torch.cuda.is_initialized():
        raise RuntimeError("The CPU Linear construction check initialized CUDA")
    return {"vllm": vllm.__version__, "vllm_file": vllm.__file__, "modules": records,
            "forward_executed": False, "weights_loaded": False, "cuda_initialized": False}


def _checkpoint_name(prefix, mapper):
    wire_name = prefix + ".weight"
    candidates = [old + wire_name[len(new):] for old, new in mapper.orig_to_new_prefix.items()
                  if wire_name.startswith(new)]
    if len(candidates) != 1 or mapper.apply_list(candidates) != [wire_name]:
        raise ValueError(f"{prefix}: the runtime mapper has no unique checkpoint name")
    return candidates[0][:-len(".weight")]


def _observed_config_class():
    from tessera.serving.config import TesseraConfig

    class ObservedTesseraConfig(TesseraConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.asked = []
            self.asked_layers = {}

        def get_quant_method(self, layer, prefix):
            self.asked.append((prefix, type(layer).__name__))
            self.asked_layers[prefix] = layer
            return super().get_quant_method(layer, prefix)

    return ObservedTesseraConfig


def selected_config(baseline_path, model_path, table_path, family):
    """Use the shared source owner and actual constructor partition metadata."""
    import fnmatch
    import importlib
    from pathlib import Path
    from tessera.serving import dense_ownership
    from tessera.serving.projection_routes import install, direct_consumer_resident_bytes
    from tessera.serving.weights_mapper import module_name_mapper

    baseline = json.loads(Path(baseline_path).read_text())
    source = json.loads((Path(model_path) / "config.json").read_text())
    table = json.loads(Path(table_path).read_text())
    module_name, _, class_name = baseline["model_class"].rpartition(".")
    model_class = getattr(importlib.import_module(module_name), class_name)
    mapper = module_name_mapper(model_class.hf_to_vllm_mapper)
    groups, ignored, selected = {}, set(), {}
    for row in baseline["linears"]:
        for instance in row["instances"]:
            prefix = instance["prefix"]
            descriptors = [entry for entry in table["modules"] if any(
                fnmatch.fnmatchcase(prefix, pattern) or fnmatch.fnmatchcase(prefix, "*." + pattern)
                for pattern in entry.get("runtime_leaves", [entry.get("runtime_leaf", "")]))]
            checkpoint = _checkpoint_name(prefix, mapper)
            if not descriptors:
                ignored.add(checkpoint)
                continue
            parent = checkpoint.rsplit(".", 1)[0]
            names = {parent + "." + member + ".weight" for descriptor in descriptors
                     for member in descriptor["source_members"]}
            members = (checkpoint + ".weight",)
            for name in sorted(names):
                owner = dense_ownership.fused_module(name, source["architectures"][0],
                                                     config=source, tensor_names=names)
                if owner is not None and owner[0] == checkpoint:
                    members = owner[1]
                    break
            sizes = instance["output_sizes"]
            padding = dense_ownership.source_padding_rows(checkpoint, members, source)
            if len(members) == len(sizes):
                source_rows = {name: size - padding.get(name, 0) for name, size in zip(members, sizes)}
            elif len(members) == 1:
                source_rows = {members[0]: sum(sizes) - padding.get(members[0], 0)}
            else:
                raise ValueError(f"{prefix}: {instance['class']} lacks explicit source-member partitions: {sizes}")
            partitions = dense_ownership.partition_members(checkpoint, members, source_rows, sizes,
                                                            padding_rows=padding)
            roles = [[member.role, member.rows] for member in partitions]
            declaration = {"family": family, "grid": "E4M3" if family == "TESSERA_FP8" else "BF16",
                           "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                           "rows": sum(size for _, size in roles), "columns": instance["input_size"],
                           "roles": roles, "wire_bytes": 1}
            groups[checkpoint] = {"targets": [checkpoint], "scheme": declaration}
            selected[prefix] = {"checkpoint": checkpoint, "roles": roles,
                                "class": instance["class"], "output_sizes": sizes,
                                "source_members": [dataclasses.asdict(member) for member in partitions],
                                "direct_consumer_bytes_if_loaded": direct_consumer_resident_bytes(
                                    prefix, family, declaration["rows"], declaration["columns"], roles)}
    for prefix, _cls in baseline["quant_config_calls"]:
        if prefix not in selected:
            ignored.add(_checkpoint_name(prefix, mapper))
    if not selected:
        raise ValueError("The selected construction view has no projection target")
    install()
    quant = _observed_config_class()(groups, tuple(sorted(ignored)), {"tp_agnostic": True})
    quant.apply_vllm_mapper(mapper)
    return quant, {"family": family, "targets": selected, "weights_loaded": False,
                   "wire_bytes_is_construction_marker": True}


def construction_views(args):
    """Run each view in a fresh process within one admitted action."""
    from pathlib import Path

    out = Path(args.out)
    baseline = out.with_name(out.stem + "-stock" + out.suffix)
    common = [sys.executable, __file__, args.model, "", "--device", args.device,
              "--runtime-image", args.runtime_image, "--max-model-len", str(args.max_model_len)]
    command = list(common)
    command[3] = str(baseline)
    subprocess.run(command, check=True)
    views = {"stock": str(baseline)}
    for family in ("TESSERA_FP8", "TESSERA_BF16"):
        path = out.with_name(out.stem + "-" + family + out.suffix)
        command = list(common)
        command[3] = str(path)
        command += ["--selection-from", str(baseline), "--module-table", args.module_table,
                    "--selected-family", family]
        subprocess.run(command, check=True)
        views[family] = str(path)
    with out.open("w") as handle:
        json.dump({"schema": "tessera.construction-views.v1", "status": "construction-only",
                   "weights_loaded": False, "forward_executed": False, "views": views}, handle, indent=1)
    return 0



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model or config-only directory (no weights are read)")
    ap.add_argument("out", help="receipt JSON to write")
    ap.add_argument("--device", default="meta",
                    help="construction device; meta allocates nothing (default)")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true",
                    help="import producer code and read actual shapes; do not construct a model")
    ap.add_argument("--linear-checks", action="store_true",
                    help="construct actual small vLLM Linears on the CPU in dry-run mode")
    ap.add_argument("--with-selection", action="store_true", help="record stock, T-8, and T-16 views in fresh processes")
    ap.add_argument("--selection-from", help="the stock census for an internal selected view")
    ap.add_argument("--module-table", help="the fixed attention projection contract")
    ap.add_argument("--selected-family", choices=("TESSERA_FP8", "TESSERA_BF16"), default="TESSERA_FP8")
    ap.add_argument("--runtime-image", required=True,
                    help="the image this census is scoped to; must equal the reference the "
                         "launcher (experiments/tessera_plugin_run.sh) declared for this container")
    args = ap.parse_args()
    if args.linear_checks and not args.dry_run:
        ap.error("--linear-checks requires --dry-run")
    if (args.with_selection or args.selection_from) and not args.module_table:
        ap.error("selected construction requires --module-table")
    if args.dry_run:
        receipt = preflight(args.model, args.runtime_image)
        if args.module_table:
            with open(args.module_table) as handle:
                table = json.load(handle)
            receipt["module_table_groups"] = len(table["modules"])
            if args.with_selection:
                from tessera.serving.dense_ownership import source_padding_rows
        if args.linear_checks:
            receipt["cpu_linear_construction"] = cpu_linear_construction(selected=args.with_selection)
            receipt["construction_performed"] = True
            receipt["cpu_linear_construction_performed"] = True
            receipt["status"] = "cpu-linear-construction-only"
        with open(args.out, "w") as handle:
            json.dump(receipt, handle, indent=1)
        print(json.dumps({"status": receipt["status"], "construction_performed": receipt["construction_performed"],
                          "model_construction_performed": receipt["model_construction_performed"],
                          "cpu_linear_construction_performed": receipt["cpu_linear_construction_performed"],
                          "num_hidden_layers": receipt["num_hidden_layers"],
                          "KDA_geometry": receipt["KDA_geometry"]}))
        return 0
    if args.with_selection:
        return construction_views(args)
    # Refuse BEFORE constructing the model: a receipt scoped to nothing is
    # not worth the build.
    stamp = runtime_stamp(args.runtime_image)

    started = time.time()
    selection = None
    quant = None
    if args.selection_from:
        quant, selection = selected_config(args.selection_from, args.model, args.module_table, args.selected_family)
    try:
        model, probe, vllm_config = build_model(args.model, args.device, args.max_model_len, quant_config=quant)
    except Exception:
        if quant is not None:
            with open(args.out + ".failure", "w") as handle:
                json.dump({"schema": "tessera.construction-failure.v1", "construction_complete": False,
                           "selection": selection, "quant_config_calls": quant.asked}, handle, indent=1)
        raise
    body = census(model, probe)
    model_class = type(model)
    receipt = {
        "schema": "tessera.construction-census.v1",
        "taken": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
        "runtime": stamp,
        "model": model_stamp(vllm_config, args.model),
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "construction_device": args.device,
        # The two tables vLLM hands a quant config so it can match a fused
        # module -- published here because a producer that must name vLLM's
        # module and not the checkpoint's leaf needs exactly them.
        "packed_modules_mapping": getattr(model_class, "packed_modules_mapping", None),
        "hf_to_vllm_mapper_unstacked": _weights_mapper_table(model_class),
        "supports_quant": _supports_quant(model_class),
        "quant_config_calls": list(probe.asked),
        "view": "selected" if selection is not None else "stock",
        "selection": selection,
        "forward_executed": False, "weights_loaded": False,
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
    if selection is not None:
        called = {prefix for prefix, _cls in probe.asked}
        receipt["selection"]["missing_calls"] = sorted(set(selection["targets"]) - called)
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=False)
    print(json.dumps(receipt["summary"], indent=1))
    for row in receipt["linears"]:
        if not row["offered_to_quant_config"]:
            print(f"  NEVER OFFERED  {row['prefix_pattern']}  "
                  f"({row['class']}, layers {row['layers']})")
    if selection is not None and receipt["selection"]["missing_calls"]:
        raise RuntimeError("An explicit projection target received no quantization method call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
