#!/usr/bin/env python3
"""Ask the pinned serving build what it does with a Tessera wire expert parameter.

WHY THIS EXISTS.  Issue #5 records, as a premise for the expert route, that
``RoutedExperts.build_expert_params_mapping`` is suffix-agnostic
(``experts.routed_experts.w13_`` / ``w2_``) "so custom suffixes route fine".
Half of that is right and the half that is wrong is the expensive half: the
MAPPING routes a custom suffix, and the LOADER then drops it.  Before anything
registers ``w13_wire`` in ``create_weights``, the runtime should be the one to
say so -- principle 14, the same reason the NVFP4 oracle probe exists beside a
reading of the oracle's source.

WHAT IT ASKS, in three legs, each a narrower question than the last:

* **mapping** -- build the expert parameter mapping for a small layer and
  record the rows.  This is where the "suffix-agnostic" claim lives: the
  mapping's ``param_name`` is a PREFIX, ``experts.routed_experts.w13_``, and
  the checkpoint's suffix is whatever follows it.
* **rewrite** -- replay the name rewrite ``load_weights`` performs before it
  calls the loader (``qual_name.replace(weight_name, param_name)``), and record
  whether the result contains ``weight`` or ``scale``.  Those two substrings
  are what ``weight_loader``'s dispatch tests, so the rewrite is where a
  suffix's fate is decided.
* **execute** -- CALL ``RoutedExperts.weight_loader`` on a wire parameter and
  on a ``weight``-named control of the same shape family, and record what it
  returns and whether the parameter was written.  A returned ``False`` is
  ``load_weights`` yielding nothing for that tensor: the wire never lands, and
  no exception says so.

The control is the point of the third leg.  A wire parameter that does not
load proves nothing on its own -- the stand-in for ``self`` here is minimal, so
a failure could be the stand-in's.  The control is the same stand-in, the same
call, one substring different in the name; if it reaches a loading branch and
the wire does not, the name is what separated them.

SCOPE.  This is a claim about the LOADER, on this build, for these parameter
names.  It is not a claim about a route (there is none), about what a served
MoE would do, or about any other vLLM version.  No GPU, no model, no port: it
imports ``routed_experts`` and calls two functions, so it needs no serve lock.
"""
from __future__ import annotations

import json
import sys

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

#: A small layer, because the mapping's shape is what is being read, not a
#: model's dimensions.  Two experts is the smallest count at which the mapping
#: has to index experts at all.
EXPERTS = 2
LAYER = "model.language_model.layers.1.mlp.experts"

#: The suffixes a Tessera expert route would register, beside the two the
#: stock methods register.  ``wire``/``wire_len`` are ``moe_layout.MoePacked``'s
#: names; ``weight``/``weight_scale`` are what ``UnquantizedFusedMoEMethod``
#: and ``CompressedTensorsW8A8Fp8MoEMethod`` register, and they are the control.
SUFFIXES = ("weight", "weight_scale", "wire", "wire_len")


class _Stand(RoutedExperts):
    """A stand-in carrying only what ``weight_loader`` reads.

    Subclassed rather than faked so every ``_load_*`` helper the dispatch may
    reach is the REAL one -- a hand-written stub would answer "did my stub have
    that method", which is a different question.  ``__init__`` is deliberately
    not the parent's: building a real ``RoutedExperts`` needs a parallel state,
    a quant config and a device.
    """

    def __init__(self, tp_size: int = 1):
        torch.nn.Module.__init__(self)
        self.quant_config = None
        self.quant_method = _StockMethod()
        self.moe_config = _MoeConfig(tp_size)
        self.layer_name = LAYER
        self._loaded_expert_biases = set()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id


class _StockMethod:
    """A quant method whose CLASS NAME is not one the loader special-cases."""


class _Parallel:
    def __init__(self, tp_size: int):
        self.tp_size = tp_size
        self.enable_eplb = False


class _MoeConfig:
    def __init__(self, tp_size: int):
        self.tp_rank = 0
        self.tp_size = tp_size
        self.is_act_and_mul = True
        self.moe_parallel_config = _Parallel(tp_size)


def _mapping():
    """The mapping the LAYER builds, not the legacy default.

    ``get_expert_mapping`` passes ``routed_experts_prefix=""``, so the real
    parameter prefixes are ``experts.w13_`` / ``experts.w2_``; the staticmethod's
    own default (``"routed_experts"``) is the legacy entry point's and would put
    a segment in the name that no live layer carries.
    """
    return RoutedExperts.build_expert_params_mapping(
        "gate_proj", "down_proj", "up_proj", num_experts=EXPERTS,
        routed_experts_prefix="")


def mapping_leg() -> dict:
    rows = _mapping()
    return {
        "rows": [list(r) for r in rows],
        "param_name_prefixes": sorted({r[0] for r in rows}),
    }


def rewrite_leg(rows) -> dict:
    """What ``load_weights`` hands ``weight_loader`` as ``weight_name``."""
    out = {}
    for suffix in SUFFIXES:
        checkpoint = f"{LAYER}.0.gate_proj.{suffix}"
        record = {"checkpoint_tensor": checkpoint, "matched": False}
        for param_name, weight_name, expert_id, shard_id in rows:
            if weight_name not in checkpoint:
                continue
            rewritten = checkpoint.replace(weight_name, param_name)
            record.update({
                "matched": True,
                "param_attribute": rewritten.removeprefix(f"{LAYER}."),
                "weight_name_seen_by_loader": rewritten,
                "contains_weight": "weight" in rewritten,
                "contains_scale": "scale" in rewritten,
                "shard_id": shard_id, "expert_id": expert_id,
            })
            break
        out[suffix] = record
    return out


def execute_leg() -> dict:
    """Call the loader on a wire parameter and on a ``weight`` control."""
    stand = _Stand()
    rows, s13, s2, hidden = EXPERTS, 64, 48, 16
    cases = {
        # The wire layout step 3 built (``tessera.moe_layout.MoePacked``).
        "w13_wire": (torch.zeros(rows, 2, s13, dtype=torch.uint8),
                     torch.arange(1, 41, dtype=torch.uint8), "w1"),
        "w2_wire": (torch.zeros(rows, s2, dtype=torch.uint8),
                    torch.arange(1, 41, dtype=torch.uint8), "w2"),
        "w13_wire_len": (torch.zeros(rows, 2, dtype=torch.long),
                         torch.tensor([40], dtype=torch.long), "w1"),
        # The control: the stock name, a shape the loader is built for.
        "w13_weight": (torch.zeros(rows, 2 * s13, hidden),
                       torch.ones(s13, hidden), "w1"),
        "w2_weight": (torch.zeros(rows, hidden, s2),
                      torch.ones(hidden, s2), "w2"),
    }
    out = {}
    for attribute, (data, loaded, shard) in cases.items():
        param = torch.nn.Parameter(data, requires_grad=False)
        before = float(param.data.to(torch.float64).abs().sum())
        record = {"shard_id": shard, "param_shape": list(param.shape),
                  "loaded_shape": list(loaded.shape)}
        try:
            returned = RoutedExperts.weight_loader(
                stand, param=param, loaded_weight=loaded,
                weight_name=f"{LAYER}.routed_experts.{attribute}",
                shard_id=shard, expert_id=0, return_success=True)
            after = float(param.data.to(torch.float64).abs().sum())
            record.update({"returned": returned, "raised": None,
                           "parameter_written": after != before})
        except Exception as exc:  # noqa: BLE001 -- the outcome IS the record
            record.update({"returned": None, "parameter_written": False,
                           "raised": f"{type(exc).__name__}: {exc}"})
        out[attribute] = record
    return out


def remedy_leg() -> dict:
    """Does ``load_weights`` call the parameter's OWN ``weight_loader``?

    If it does, a route registers its wire parameters with a loader of its own
    and the dispatch above is never in the path -- which is the difference
    between "this cannot be loaded" and "this must be loaded by the route".
    Asked by execution: the parameter below carries a callable that records its
    call, and only ``load_weights``'s body is under test (the mapping is the
    layer's own, and nothing else is stubbed).
    """
    stand = _Stand()
    calls = []

    def our_loader(param, loaded_weight, weight_name, shard_id, expert_id,
                   return_success=False):
        calls.append({"weight_name": weight_name, "shard_id": shard_id,
                      "expert_id": expert_id,
                      "loaded_shape": list(loaded_weight.shape)})
        param.data[expert_id, 0, :loaded_weight.numel()] = loaded_weight
        return True if return_success else None

    wire = torch.nn.Parameter(torch.zeros(EXPERTS, 2, 64, dtype=torch.uint8),
                              requires_grad=False)
    wire.weight_loader = our_loader
    stand.register_parameter("w13_wire", wire)
    rows = _mapping()
    stand.get_expert_mapping = lambda include_fused=False: rows  # type: ignore[assignment]

    yielded = list(RoutedExperts.load_weights(stand, [
        ("0.gate_proj.wire", torch.arange(1, 41, dtype=torch.uint8)),
    ]))
    return {"yielded": yielded, "calls": calls,
            "parameter_written": bool(int(wire.data.sum()))}


def main() -> int:
    import vllm

    rows = _mapping()
    record = {
        "vllm": getattr(vllm, "__version__", "unknown"),
        "torch": torch.__version__,
        "mapping": mapping_leg(),
        "rewrite": rewrite_leg(rows),
        "execute": execute_leg(),
    }
    try:
        record["remedy"] = remedy_leg()
    except Exception as exc:  # noqa: BLE001 -- an unanswered leg is a recorded one
        record["remedy"] = {"raised": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(record, indent=2))
    print("TESSERA_MOE_LOADER_JSON " + json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
