"""``tools/tessera_route_census.py`` joins its records in MODULE space.

The census reads a route record off every entry of ``named_modules()`` and
matches it to the families ``config_groups`` declares.  Those two are written
in different namespaces whenever the model class carries an
``hf_to_vllm_mapper``: the records are in the namespace vLLM built, the targets
in the checkpoint's.  Every census taken before these tests was on Qwen3-0.6B,
whose class declares no mapper -- so the two spaces coincided and the omission
could not show.  On ``Glm5NextForConditionalGeneration`` the mapper is
``{"model.language_model." -> "language_model.model.", ...}`` and nothing would
have joined, which the census would have reported as every served module
lacking a declaration: the opposite of what is true.

These pin the semantics of the census's USE of the mapper.  The mapper here is
a stub; that the replay matches vLLM's own ``WeightsMapper`` is attested
elsewhere (``tests/test_serving_name_mapping.py``, and the census asks the live
model class for its mapper rather than restating a table).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"


def _tool():
    spec = importlib.util.spec_from_file_location("tessera_route_census", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # its top level imports stdlib only
    return module


class _Unstacked:
    """The one method the census calls on vLLM's unstacked mapper."""

    def __init__(self, prefixes, drop=()):
        self._prefixes = dict(prefixes)
        self._drop = set(drop)

    def apply_list(self, names):
        out = []
        for name in names:
            if name in self._drop:
                continue
            for old, new in self._prefixes.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            out.append(name)
        return out


class _Mapper:
    def __init__(self, unstacked):
        self._unstacked = unstacked

    def get_unstacked_mapper(self):
        return self._unstacked


class _Model:
    def __init__(self, mapper=None):
        if mapper is not None:
            self.hf_to_vllm_mapper = mapper


def test_a_model_with_no_mapper_reports_no_translation():
    """``None`` means checkpoint space IS module space -- Qwen3-0.6B's case."""
    assert _tool().declared_in_module_space(_Model(), ["model.layers.0.mlp.down_proj"]) is None


def test_a_mapped_architecture_translates_every_target():
    mapper = _Mapper(_Unstacked({"model.language_model.": "language_model.model.",
                                 "model.visual.": "visual."}))
    targets = ["model.language_model.layers.1.mlp.experts",
               "model.language_model.layers.0.mlp.down_proj",
               "model.visual.blocks.3.mlp.down_proj"]
    assert _tool().declared_in_module_space(_Model(mapper), targets) == {
        "model.language_model.layers.1.mlp.experts": "language_model.model.layers.1.mlp.experts",
        "model.language_model.layers.0.mlp.down_proj":
            "language_model.model.layers.0.mlp.down_proj",
        "model.visual.blocks.3.mlp.down_proj": "visual.blocks.3.mlp.down_proj"}


def test_a_target_the_mapper_drops_is_reported_as_no_module():
    """A dropped target is not the identity: the runtime builds nothing for it."""
    dead = "model.dead.layers.0.proj"
    mapper = _Mapper(_Unstacked({"model.": "lm."}, drop=[dead]))
    got = _tool().declared_in_module_space(_Model(mapper), [dead, "model.layers.0.proj"])
    assert got[dead] is None
    assert got["model.layers.0.proj"] == "lm.layers.0.proj"


@pytest.mark.parametrize("target", ["Linear", "re:.*down_proj"])
def test_a_class_name_or_a_regex_target_is_left_alone(target):
    """compressed-tensors' own target shapes: only a dotted path is a path."""
    mapper = _Mapper(_Unstacked({"model.": "lm."}))
    assert _tool().declared_in_module_space(_Model(mapper), [target]) == {target: target}
