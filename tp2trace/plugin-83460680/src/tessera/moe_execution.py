"""Explicit research execution metadata shared by producer and serving readers.

This grammar selects an existing runtime owner; it does not qualify a wire,
change encoder inputs, or establish a production serving cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .serving.scheme import (MOE_BUILDERS, ROUTES, STRUCTURE_ROUTED_MOE, TESSERA_BF16,
                             TESSERA_FP8, route_for_grid)

#: The routed ``(family, grid)`` pairs the selected owner decodes.  A routed
#: stack on any OTHER family is not this block's to serve: it is dispatched to
#: that family's own production builder (``scheme.MOE_BUILDERS``) whether or
#: not the checkpoint carries this block, and the block neither covers nor
#: refuses it (tessera#492: an NVFP4 expert stack beside selected FP8 ones).
SELECTED_TARGETS = ((TESSERA_FP8, "E4M3"), (TESSERA_BF16, "BF16"))


@dataclass(frozen=True)
class ResearchSelectedMoeConfig:
    """Explicit bounded research construction, including versioned checkpoints.

    This does not name a production residency mode or a qualified runtime cell.
    Only an explicit checkpoint block asks ``TesseraConfig`` to supply it.
    The bound limits decoder
    temporaries, not the final selected FP8 stack or whole-engine workspace.
    """

    max_experts_per_chunk: int
    decode_backend: str = "torch"
    expected_tensor_parallel_size: int = 1

    def __post_init__(self):
        if type(self.max_experts_per_chunk) is not int or self.max_experts_per_chunk <= 0:
            raise ValueError("max_experts_per_chunk must be a positive integer")
        if type(self.expected_tensor_parallel_size) is not int or self.expected_tensor_parallel_size not in (1, 2):
            raise ValueError("expected_tensor_parallel_size must be exactly 1 or 2")
        if self.decode_backend not in ("torch", "triton"):
            raise ValueError(f"unknown selected window backend {self.decode_backend!r}")

    @classmethod
    def from_checkpoint(cls, block: Mapping) -> "ResearchSelectedMoeConfig":
        """The closed artifact grammar; omission is handled by the caller."""
        fields = {"schema", "max_experts_per_chunk", "decode_backend",
                  "expected_tensor_parallel_size"}
        if not isinstance(block, Mapping) or set(block) != fields:
            raise ValueError(f"research_selected_moe requires exactly {sorted(fields)}")
        if block["schema"] != "tessera.research_selected_moe.v1":
            raise ValueError("research_selected_moe has an unknown schema")
        try:
            return cls(**{name: block[name] for name in fields if name != "schema"})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"research_selected_moe: {exc}") from exc

    def as_checkpoint(self) -> dict:
        return {"schema": "tessera.research_selected_moe.v1",
                "max_experts_per_chunk": self.max_experts_per_chunk,
                "decode_backend": self.decode_backend,
                "expected_tensor_parallel_size": self.expected_tensor_parallel_size}

    @staticmethod
    def applies_to(scheme: Mapping) -> bool:
        """Is this routed scheme one the selected owner serves?

        ``TesseraConfig.get_quant_method`` asks this per stack: a selected
        pair goes to the research owner, any other routed family to its own
        production builder.  Structure is the caller's to check.
        """
        return (scheme.get("family"), scheme.get("grid")) in SELECTED_TARGETS

    def require_targets(self, target_schemes: Mapping, mode: str) -> None:
        """Fail before loading; runtime geometry/backend guards remain at construction.

        Every routed target must be one this block serves OR one with its
        own production builder; at least one must be this block's, or the
        block names nothing.  A routed family with neither is refused here by
        name rather than at load.
        """
        if mode != "resident":
            raise ValueError("research_selected_moe requires resident mode")
        routed = {name: scheme for name, scheme in target_schemes.items()
                  if scheme.get("structure") == STRUCTURE_ROUTED_MOE}
        if not routed:
            raise ValueError("research_selected_moe requires a declared routed_moe target")
        selected = [name for name, scheme in routed.items() if self.applies_to(scheme)]
        for name, scheme in routed.items():
            if name in selected or scheme.get("family") in MOE_BUILDERS:
                continue
            raise ValueError(f"research_selected_moe target {name!r} requires "
                             "TESSERA_FP8/E4M3 or TESSERA_BF16/BF16, or a family with its "
                             f"own expert builder ({sorted(MOE_BUILDERS)})")
        if not selected:
            raise ValueError("research_selected_moe names no routed target it serves "
                             "(TESSERA_FP8/E4M3 or TESSERA_BF16/BF16); every declared "
                             "routed stack takes its own production builder")

    def require_wire_recipe(self, *, grid: str, q256: int, body: str,
                            plane: str, span: int, target: str) -> str:
        """Research selected decoder reach, separate from production cells.

        A readable dense recipe is necessary for the selected owner but says
        nothing about TP2 geometry, runtime execution, or device qualification.
        Those are checked at construction and in served validation.
        """
        from .serving.contract import reader_accepts, reader_rate_grid

        family = route_for_grid(grid)
        if (family, grid) not in ((TESSERA_FP8, "E4M3"), (TESSERA_BF16, "BF16")):
            raise ValueError(f"research_selected_moe target {target!r}: no selected "
                             f"decoder for grid {grid!r}")
        route = ROUTES[family]
        if (body, plane, int(span)) != (route["body"], route["plane"], route["span"]):
            raise ValueError(f"research_selected_moe target {target!r}: {family} requires "
                             f"{route['body']}/{route['plane']}/span-{route['span']}, "
                             f"got {body}/{plane}/span-{span}")
        found = reader_rate_grid(family, grid)
        if found is None or not reader_accepts(int(q256), *found[1:]):
            raise ValueError(f"research_selected_moe target {target!r}: q256={q256} "
                             f"is outside the {family} selected decoder's published reader range")
        return family


@dataclass(frozen=True)
class ResearchSelectedMoeInput:
    """One immutable input snapshot, carried through export and partition merge.

    The exact UTF-8 text and its digest bind whitespace as well as values. The
    checkpoint contains only the validated execution object. Neither record
    is an encoder-source identity or a device qualification.
    """

    input_utf8: str
    config: ResearchSelectedMoeConfig

    @classmethod
    def read(cls, path: Path) -> "ResearchSelectedMoeInput":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ResearchSelectedMoeInput":
        def unique_pairs(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f"research_selected_moe duplicates field {key!r}")
                out[key] = value
            return out
        try:
            text = raw.decode("utf-8")
            block = json.loads(text, object_pairs_hook=unique_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"research_selected_moe input must be UTF-8 JSON: {exc}") from exc
        return cls(text, ResearchSelectedMoeConfig.from_checkpoint(block))

    def record(self) -> dict:
        return {"input_sha256": hashlib.sha256(self.input_utf8.encode("utf-8")).hexdigest(),
                "input_utf8": self.input_utf8, "config": self.config.as_checkpoint()}

    @classmethod
    def from_record(cls, record: Mapping) -> "ResearchSelectedMoeInput":
        if (not isinstance(record, Mapping)
                or set(record) != {"input_sha256", "input_utf8", "config"}
                or not isinstance(record["input_utf8"], str)):
            raise ValueError("research_selected_moe input record has missing or unknown fields")
        snapshot = cls.from_bytes(record["input_utf8"].encode("utf-8"))
        # Python equates True/1 and 2.0/2; the carried JSON has the same typed
        # grammar as the original input even when those values compare equal.
        ResearchSelectedMoeConfig.from_checkpoint(record["config"])
        if snapshot.record() != record:
            raise ValueError("research_selected_moe input record digest/content mismatch")
        return snapshot
