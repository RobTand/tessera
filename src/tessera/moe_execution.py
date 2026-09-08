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

from .serving.scheme import STRUCTURE_ROUTED_MOE, TESSERA_FP8

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

    def require_targets(self, target_schemes: Mapping, mode: str) -> None:
        """Fail before loading; runtime geometry/backend guards remain at construction."""
        if mode != "resident":
            raise ValueError("research_selected_moe requires resident mode")
        routed = {name: scheme for name, scheme in target_schemes.items()
                  if scheme.get("structure") == STRUCTURE_ROUTED_MOE}
        if not routed:
            raise ValueError("research_selected_moe requires a declared routed_moe target")
        for name, scheme in routed.items():
            if scheme.get("family") != TESSERA_FP8 or scheme.get("grid") != "E4M3":
                raise ValueError(f"research_selected_moe target {name!r} requires TESSERA_FP8/E4M3")


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
        if snapshot.record() != record:
            raise ValueError("research_selected_moe input record digest/content mismatch")
        return snapshot
