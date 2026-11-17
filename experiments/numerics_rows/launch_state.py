"""Bind worker-start environment inputs to the frozen launch-lever policy."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.numerics_rows.common import Invalid, sha256

LEVERS = Path(__file__).with_name("launch-levers.tsv")
PREFIXES = ("TESSERA_", "TRITON_", "TORCH_", "PYTORCH_", "CUBLAS_", "CUBLASLT_", "CUDNN_", "NVIDIA_")
IMPORTS = ("tessera", "torch", "triton", "vllm")
TOOLCHAIN = ("PATH", "CUDA_HOME", "CUDA_PATH", "PYTORCH_NVCC", "PYTORCH_ROCM_ARCH", "ROCM_HOME", "ROCM_PATH")
MATH_PATHS = (
    "backends.cuda.matmul.allow_tf32", "backends.cudnn.allow_tf32",
    "backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
    "backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
    "backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k",
    "backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k",
    "backends.cuda.matmul.allow_fp16_accumulation",
    "backends.cudnn.deterministic", "backends.cudnn.benchmark",
    "backends.fp32_precision", "backends.cuda.matmul.fp32_precision",
    "backends.cudnn.conv.fp32_precision", "backends.cudnn.rnn.fp32_precision",
)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":")).encode("ascii")).hexdigest()


def math_settings(torch):
    settings = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
    for path in MATH_PATHS:
        value = torch
        for part in path.split("."):
            value = getattr(value, part, None)
        if value is not None:
            settings[path] = value
    cuda = getattr(getattr(torch, "backends", None), "cuda", None)
    preferred = getattr(cuda, "preferred_blas_library", None)
    if callable(preferred):
        settings["preferred_blas_library"] = str(preferred())
    return settings


def digest(mapping):
    """SHA-256 of sorted compact ensure_ascii JSON, ASCII, with no final newline.

    The object maps included environment names to their exact string values.
    Absent names are omitted. Empty strings are retained. No value is parsed.
    """
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise Invalid("launch state must map names to exact strings")
    return json_digest(mapping)


class LaunchState:
    def __init__(self, path=LEVERS, *, fresh=False):
        imported = sorted({name.split(".")[0] for name in sys.modules if name.split(".")[0] in IMPORTS})
        if fresh and imported:
            raise Invalid("launch state requires a fresh worker before its first Tessera import or torch, Triton or vLLM import: " + ", ".join(imported))
        self.path, self.policy = Path(path), {}
        for row in csv.DictReader(self.path.read_text().splitlines(), delimiter="\t"):
            name, policy = row.get("name"), row.get("launch_state")
            if not isinstance(name, str) or not name:
                raise Invalid("launch lever policy has no environment name")
            if policy not in ("include", "exclude", "not launch-changing") or self.policy.get(name, policy) != policy:
                raise Invalid("launch lever policy is malformed or inconsistent")
            self.policy[name] = policy
        if not self.policy or self.policy.get("TESSERA_SERVE_MODE") != "include":
            raise Invalid("launch lever policy must retain the residency selector")
        if any(self.policy.get(name) != "exclude" for name in TOOLCHAIN):
            raise Invalid("launch lever policy must exclude the seven toolchain diagnostics")
        if self.policy.get("PRISMAQUANT_DEV_MODE") != "include":
            raise Invalid("launch lever policy must include PRISMAQUANT_DEV_MODE")
        self.levers_sha256 = sha256(self.path)
        self.mapping = self.current()
        self.sha256 = digest(self.mapping)
        self.torch = None

    def bind_math(self, torch):
        """Snapshot effective settings after torch import, before Tessera import.

        Keep the settings fixed, without changing the pinned image's defaults.
        Every caller uses check() before preparation and native execution.
        """
        if self.torch is not None:
            self.check()
            return
        self.torch = torch
        self.math = math_settings(torch)
        self.math_sha256 = json_digest(self.math)

    def current(self):
        return {k: v for k, v in sorted(os.environ.items()) if (
            self.policy.get(k) != "exclude" and (k.startswith(PREFIXES) or self.policy.get(k) == "include"))}

    def check(self):
        if self.current() != self.mapping:
            raise Invalid("launch state changed after worker start")
        self.check_math()

    def check_math(self):
        if self.torch is not None and math_settings(self.torch) != self.math:
            raise Invalid("dependency math settings changed after worker start")

    def fields(self):
        result = {"launch_state": dict(self.mapping), "launch_state_sha256": self.sha256,
                  "launch_levers_sha256": self.levers_sha256}
        if self.torch is not None:
            result.update(dependency_math=dict(self.math), dependency_math_sha256=self.math_sha256)
        return result

    def diagnostics(self):
        return {name: os.environ[name] for name in sorted(self.policy) if self.policy[name] == "exclude" and name in os.environ}


def check_binding(record, expected):
    for field in ("launch_state", "launch_state_sha256", "launch_levers_sha256"):
        if record.get(field) != expected[field]:
            raise Invalid(f"C1 launch state or lever file differs: {field}")


def check_math_binding(record, expected):
    settings = record.get("dependency_math")
    if not isinstance(settings, dict) or not settings or json_digest(settings) != record.get("dependency_math_sha256"):
        raise Invalid("dependency math settings are missing or malformed")
    if any(record.get(k) != expected.get(k) for k in ("dependency_math", "dependency_math_sha256")):
        raise Invalid("dependency math settings differ")
