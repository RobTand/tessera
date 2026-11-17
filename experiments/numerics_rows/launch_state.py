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
        self.path, self.policy, self.public = Path(path), {}, {}
        for row in csv.DictReader(self.path.read_text().splitlines(), delimiter="\t"):
            name, policy = row.get("name"), row.get("launch_state")
            if not isinstance(name, str) or not name:
                raise Invalid("launch lever policy has no environment name")
            if policy not in ("include", "exclude", "not launch-changing") or self.policy.get(name, policy) != policy:
                raise Invalid("launch lever policy is malformed or inconsistent")
            if policy == "exclude" and not all(row.get(k) not in (None, "", "-") for k in ("exclusion_reason", "evidence", "file:line")):
                raise Invalid("launch exclusion has no cited reason: " + name)
            self.policy[name] = policy
            values = row.get("public_values", "-")
            if values != "-":
                allowed = json.loads(values)
                if policy != "include" or not (name.startswith("TESSERA_") or name == "PRISMAQUANT_DEV_MODE") or not isinstance(allowed, list) or any(not isinstance(v, str) for v in allowed):
                    raise Invalid("malformed non-secret enum policy: " + name)
                if name in self.public and self.public[name] != allowed:
                    raise Invalid("inconsistent non-secret enum policy: " + name)
                self.public[name] = allowed
        if not self.policy or self.policy.get("TESSERA_SERVE_MODE") != "include":
            raise Invalid("launch lever policy must retain the residency selector")
        if any(self.policy.get(name) != "exclude" for name in TOOLCHAIN):
            raise Invalid("launch lever policy must exclude the seven toolchain diagnostics")
        if self.policy.get("PRISMAQUANT_DEV_MODE") != "include":
            raise Invalid("launch lever policy must include PRISMAQUANT_DEV_MODE")
        self.levers_sha256 = sha256(self.path)
        self.start_names = set(os.environ)
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
        return {k: v for k, v in sorted(os.environ.items()) if self.policy.get(k) != "exclude"}

    def check(self):
        changed = sorted(k for k, v in self.mapping.items() if os.environ.get(k) != v)
        if changed:
            raise Invalid("launch state changed after worker start: " + ", ".join(changed))
        self.check_math()

    def check_math(self):
        if self.torch is not None and math_settings(self.torch) != self.math:
            raise Invalid("dependency math settings changed after worker start")

    def fields(self):
        from experiments.numerics_rows.start_env import specification_sha256

        result = {"launch_state_names": sorted(self.mapping), "launch_state_sha256": self.sha256,
                  "launch_levers_sha256": self.levers_sha256,
                  "start_environment_spec_sha256": specification_sha256(),
                  "launch_selectors": self.public_values(self.mapping)}
        if self.torch is not None:
            result.update(dependency_math=dict(self.math), dependency_math_sha256=self.math_sha256)
        return result

    def diagnostics(self):
        return {"excluded_names": sorted(k for k in os.environ if self.policy.get(k) == "exclude"),
                "added_names": sorted(set(os.environ) - self.start_names)}

    def public_values(self, mapping):
        """Expose only exact boolean/enumerated values named by the frozen policy."""
        return {k: mapping[k] for k in sorted(mapping) if k in self.public and mapping[k] in self.public[k]}

    def latched_diagnostic(self, mapping):
        return {"names": sorted(mapping), "non_secret_values": self.public_values(mapping)}


def check_fields(record):
    """Validate names-only evidence. Values remain solely in the live worker."""
    names = record.get("launch_state_names")
    if not isinstance(names, list) or any(not isinstance(k, str) or not k for k in names) or names != sorted(set(names)):
        raise Invalid("launch state names are missing or malformed")
    if "launch_state" in record:
        raise Invalid("launch state values must not be recorded")
    for key in ("launch_state_sha256", "launch_levers_sha256", "start_environment_spec_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise Invalid("launch state binding is missing or malformed: " + key)
    selectors = record.get("launch_selectors")
    if not isinstance(selectors, dict) or set(selectors) - set(names):
        raise Invalid("launch state selectors are missing or malformed")
    public = LaunchState().public
    if any(not isinstance(v, str) or k not in public or v not in public[k] for k, v in selectors.items()):
        raise Invalid("launch state diagnostic contains a name outside the non-secret list")


def check_binding(record, expected, label="C1"):
    try:
        check_fields(record)
        check_fields(expected)
    except Invalid as error:
        raise Invalid(label + " launch state or lever file: " + str(error)) from error
    for field in ("launch_state_names", "launch_state_sha256", "launch_levers_sha256", "launch_selectors", "start_environment_spec_sha256"):
        if record.get(field) != expected[field]:
            raise Invalid(f"{label} launch state or lever file differs: {field}")


def check_math_binding(record, expected):
    settings = record.get("dependency_math")
    if not isinstance(settings, dict) or not settings or json_digest(settings) != record.get("dependency_math_sha256"):
        raise Invalid("dependency math settings are missing or malformed")
    if any(record.get(k) != expected.get(k) for k in ("dependency_math", "dependency_math_sha256")):
        raise Invalid("dependency math settings differ")
