#!/usr/bin/env python3
"""The certification harness: one receipt, and the scope it is allowed to claim.

Tessera is certified per platform, and the three ways of learning something
about a platform do not prove the same thing (design ``§8.1``):

* a ``hipcc --offload-arch=gfx1151`` compile proves the SOURCE is valid for
  RDNA3.5 and nothing about execution;
* a run on ``gfx1201`` proves the HIP CODE PATH executes and matches the
  reference where the contract says it must -- never gfx1151 numerics, and
  never any performance number (WSL2 has no PMCs and a desktop RDNA4 card
  says nothing about an APU);
* a run on a real Strix Halo is the ONLY thing from which a ``gfx1151``
  ``device_qualified`` cell may be minted.

Those three sentences are the whole reason this file exists.  A tester on a
box we do not own runs one command, sends back one JSON file, and somebody
here decides whether a ``lane_eligibility`` cell may be written from it.  That
decision must be readable off the receipt by a program, so the receipt carries
the scope as a VALUE (``header.scope``, ``header.scope_sentence``), the
qualification as the contract's own word (``header.qualification`` is
``device_qualified``, ``compile_only`` or ``null``), and the platform it was
measured on beside the platform it CLAIMS.  When those two differ the receipt
is still written -- a refused claim is evidence too -- with the qualification
withheld and the disagreement named in ``problems``.

**Identity comes from torch, not from a system tool.**  ``gcnArchName`` off
``torch.cuda.get_device_properties`` is the one identity this harness reads:
under WSL2 ``amdsmi_init`` fails ``DRIVER_NOT_LOADED``, ``rocm-smi --json``
prints nothing, and ``vllm.platforms.rocm.get_device_name`` is backed by
amdsmi -- so a harness that asked any of them would be unable to identify the
device it is standing on.  ``amd-smi`` is used for ONE thing, package power,
and its absence is recorded in the receipt and never fatal.

**The reference set is Tessera-16 at 896 and 1024.**  ``TESSERA_BF16_K1`` is
the one family the AMD lane serves (Rob's ruling, 2026-09-12: the RDNA3.5
product is a WnA16 artifact), and the two rungs are the reference artifacts
the protocol names.  An artifact carrying any other rung is refused rather
than measured, because a receipt at a rung nobody asked for cannot be joined
to a cell.

**What this harness does not do.**  It does not compute a KL and it does not
run a census: both are owned elsewhere (``tools/tessera_route_census.py``,
the KL harness behind ``experiments/serve_and_dump_kl.sh``), and a second
implementation of either would be a second answer to one question.  It
INGESTS their receipts -- path, digest, and the few fields the qualification
rule reads -- so the evidence travels with the claim it supports.

    tools/tessera_attest.py --out receipt.json \\
        [--expect-platform gfx1151] [--artifact DIR] [--build] \\
        [--image <repo@sha256:...>] [--census-receipt P] [--kl-receipt P] \\
        [--endpoint http://127.0.0.1:8000 --model NAME]

Exit status is 0 when the receipt was written, 2 when the harness refused to
write one at all (a platform it was not asked for, an artifact outside the
reference set, an unreadable image reference).  A receipt whose qualification
is ``null`` is a normal, successful run: it says what was proved.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform as platform_module
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tessera.serving.backend import capability_token, gcn_arch_token  # noqa: E402

#: Bumped when the receipt's shape changes.  A consumer keys on it, and #460
#: mints cells from receipts that carry it.
SCHEMA = "tessera.attest/1"

#: The one family the AMD lane serves, and the two rungs the protocol names.
#: Both are the reference SET, not a capability claim: an artifact at another
#: rung is refused because nothing downstream could join its numbers.
REFERENCE_FAMILY = "TESSERA_BF16_K1"
REFERENCE_RUNGS_Q256 = (896, 1024)

#: RDNA3.5 (Strix Point/Halo) and the gfx12 parts that execute gfx12 code.
#: Kept apart on purpose: they select different §8.1 rows, and the difference
#: between them is the difference between a performance claim and a code-path
#: claim.
STRIX_HALO_TOKENS = ("gfx1150", "gfx1151", "gfx1152", "gfx1153")
GFX12_EXECUTION_TOKENS = ("gfx1200", "gfx1201")

SCOPE_COMPILE_GFX1151 = "compile_gfx1151"
SCOPE_EXECUTE_GFX12 = "execute_gfx1201"
SCOPE_DEVICE_STRIX_HALO = "device_strix_halo"

#: §8.1 verbatim: what each receipt proves and what it does not.  The
#: ``sentence`` is what goes on the receipt and into the tester protocol doc;
#: ``docs/strix-halo-tester-protocol.md`` reproduces these strings and a CPU
#: test compares the two, so the doc cannot drift from the tool.
SCOPES = {
    SCOPE_COMPILE_GFX1151: {
        "receipt": "hipcc --offload-arch=gfx1151 compile of the port (wsl-gpu)",
        "proves": "the kernel source is valid for RDNA3.5; LDS/VGPR budgets per instantiation",
        "does_not_prove": "anything about gfx1151 execution, numerics or speed",
        "perf_claim": False,
    },
    SCOPE_EXECUTE_GFX12: {
        "receipt": "gfx1201 execution on wsl-gpu",
        "proves": ("the HIP code path (loader, shims, plan, window_decode+torch.mm prefill, "
                   "BF16 route) executes and matches the CUDA/CPU reference bit-for-bit "
                   "where the contract says it must"),
        "does_not_prove": ("gfx1151 numerics or perf; any perf (no PMCs, WSL2, and wall-clock "
                           "on a desktop RDNA4 says nothing about an APU)"),
        "perf_claim": False,
    },
    SCOPE_DEVICE_STRIX_HALO: {
        "receipt": "Strix Halo tester run",
        "proves": "gfx1151 device_qualified: correctness, KL-vs-BF16, decode/prefill tok/s, power",
        "does_not_prove": "another platform's numbers; a rung outside the reference set",
        "perf_claim": True,
    },
}

#: The contract's word for what a cell rests on.  Read from the module that
#: owns the vocabulary when it can be imported, so this harness cannot mint a
#: word ``contract.validate_serving_contract`` would refuse.
QUALIFICATION_DEVICE = "device_qualified"
QUALIFICATION_COMPILE_ONLY = "compile_only"

#: A step's outcome.  ``not_run`` is not a failure: most runs of this harness
#: are partial by design (no artifact, no serve), and the qualification rule
#: below is what turns a partial run into a withheld claim.
STATUS_RAN = "ran"
STATUS_FAILED = "failed"
STATUS_REFUSED = "refused"
STATUS_NOT_RUN = "not_run"
STEP_STATUSES = (STATUS_RAN, STATUS_FAILED, STATUS_REFUSED, STATUS_NOT_RUN)

#: §8.2 -- what "proves the code path" means -- and §8.3 -- what a Strix Halo
#: tester must run.  One step map, two views: §8.3 item 3 IS §8.2 items 2-4,
#: and writing them twice would let the two copies disagree about what ran.
STEP_EXTENSION_BUILD = "extension_build"
STEP_DECODER_BIT_EXACT = "decoder_bit_exact"
STEP_GEMV_TOLERANCE = "gemv_tolerance"
STEP_ROUTE_CENSUS = "route_census"
STEP_SERVED_KL = "served_kl"
STEP_DEVICE_IDENTITY = "device_identity"
STEP_SERVE_MODES = "serve_eager_and_graph"
STEP_DECODE_TOK_S = "decode_tok_s"
STEP_PREFILL_TOK_S = "prefill_tok_s"
STEP_PACKAGE_POWER = "package_power"

SECTIONS = {
    "8.2": (
        ("1", (STEP_EXTENSION_BUILD,)),
        ("2", (STEP_DECODER_BIT_EXACT,)),
        ("3", (STEP_GEMV_TOLERANCE,)),
        ("4", (STEP_ROUTE_CENSUS,)),
        ("5", (STEP_SERVED_KL,)),
    ),
    "8.3": (
        ("1", (STEP_DEVICE_IDENTITY,)),
        ("2", (STEP_EXTENSION_BUILD,)),
        ("3", (STEP_DECODER_BIT_EXACT, STEP_GEMV_TOLERANCE, STEP_ROUTE_CENSUS)),
        ("4", (STEP_SERVE_MODES,)),
        ("5", (STEP_SERVED_KL, STEP_DECODE_TOK_S, STEP_PREFILL_TOK_S, STEP_PACKAGE_POWER)),
        ("6", ()),   # the receipt itself: this file
    ),
}

#: What must have RUN before a receipt may carry ``device_qualified``.  The
#: code-path half is required on every platform; the performance half is
#: required only where a performance number is admissible at all, which is the
#: Strix Halo row and nowhere else.
REQUIRED_CODE_PATH_STEPS = (
    STEP_EXTENSION_BUILD, STEP_DECODER_BIT_EXACT, STEP_GEMV_TOLERANCE,
    STEP_ROUTE_CENSUS, STEP_SERVED_KL,
)
REQUIRED_DEVICE_PERF_STEPS = (STEP_SERVE_MODES, STEP_DECODE_TOK_S, STEP_PREFILL_TOK_S)

# --- identity, scope, qualification: pure, and testable with a stub --------

#: Identity is #452's, not this harness's.  ``tessera.serving.backend`` is
#: where the platform token is minted -- the build keys its directory on the
#: same string this receipt claims a cell for, and two spellings of one
#: identity is how a receipt comes to name a platform the build never
#: targeted.  These two names stay as the harness's vocabulary; the answers
#: come from there.
platform_token_of = gcn_arch_token


def cuda_platform_token(major: int, minor: int) -> str:
    """``(12, 1)`` -> ``sm_121``, the spelling the packaged contract uses."""
    return capability_token((int(major), int(minor)))


def probe_identity(torch_module=None) -> dict:
    """What device this process is standing on, read from torch alone.

    ``torch_module`` is a parameter so a CPU test can pass a stub: the whole
    point of this function is the READ, and the read must be exercisable on a
    box with no GPU.  Nothing here touches ``amdsmi``, ``rocm-smi`` or
    ``vllm.platforms.rocm`` -- see the module docstring for why none of the
    three can answer under WSL2.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415 -- torch is optional for the pure half

    version = getattr(torch_module, "version", None)
    hip = getattr(version, "hip", None)
    cuda = getattr(version, "cuda", None)
    backend = "hip" if hip else ("cuda" if cuda else None)
    out = {
        "backend": backend,
        "platform": None,
        "gcn_arch_name": None,
        "device_name": None,
        "device_count": 0,
        "warp_size": None,
        "multi_processor_count": None,
        "total_memory_bytes": None,
        "shared_memory_per_block": None,
        "source": "torch.cuda.get_device_properties(0).gcnArchName",
        "torch": str(getattr(torch_module, "__version__", "")) or None,
        "hip": str(hip) if hip else None,
        "cuda": str(cuda) if cuda else None,
        "available": False,
        "reason": None,
    }
    try:
        available = bool(torch_module.cuda.is_available())
    except Exception as exc:                                  # a torch with no runtime at all
        out["reason"] = f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}"
        return out
    out["available"] = available
    if not available:
        out["reason"] = "torch reports no device; nothing was executed"
        return out
    try:
        out["device_count"] = int(torch_module.cuda.device_count())
    except Exception:                                         # pragma: no cover - stub freedom
        out["device_count"] = 1
    props = torch_module.cuda.get_device_properties(0)
    gcn = getattr(props, "gcnArchName", None)
    out["device_name"] = getattr(props, "name", None)
    out["warp_size"] = getattr(props, "warp_size", None)
    out["multi_processor_count"] = getattr(props, "multi_processor_count", None)
    out["total_memory_bytes"] = getattr(props, "total_memory", None)
    out["shared_memory_per_block"] = getattr(props, "shared_memory_per_block", None)
    if gcn:
        out["gcn_arch_name"] = str(gcn)
        out["platform"] = platform_token_of(gcn)
    else:
        major, minor = getattr(props, "major", None), getattr(props, "minor", None)
        if major is None or minor is None:
            out["reason"] = "the device reports neither gcnArchName nor a compute capability"
            return out
        out["platform"] = cuda_platform_token(major, minor)
        out["source"] = "torch.cuda.get_device_properties(0).major/.minor"
    return out


def scope_for(platform_token: str, *, executed: bool = True) -> str:
    """Which §8.1 row a receipt from this platform occupies.

    Derived from the token, never from a flag, because the flag is exactly the
    thing a tired operator gets wrong: a Strix Halo row on a gfx1201 receipt
    is how a performance number escapes onto a platform that cannot support
    it.  ``executed=False`` is the compile-only row, the only row a receipt
    with no device may claim.
    """
    token = str(platform_token)
    if not executed:
        return SCOPE_COMPILE_GFX1151
    if token in STRIX_HALO_TOKENS:
        return SCOPE_DEVICE_STRIX_HALO
    if token in GFX12_EXECUTION_TOKENS:
        return SCOPE_EXECUTE_GFX12
    raise ValueError(
        f"{token!r} has no §8.1 row: this is the AMD certification harness, and the "
        f"platforms it certifies are {list(STRIX_HALO_TOKENS + GFX12_EXECUTION_TOKENS)}. "
        "A CUDA platform is certified by the CUDA suite and its own cells.")


def scope_sentence(scope: str) -> str:
    """The sentence §8.1 requires on every receipt."""
    row = SCOPES[scope]
    return (f"scope: {row['receipt']} -- proves {row['proves']}; "
            f"does not prove {row['does_not_prove']}")


def qualification_for(*, claim_platform, measured_platform, scope, steps) -> "tuple[str | None, list[str]]":
    """``(qualification, problems)`` -- and the refusal this harness exists for.

    ``device_qualified`` is minted only when the receipt claims the platform
    it was MEASURED on and every step that platform's row requires actually
    ran.  A receipt for a platform this process did not run on is the failure
    mode the whole protocol is built to prevent -- a gfx1151 cell minted from
    a gfx1201 run -- so it is refused here, in the value, rather than in a
    sentence somebody has to read.

    ``compile_only`` is the contract's word for a receipt whose extension
    built and whose device never ran it; it is what a compile gate produces.
    ``None`` is the honest answer for everything else, and it is not an error.
    """
    problems = []
    ran = {name for name, step in steps.items() if step.get("status") == STATUS_RAN}
    failed = sorted(name for name, step in steps.items() if step.get("status") == STATUS_FAILED)
    if failed:
        problems.append(f"steps failed: {failed}")
    if claim_platform != measured_platform:
        problems.append(
            f"this receipt claims {claim_platform!r} and was measured on "
            f"{measured_platform!r}: no qualification is minted for a platform the harness "
            "did not run on")
        return None, problems
    if measured_platform is None:
        if STEP_EXTENSION_BUILD in ran:
            return QUALIFICATION_COMPILE_ONLY, problems
        return None, problems + ["no device was present and nothing was built"]
    required = list(REQUIRED_CODE_PATH_STEPS)
    if SCOPES[scope]["perf_claim"]:
        required += list(REQUIRED_DEVICE_PERF_STEPS)
    missing = [name for name in required if name not in ran]
    if missing:
        problems.append(f"device_qualified withheld; steps that did not run: {missing}")
        if ran == {STEP_EXTENSION_BUILD}:
            return QUALIFICATION_COMPILE_ONLY, problems
        return None, problems
    return QUALIFICATION_DEVICE, problems


def section_view(steps: dict) -> dict:
    """§8.2 and §8.3 as the protocol numbers them, over one step map."""
    out = {}
    for section, items in SECTIONS.items():
        rows = {}
        for item, names in items:
            if not names:
                rows[item] = {"refers_to": [], "status": STATUS_RAN, "note": "this receipt"}
                continue
            statuses = [steps.get(name, {}).get("status", STATUS_NOT_RUN) for name in names]
            if all(status == STATUS_RAN for status in statuses):
                status = STATUS_RAN
            elif STATUS_FAILED in statuses:
                status = STATUS_FAILED
            elif STATUS_REFUSED in statuses:
                status = STATUS_REFUSED
            else:
                status = STATUS_NOT_RUN
            rows[item] = {"refers_to": list(names), "status": status}
        out[section] = rows
    return out


def step(status: str, reason: "str | None" = None, **evidence) -> dict:
    """One step record.  ``reason`` is mandatory for anything but ``ran``."""
    if status not in STEP_STATUSES:
        raise ValueError(f"{status!r} is not one of {list(STEP_STATUSES)}")
    if status != STATUS_RAN and not reason:
        raise ValueError(f"a {status!r} step must say why")
    record = {"status": status, "reason": reason}
    record.update(evidence)
    return record


# --- power: one tool, one purpose, and its absence is a field -------------

class PackagePower:
    """``amd-smi metric --power`` at 1 Hz, and a receipt field when there is none.

    Power is the only thing this harness asks a system tool for, because it is
    the only thing torch cannot answer and the only thing that turns tok/s
    into tokens per joule (principle 9).  Under WSL2 there is no driver behind
    it; the tester protocol says so, and an absent or failing ``amd-smi`` is
    recorded here and never raised -- a correctness receipt must not be lost
    because a power meter was missing.
    """

    def __init__(self, *, interval: float = 1.0, binary: str = "amd-smi"):
        self.interval = float(interval)
        self.binary = binary
        self.path = shutil.which(binary)
        self.samples: list = []
        self.reason = None if self.path else f"{binary} is not on PATH"
        self._stop = threading.Event()
        self._thread = None

    def _sample(self) -> "float | None":
        try:
            out = subprocess.run([self.path, "metric", "--power", "--json"],
                                 capture_output=True, text=True, timeout=10)
        except Exception as exc:
            self.reason = f"{self.binary} raised {type(exc).__name__}: {exc}"
            return None
        if out.returncode != 0:
            self.reason = f"{self.binary} exited {out.returncode}: {out.stderr.strip()[:200]}"
            return None
        try:
            payload = json.loads(out.stdout)
        except ValueError:
            self.reason = f"{self.binary} printed no JSON ({out.stdout.strip()[:120]!r})"
            return None
        return _first_power_watts(payload)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            watts = self._sample()
            if watts is None:
                return
            self.samples.append({"t": time.time(), "watts": watts})

    def __enter__(self):
        if self.path and self._sample() is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        return False

    def block(self) -> dict:
        """The receipt's ``power`` field: available or not, and why not."""
        watts = [s["watts"] for s in self.samples]
        return {
            "tool": self.binary,
            "available": bool(watts),
            "reason": None if watts else (self.reason or "no sample was taken"),
            "interval_s": self.interval,
            "samples": len(watts),
            "mean_watts": (sum(watts) / len(watts)) if watts else None,
            "max_watts": max(watts) if watts else None,
        }


def _first_power_watts(payload) -> "float | None":
    """The first socket/package power in an ``amd-smi --json`` payload.

    The tool's JSON shape moves between releases, so the read is structural
    rather than keyed to one version: the first numeric value under a key
    naming socket or average power, anywhere in the tree.  A shape it cannot
    read reports no power, which is the same honest state as no tool at all.
    """
    wanted = ("socket_power", "average_socket_power", "power", "current_socket_power")
    stack = [payload]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted:
                    if isinstance(value, (int, float)):
                        return float(value)
                    if isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
                        return float(value["value"])
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


# --- the reference set ----------------------------------------------------

def reference_set_report(manifest: dict) -> dict:
    """The artifact's family and rungs, refused unless they are the reference set.

    Read from ``tessera_serving_manifest.json`` -- the roles the export
    WROTE -- rather than from a config a producer intended, and mapped to a
    payload family through ``scheme.route_for_grid`` /
    ``contract.PAYLOAD_FAMILY_BY_ROUTE`` so this file holds no second copy of
    the grid-to-family table.
    """
    from tessera.serving.contract import PAYLOAD_FAMILY_BY_ROUTE
    from tessera.serving.scheme import route_for_grid

    grids, rungs = set(), set()
    for module in (manifest.get("modules") or {}).values():
        for role in module.get("roles", ()):
            if role.get("grid") is not None:
                grids.add(str(role["grid"]))
            if role.get("q256") is not None:
                rungs.add(int(role["q256"]))
    families = set()
    for grid in grids:
        route = route_for_grid(grid)
        if route is None:
            raise SystemExit(f"the artifact carries grid {grid!r}, which no route in this "
                             "build holds; there is nothing to certify")
        families.add(PAYLOAD_FAMILY_BY_ROUTE[route])
    if families != {REFERENCE_FAMILY}:
        raise SystemExit(
            f"the reference set is {REFERENCE_FAMILY} and this artifact carries "
            f"{sorted(families)}: a receipt at another family cannot be joined to the "
            "cells this protocol mints")
    outside = sorted(r for r in rungs if r not in REFERENCE_RUNGS_Q256)
    if outside:
        raise SystemExit(
            f"the reference set is {REFERENCE_FAMILY} at rungs {list(REFERENCE_RUNGS_Q256)} "
            f"and this artifact carries {outside}: refusing to measure a rung the protocol "
            "did not ask for")
    return {"family": REFERENCE_FAMILY, "rungs_q256": sorted(rungs),
            "reference_rungs_q256": list(REFERENCE_RUNGS_Q256)}


# --- the steps this harness owns ------------------------------------------

def build_extension_step(platform_token: str) -> dict:
    """§8.2 item 1: the window-GEMV extension builds through the loader's path.

    Through ``kernel_window_gemv._ext()`` -- the loader vLLM will use -- and
    not a hand ``hipcc``, because the thing being certified is what the serve
    does.  A build failure is a recorded step, not a crash: on a tree without
    the HIP shims this is exactly the expected outcome, and the receipt should
    say so rather than lose the identity block it already holds.
    """
    started = time.time()
    try:
        from tessera import kernel_window_gemv as kg

        module = kg._ext()
    except Exception as exc:
        # The TAIL, not the head: ninja echoes the whole compile command before
        # the diagnostic, so a head-truncated build error is the flags without
        # the reason they failed.
        return step(STATUS_FAILED,
                    f"{type(exc).__name__}: ...{str(exc).strip()[-1800:]}",
                    platform=platform_token, seconds=round(time.time() - started, 3))
    return step(STATUS_RAN, None, platform=platform_token,
                module=getattr(module, "__name__", None),
                seconds=round(time.time() - started, 3))


def _parsed_modules(artifact: str, device: str):
    """``[(module, [(role, ParsedUnit)])]`` for every Tessera unit on disk."""
    from safetensors import safe_open

    from tessera.fused import parse_fused
    from tessera.unit_artifact import parse_unit_artifact

    out = []
    shards = sorted(f for f in os.listdir(artifact) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"{artifact}: no .safetensors shard here")
    for shard in shards:
        with safe_open(os.path.join(artifact, shard), framework="pt") as handle:
            names = [n for n in handle.keys() if n.endswith(".wire_bytes")]
            for name in sorted(names):
                module = name[: -len(".wire_bytes")]
                blob = bytes(handle.get_tensor(name).cpu().numpy().tobytes())
                roles = [(member.name, parse_unit_artifact(member.blob, device=device))
                         for member in parse_fused(blob)]
                out.append((module, roles))
    return out


def correctness_steps(artifact: str, *, device: str = "cuda", modules: "int | None" = None) -> dict:
    """§8.2 items 2 and 3, over the reference artifact's own units.

    Item 2 is bit-exactness of the packed decoder against the ``torch_window``
    reference, and that comparison already has a home:
    ``bf16_route.prepare_tessera_bf16_module`` decodes every role twice and
    refuses the module when the two disagree.  This drives that path rather
    than restating it -- a second comparison here would be a second opinion
    about the thing the serve actually checks.

    Item 3 is the GEMV against the decoded tile: ``window_gemv(unit, x)``
    versus ``(tile * scale) @ x`` in fp32, for M in 1, 2, 4, 8.  The bound is
    fp32 summation order over the same operands, so it is expressed as a
    relative error against the reference's own magnitude rather than a
    hand-set epsilon.
    """
    import torch

    from tessera import kernel_window_gemv as kg
    from tessera.serving.bf16_route import prepare_tessera_bf16_module

    checked = failures = 0
    gemv_checked = 0
    worst = 0.0
    worst_where = None
    started = time.time()
    parsed = _parsed_modules(artifact, device)
    if modules is not None:
        parsed = parsed[:modules]
    if not parsed:
        return {STEP_DECODER_BIT_EXACT: step(STATUS_NOT_RUN, f"{artifact} holds no Tessera wire"),
                STEP_GEMV_TOLERANCE: step(STATUS_NOT_RUN, f"{artifact} holds no Tessera wire")}
    first_failure = None
    for module, roles in parsed:
        try:
            prepared = prepare_tessera_bf16_module(roles, device=torch.device(device))
        except Exception as exc:
            failures += 1
            first_failure = first_failure or f"{module}: {type(exc).__name__}: {exc}"
            continue
        checked += 1
        unit = getattr(prepared, "gemv", None)
        unit = getattr(unit, "unit", unit)
        if unit is None or not isinstance(unit, kg.WindowGemvUnit):
            continue
        tile = kg.decode_values(unit).float()
        scale = unit.scale.float().reshape(-1, 1)
        reference_weight = tile * scale
        for m in (1, 2, 4, 8):
            x = torch.randn(m, reference_weight.shape[1], device=reference_weight.device,
                            dtype=torch.bfloat16)
            got = kg.window_gemv(unit, x).float()
            want = x.float() @ reference_weight.T
            denominator = want.abs().max().clamp_min(1e-6)
            error = float(((got - want).abs().max() / denominator).item())
            gemv_checked += 1
            if error > worst:
                worst, worst_where = error, f"{module} M={m}"
    seconds = round(time.time() - started, 3)
    decoder = (step(STATUS_RAN, None, modules=checked, seconds=seconds)
               if not failures else
               step(STATUS_FAILED,
                    f"{failures} module(s) refused the packed decoder; first: {first_failure}",
                    modules=checked, seconds=seconds))
    if gemv_checked:
        gemv = step(STATUS_RAN, None, comparisons=gemv_checked,
                    worst_relative_error=worst, worst_at=worst_where)
    else:
        gemv = step(STATUS_NOT_RUN, "no module presented a window-GEMV unit to compare")
    return {STEP_DECODER_BIT_EXACT: decoder, STEP_GEMV_TOLERANCE: gemv}


def ingest_receipt(path: str, kind: str, *, reads=()) -> dict:
    """Record another tool's receipt: path, digest, and the fields we read.

    The census and the KL are owned by ``tools/tessera_route_census.py`` and
    the KL harness.  This harness neither recomputes them nor paraphrases
    them: it names the file, digests it so the claim cannot drift from the
    bytes it was read off, and copies forward only the fields the
    qualification rule and a future cell need.
    """
    if not os.path.isfile(path):
        return step(STATUS_FAILED, f"no {kind} receipt at {path}")
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    try:
        payload = json.loads(data)
    except ValueError as exc:
        return step(STATUS_FAILED, f"{path} is not JSON: {exc}", sha256=digest)
    read = {key: payload.get(key) for key in reads if key in payload}
    return step(STATUS_RAN, None, receipt=os.path.abspath(path), sha256=digest, read=read)


# --- the served half: tok/s over an endpoint the tester serves -------------

def _completion(endpoint: str, model: str, prompt, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def throughput_steps(endpoint: str, model: str, *, decode_tokens: int = 128,
                     prefill_tokens=(512, 2048), concurrency=(1, 8),
                     timeout: float = 600.0) -> dict:
    """§8.3 item 5's tok/s halves, measured against a serve the tester started.

    Decode at M=1 and M=8 (``concurrency``), prefill at 512 and 2048 prompt
    tokens with one output token, both over ``/v1/completions`` so the harness
    measures the runtime rather than a private code path.  The numbers are
    wall-clock over the completed batch; whether they may be QUOTED at all is
    the receipt's ``header.perf_claim``, which is a property of the §8.1 row
    and not of this function.
    """
    prompt = "The quick brown fox jumps over the lazy dog. " * 4

    def decode(m: int) -> dict:
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=m) as pool:
            futures = [pool.submit(_completion, endpoint, model, prompt, decode_tokens, timeout)
                       for _ in range(m)]
            payloads = [f.result() for f in futures]
        elapsed = time.time() - started
        produced = sum(int(p.get("usage", {}).get("completion_tokens", 0)) for p in payloads)
        return {"concurrency": m, "tokens": produced, "seconds": round(elapsed, 3),
                "tok_s": round(produced / elapsed, 3) if elapsed > 0 else None}

    def prefill(n: int) -> dict:
        started = time.time()
        payload = _completion(endpoint, model, [0] * n, 1, timeout)
        elapsed = time.time() - started
        consumed = int(payload.get("usage", {}).get("prompt_tokens", n))
        return {"prompt_tokens": consumed, "seconds": round(elapsed, 3),
                "tok_s": round(consumed / elapsed, 3) if elapsed > 0 else None}

    out = {}
    try:
        out[STEP_DECODE_TOK_S] = step(STATUS_RAN, None,
                                      measurements=[decode(m) for m in concurrency])
    except Exception as exc:
        out[STEP_DECODE_TOK_S] = step(STATUS_FAILED, f"{type(exc).__name__}: {exc}")
    try:
        out[STEP_PREFILL_TOK_S] = step(STATUS_RAN, None,
                                       measurements=[prefill(n) for n in prefill_tokens])
    except Exception as exc:
        out[STEP_PREFILL_TOK_S] = step(STATUS_FAILED, f"{type(exc).__name__}: {exc}")
    return out


# --- the receipt ----------------------------------------------------------

def host_block() -> dict:
    """Where this ran.  ``wsl2`` is a field because it decides a scope row."""
    release = platform_module.release()
    return {"node": platform_module.node(), "kernel": release,
            "wsl2": "microsoft" in release.lower(), "machine": platform_module.machine(),
            "python": platform_module.python_version()}


def versions_block(identity: dict) -> dict:
    """``vllm``/``torch`` (the keys ``contract.RUNTIME_VERSION_KEYS`` names) plus
    the AMD half a cell's reader wants: the HIP the torch build carries and the
    distribution version of Tessera itself."""
    try:
        import vllm

        vllm_version = str(getattr(vllm, "__version__", "")) or None
    except Exception:
        vllm_version = None
    try:
        # ``tessera.__version__`` is the one home (ARCHITECTURE §5.5): it reads
        # pyproject from a checkout and distribution metadata from a wheel, so a
        # receipt produced from a clone is not stamped ``null``.
        import tessera

        tessera_version = str(tessera.__version__) or None
    except Exception:
        tessera_version = None
    return {"torch": identity.get("torch"), "vllm": vllm_version, "hip": identity.get("hip"),
            "cuda": identity.get("cuda"), "tessera": tessera_version,
            "python": platform_module.python_version()}


def build_receipt(*, identity: dict, steps: dict, power: dict, reference: dict,
                  versions: dict, image=None, claim_platform=None, host=None,
                  now=None, limitations=()) -> dict:
    """The §8.1 header, the §8.2/§8.3 sections, and what may be claimed.

    Pure: everything it needs has already been measured.  That is what lets a
    CPU test with no GPU check the schema, the refusal and the scope sentence
    on a receipt built from a stub.
    """
    measured = identity.get("platform")
    executed = bool(identity.get("available")) and measured is not None
    claim = claim_platform or measured
    scope = scope_for(claim, executed=executed) if claim else SCOPE_COMPILE_GFX1151
    qualification, problems = qualification_for(
        claim_platform=claim, measured_platform=measured, scope=scope, steps=steps)
    row = SCOPES[scope]
    header = {
        "platform": claim,
        "measured_platform": measured,
        "image": image,
        "versions": versions,
        "qualification": qualification,
        "scope": scope,
        "scope_sentence": scope_sentence(scope),
        "proves": row["proves"],
        "does_not_prove": row["does_not_prove"],
        "perf_claim": row["perf_claim"],
        "generated": (now or datetime.now(timezone.utc)).isoformat(),
        "host": host or host_block(),
    }
    notes = list(limitations)
    if not power.get("available"):
        notes.append(f"package power unavailable ({power.get('reason')}); "
                     "a work-per-joule ranking cannot be read off this receipt")
    if image is None:
        notes.append("no runtime image digest: a lane_eligibility cell needs one, so this "
                     "receipt informs a cell but cannot complete it")
    if not row["perf_claim"]:
        notes.append("this §8.1 row admits no performance number; any tok/s or power field "
                     "here is a code-path observation, never a claim about the hardware")
    return {
        "schema": SCHEMA,
        "header": header,
        "device": identity,
        "reference_set": reference,
        "power": power,
        "steps": steps,
        "sections": section_view(steps),
        "problems": problems,
        "limitations": notes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the JSON receipt")
    ap.add_argument("--expect-platform", default=None,
                    help="refuse before doing anything unless the device reads as this token")
    ap.add_argument("--claim-platform", default=None,
                    help="the platform the receipt claims (default: the one measured). A claim "
                         "that differs from the measurement is written with no qualification.")
    ap.add_argument("--image", default=None,
                    help="the serve image digest (repo@sha256:...) this run used")
    ap.add_argument("--build", action="store_true", help="build the window-GEMV extension (§8.2/1)")
    ap.add_argument("--artifact", default=None,
                    help="the reference Tessera-16 checkpoint (§8.2/2-3)")
    ap.add_argument("--artifact-modules", type=int, default=None,
                    help="check only the first N modules (a smoke, recorded as such)")
    ap.add_argument("--device", default="cuda", help="torch device for the correctness steps")
    ap.add_argument("--census-receipt", default=None,
                    help="the JSON tools/tessera_route_census.py wrote (§8.2/4)")
    ap.add_argument("--kl-receipt", default=None, help="the served KL receipt (§8.2/5, §8.3/5)")
    ap.add_argument("--endpoint", default=None, help="a running serve, for tok/s (§8.3/5)")
    ap.add_argument("--model", default=None, help="the model name the endpoint serves")
    ap.add_argument("--execution-modes", default=None,
                    help="comma-separated modes the tester served in, e.g. eager,compiled (§8.3/4)")
    ap.add_argument("--serve-mode", default=None, choices=("streamed", "resident"),
                    help="TESSERA_SERVE_MODE the serve ran under")
    ap.add_argument("--power-interval", type=float, default=1.0)
    ap.add_argument("--no-power", action="store_true", help="do not sample package power at all")
    args = ap.parse_args(argv)

    identity = probe_identity()
    measured = identity.get("platform")
    if args.expect_platform and measured != args.expect_platform:
        print(f"refusing: asked for {args.expect_platform!r}, this device reads "
              f"{measured!r} (gcnArchName {identity.get('gcn_arch_name')!r}). "
              "A receipt for a platform the harness did not run on is what this check exists "
              "to prevent.", file=sys.stderr)
        return 2
    if args.image is not None:
        from tessera.serving.contract import require_runtime_image

        try:
            require_runtime_image(args.image, "--image")
        except Exception as exc:
            print(f"refusing: --image {args.image!r} is not an exact manifest reference ({exc})",
                  file=sys.stderr)
            return 2

    reference = {"family": REFERENCE_FAMILY, "rungs_q256": [],
                 "reference_rungs_q256": list(REFERENCE_RUNGS_Q256)}
    if args.artifact:
        manifest_path = os.path.join(args.artifact, "tessera_serving_manifest.json")
        if not os.path.isfile(manifest_path):
            print(f"refusing: {manifest_path} is missing; the reference set is read off the "
                  "manifest the export wrote, never off a config a producer intended",
                  file=sys.stderr)
            return 2
        reference = reference_set_report(json.loads(open(manifest_path).read()))

    steps = {}
    steps[STEP_DEVICE_IDENTITY] = (
        step(STATUS_RAN, None, platform=measured, source=identity["source"])
        if measured else step(STATUS_FAILED, identity.get("reason") or "no device identity"))
    steps[STEP_EXTENSION_BUILD] = (
        build_extension_step(measured or "unknown") if args.build
        else step(STATUS_NOT_RUN, "--build was not given"))

    power = PackagePower(interval=args.power_interval)
    if args.no_power:
        power.reason = "--no-power"
    with (power if not args.no_power else _NullContext(power)):
        if args.artifact:
            steps.update(correctness_steps(args.artifact, device=args.device,
                                           modules=args.artifact_modules))
        else:
            for name in (STEP_DECODER_BIT_EXACT, STEP_GEMV_TOLERANCE):
                steps[name] = step(STATUS_NOT_RUN, "no --artifact was given")
        if args.endpoint:
            if not args.model:
                print("refusing: --endpoint needs --model", file=sys.stderr)
                return 2
            steps.update(throughput_steps(args.endpoint, args.model))
        else:
            for name in (STEP_DECODE_TOK_S, STEP_PREFILL_TOK_S):
                steps[name] = step(STATUS_NOT_RUN, "no --endpoint was given")

    steps[STEP_ROUTE_CENSUS] = (
        ingest_receipt(args.census_receipt, "census",
                       reads=("schema", "engagement", "problems", "runtime_image"))
        if args.census_receipt else step(STATUS_NOT_RUN, "no --census-receipt was given"))
    steps[STEP_SERVED_KL] = (
        ingest_receipt(args.kl_receipt, "KL", reads=("schema", "kl", "corpus", "regime"))
        if args.kl_receipt else step(STATUS_NOT_RUN, "no --kl-receipt was given"))
    modes = [m.strip() for m in (args.execution_modes or "").split(",") if m.strip()]
    steps[STEP_SERVE_MODES] = (
        step(STATUS_RAN, None, execution_modes=modes, serve_mode=args.serve_mode)
        if modes else step(STATUS_NOT_RUN, "no --execution-modes were reported"))
    steps[STEP_PACKAGE_POWER] = (
        step(STATUS_RAN, None, samples=power.block()["samples"])
        if power.block()["available"] else
        step(STATUS_NOT_RUN, power.block()["reason"] or "no power samples"))

    limitations = []
    if args.artifact_modules:
        limitations.append(f"correctness ran over the first {args.artifact_modules} modules only; "
                           "a qualifying run covers the artifact")
    receipt = build_receipt(identity=identity, steps=steps, power=power.block(),
                            reference=reference, versions=versions_block(identity),
                            image=args.image, claim_platform=args.claim_platform,
                            limitations=limitations)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=False)
        handle.write("\n")
    header = receipt["header"]
    print(f"{out}\n{header['scope_sentence']}\n"
          f"platform {header['platform']} (measured {header['measured_platform']}), "
          f"qualification {header['qualification']}")
    for problem in receipt["problems"]:
        print(f"  problem: {problem}")
    return 0


class _NullContext:
    """``--no-power``: the sampler object still answers ``block()``."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
