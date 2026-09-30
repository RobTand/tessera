"""Which token counts a CUDA-graph serve replays in a larger graph, and which ops it switches.

Two backends ask the same questions of a vLLM configuration: the research GLM53
NoPE backend (``glm53_nope``, tessera#508) and the serving path on the vLLM
nightly stack, which runs the image's own GLM5-next attention (tessera#702).
Both answer whether a graph serve runs eager's arithmetic, and the answer has
the same two halves on both:

- the OP IMPLEMENTATIONS a compilation mode selects.  GLM5-next is not
  torch-compiled by vLLM ("does not support torch.compile"), yet any
  compilation mode other than NONE resolves ``custom_ops`` to ``["none"]`` and
  the two RMS-norm IR ops to ``native`` rather than eager's ``["all"]`` and
  ``["vllm_c", "native"]`` (``config/vllm.py``; ``platforms/cuda.py``,
  ``get_default_ir_op_priority``), so the graph serve computes other
  arithmetic than the eager one it is compared with;
- the PADDED REPLAY.  vLLM's V2 runner (``CudaGraphManager``) runs a batch of n
  tokens in the smallest captured graph of its family that holds it, and some
  kernels pick their reduction by token count, so a batch replayed in a larger
  graph is another computation.

Pure: no vLLM import at module level.  Enum members are compared by NAME and a
resolved mode is constructed from the input's own enum class, so the CPU tests
drive these with doubles and the answers are vLLM's own members at serve time.
"""
from __future__ import annotations

from typing import Any

__all__ = ["EAGER_IR_OPS", "graph_mode", "graph_verdict", "num_draft_tokens",
           "op_implementation_gap", "padded_families", "padded_token_counts", "receipt_for",
           "report_once", "runner_digests", "serve_gap"]

#: The op implementations compilation mode NONE resolves, which the eager
#: reference runs: vLLM appends custom op ``"all"`` unless inductor compiles
#: (``config/vllm.py``), and CUDA orders both IR ops ``vllm_c`` before
#: ``native`` without codegen (``platforms/cuda.py``,
#: ``get_default_ir_op_priority``). Every other compilation mode defaults to
#: ``"none"`` and ``native``.
EAGER_IR_OPS = ("rms_norm", "fused_add_rms_norm")


def _name(member: Any) -> str | None:
    return getattr(member, "name", None)


def graph_mode(compilation: Any) -> Any:
    """The CUDA-graph mode vLLM will run for the FlashInfer sparse MLA backends.

    ``resolve_cudagraph_mode_and_sizes`` settles the mode after the backend is
    chosen, from the least capable metadata builder. The sparse MLA builder
    (``FlashInferMLASparseMetadataBuilder``) supports ``UNIFORM_BATCH``, never
    ``ALWAYS``, so a ``FULL`` request loses its mixed-batch half there: to
    ``FULL_AND_PIECEWISE`` when attention is a splitting op, and to
    ``FULL_DECODE_ONLY`` otherwise. The verdict judges the mode that will run.
    Returns ``None`` for an unset mode (vLLM's NONE).
    """
    mode = compilation.cudagraph_mode
    if mode is None:
        return None
    if _name(mode) == "FULL":
        return (type(mode).FULL_AND_PIECEWISE
                if compilation.splitting_ops_contain_attention()
                else type(mode).FULL_DECODE_ONLY)
    return mode


def num_draft_tokens(config: Any) -> int:
    spec = config.speculative_config
    return int(spec.num_speculative_tokens or 0) if spec is not None else 0


def padded_families(config: Any, graph: Any) -> list[tuple[str, list[int]]]:
    """Each graph family's token counts that replay a larger captured graph than their own.

    vLLM's V2 runner (``CudaGraphManager._init_candidates``) runs a batch of n
    tokens in the smallest captured graph of its family that holds it; a
    FULL candidate comes before a piecewise one, and a batch larger than
    every graph of its family runs eager, unpadded. With k draft tokens a
    decode request carries q = 1 + k query tokens, and the families are:

    - the FULL graphs for uniform decode batches (``FULL_DECODE_ONLY``, and
      ``FULL_AND_PIECEWISE``'s FULL half): each capture size rounded up to
      whole requests of q tokens, kept up to ``max_num_seqs`` requests and
      the largest capture size. Without a drafter this is the decode batch
      of n requests; with one, the target's verification of n requests, and
      the drafter's first step, which dispatches on the target's padded
      count (``AutoRegressiveSpeculator.propose``) and so pads with it;
    - the drafter's later steps, k >= 2 (``init_cudagraph_manager``): FULL
      graphs of one token per request, at the capture sizes up to
      ``max_num_seqs``, in any mode with FULL decode graphs; eager otherwise;
    - mixed batches (``PIECEWISE``, and ``FULL_AND_PIECEWISE``'s piecewise
      half): every capture size. Under ``PIECEWISE`` uniform batches take
      these graphs too.

    These are the families of the autoregressive speculator, which vLLM runs
    for method ``mtp``.
    """
    sizes = sorted(set(config.compilation_config.cudagraph_capture_sizes or ()))
    name = _name(graph)
    if not sizes or name in (None, "NONE"):
        return []
    max_num_seqs = config.scheduler_config.max_num_seqs
    draft = num_draft_tokens(config)
    query = 1 + draft
    families = []
    if name in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
        ceiling = min(max_num_seqs * query, sizes[-1])
        uniform = {-(-n // query) * query for n in sizes}
        uniform = sorted(n for n in uniform if n <= ceiling)
        if uniform:
            families.append(("target verification and draft prefill" if draft else "decode",
                             [n * query for n in range(1, max_num_seqs + 1)
                              if n * query < uniform[-1] and n * query not in uniform]))
        if draft >= 2:
            steps = [n for n in sizes if n <= max_num_seqs]
            if steps:
                families.append(("draft decode",
                                 [n for n in range(1, steps[-1]) if n not in steps]))
    if name in ("PIECEWISE", "FULL_AND_PIECEWISE"):
        families.append(("mixed", [n for n in range(1, sizes[-1]) if n not in sizes]))
    return [(family, counts) for family, counts in families if counts]


def padded_token_counts(config: Any, graph: Any) -> list[int]:
    """Token counts that replay a larger captured graph than their own, in any family."""
    return sorted({n for _, counts in padded_families(config, graph) for n in counts})


def op_implementation_gap(config: Any) -> str | None:
    """How this configuration's op implementations differ from eager's, or None."""
    custom = list(config.compilation_config.custom_ops or ())
    gaps = []
    if "all" not in custom or any(str(op).startswith("-") for op in custom):
        gaps.append(f"custom_ops resolves to {custom}, not ['all']")
    priority = config.kernel_config.ir_op_priority
    for op in EAGER_IR_OPS:
        order = list(getattr(priority, op, None) or ())
        if order[:1] != ["vllm_c"]:
            gaps.append(f"IR op {op} resolves to {order}, not ['vllm_c', 'native']")
    return "; ".join(gaps) or None


# --- the serve-time consult (tessera#702) ------------------------------------

#: Graph verdicts already reported by this process (one line per distinct verdict).
_REPORTED: set = set()


def _plain(value: Any) -> Any:
    """A vLLM config value as JSON would spell it: enum members by name, sequences as lists."""
    import enum

    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    # Before the scalar case: vLLM's modes are int-valued enums, and a
    # published receipt spells them by name, as the CLI takes them.
    if isinstance(value, enum.Enum):
        return value.name
    return value


def _differences(running: Any, attested: Any, path: str) -> list[str]:
    """Each attested field the running config object does not carry verbatim."""
    out = []
    for key, want in attested.items():
        got = getattr(running, key, None)
        if isinstance(want, dict):
            out.extend(_differences(got, want, f"{path}.{key}"))
        elif _plain(got) != want:
            out.append(f"{path}.{key} is {_plain(got)!r}, not {want!r}")
    return out


def runner_digests(paths: Any) -> dict[str, str | None]:
    """sha256 of each vLLM source path (relative to the ``vllm`` package), None if absent."""
    import hashlib
    from pathlib import Path

    import vllm

    root = Path(vllm.__file__).parent
    out = {}
    for relative in paths:
        path = root / relative
        out[relative] = (hashlib.sha256(path.read_bytes()).hexdigest()
                         if path.is_file() else None)
    return out


def receipt_for(config: Any, receipts: Any, digests: Any = None) -> Any:
    """The graph receipt measured on this runtime's graph path and model type, or None."""
    digests = runner_digests if digests is None else digests
    model_type = getattr(getattr(config.model_config, "hf_text_config", None),
                         "model_type", None)
    for receipt in receipts:
        if receipt["model_type"] != model_type:
            continue
        if digests(receipt["runner_sha256"]) == dict(receipt["runner_sha256"]):
            return receipt
    return None


def serve_gap(config: Any, receipt: Any, environ: Any = None) -> str | None:
    """How a graph serve differs from the one ``receipt`` measured equal to eager, or None."""
    import os

    environ = os.environ if environ is None else environ
    serve = receipt["serve"]
    gaps = _differences(config.compilation_config, serve["compilation_config"],
                        "compilation_config")
    gaps += _differences(config.kernel_config, serve["kernel_config"], "kernel_config")
    for key, want in serve["env"].items():
        if environ.get(key) != want:
            gaps.append(f"environment {key} is {environ.get(key)!r}, not {want!r}")
    ceiling = config.scheduler_config.max_num_seqs * (1 + num_draft_tokens(config))
    sizes = sorted(set(config.compilation_config.cudagraph_capture_sizes or ()))
    missing = [n for n in range(1, ceiling + 1) if n not in sizes]
    if missing:
        gaps.append(f"capture sizes {sizes} leave token counts {missing} of 1..{ceiling} "
                    "to replay a larger graph (every count must be captured)")
    spec = config.speculative_config
    if spec is not None:
        measured = {(s["method"], s["num_speculative_tokens"]) for s in receipt["speculative"]}
        if (spec.method, num_draft_tokens(config)) not in measured:
            gaps.append(f"no receipt measures drafter {spec.method!r} at "
                        f"{num_draft_tokens(config)} draft tokens under graphs "
                        f"(measured: {sorted(measured) or 'none'})")
    return "; ".join(gaps) or None


def graph_verdict(config: Any, receipts: Any, digests: Any = None,
                  environ: Any = None) -> tuple[str | None, str | None]:
    """``(receipt id or None, gap or None)`` for this serve's execution.

    An eager serve (CUDA-graph mode NONE and compilation mode NONE) runs eager's
    arithmetic by definition and needs no receipt: ``(None, None)``.  A graph
    serve is claimed equal to eager only when a receipt measured on this
    runtime's graph path and model type exists and the serve reproduces its
    configuration.
    """
    graph = graph_mode(config.compilation_config)
    compiled = _name(getattr(config.compilation_config, "mode", None)) not in (None, "NONE")
    if _name(graph) in (None, "NONE") and not compiled:
        return None, None
    receipt = receipt_for(config, receipts, digests)
    if receipt is None:
        return None, ("no graph receipt measures this runtime's graph path for model type "
                      f"{getattr(config.model_config.hf_text_config, 'model_type', None)!r} "
                      "(runtime_contract.json lane_eligibility.graph_receipts, tessera#702)")
    gap = serve_gap(config, receipt, environ)
    if gap is None:
        return receipt["id"], None
    return receipt["id"], (f"this serve is not the graph serve receipt {receipt['id']!r} "
                           f"measured equal to eager ({receipt['equivalence']['receipt']}): "
                           f"{gap}")


def report_once(config: Any, receipts: Any, *, backend: str = "vllm") -> None:
    """Say once per process whether this serve runs eager's arithmetic, and if not, why."""
    import sys

    from .telemetry import record_backend_execution_identity

    rid, gap = graph_verdict(config, receipts)
    if rid is None and gap is None:
        return  # an eager serve: nothing to claim, and its route trace stays as it was
    graph = graph_mode(config.compilation_config)
    record_backend_execution_identity(
        backend=backend, compilation_mode=_name(config.compilation_config.mode) or "NONE",
        cuda_graph_mode=_name(graph) or "NONE", eager_equivalence_gap=gap)
    key = (rid, gap)
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    if gap is None:
        print(f"Tessera: CUDA-graph mode {_name(graph)} reproduces graph receipt {rid!r}: "
              "runs eager's arithmetic (tessera#702)", file=sys.stderr, flush=True)
    else:
        print(f"Tessera: WARNING: this serve's outputs are not claimed equal to eager's; "
              f"measure its quality on it: {gap}", file=sys.stderr, flush=True)
