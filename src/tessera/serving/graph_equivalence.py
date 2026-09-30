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

__all__ = ["EAGER_IR_OPS", "graph_mode", "num_draft_tokens", "op_implementation_gap",
           "padded_families", "padded_token_counts"]

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

