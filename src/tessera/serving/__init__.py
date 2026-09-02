"""Tessera's vLLM serving plugin: the runtime that reads Tessera bytes.

Tessera is a wire format.  Until now the only runtime that could serve it was
Gridbook, which imported Tessera's reader and added the serving half.  This
package IS that serving half, owned by the format it serves: a vLLM
``general_plugins`` entry point registering ``quant_method = "tessera"``, two
dense routes (NVFP4 W4A4 and per-channel FP8 W8A8), a streamed window decoder,
the span-2 NVFP4 CUDA decoder, the route telemetry a census reads, and the
``runtime_contract.json`` a producer reads to decide whether a rung is
servable at all.  Nothing here imports ``gridbook``; a test asserts it.

WHAT SELECTS IT.  The checkpoint: ``quantization_config.quant_method:
"tessera"``.  vLLM's own dispatch then builds :class:`config.TesseraConfig`.
The single operator knob is the residency, ``TESSERA_SERVE_MODE=resident|
streamed`` (``lane``), which is declared rather than defaulted because it
changes the footprint the artifact occupies, and is folded into vLLM's
compile-cache key because it changes the traced forward.

WHAT IT NEEDS.  A vLLM process on a device whose compiled ops include the
quantizers each route uses; the NVFP4 route additionally builds one CUDA
extension on first use (``ext``).  The FP8 route is pure torch.  Nothing here
is imported unless a Tessera checkpoint is being served: this module imports
neither torch nor vLLM at module level, so a producer can read the packaged
contract on a machine with no GPU.

SCOPE.  Dense Linears, TP=1, both residency modes, eager and compiled.  Routed
MoE, tensor parallelism above one and expert parallelism are refused by name
(``config.TesseraConfig.get_quant_method``), never degraded.
"""
from __future__ import annotations

__all__ = ["__version__", "QUANT_METHOD", "register"]

__version__ = "0.1.0"

#: The ``quantization_config.quant_method`` value this plugin registers under.
QUANT_METHOD = "tessera"


def register() -> None:
    """vLLM's ``general_plugins`` entry point.

    Registers the quantization config under ``tessera``.  Idempotent across
    repeated plugin loads (vLLM loads plugins in the engine core and in every
    worker process).
    """
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .config import TesseraConfig

    try:
        register_quantization_config(QUANT_METHOD)(TesseraConfig)
    except ValueError:
        # Already registered.
        pass
