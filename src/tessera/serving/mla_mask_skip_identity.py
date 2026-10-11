"""The eager mask-skip identity shared by the emitter and its audit."""

MASK_SKIP_VOCABULARY = {
    "MASK_SKIP_POLICY": "mla_mask_skip_eager_dispatch",
    "MASK_SKIP_KIND": "attention_backend",
    "MASK_SKIP_CONTRACT": "stock_kernel_overrides",
    "NATIVE_DECODER": "native_mg_mask_skip",
    "STOCK_DECODER": "stock",
    "NATIVE_SYMBOL": "tessera_mla_prefill_mg_l0",
    "STOCK_SYMBOL": "stock",
    "NATIVE_SCHEDULE": "mg_mask_skip_pass_buffers",
    "STOCK_SCHEDULE": "stock_mg",
}
_NATIVE = (MASK_SKIP_VOCABULARY["NATIVE_DECODER"], MASK_SKIP_VOCABULARY["NATIVE_SCHEDULE"])
_STOCK = (MASK_SKIP_VOCABULARY["STOCK_DECODER"], MASK_SKIP_VOCABULARY["STOCK_SCHEDULE"])


def emit_mask_skip_route(layer, *, shape: str, symbol: str, reason: str | None = None) -> None:
    """Record the actual symbol with the native or stock dispatch identity."""
    from .telemetry import emit_route

    decoder, schedule = _STOCK if reason else _NATIVE
    emit_route(layer, kind=MASK_SKIP_VOCABULARY["MASK_SKIP_KIND"],
               policy=MASK_SKIP_VOCABULARY["MASK_SKIP_POLICY"], symbol=symbol,
               shape=shape, contract=MASK_SKIP_VOCABULARY["MASK_SKIP_CONTRACT"],
               state="served", reason=reason, decoder=decoder, kernel_schedule=schedule)
