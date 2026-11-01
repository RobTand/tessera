"""Extend the existing graph observer with actual native piece-major arguments."""
from pathlib import Path

# Reuse its histogram, import patcher, serving predicate and serialized writer.
_source = Path("/digest/usercustomize.py")
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())


def _ga_patch_routed(module):
    original_ext = module._ext

    def extension(library):
        native = original_ext(library)
        if not getattr(native, "_ga702_piece_major_observed", False):
            original_forward = native.routed_fused_forward

            def forward(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                # The existing native contract has 33 arguments; the actual
                # boolean at index 20 selects the resident word reader.
                if (getattr(_GA["serving"], "on", False) and len(args) == 33
                        and type(args[20]) is bool):
                    key = (f"tessera.routed_fused|native-forward|library={library}"
                           f"|mode={args[0]}|tokens={args[2].shape[0]}"
                           f"|piece_major={int(args[20])}|block_rows={args[32]}")
                    with _GA["lock"]:
                        _GA["counts"][key] = _GA["counts"].get(key, 0) + 1
                        _GA["dirty"] = True
                return result

            native.routed_fused_forward = forward
            native._ga702_piece_major_observed = True
        return native

    module._ext = extension


if "_GA_PATCHES" in globals():
    _GA_PATCHES["tessera.routed_fused"] = _ga_patch_routed
