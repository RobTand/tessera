"""Synthetic wires shared by H1 and the dense GPU tests."""

def _scheme(family, rows, cols, roles, wire_bytes, q256=1024):
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8

    e4m3 = family == "e4m3"
    return {"family": TESSERA_FP8 if e4m3 else TESSERA_BF16, "grid": "E4M3" if e4m3 else "BF16",
            "body": "WINDOW", "plane": "CHANNEL", "q256": q256, "rows": rows, "columns": cols,
            "wire_bytes": wire_bytes, "roles": [[n, r] for n, r in roles]}


def encode_module(family, roles, cols, q256=1024, seed=0):
    """Encode the roles and return the blob, scheme and FP64 effective weights.

    Each family keeps raw decoded values and its FP32 row scale separate.
    The reference multiplies these operands in FP64 without a BF16 conversion.
    """
    import torch
    from tessera import fused, export, stock, decode, alphabet
    torch.manual_seed(seed)
    blobs, refs = [], []
    for i, (name, rows) in enumerate(roles):
        w = torch.randn(rows, cols, device="cuda") * 0.02
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        grid = alphabet.E4M3_GRID if family == "e4m3" else alphabet.BF16_GRID
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=grid, q256=q256, name=name, verify=False)
        if family == "e4m3":
            tiles = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
            refs.append(tiles["weight"].to("cuda").double()
                        * tiles["weight_scale"].to("cuda").double().reshape(-1, 1))
        else:
            values, scale = decode.materialize_bf16(unit, forests, export.DEFAULT_CODE)
            refs.append(values.to("cuda").double() * scale.to("cuda").double()[:, None])
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(family, sum(r for _, r in roles), cols, roles, len(blob), q256)
    return blob, scheme, torch.cat(refs)

