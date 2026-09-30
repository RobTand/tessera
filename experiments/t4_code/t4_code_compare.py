"""T-4 code question: which body spends the E2M1 symbol bits best under the
mandatory group-scale plane?

The FP4 block-scaled MMA on sm_121 reads its weight operand as E2M1 codes with
a per-16 UE4M3 block scale, so every T-4 wire must decode to E2M1 symbols over
a group-scale plane.  Two bodies do that today, both over the LUT16 plane
(a 4-bit index per 16 weights into a per-unit 16-entry E4M3 table):

* (a) span-2 TCQ, the served routed wire (``export.TCQ_RECIPE``), one code
  rate per unit on the native decoder: q256 128..896 in steps of 128;
* (b) the window body (``export.E2M1X2_SUBCAP_RECIPE``, L=12 by default),
  which ``wire_recipe`` already names below the cap for research encodes and
  which the kernel lane decodes bit-exactly.  It spends no label plane, so it
  is matched to (a) at q256 + 64, and it reaches q256 1024 (the whole E2M1x2
  grid, 4.0 body bits per weight).

Every arm is the production encoder (``encode_linear``) -> bytes ->
``read_unit_artifact``, priced at the bytes written.  Legs, on held-out rows
(the last ``--eval-rows`` of each capture; the rest are the fit rows):

* ``wt``   relative Frobenius weight error;
* ``out``  relative output error, bf16-captured activations, weight-only
           (``||X dW^T|| / ||X W^T||``: the activation-weighted error, i.e. the
           Hessian-weighted error with H = X^T X);
* ``a4s``  executed W4A4 with the STATIC per-module global activation scale
           (capacity/amax over the fit rows, the export's calibration), through
           vLLM's own ``scaled_fp4_quant`` -- the served contract
           ``e2m1_group16_ue4m3_static``;
* ``a4d``  the same with a DYNAMIC per-token global (capacity/amax of the
           token), through a torch reference quantiser whose agreement with
           ``scaled_fp4_quant`` at the static global is recorded per tensor;
* ``a4s_ref`` the static global through that same reference, so static and
           dynamic are compared on one quantiser (``a4s_ref`` vs ``a4d``).

No PrismaQuant import: the a4 legs use vLLM's operator and a local reference.

Activation proxies (named in every row): routed-expert down_proj reads the
expert's own SwiGLU-with-clamp of the captured MoE input through its BF16
gate/up, over every captured token (routing not applied); MLA q_a/kv_a read the
preceding linear-attention layer's captured attention input; q_b reads that
proxy through this layer's q_a and q_a_layernorm; o_proj has no capture and is
weight-only.

    python3 experiments/t4_code/t4_code_compare.py --set experts --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.export import DEFAULT_SCALE_REFIT, encode_linear
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.unit_artifact import read_unit_artifact

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
P = "model.language_model.layers."
GRID = tuple_grid(E2M1_GRID, 2)
TCQ, WINDOW, LUT = BodyKind.TCQ, BodyKind.WINDOW, ScalePlaneKind.LUT
SWIGLU_LIMIT = 10.0
CAPACITY = 448.0 * 6.0          # E4M3 max * E2M1 max: the NVFP4 global's numerator
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# (arm label, body, q256, window bits).  TCQ at every native-decoder rung; the
# window at the matched q256 (+64: no label plane) and at the grid's top.
ARMS = ([(f"tcq q{q}", TCQ, q, None) for q in (128, 256, 384, 512, 640, 768, 896)]
        + [(f"win12 q{q}", WINDOW, q, 12) for q in (192, 320, 448, 576, 704, 832, 960, 1024)]
        + [(f"win14 q{q}", WINDOW, q, 14) for q in (576, 960)])


def cap(name: str) -> str:
    return f"{ACT}/{name.replace('.', '__')}.pt"


def tensor_sets():
    """``{set: [(tag, weight key, row slice, activation spec)]}``."""
    experts = []
    for layer in (5, 20, 42):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{P}{layer}.mlp.experts.0.{proj}.weight"
            act = (("moe_in", layer) if proj != "down_proj" else ("expert_swiglu", layer, 0))
            experts.append((f"L{layer}.e0.{proj}", key, None, act))
    dense = [
        ("L0.mlp.gate_proj", f"{P}0.mlp.gate_proj.weight", 2048, ("capture", f"{P}0.mlp.gate_proj")),
        ("L0.mlp.up_proj", f"{P}0.mlp.up_proj.weight", 2048, ("capture", f"{P}0.mlp.up_proj")),
        ("L0.mlp.down_proj", f"{P}0.mlp.down_proj.weight", 1024, ("capture", f"{P}0.mlp.down_proj")),
        ("L10.shared.gate_proj", f"{P}10.mlp.shared_experts.gate_proj.weight", None,
         ("capture", f"{P}10.mlp.shared_experts.gate_proj")),
        ("L10.shared.up_proj", f"{P}10.mlp.shared_experts.up_proj.weight", None,
         ("capture", f"{P}10.mlp.shared_experts.up_proj")),
        ("L10.shared.down_proj", f"{P}10.mlp.shared_experts.down_proj.weight", None,
         ("capture", f"{P}10.mlp.shared_experts.down_proj")),
    ]
    attn = [
        ("L3.attn.q_a_proj", f"{P}3.self_attn.q_a_proj.weight", None, ("attn_in", 2)),
        ("L3.attn.kv_a_proj_with_mqa", f"{P}3.self_attn.kv_a_proj_with_mqa.weight", None,
         ("attn_in", 2)),
        ("L3.attn.q_b_proj", f"{P}3.self_attn.q_b_proj.weight", 4096, ("q_b_in", 3, 2)),
        ("L3.attn.o_proj", f"{P}3.self_attn.o_proj.weight", 1024, None),
    ]
    return {"experts": experts, "dense": dense, "attn": attn}


class Source:
    def __init__(self):
        self.index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]

    def get(self, key):
        with safe_open(f"{SRC}/{self.index[key]}", framework="pt") as f:
            return f.get_tensor(key)


def load_capture(name):
    blob = torch.load(cap(name), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "inputs" not in blob:
        raise SystemExit(f"{cap(name)}: expected a dict with 'inputs', got "
                         f"{list(blob) if isinstance(blob, dict) else type(blob)}")
    return blob["inputs"]


def rms_norm(x, weight, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def activations(src, spec, dev):
    """``(x_fit, x_eval, proxy label)`` in fp32 on ``dev``, or ``None``."""
    if spec is None:
        return None
    kind = spec[0]
    if kind == "moe_in":
        x = load_capture(f"{P}{spec[1]}.mlp.experts")
        label = "captured MoE input"
    elif kind == "capture":
        x = load_capture(spec[1])
        label = "captured input"
    elif kind == "expert_swiglu":
        layer, e = spec[1], spec[2]
        x0 = load_capture(f"{P}{layer}.mlp.experts").to(dev).float()
        g = x0 @ src.get(f"{P}{layer}.mlp.experts.{e}.gate_proj.weight").to(dev).float().T
        u = x0 @ src.get(f"{P}{layer}.mlp.experts.{e}.up_proj.weight").to(dev).float().T
        g = g.clamp(max=SWIGLU_LIMIT)
        u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        x = (g * torch.sigmoid(g) * u).to(torch.bfloat16).cpu()
        del x0, g, u
        label = "PROXY: expert SwiGLU(clamp 10) of the captured MoE input, all tokens"
    elif kind == "attn_in":
        x = load_capture(f"{P}{spec[1]}.self_attn.forget_gate.f_a_proj")
        label = f"PROXY: layer {spec[1]} captured attention input"
    elif kind == "q_b_in":
        layer, near = spec[1], spec[2]
        x0 = load_capture(f"{P}{near}.self_attn.forget_gate.f_a_proj").to(dev).float()
        qa = x0 @ src.get(f"{P}{layer}.self_attn.q_a_proj.weight").to(dev).float().T
        x = rms_norm(qa, src.get(f"{P}{layer}.self_attn.q_a_layernorm.weight").to(dev)
                     ).to(torch.bfloat16).cpu()
        del x0, qa
        label = f"PROXY: layer {near} attention input -> L{layer} q_a -> q_a_layernorm"
    else:
        raise SystemExit(f"unknown activation spec {spec}")
    return x, label


# ---------------------------------------------------------------- NVFP4 A-side
def e2m1_decode(codes):
    mag = E2M1_VALUES.to(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def unpack_fp4(packed, cols):
    lo = packed & 0xF
    hi = packed >> 4
    return torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], cols)


def nvfp4_qdq_vllm(x_bf16, gs):
    """vLLM's own quantiser at a scalar global, dequantised (linear layout)."""
    from tessera.kernel_a4 import a4_quantize_activation

    packed, scales = a4_quantize_activation(x_bf16.contiguous(), gs)
    m, k = x_bf16.shape
    codes = unpack_fp4(packed.reshape(m, k // 2), k)
    sf = scales.reshape(m, k // 16).float()
    deq = e2m1_decode(codes).reshape(m, k // 16, 16) * (sf / gs.reshape(1, 1)).unsqueeze(-1)
    return deq.reshape(m, k), codes, scales.view(torch.uint8).reshape(m, k // 16)


def e2m1_round(v):
    """Round-to-nearest-even onto the E2M1 grid, saturating at 6 (the cvt's rule)."""
    a = v.abs().clamp(max=6.0)
    grid = E2M1_VALUES.to(v.device)
    # nearest level; ties to the even code (index parity)
    d = (a.unsqueeze(-1) - grid).abs()
    best = d.min(dim=-1, keepdim=True).values
    cand = (d == best)
    idx = torch.arange(8, device=v.device)
    # among tied candidates prefer the even index
    score = cand.float() * 2 + (idx % 2 == 0).float() * cand.float()
    code = score.argmax(dim=-1)
    # the cvt keeps the sign of a value that rounds to zero (-0 is code 8)
    return code | ((v < 0).long() << 3)


def nvfp4_qdq_ref(x_bf16, gs_rows):
    """Reference NVFP4 activation quantiser with a per-row global ``gs_rows``
    ([M, 1] fp32), mirroring the kernel's arithmetic: block scale
    e4m3(amax/6 * gs), then x * (1 / (sf / gs)) rounded onto E2M1."""
    x = x_bf16.float()
    m, k = x.shape
    xg = x.reshape(m, k // 16, 16)
    amax = xg.abs().amax(-1)
    sf = (gs_rows * (amax * (1.0 / 6.0))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sff = sf.float()
    out_scale = torch.where(sff > 0, 1.0 / (sff * (1.0 / gs_rows)), torch.zeros_like(sff))
    codes = e2m1_round(xg * out_scale.unsqueeze(-1))
    deq = e2m1_decode(codes) * (sff / gs_rows).unsqueeze(-1)
    return deq.reshape(m, k), codes.reshape(m, k), sf.view(torch.uint8)


def a_side(x_fit, x_ev):
    """Static and dynamic A-side, plus their activation error and range events."""
    amax_fit = x_fit.float().abs().max()
    gs = (CAPACITY / amax_fit).reshape(1).float()
    xb = x_ev.to(torch.bfloat16)
    x32 = xb.float()
    deq_s, codes_s, sf_s = nvfp4_qdq_vllm(xb, gs)
    # the reference must reproduce vLLM's op at the static global, byte for byte
    deq_r, codes_r, sf_r = nvfp4_qdq_ref(xb, gs.reshape(1, 1).expand(xb.shape[0], 1))
    ref_match = {"codes_equal": bool(torch.equal(codes_s.long() & 0xF, codes_r.long() & 0xF)),
                 "scales_equal": bool(torch.equal(sf_s, sf_r)),
                 "code_mismatch_frac": float((codes_s.long() != codes_r.long()).float().mean()),
                 "scale_mismatch_frac": float((sf_s != sf_r).float().mean())}
    amax_tok = x32.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    gs_tok = CAPACITY / amax_tok
    deq_d, _codes_d, sf_d = nvfp4_qdq_ref(xb, gs_tok)
    a_side.static_ref = deq_r
    blocks = x32.reshape(x32.shape[0], -1, 16).abs().amax(-1)
    live = blocks > 0

    def events(sf_bytes):
        sf = sf_bytes.view(torch.float8_e4m3fn).float()
        return {"saturated_blocks": int((sf >= 448.0).sum()),
                "zero_scale_live_blocks": int(((sf == 0) & live).sum()),
                "subnormal_scale_blocks": int(((sf > 0) & (sf < 2.0 ** -6)).sum()),
                "blocks": int(live.numel())}

    nx = x32.norm()
    return {
        "gs_static": float(gs), "amax_fit": float(amax_fit),
        "amax_eval": float(x32.abs().max()),
        "tokens_over_fit_amax": int((amax_tok.squeeze(-1) > amax_fit).sum()),
        "ref_vs_vllm_static": ref_match,
        "act_err_static": float((deq_s - x32).norm() / nx),
        "act_err_static_ref": float((deq_r - x32).norm() / nx),
        "act_err_dynamic": float((deq_d - x32).norm() / nx),
        "events_static": events(sf_s), "events_dynamic": events(sf_d),
    }, deq_s, deq_d


def nvfp4_weight_rtn(w):
    """Stock NVFP4 weight RTN (per-16 E4M3 scale, one fp32 global): 4.5 bpp."""
    gs = (CAPACITY / w.abs().max()).reshape(1, 1).float()
    deq, _c, _s = nvfp4_qdq_ref(w.to(torch.bfloat16), gs.expand(w.shape[0], 1))
    return deq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["experts", "dense", "attn"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--refit", type=int, default=DEFAULT_SCALE_REFIT)
    ap.add_argument("--arms", nargs="*", default=None, help="substring filters on arm labels")
    ap.add_argument("--tensors", nargs="*", default=None, help="substring filters on tensor tags")
    ap.add_argument("--slice-cols", type=int, default=None, help="smoke only: cut columns")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t4_code_{a.set}.json"
    dev = "cuda"
    torch.manual_seed(0)
    src = Source()
    arms = [arm for arm in ARMS if not a.arms or any(f in arm[0] for f in a.arms)]
    root = Path(__file__).resolve().parent.parent.parent / "src" / "tessera"
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    out = {"args": vars(a), "encoder_digest": h.hexdigest()[:16],
           "tessera_head": os.environ.get("TESSERA_HEAD"), "image": os.environ.get("ORACLE_IMAGE"),
           "host": os.environ.get("HOST_NAME"), "device": torch.cuda.get_device_name(),
           "torch": torch.__version__, "start_unix": time.time(), "tensors": {}}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    for tag, key, rows, spec in tensor_sets()[a.set]:
        if a.tensors and not any(f in tag for f in a.tensors):
            continue
        t_tensor = time.time()
        w = src.get(key)
        if rows is not None:
            w = w[:rows]
        if a.slice_cols:
            w = w[:, : a.slice_cols]
        w = w.contiguous().to(dev).float()
        acts = activations(src, spec, dev)
        rec_t = {"weight": key, "shape": list(w.shape), "row_slice": rows, "arms": {}}
        if acts is not None:
            x, label = acts
            if a.slice_cols:
                x = x[:, : a.slice_cols]
            n_fit = x.shape[0] - a.eval_rows
            x_fit = x[:n_fit].to(dev)
            x_ev = x[n_fit:].to(dev).float()
            rec_t["activation"] = label
            aside, xq_s, xq_d = a_side(x_fit, x_ev)
            xq_r = a_side.static_ref
            rec_t["a_side"] = aside
            y = x_ev @ w.T
            ny = y.norm()
            del x, x_fit
        else:
            rec_t["activation"] = "none: weight-only (no capture of this module's input)"
            x_ev = xq_s = xq_d = xq_r = y = ny = None
        nw = w.norm()
        log(f"\n== {tag} {tuple(w.shape)}  act: {rec_t['activation']}")
        if acts is not None:
            log(f"   A-side: static act err {rec_t['a_side']['act_err_static']:.5f}  dynamic "
                f"{rec_t['a_side']['act_err_dynamic']:.5f}  ref==vLLM "
                f"{rec_t['a_side']['ref_vs_vllm_static']}")
        log(f"   {'arm':<14} {'bpp':>6} {'wt':>8} {'out':>8} {'a4s':>8} {'a4d':>8} {'s':>6}")

        def rec(arm, hat, bpp, secs, **extra):
            r = {"bpp": bpp, "wt": float((hat - w).norm() / nw), "secs": secs, **extra}
            if y is not None:
                r["out"] = float((x_ev @ hat.T - y).norm() / ny)
                r["a4s"] = float((xq_s @ hat.T - y).norm() / ny)
                r["a4d"] = float((xq_d @ hat.T - y).norm() / ny)
                r["a4s_ref"] = float((xq_r @ hat.T - y).norm() / ny)
            rec_t["arms"][arm] = r
            fmt = lambda k: f"{r[k]:8.5f}" if k in r else "       -"  # noqa: E731
            log(f"   {arm:<14} {bpp:6.3f} {r['wt']:8.5f} {fmt('out')} {fmt('a4s')} {fmt('a4d')} "
                f"{secs:6.1f}")

        rec("nvfp4 rtn", nvfp4_weight_rtn(w), 4.5, 0.0)
        for label, body, q, bits in arms:
            t0 = time.time()
            try:
                kw = dict(grid=GRID, q256=q, body=body, scale_plane=LUT, scale_refit=a.refit)
                if body is TCQ:
                    kw["span"] = 2
                else:
                    kw["window_bits"] = bits
                unit = encode_linear(w, name=tag, **kw)
                torch.cuda.synchronize()
                hat = read_unit_artifact(unit.blob, device=dev).float()
                rec(label, hat, 8 * len(unit.blob) / w.numel(), time.time() - t0,
                    blob_bytes=len(unit.blob), q256=q, body=body.value if hasattr(body, "value")
                    else str(body), window_bits=bits)
                del hat, unit
            except Exception as exc:  # noqa: BLE001
                rec_t["arms"][label] = {"error": repr(exc)[:600], "q256": q}
                log(f"   {label:<14} ERROR {repr(exc)[:300]}")
            torch.cuda.empty_cache()
        rec_t["secs"] = time.time() - t_tensor
        out["tensors"][tag] = rec_t
        out_path.write_text(json.dumps(out, indent=1))
        out_path.with_suffix(".log").write_text("\n".join(lines) + "\n")
        del w, x_ev, xq_s, xq_d, xq_r, y
        torch.cuda.empty_cache()

    out["end_unix"] = time.time()
    out_path.write_text(json.dumps(out, indent=1))
    out_path.with_suffix(".log").write_text("\n".join(lines) + "\n")
    errors = [(t, arm) for t, r in out["tensors"].items() for arm, v in r["arms"].items()
              if "error" in v]
    log(f"\nDONE {len(out['tensors'])} tensors; arm errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
