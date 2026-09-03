"""The measured rate-distortion curve of the seven Linears issue #1 priced.

WHAT THIS IS.  Tessera issue #4 asks for one table: for each of the seven
layer-0 Qwen3-0.6B Linears PrismaQuant actually measured, and for each rung the
allocator could have chosen, the distortion that rung really costs -- so every
candidate cost can be scored against it offline instead of argued about.

THE GEOMETRY IS THE RECEIPT'S.  ``docs/measurements/tessera-allocated-served-2026-09-02.md``
section 7 built a *separator pair*: the same seven layer-0 Linears, once at the
allocator's rungs and once at the byte-matched uniform R1006, with every other
body Linear left BF16 in both arms.  That pair is what isolated the cost model
from the broadcast-to-28-layers assumption, and it is what this sweep extends:
one unit moves at a time, the other six stay at R1006, everything else stays
BF16.

THE PROXY, AND WHY IT IS FAITHFUL.  The receipt's numbers came off a vLLM serve.
Serving 100-odd arms is not affordable and is not necessary, because the thing
under test is entirely local: the seven modules' *arithmetic*.  This harness
runs the same arithmetic in-process --

  * the weight side is the wire's own: ``export.encode_linear_planes`` at HEAD
    (proven byte-equal to the priced blobs' decoded tiles by
    ``encoder_identity.py``) then ``decode.materialize_fp8``, which is the
    exact pair ``serving/fp8_route.py`` hands the GEMM;
  * the activation side is the route's declared contract,
    ``fp8_per_token_dynamic``: per-token absmax / 448 with vLLM's own
    ``min_scaling_factor = 1/(448*512)`` floor, saturating cast to E4M3;
  * the GEMM is ``torch._scaled_mm`` with rowwise A and B scales and a bf16
    output -- literally the call ``TesseraFp8LinearMethod.apply`` makes.

What differs from the serve is everything *outside* those seven modules: HF's
forward rather than vLLM's.  That difference cancels, because the teacher here
is the same HF forward with the seven modules unquantized -- both arms of every
comparison share it.  ``--anchors`` re-runs the four arms the receipt served
(uniform R1006, the allocation, the allocation with ``down_proj`` restored, and
uniform with ``down_proj`` alone cut) so the proxy's agreement with the served
numbers is measured rather than asserted.

METRICS.  Two, both against the same local BF16 teacher on the same 4088 scored
positions of ``corpus_qwen_n8_s512.json``:

  ``kl_full``    exact full-vocabulary KL(teacher || student).  The primary
                 number: nothing is coarsened, so a third-digit difference is a
                 third-digit difference.
  ``kl_top1024`` the served instrument's estimator -- KL lumped over the
                 teacher's top-1024 support plus one tail cell -- for
                 comparability with the receipt.  A lower bound by the
                 data-processing inequality, as there.

Weights-only throughout (no ``--hessian``, no LDLQ, ``scale_refit`` at the
exporter's default), because the allocation being audited was priced
weights-only and the served arms were exported that way.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import E4M3_GRID
from tessera.decode import materialize_fp8
from tessera.export import encode_linear_planes, wire_recipe

MODEL = Path("/home/rob/models/Qwen3-0.6B")
CORPUS = Path("/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json")

ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
UNIFORM_RUNG = 1006
ALLOC = {"self_attn.q_proj": 1083, "self_attn.k_proj": 1083, "self_attn.v_proj": 1083,
         "self_attn.o_proj": 934, "mlp.gate_proj": 1107, "mlp.up_proj": 1107,
         "mlp.down_proj": 749}
#: The served rungs and controls, PrismaQuant's own measured E4M3 anchors
#: (the campaign's adaptive rounds land within a few q256 of these per unit),
#: and fill-ins that put >= 4 samples between R749 and R1262 where the
#: allocator traded.
RUNGS = (320, 512, 640, 749, 826, 900, 934, 970, 1006, 1044,
         1083, 1107, 1150, 1200, 1262, 1340)

FP8_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


class TesseraFp8Linear(torch.nn.Module):
    """``serving/fp8_route.py``'s ``apply``, out of a serve.

    The tile and the row scale are mutable so one installed module serves every
    arm: an arm is a swap of two tensors, not a rebuilt model.
    """

    def __init__(self, bias: "torch.Tensor | None", out_features: int) -> None:
        super().__init__()
        self.bias = bias
        self.out_features = out_features
        self.w_fp8: torch.Tensor | None = None
        self.w_bf16: torch.Tensor | None = None
        self.scale_b: torch.Tensor | None = None

    def load(self, tile: torch.Tensor, row_scale: torch.Tensor) -> None:
        self.w_bf16 = None
        self.w_fp8 = tile.view(torch.float8_e4m3fn)
        self.scale_b = row_scale.reshape(1, self.out_features).contiguous()

    def load_bf16(self, weight: torch.Tensor) -> None:
        """Activation-side only: an exact BF16 weight under the same A contract.

        The floor no weight rate can go below, because the route quantizes the
        activations whatever the wire says.
        """
        self.w_fp8 = None
        self.w_bf16 = weight.to(torch.bfloat16).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        x2 = x2.contiguous()
        amax = x2.abs().amax(dim=-1, keepdim=True).float()
        scale_a = torch.clamp(amax / FP8_MAX, min=FP8_MIN_SCALE)
        a_q = (x2.float() / scale_a).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        if self.w_fp8 is None:
            y = ((a_q.float() * scale_a) @ self.w_bf16.float().t()).to(torch.bfloat16)
        else:
            y = torch._scaled_mm(a_q, self.w_fp8.t(), scale_a=scale_a,
                                 scale_b=self.scale_b, out_dtype=torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig[:-1], self.out_features)


def encode_tile(weight: torch.Tensor, rung: int, name: str, scale_refit=None):
    """The wire at ``rung``, decoded to the per-channel FP8 pair the route multiplies."""
    extra = {} if scale_refit is None else {"scale_refit": int(scale_refit)}
    exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=rung, name=name, verify=True, **extra)
    tile, row_scale = materialize_fp8(unit, forests, None)
    return exported, tile.contiguous(), row_scale.float().contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/home/rob/tmp/ts-rung-rd-out/tiles")
    ap.add_argument("--scale-refit", type=int, default=None,
                    help="override the encoder's LS row-scale refit passes "
                         "(candidate 3: does the mispricing track refit gain?). "
                         "None = the exporter default, which is what served.")
    ap.add_argument("--rungs", default=",".join(str(r) for r in RUNGS))
    ap.add_argument("--headroom", action="store_true",
                    help="Instead of the rung sweep, measure per unit the two "
                         "arms that bound what any rung above R1006 could ever "
                         "buy: that unit at per-channel FP8 RTN (the wire's "
                         "rate-to-infinity asymptote) and at BF16 (no weight "
                         "quantization at all), the other six held at R1006.")
    ap.add_argument("--positions", default=None,
                    help="also write per-scored-position kl_full for every arm "
                         "to this .npz, so a paired bootstrap can say which "
                         "differences at the top of the curve are real")
    args = ap.parse_args()
    rungs = tuple(int(r) for r in args.rungs.split(","))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)

    corpus = json.loads(CORPUS.read_text())
    chunks = torch.tensor(corpus["chunks"], dtype=torch.long, device=device)
    n_chunks, seqlen = chunks.shape
    scored = n_chunks * (seqlen - 1)
    assert scored == corpus["scored_positions"], (scored, corpus["scored_positions"])

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    # ---- the wire, per (unit, rung) -----------------------------------
    tiles: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    wire_bytes: dict[tuple[str, int], int] = {}
    recipes: dict[int, dict] = {}
    t0 = time.time()
    source: dict[str, torch.Tensor] = {}
    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as handle:
        for role in ROLES:
            w = handle.get_tensor(f"model.layers.0.{role}.weight").to(device, torch.float32)
            source[role] = w
            for rung in rungs:
                key = f"{role.replace('.', '__')}_R{rung}"
                path = cache / f"{key}.pt"
                if path.exists():
                    blob = torch.load(path, map_location=device)
                    tiles[(role, rung)] = (blob["tile"].view(torch.uint8).contiguous(),
                                           blob["scale"].contiguous())
                    wire_bytes[(role, rung)] = int(blob["bytes"])
                else:
                    t1 = time.time()
                    exported, tile, row_scale = encode_tile(
                        w, rung, f"model.layers.0.{role}", args.scale_refit)
                    torch.save({"tile": tile.cpu(), "scale": row_scale.cpu(),
                                "bytes": exported.exact_bytes}, path)
                    tiles[(role, rung)] = (tile, row_scale)
                    wire_bytes[(role, rung)] = exported.exact_bytes
                    print(f"encoded {role:22s} R{rung:5d} {exported.exact_bytes:9d}B "
                          f"in {time.time() - t1:6.1f}s", flush=True)
                if rung not in recipes:
                    r = wire_recipe(E4M3_GRID, rung)
                    recipes[rung] = {"body": str(r.body), "scale_plane": str(r.scale_plane),
                                     "span": r.span, "window_bits": r.window_bits}
    print(f"wire ready in {time.time() - t0:.1f}s", flush=True)

    # ---- teacher -------------------------------------------------------
    @torch.no_grad()
    def logits_for(chunk: torch.Tensor) -> torch.Tensor:
        return model(chunk.unsqueeze(0)).logits[0, : seqlen - 1].float()

    teacher_lp = torch.empty((n_chunks, seqlen - 1, model.config.vocab_size),
                             dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n_chunks):
            teacher_lp[i] = torch.log_softmax(logits_for(chunks[i]), dim=-1)
    teacher_p = teacher_lp.exp()
    top_v, top_i = teacher_lp.topk(1024, dim=-1)
    top_p = top_v.exp()
    top_tail_p = (1.0 - top_p.sum(-1)).clamp_min(0.0)
    print("teacher ready", flush=True)

    # ---- install the seven route modules -------------------------------
    layer0 = model.model.layers[0]
    holders = {}
    for role in ROLES:
        parent_name, leaf = role.split(".")
        parent = getattr(layer0, parent_name)
        old = getattr(parent, leaf)
        holder = TesseraFp8Linear(old.bias, old.out_features).to(device)
        setattr(parent, leaf, holder)
        holders[role] = holder

    def set_arm(assignment: dict[str, int]) -> None:
        for role in ROLES:
            tile, row_scale = tiles[(role, assignment[role])]
            holders[role].load(tile, row_scale)

    per_position: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def measure() -> dict:
        kl_full = 0.0
        kl_top = 0.0
        agree = 0
        pos = torch.empty(scored, dtype=torch.float32, device=device)
        for i in range(n_chunks):
            slp = torch.log_softmax(logits_for(chunks[i]), dim=-1)
            rowwise = (teacher_p[i] * (teacher_lp[i] - slp)).sum(-1)
            pos[i * (seqlen - 1):(i + 1) * (seqlen - 1)] = rowwise
            kl_full += float(rowwise.sum())
            sv = slp.gather(-1, top_i[i])
            sp = sv.exp()
            s_tail = (1.0 - sp.sum(-1)).clamp_min(1e-30)
            t_tail = top_tail_p[i].clamp_min(1e-30)
            kl_top += float((top_p[i] * (top_v[i] - sv)).sum()
                            + (t_tail * (t_tail / s_tail).log()).sum())
            agree += int((slp.argmax(-1) == teacher_lp[i].argmax(-1)).sum())
        measure.last_positions = pos.cpu()
        return {"kl_full": kl_full / scored, "kl_top1024": kl_top / scored,
                "top1_agree": agree / scored}

    results = {"schema": "tessera.rung_rd_table/1", "rungs": list(rungs),
               "scale_refit_override": args.scale_refit,
               "roles": list(ROLES), "uniform_rung": UNIFORM_RUNG, "alloc": ALLOC,
               "recipes": recipes,
               "wire_bytes": {f"{r}|{q}": v for (r, q), v in wire_bytes.items()},
               "arms": []}

    def run(name: str, assignment: dict[str, int], kind: str) -> dict:
        set_arm(assignment)
        t = time.time()
        m = measure()
        per_position[name] = measure.last_positions
        row = {"arm": name, "kind": kind, "assignment": dict(assignment), **m,
               "seconds": time.time() - t,
               "wire_bytes": sum(wire_bytes[(r, assignment[r])] for r in ROLES)}
        results["arms"].append(row)
        print(f"{name:34s} kl_full {m['kl_full']:.6f}  kl_top1024 {m['kl_top1024']:.6f}  "
              f"top1 {m['top1_agree']*100:.2f}%  {row['wire_bytes']}B", flush=True)
        return row

    def run_floor(name: str, loader) -> dict:
        loader()
        t = time.time()
        m = measure()
        per_position[name] = measure.last_positions
        row = {"arm": name, "kind": "floor", "assignment": {}, **m,
               "seconds": time.time() - t, "wire_bytes": 0}
        results["arms"].append(row)
        print(f"{name:34s} kl_full {m['kl_full']:.6f}  kl_top1024 {m['kl_top1024']:.6f}  "
              f"top1 {m['top1_agree']*100:.2f}%", flush=True)
        return row

    def load_rtn_fp8() -> None:
        """Per-channel FP8 RTN: the rate-to-infinity asymptote of the E4M3 wire."""
        for role in ROLES:
            w = source[role]
            s_row = (w.abs().amax(dim=-1) / FP8_MAX).clamp_min(1e-12).float()
            tile = (w / s_row[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            holders[role].load(tile.view(torch.uint8), s_row)

    def load_bf16_weights() -> None:
        for role in ROLES:
            holders[role].load_bf16(source[role])

    run_floor("floor:activation_only(W16A8)", load_bf16_weights)
    run_floor("floor:fp8_rtn_per_channel(W8A8)", load_rtn_fp8)

    uniform = {r: UNIFORM_RUNG for r in ROLES}
    # the four arms the receipt served, first, so the proxy is validated before
    # anything is read off the sweep
    run("anchor:uniform_R1006", uniform, "anchor")
    run("anchor:allocated", dict(ALLOC), "anchor")
    run("anchor:alloc_down_restored", {**ALLOC, "mlp.down_proj": UNIFORM_RUNG}, "anchor")
    run("anchor:uniform_down749", {**uniform, "mlp.down_proj": 749}, "anchor")

    if args.headroom:
        # What is there left to win above R1006?  Two arms per unit, the other
        # six held at R1006: the unit at the wire's rate-to-infinity asymptote
        # (per-channel FP8 RTN, which is what an E4M3 body converges to as the
        # trellis stops making errors) and the unit at BF16 (no weight
        # quantization at all).  The gap from that unit's R1006 arm to its RTN
        # arm is an upper bound on everything any finer rung could buy for it.
        def run_mixed(name: str, role: str, loader) -> dict:
            set_arm(uniform)
            loader(role)
            t = time.time()
            m = measure()
            per_position[name] = measure.last_positions
            row = {"arm": name, "kind": "headroom", "assignment": {**uniform},
                   "moved_role": role, **m, "seconds": time.time() - t,
                   "wire_bytes": sum(wire_bytes[(r, UNIFORM_RUNG)]
                                     for r in ROLES if r != role)}
            results["arms"].append(row)
            print(f"{name:34s} kl_full {m['kl_full']:.6f}  "
                  f"kl_top1024 {m['kl_top1024']:.6f}  "
                  f"top1 {m['top1_agree']*100:.2f}%", flush=True)
            return row

        def one_rtn(role: str) -> None:
            w = source[role]
            s_row = (w.abs().amax(dim=-1) / FP8_MAX).clamp_min(1e-12).float()
            tile = (w / s_row[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            holders[role].load(tile.view(torch.uint8), s_row)

        def one_bf16(role: str) -> None:
            holders[role].load_bf16(source[role])

        for role in ROLES:
            run_mixed(f"headroom:{role}@fp8_rtn", role, one_rtn)
            run_mixed(f"headroom:{role}@bf16", role, one_bf16)
    else:
        # the sweep: one unit moves, the other six hold at R1006
        for role in ROLES:
            for rung in rungs:
                if rung == UNIFORM_RUNG:
                    continue
                run(f"sweep:{role}@R{rung}", {**uniform, role: rung}, "sweep")

    Path(args.out).write_text(json.dumps(results, indent=2))
    if args.positions:
        import numpy as np
        np.savez_compressed(args.positions,
                            **{k: v.numpy() for k, v in per_position.items()})
        print("wrote", args.positions)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
