"""Stock-oracle crosscheck for the REAL window MoE method.

Drives `build_tessera_moe_method` through create_weights -> every loader
callback -> process_weights_after_loading -> apply and compares the ROUTED
output against **actual stock vLLM `fused_experts`** on **independently
decoded reference weights** (the materialising reader / read_unit_artifact,
never the native path's own bundles):

* FP8 ordinary route, TP1 and both TP2 geometry cuts (rank-local weights and
  the rank-local stock result; no all-reduce in this component check);
* folded BF16 through the research route (TP2 rank 0).

Stock is the oracle for the boundary in question: dynamic per-token FP8 at
both stages, the per-route bf16 down output and `moe_sum` rounding.  The
numbers are reported, not buried: each arm prints max abs, max relative and
the deviation in bf16 ulps of the reference maximum.

Run directly (actual vLLM work is exempt from PrismaBuild), in the pinned
image:

  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests \
    -v <worktree>:/work:ro --entrypoint bash <image> -lc \
    "python3 -m pip install -q --user pytest; python3 /work/tests/native_window_moe_stock_crosscheck.py"
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/tests")

from tessera import native_window_moe as nwm               # noqa: E402
from tessera.serving import moe_route                       # noqa: E402
from tessera.serving.scheme import validate_tessera_moe_scheme  # noqa: E402
from test_serving_moe_route import _stack, HIDDEN, INTER, EXPERTS   # noqa: E402
from test_native_window_moe_method import (                 # noqa: E402
    _load_all, _native_layer, bf16_wires_native_data)


def _bf16_ulp(value: float):
    """The spacing to the NEXT bf16 value, honestly.

    ``torch.nextafter`` moves one representable step in bf16, so this is the
    spacing at ``value``'s exponent -- not half of it (an fp32 bit bump of
    0x8000 is half a bf16 ulp) and not an assumed unit.  A zero magnitude has
    no meaningful ulp ratio; ``None`` says so instead of dividing by a
    subnormal spacing.
    """
    if value == 0.0:
        return None
    t = torch.tensor(abs(value), dtype=torch.bfloat16)
    nxt = torch.nextafter(t, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return float(nxt) - float(t)


def _arm(name, native, stock, report):
    diff = (native.float() - stock.float()).abs()
    mag = float(stock.float().abs().max())
    ulp = _bf16_ulp(mag)
    ulps = None if (ulp in (None, 0.0)) else float(diff.max() / ulp)
    entry = {
        "arm": name,
        "max_abs": float(diff.max()),
        "max_over_mag": float(diff.max() / max(mag, 1e-6)),
        "bf16_ulps_of_max": ulps,
        "shapes": [tuple(native.shape), tuple(stock.shape)],
    }
    report["arms"].append(entry)
    print("ARM", json.dumps(entry), flush=True)
    # A CHOSEN DIAGNOSTIC SCREEN, not a derived bound: the chain runs through a
    # nonlinear activation and two quantized stages, so 4 x 0.5 ulp does not
    # compose into an end-to-end proof.  It is tight enough to catch a wrong
    # clamp or a wrong projection and labelled as a screen in the report.
    entry["tolerance"] = {"screen_bf16_ulps_of_max": 4.0,
                          "derivation": "chosen screen, not a composed bound"}
    return ulps is not None and ulps <= 4.0


def _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size,
               apply_router_weight_on_input=False, swiglu_limit=None):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    w1 = torch.stack([
        torch.cat([r["gate"]["weight"][lo:hi], r["up"]["weight"][lo:hi]]) for r in reference
    ]).cuda().contiguous()
    s1 = torch.stack([
        torch.cat([r["gate"]["weight_scale"][lo:hi], r["up"]["weight_scale"][lo:hi]]) for r in reference
    ]).cuda().contiguous()
    w2 = torch.stack([r["down"]["weight"][:, lo:hi] for r in reference]).cuda().contiguous()
    s2 = torch.stack([r["down"]["weight_scale"] for r in reference]).cuda().contiguous()
    return fused_experts(
        x, w1, w2, weights, ids, activation=MoEActivation.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
        quant_config=quant(w1_scale=s1, w2_scale=s2, swiglu_limit=swiglu_limit))


def _fp8_quant_config():
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend, make_fp8_moe_quant_config)

    def build(w1_scale, w2_scale, swiglu_limit=None):
        return make_fp8_moe_quant_config(
            fp8_backend=Fp8MoeBackend.TRITON, w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=None, a2_scale=None, per_act_token_quant=True,
            per_out_ch_quant=True, block_shape=None,
            gemm1_alpha=None, gemm1_beta=None, swiglu_limit=swiglu_limit, layer=None)
    return build


def canonical_units(fixture, layers=(3, 4), experts=(0, 1), verify=True):
    """The FIXTURE'S OWN canonical experts, not synthetic wires.

    Groups are found by EXACT target: the checkpoint's ``config_groups`` are
    keyed by sanitized names, and each entry carries its real ``targets`` list
    and its own ``scheme``, so the lookup matches the dotted module path inside
    ``targets`` and requires it to be unique -- not by guessing a key spelling.

    Every projection container is read from the shard the index names and its
    sha256 and byte count are checked against the manifest entry that recorded
    them, so the bytes named are the bytes served.  Each container is then
    parsed with the SHARED compact reader, against its own declaration.

    Returns ``{layer: {"family", "declared", "target", "experts": {e: {...}}}}``
    where each expert carries the raw ``bytes`` (what the loader callbacks take)
    and the parsed wires.
    """
    import hashlib, json, os
    from safetensors import safe_open
    from tessera.serving import scheme as _scheme

    cfg = json.load(open(os.path.join(fixture, "config.json")))
    quant = cfg["quantization_config"]
    index = json.load(open(os.path.join(fixture, "model.safetensors.index.json")))["weight_map"]
    manifest = json.load(open(os.path.join(fixture, "tessera_derivative_manifest.json")))
    recorded = {entry["tensor"]: entry for entry in manifest["tensors"]}

    out = {}
    for layer in layers:
        target = f"model.language_model.layers.{layer}.mlp.experts"
        hits = [entry for entry in quant["config_groups"].values()
                if target in (entry.get("targets") or [])]
        if len(hits) != 1:
            raise SystemExit(f"{target}: {len(hits)} config_groups name it; a "
                             f"canonical read requires exactly one")
        # ``validate_tessera_moe_scheme`` takes the GROUP itself: the module
        # facts (family, grid, body, plane) and the two groups live at its top.
        declared = _scheme.validate_tessera_moe_scheme(hits[0]["scheme"], target)
        w13_roles = _scheme.expert_role_declarations(declared["groups"]["w13"])
        w2_roles = _scheme.expert_role_declarations(declared["groups"]["w2"])
        by_role = {row["roles"][0][0]: row for row in w13_roles + w2_roles}
        containers = {}
        for e in experts:
            raw, parsed, materialized = {}, {}, {}
            for role_key, tensor_role in (("w1", "gate_proj"), ("w3", "up_proj"),
                                          ("w2", "down_proj")):
                name = f"{target}.{e}.{tensor_role}.wire"
                with safe_open(os.path.join(fixture, index[name]), framework="pt") as f:
                    blob = f.get_tensor(name).numpy().tobytes()
                entry = recorded.get(name)
                if verify:
                    digest = hashlib.sha256(blob).hexdigest()
                    if entry is None or entry["sha256"] != digest:
                        raise SystemExit(
                            f"sha256 mismatch for {name}: manifest "
                            f"{None if entry is None else entry['sha256'][:12]}, bytes {digest[:12]}")
                    if int(entry["bytes"]) != len(blob):
                        raise SystemExit(f"byte count mismatch for {name}")
                declaration = by_role.get(tensor_role)
                if declaration is None:
                    raise SystemExit(f"{target}: no declaration for {tensor_role}; "
                                     f"has {sorted(by_role)}")
                raw[role_key] = blob
                # The compact reader is what the native intake consumes; the
                # materialising reader is the independent reference.  Both are
                # kept, so the stock decode never runs on the parser under test.
                parsed[role_key] = _scheme.parse_compact_tessera_expert_blob(
                    blob, declaration, target, device="cpu")
                materialized[role_key] = _scheme.parse_tessera_expert_blob(
                    blob, declaration, target, device="cpu")
            containers[e] = {"raw": raw, "parsed": parsed, "materialized": materialized,
                             "manifest": {k: recorded.get(f"{target}.{e}."
                                                          f"{ {'w1':'gate_proj','w3':'up_proj','w2':'down_proj'}[k] }.wire")
                                          for k in raw}}
        out[layer] = {"family": declared["family"], "declared": declared,
                      "target": target, "experts": containers}
    return out


def canonical_reference_tensors(units, layer, device="cpu"):
    """REFERENCE TENSORS for every projection -- gate, up AND down.

    FP8 (layer 3): ``prepare_tessera_moe_experts`` materialises the stock
    per-channel stack; its ``w13`` rows are split into gate and up at the
    expert's intermediate size, ``w2`` is down.
    BF16 (layer 4): each role is decoded independently with
    ``decode.materialize_bf16_folded`` on the MATERIALISING parse (the
    reference reader), which is the twin's rendering -- one bf16 rounding with
    the row scale folded in.

    Returns ``{"tensors": {...}, "meta": {...}}``: tensors stay out of the JSON
    a receipt prints, and the metadata is shapes only -- it is a preparation
    report, not a decode.
    """
    from tessera.serving import moe_route as _mr
    from tessera.decode import materialize_bf16_folded

    u = units[layer]
    experts = sorted(u["experts"])
    declared = u["declared"]
    # ``inter`` is the module's intermediate size; the stock w13 stack is
    # [2*inter, K] and is split gate-first for the per-role reference tensors.
    inter = int(declared["groups"]["w13"]["rows"]) // 2

    def _fold(expert_entry, role):
        # the MATERIALISING parse, never the compact one under test
        parsed = expert_entry["materialized"][role][0][1]
        return materialize_bf16_folded(parsed.unit, parsed.forests, parsed.code)

    if u["family"] == "TESSERA_FP8":
        bounded = dict(declared)
        bounded["experts"] = len(experts)
        blobs = {"w13": [[u["experts"][e]["raw"]["w1"], u["experts"][e]["raw"]["w3"]]
                         for e in experts],
                 "w2": [[u["experts"][e]["raw"]["w2"]] for e in experts]}
        prepared = _mr.prepare_tessera_moe_experts(blobs, bounded, u["target"],
                                                   device=device)
        tensors = {"w1": [prepared.w13_weight[e][:inter] for e in range(len(experts))],
                   "w3": [prepared.w13_weight[e][inter:2 * inter] for e in range(len(experts))],
                   "w2": [prepared.w2_weight[e] for e in range(len(experts))],
                   "s1": [prepared.w13_weight_scale[e][:inter] for e in range(len(experts))],
                   "s3": [prepared.w13_weight_scale[e][inter:2 * inter] for e in range(len(experts))],
                   "s2": [prepared.w2_weight_scale[e] for e in range(len(experts))]}
    else:
        tensors = {"w1": [_fold(u["experts"][e], "w1") for e in experts],
                   "w3": [_fold(u["experts"][e], "w3") for e in experts],
                   "w2": [_fold(u["experts"][e], "w2") for e in experts]}
    meta = {"family": u["family"], "experts": experts, "intermediate": inter,
            "note": "reference tensors materialised independently of the compact intake",
            "shapes": {k: [list(v.shape) for v in vs] for k, vs in tensors.items()}}
    return {"tensors": tensors, "meta": meta}


def canonical_execution(fixture, layers=(3, 4), experts=(0, 1), clamp=10.0):
    """The clamp-active arms at the FIXTURE'S geometry, both TP2 cuts.

    H, I and E come from the units actually loaded -- H is ``w13``'s columns, I
    is half its rows, E is the experts carried -- never from this file's
    synthetic HIDDEN/INTER globals, which is the whole point of the mode.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.config import set_current_vllm_config
    import types as _types

    # A canonical path is a measurement, so its inputs are reproducible and
    # its routing weights are neither uniform nor unit: all-ones weights make
    # apply_router_weight_on_input mathematically degenerate, which would pass
    # either placement without distinguishing them.
    torch.manual_seed(4242)
    # The real modular kernel allocates through vLLM's workspace manager,
    # which the GPU model runner normally initialises.  A harness that calls
    # ``FusedMoEKernel.apply`` directly must do the same or the kernel asserts
    # before any arithmetic.
    from vllm.v1.worker.workspace import init_workspace_manager
    if torch.cuda.is_available():
        init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))

    units = canonical_units(fixture, layers=layers, experts=experts)
    report = {"canonical": True, "fixture": fixture, "clamp": clamp,
              "seed": 4242, "arms": []}
    ok = True
    for layer in layers:
        u = units[layer]
        ref = canonical_reference_tensors(units, layer)
        tenor = ref["tensors"]
        declared = u["declared"]
        H = int(declared["groups"]["w13"]["columns"])
        inter = int(declared["groups"]["w13"]["rows"]) // 2
        E = len(u["experts"])
        report["arms"].append({"layer": layer, "family": u["family"], "H": H,
                               "intermediate": inter, "experts": E,
                               "reference": ref["meta"]["shapes"]})
        x = (torch.randn(8, H) * 8.0).bfloat16().cuda()
        ids = torch.zeros(8, 2, dtype=torch.int32, device="cuda")
        ids[:, 1] = min(1, E - 1)
        weights = (0.2 + 0.6 * torch.rand(8, 2, device="cuda")).float()
        weights[1::2, 0] = 0.2
        weights[0::2, 1] = 0.8
        report.setdefault("routing_weights", []).append(
            {"layer": layer, "distinct": int(torch.unique(weights).numel()),
             "min": float(weights.min()), "max": float(weights.max())})
        quant_fp8 = _fp8_quant_config()
        for tp_rank, tp_size in ((0, 2), (1, 2)):
            local = inter // tp_size
            lo, hi = tp_rank * local, (tp_rank + 1) * local
            layer_stub = _native_layer(tp_rank=tp_rank, tp_size=tp_size,
                                       hidden=H, inter=inter, experts=E)
            layer_stub.swiglu_limit = clamp
            kwargs = {}
            if u["family"] == "TESSERA_BF16":
                kwargs["research_selected"] = moe_route.ResearchSelectedMoeConfig(
                    max_experts_per_chunk=8, expected_tensor_parallel_size=2,
                    decode_backend="triton")
            # The builder and the stub must agree with create_weights' E.  The
            # checkpoint's 288 stays in provenance; the declaration is bounded
            # in expert COUNT only and every other fact is the checkpoint's.
            bounded = dict(declared)
            bounded["experts"] = E
            with set_current_vllm_config(_types.SimpleNamespace(
                    model_config=_types.SimpleNamespace(enforce_eager=True))):
                method = moe_route.build_tessera_moe_method(
                    bounded, u["target"], "resident", layer_stub, **kwargs)
            # MEASURE state, do not infer it: the finalizer deliberately clears
            # ``_rank_local_intake`` before ``finish``, so a None afterwards says
            # nothing about which branch ran.
            intake_now = getattr(method, "_rank_local_intake", None)
            intake_before = None
            if intake_now is not None:
                intake_before = {"type": type(intake_now).__name__,
                                 "tp_rank": getattr(intake_now, "tp_rank", None),
                                 "tp_size": getattr(intake_now, "tp_size", None)}
                for attr in ("_axis", "axis", "plans", "declared", "_declared"):
                    value = getattr(intake_now, attr, None)
                    if isinstance(value, dict):
                        intake_before[attr] = sorted(value)[:6]
            report.setdefault("provenance", []).append(
                {"layer": layer, "declared_experts": declared.get("experts"),
                 "decoded_experts": E,
                 "create_weights": {"experts": E, "hidden": H,
                                    "intermediate_per_partition": local},
                 "compact_ready": bool(getattr(method, "_compact_ready", None)),
                 "intake_before_finalize": intake_before,
                 "note": "count narrowed for a bounded decode; geometry untouched"})
            method.create_weights(layer_stub, E, H, local, torch.bfloat16)
            # ``_load_all`` walks one blob per shard id: w13 takes the gate/up
            # pair in order, w2 takes the single down blob (NOT a list of one).
            _load_all(method, layer_stub,
                      [[u["experts"][e]["raw"]["w1"], u["experts"][e]["raw"]["w3"]]
                       for e in sorted(u["experts"])],
                      [u["experts"][e]["raw"]["w2"] for e in sorted(u["experts"])])
            method.process_weights_after_loading(layer_stub)
            # what the native path ACTUALLY holds after finalize
            native_geom = {}
            for part in ("gate_up", "gate", "up", "down"):
                bundle = getattr(method._native, part, None)
                if bundle is not None:
                    native_geom[part] = {"rows": getattr(bundle, "rows", None),
                                         "cols": getattr(bundle, "cols", None),
                                         "experts": getattr(bundle, "experts", None)}
            report.setdefault("provenance", [])[-1]["native_after_finalize"] = {
                "adapter": type(method._native).__name__,
                "intake_after_finalize": type(getattr(method, "_rank_local_intake", None)).__name__
                if getattr(method, "_rank_local_intake", None) is not None else None,
                "geometry": native_geom,
                "layer_w13_weight": (list(layer_stub.w13_weight.shape)
                                     if getattr(layer_stub, "w13_weight", None) is not None
                                     and hasattr(layer_stub.w13_weight, "shape") else None)}
            if u["family"] == "TESSERA_FP8":
                w1 = torch.stack([torch.cat([tenor["w1"][e][lo:hi], tenor["w3"][e][lo:hi]])
                                  for e in range(E)]).cuda().contiguous()
                s1 = torch.stack([torch.cat([tenor["s1"][e][lo:hi], tenor["s3"][e][lo:hi]])
                                  for e in range(E)]).cuda().contiguous()
                w2 = torch.stack([tenor["w2"][e][:, lo:hi] for e in range(E)]).cuda().contiguous()
                s2 = torch.stack([tenor["s2"][e] for e in range(E)]).cuda().contiguous()
            else:
                w1 = torch.stack([torch.cat([tenor["w1"][e][lo:hi], tenor["w3"][e][lo:hi]])
                                  for e in range(E)]).cuda().contiguous()
                s1 = None
                w2 = torch.stack([tenor["w2"][e][:, lo:hi] for e in range(E)]).cuda().contiguous()
                s2 = None
            ones_ids = torch.zeros(4, 1, dtype=torch.int32, device="cuda")
            ones_w = torch.ones(4, 1, device="cuda")
            if method._native.gate_up is not None:
                gu = method._native.gate_up(x[:4], ones_ids, ones_w, preserve=True)
            else:
                gu = torch.cat([method._native.gate(x[:4], ones_ids, ones_w, preserve=True),
                                method._native.up(x[:4], ones_ids, ones_w, preserve=True)], dim=-1)
            # The fused gate/up stack is 2*LOCAL wide on a rank, not 2*inter:
            # splitting at the full intermediate size leaves the up half empty
            # and the arm would compare a real gate against a zero block.
            act = _activation_arm(
                f"canonical_L{layer}_tp2_rank{tp_rank}_activation_vs_stock_op",
                lambda g, u_, lim: nwm._silu_and_mul(g, u_, clamp_limit=lim),
                gu[:, 0, :local], gu[:, 0, local:], clamp, report)
            active = act["activity"]
            ok &= (active["gate_over_limit"] > 0 and active["up_under_neg_limit"] > 0
                   and active["up_over_limit"] > 0)
            ok &= bool(act.get("_ok"))
            # STAGED localization per rank: native gate/up vs the reference
            # tensors the arms are built from, before any activation.
            report.setdefault("stages", [])
            # ---- staged localization, COMPLETE -------------------------
            # Cover gate AND up, BOTH experts, non-unit routing, and the
            # composition the full arm actually performs.  A stage check that
            # looks at expert 0's gate under unit weights cannot say anything
            # about routing or the fused expert path.
            rids = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0],
                                 [0, 0], [1, 1]], dtype=torch.int32, device="cuda")
            for expert in range(E):
                e_ids = torch.full((4, 1), expert, dtype=torch.int32, device="cuda")
                e_w = torch.ones(4, 1, device="cuda")
                if method._native.gate_up is not None:
                    gu_e = method._native.gate_up(x[:4], e_ids, e_w, preserve=True)
                    n_g, n_u = gu_e[:, 0, :local], gu_e[:, 0, local:]
                else:
                    n_g = method._native.gate(x[:4], e_ids, e_w, preserve=True)[:, 0]
                    n_u = method._native.up(x[:4], e_ids, e_w, preserve=True)[:, 0]
                if u["family"] == "TESSERA_FP8":
                    xq = torch.empty_like(x[:4], dtype=torch.float8_e4m3fn)
                    xs = torch.empty((4, 1), dtype=torch.float32, device=x.device)
                    torch.ops._C.dynamic_per_token_scaled_fp8_quant(xq, x[:4], xs, None)
                    r_g = torch._scaled_mm(xq, w1[expert][:local].t(), xs,
                                           s1[expert][:local].reshape(1, -1).contiguous(),
                                           out_dtype=torch.bfloat16)
                    r_u = torch._scaled_mm(xq, w1[expert][local:].t(), xs,
                                           s1[expert][local:].reshape(1, -1).contiguous(),
                                           out_dtype=torch.bfloat16)
                    n_act = nwm._silu_and_mul(n_g, n_u, clamp_limit=clamp)
                    r_act = _stock_activation(r_g, r_u, clamp)
                    flat_act = n_act.reshape(-1, local).contiguous()
                    d_ids = torch.full((flat_act.shape[0], 1), expert, dtype=torch.int32, device="cuda")
                    d_w = torch.ones(flat_act.shape[0], 1, device="cuda")
                    n_dn = method._native.down(flat_act, d_ids, d_w, route_input=True)
                    aq = torch.empty_like(flat_act, dtype=torch.float8_e4m3fn)
                    asc = torch.empty((flat_act.shape[0], 1), dtype=torch.float32, device="cuda")
                    torch.ops._C.dynamic_per_token_scaled_fp8_quant(aq, flat_act, asc, None)
                    r_dn = torch._scaled_mm(aq, w2[expert].t(), asc,
                                            s2[expert].reshape(1, -1).contiguous(),
                                            out_dtype=torch.bfloat16)
                    report["stages"].append(
                        {"layer": layer, "tp_rank": tp_rank, "expert": expert,
                         "stage": "gate", "max_abs": float((n_g.float() - r_g.float()).abs().max()),
                         "mag": float(r_g.float().abs().max()),
                         "gate_cols": list(n_g.shape), "up_cols": list(n_u.shape)})
                    report["stages"].append(
                        {"layer": layer, "tp_rank": tp_rank, "expert": expert,
                         "stage": "up", "max_abs": float((n_u.float() - r_u.float()).abs().max()),
                         "mag": float(r_u.float().abs().max())})
                    report["stages"].append(
                        {"layer": layer, "tp_rank": tp_rank, "expert": expert,
                         "stage": "activation", "max_abs": float((n_act.float() - r_act.float()).abs().max()),
                         "mag": float(r_act.float().abs().max())})
                    report["stages"].append(
                        {"layer": layer, "tp_rank": tp_rank, "expert": expert,
                         "stage": "down_on_act", "max_abs": float((n_dn.float() - r_dn.float()).abs().max()),
                         "mag": float(r_dn.float().abs().max())})
            # ---- NEGATIVE EVIDENCE: the non-modular path drops the clamp.
            # ``fused_experts`` calls ``apply_moe_activation`` with no
            # ``activation_config``, so ``gemm1_clamp_limit`` never reaches the
            # activation.  This probe is retained to demonstrate that, which is
            # why the arms below use ``_stock_modular_reference`` (the real
            # ``FusedMoEKernel.apply``) instead of ``fused_experts``.
            if u["family"] == "TESSERA_FP8":
                qc_probe = quant_fp8(w1_scale=s1, w2_scale=s2, swiglu_limit=clamp)
                fused_clamped = fused_experts(x[:4], w1, w2, torch.ones(4, 2, device="cuda"),
                                              rids[:4], activation=MoEActivation.SILU,
                                              global_num_experts=E, quant_config=qc_probe)
                qc_uncapped = quant_fp8(w1_scale=s1, w2_scale=s2, swiglu_limit=None)
                fused_uncapped = fused_experts(x[:4], w1, w2, torch.ones(4, 2, device="cuda"),
                                               rids[:4], activation=MoEActivation.SILU,
                                               global_num_experts=E, quant_config=qc_uncapped)
                report["stages"].append(
                    {"layer": layer, "tp_rank": tp_rank,
                     "stage": "fused_experts_respects_swiglu_limit",
                     "clamped_vs_uncapped_max_abs": float(
                         (fused_clamped.float() - fused_uncapped.float()).abs().max()),
                     "clamped_mag": float(fused_clamped.float().abs().max()),
                     "byte_identical": bool(torch.equal(fused_clamped, fused_uncapped)),
                     "note": "0 here would mean fused_experts IGNORED the clamp"})
            for weight_input in (False, True):
                # The stock modular kernel only implements
                # ``apply_router_weight_on_input`` for topk=1 (it asserts so).
                # Comparing that placement therefore requires a topk=1 routing
                # on BOTH sides; with topk=2 only the gemm2 placement is
                # comparable against this stock path.
                if weight_input and ids.shape[1] != 1:
                    ids_wi = ids[:, :1].contiguous()
                    w_wi = weights[:, :1].contiguous()
                else:
                    ids_wi, w_wi = ids, weights
                layer_stub.apply_router_weight_on_input = weight_input
                native = method.apply(layer_stub, x, w_wi, ids_wi, None, None)
                qc = (quant_fp8(w1_scale=s1, w2_scale=s2, swiglu_limit=clamp)
                      if s1 is not None else
                      FusedMoEQuantConfig.make(gemm1_clamp_limit=clamp))
                # ``fused_experts`` calls ``apply_moe_activation`` with NO
                # ``activation_config``, so it DROPS ``gemm1_clamp_limit`` --
                # demonstrated below by a clamped run that is byte-identical to
                # an uncapped one.  The stock clamped semantics live on the
                # modular path, which builds ``ApplyMoEActivationConfig`` from
                # the quant config and passes it.  The arm applies the stock
                # stage explicitly on the fused kernel's own gemm1 output, so
                # the composition is graded against the operation the model
                # actually specifies rather than against a clamp-less stand-in.
                stock = _stock_modular_reference(
                    x, w1, w2, w_wi, ids_wi, family=u["family"], clamp=clamp,
                    experts=E, apply_router_weight_on_input=weight_input,
                    w1_scale=s1, w2_scale=s2,
                    moe_config=method.moe)
                report.setdefault("clamp_wiring", []).append(
                    {"layer": layer, "tp_rank": tp_rank, "weight_input": weight_input,
                     "stock_used": "FusedMoEKernel.apply via make_fp8_moe_kernel / "
                                   "make_unquantized_moe_kernel with the clamp in the "
                                   "quant config (modular path)"})
                ok &= _arm(f"canonical_L{layer}_tp2_rank{tp_rank}"
                           f"_clamped_weight_input={int(weight_input)}", native, stock, report)
    report["all_arms_active_and_bounded"] = bool(ok)
    return report


def _stock_modular_reference(x, w1, w2, weights, ids, *, family, clamp,
                             experts, apply_router_weight_on_input,
                             w1_scale=None, w2_scale=None, moe_config=None):
    """THE ACTUAL STOCK KERNEL, clamp included.

    Not a composition and not ``fused_experts``: ``FusedMoEKernel.apply``
    (``modular_kernel.py``) is what the serving lane itself calls, and its
    activation path passes ``ApplyMoEActivationConfig``, which is where
    ``gemm1_clamp_limit`` survives.  The non-modular ``fused_experts`` calls
    ``apply_moe_activation`` with NO config and silently drops the clamp --
    measured, not inferred (see the ``fused_experts_respects_swiglu_limit``
    stage in this file, which is retained as the negative evidence).

    FP8 (layer 3): ``make_fp8_moe_kernel`` with the stock per-channel quant
    config carrying ``swiglu_limit``.
    BF16 (layer 4): ``make_unquantized_moe_kernel`` with
    ``gemm1_clamp_limit``.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    # The backend and expert class are selected HERE, independently, with the
    # stock oracle's own factories.  They must NOT be read off the native
    # method: ``moe_route.py`` sets ``self.fp8_backend = self.bf16_backend =
    # self.experts_cls = None`` in native mode by design, so reusing them
    # cannot instantiate anything.
    if family == "TESSERA_FP8":
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            make_fp8_moe_kernel, make_fp8_moe_quant_config,
            select_fp8_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTokenSym, kFp8StaticChannelSym)
        backend, experts_cls = select_fp8_moe_backend(
            config=moe_config, weight_key=kFp8StaticChannelSym,
            activation_key=kFp8DynamicTokenSym, allow_vllm_cutlass=True)
        quant = make_fp8_moe_quant_config(
            fp8_backend=backend, w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=None, a2_scale=None, per_act_token_quant=True,
            per_out_ch_quant=True, block_shape=None,
            gemm1_alpha=None, gemm1_beta=None, swiglu_limit=clamp, layer=None)
        kernel = make_fp8_moe_kernel(
            moe_quant_config=quant, moe_config=moe_config,
            fp8_backend=backend, experts_cls=experts_cls,
            routing_tables=None)
    else:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend, make_unquantized_moe_kernel,
            select_unquantized_moe_backend)
        backend, experts_cls = select_unquantized_moe_backend(moe_config=moe_config)
        quant = FusedMoEQuantConfig.make(gemm1_clamp_limit=clamp)
        kernel = make_unquantized_moe_kernel(
            quant_config=quant, moe_config=moe_config,
            backend=backend, experts_cls=experts_cls,
            routing_tables=None)
    ids_arg = ids.int()
    return kernel.apply(x, w1, w2, weights, ids_arg,
                        activation=MoEActivation.SILU,
                        global_num_experts=experts,
                        expert_map=None,
                        apply_router_weight_on_input=apply_router_weight_on_input)


def stock_factory_construction_check(fixture, layers=(3, 4), experts=(0, 1)):
    """Actually CONSTRUCT the stock reference, on CPU, before any GPU run.

    The geometry self-check proves none of this: it never builds the adapter or
    the stock kernel.  Two real defects lived here -- the previous version read
    ``method.fp8_backend``/``method.experts_cls``, which ``moe_route.py`` sets to
    ``None`` in native mode by design, and it passed the string ``"auto"`` where
    ``make_unquantized_moe_kernel`` wants a selected ``UnquantizedMoeBackend``.
    Both are only caught by instantiating.

    Runs the REAL selection + factory for each family's units.  No kernel is
    launched (that needs CUDA); what is exercised is the wiring that failed.

    NOTE: the stock backend SELECTION needs a visible CUDA device -- with no
    active driver, Triton is disabled and ``select_fp8_moe_backend`` returns
    ``Fp8MoeBackend.NONE``.  So this check is cheap but not CPU-only: run it in
    the image WITH ``--gpus all``.  What it still avoids is any kernel launch.
    """
    import os
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        make_unquantized_moe_kernel, select_unquantized_moe_backend)

    units = canonical_units(fixture, layers=layers, experts=experts)
    out = {}
    for layer in layers:
        u = units[layer]
        declared = u["declared"]
        E = len(u["experts"])
        H = int(declared["groups"]["w13"]["columns"])
        inter = int(declared["groups"]["w13"]["rows"]) // 2
        stub = _native_layer(tp_rank=0, tp_size=2, hidden=H, inter=inter, experts=E)
        facts = {"family": u["family"], "E": E, "H": H, "inter": inter,
                 "stub_num_experts": stub.moe_config.num_experts,
                 "stub_local_intermediate": stub.moe_config.intermediate_size_per_partition}
        if stub.moe_config.num_experts != E:
            raise SystemExit(f"L{layer}: stub config says {stub.moe_config.num_experts} experts, weights carry {E}")
        if stub.moe_config.intermediate_size_per_partition != inter // 2:
            raise SystemExit(f"L{layer}: stub local intermediate "
                             f"{stub.moe_config.intermediate_size_per_partition} != {inter // 2}")
        if u["family"] == "TESSERA_FP8":
            from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                select_fp8_moe_backend)
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8DynamicTokenSym, kFp8StaticChannelSym)
            backend, cls = select_fp8_moe_backend(
                config=stub.moe_config, weight_key=kFp8StaticChannelSym,
                activation_key=kFp8DynamicTokenSym, allow_vllm_cutlass=True)
            facts["selected"] = str(getattr(backend, "value", backend))
            facts["experts_cls"] = getattr(cls, "__name__", None)
            if backend is None or cls is None:
                raise SystemExit(f"L{layer}: fp8 stock selection returned {backend}/{cls}")
        else:
            backend, cls = select_unquantized_moe_backend(moe_config=stub.moe_config)
            facts["selected"] = str(getattr(backend, "value", backend))
            facts["experts_cls"] = getattr(cls, "__name__", None)
            if backend is None or cls is None:
                raise SystemExit(f"L{layer}: unquantized stock selection returned {backend}/{cls}")
        out[layer] = facts
    return {"construction": out, "ok": True}


def canonical_self_check(fixture, layers=(3, 4), experts=(0, 1)):
    """CPU self-check of the canonical wiring facts, without the vLLM runtime.

    Catches, before a GPU is touched, the three wiring bugs that a shape check
    can see: the expert count the builder is given must equal the count
    ``create_weights`` allocates, the fused gate/up width on a rank must be
    twice the rank-local intermediate (splitting at the full intermediate
    leaves the up half empty), and the blob list the loader callbacks take must
    have one entry per shard id.  Nothing here executes a kernel.
    """
    problems = []
    units = canonical_units(fixture, layers=layers, experts=experts)
    checked = {}
    for layer in layers:
        u = units[layer]
        declared = u["declared"]
        E = len(u["experts"])
        inter = int(declared["groups"]["w13"]["rows"]) // 2
        H = int(declared["groups"]["w13"]["columns"])
        bounded = dict(declared)
        bounded["experts"] = E
        facts = {"layer": layer, "family": u["family"], "H": H, "inter": inter,
                 "declared_experts": declared["experts"], "bounded_experts": E}
        if bounded["experts"] != E:
            problems.append(f"L{layer}: bounded declaration does not carry E")
        if declared["experts"] != 288:
            problems.append(f"L{layer}: expected the checkpoint's 288, got {declared['experts']}")
        for tp_size in (1, 2):
            local = inter // tp_size
            fused_width = 2 * local
            if fused_width == 2 * inter and tp_size == 2:
                problems.append(f"L{layer}: rank-local gate/up width is the module's")
            # the split the arms use, at this rank's width
            gate_cols, up_cols = fused_width // 2, fused_width - fused_width // 2
            if gate_cols != local or up_cols != local:
                problems.append(
                    f"L{layer} tp{tp_size}: split at LOCAL gives {gate_cols}/{up_cols}, "
                    f"expected {local}/{local}")
            if up_cols == 0:
                problems.append(f"L{layer} tp{tp_size}: up half would be empty")
            facts[f"tp{tp_size}_local"] = local
            facts[f"tp{tp_size}_fused_gate_up_width"] = fused_width
        # the loader blob contract: w13 is (gate, up), w2 is ONE blob
        per_expert_w13 = [[u["experts"][e]["raw"]["w1"], u["experts"][e]["raw"]["w3"]]
                          for e in sorted(u["experts"])]
        per_expert_w2 = [u["experts"][e]["raw"]["w2"] for e in sorted(u["experts"])]
        if len(per_expert_w2) != E or any(isinstance(b, list) for b in per_expert_w2):
            problems.append(f"L{layer}: w2 blobs are not one blob per expert")
        if len(per_expert_w13) != E or any(len(pair) != 2 for pair in per_expert_w13):
            problems.append(f"L{layer}: w13 blobs are not a gate/up pair per expert")
        facts["w2_blob_bytes"] = [len(b) for b in per_expert_w2]
        checked[layer] = facts
    return {"checked": checked, "problems": problems, "ok": not problems}


def _stock_activation(gate, up, limit):
    """THE STOCK STAGE, the op itself: ``activation.py``'s
    ``silu_and_mul_with_clamp`` (SILU + clamp resolves to
    ``torch.ops._C.silu_and_mul_with_clamp``) on the bf16 gemm1 output, which
    is what the reference path feeds it.  This is an oracle, not a restatement:
    the native lane is compared against the kernel, not the formula."""
    import vllm  # noqa: F401  (registers _C)
    d = gate.shape[-1]
    inp = torch.cat([gate, up], dim=-1).to(torch.bfloat16).contiguous()
    out = torch.empty_like(inp[..., :d])
    torch.ops._C.silu_and_mul_with_clamp(out, inp, float(limit), 1.0, 0.0)
    return out


def _clamp_activity(gate, up, limit):
    """How many accumulator entries the clamp actually saturates, per branch.
    A clamp arm whose inputs never cross the limit proves nothing."""
    flat_g, flat_u = gate.float().reshape(-1), up.float().reshape(-1)
    return {"gate_over_limit": int((flat_g > limit).sum()),
            "gate_under_neg_limit": int((flat_g < -limit).sum()),
            "up_over_limit": int((flat_u > limit).sum()),
            "up_under_neg_limit": int((flat_u < -limit).sum()),
            "entries": int(flat_g.numel()), "limit": float(limit)}


def _activation_arm(name, native_gate_up, gate, up, limit, report):
    """Native clamp+silu vs the stock op at the SAME accumulators.

    The lane's documented single rounding is the only intended difference, so
    the deviation is reported in true bf16 ulps of the reference maximum --
    the same unit the end-to-end arms use."""
    got = native_gate_up(gate, up, limit)
    want = _stock_activation(gate, up, limit)
    diff = (got.float() - want.float()).abs()
    mag = float(want.float().abs().max())
    ulp = _bf16_ulp(mag)
    ulps = None if not ulp else float(diff.max() / ulp)
    # The lane's documented single rounding is the only intended difference
    # from the stock op, so the stage is gated: one ulp of the reference
    # maximum, the same unit the staged comparison uses.
    within = ulps is not None and ulps <= 1.0
    arm = {"arm": name, "max_abs": float(diff.max()), "mag": mag,
           "bf16_ulps": ulps, "within_1_ulp_of_stock_op": within,
           "tolerance": {"screen_bf16_ulps_of_max": 1.0,
                         "derivation": "the lane's documented single rounding"},
           "activity": _clamp_activity(gate, up, limit),
           "shapes": [list(gate.shape), list(up.shape)]}
    report.setdefault("clamp_arms", []).append(arm)
    print("CLAMP", json.dumps(arm), flush=True)
    arm["_ok"] = within
    return arm


def _staged(reference, x, ids, weights, tp_rank=0, tp_size=1, dtype=torch.float64):
    """Per-stage reference (fp64) on the same decoded weights and quantized
    operands, so a stage's deviation can be attributed and budgeted.

    Returns (gate_up, act_bf16, down, final) with the SAME casts the stock
    pipeline performs: fp32 accumulate -> bf16 per-route gemm1 output -> fp32
    activation cast once -> per-route bf16 gemm2 output -> fp32 weighted sum
    -> bf16.
    """
    from vllm import _custom_ops  # noqa: F401

    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    t_tokens, top_k = ids.shape
    s1s, acts, downs, finals = [], [], [], []
    for token in range(t_tokens):
        q = torch.empty((1, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, x[token].reshape(1, -1), s, None)
        xg = (q.reshape(-1).float() * s.reshape(-1)[0]).to(dtype)
        row1, row2, rowf = [], [], None
        for choice in range(top_k):
            e = int(ids[token, choice])
            ref = reference[e]
            gate = ref["gate"]["weight"][lo:hi].to(x.device).to(dtype) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device).to(dtype)
            up = ref["up"]["weight"][lo:hi].to(x.device).to(dtype) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device).to(dtype)
            down = ref["down"]["weight"][:, lo:hi].to(x.device).to(dtype) \
                * ref["down"]["weight_scale"].to(x.device).to(dtype)
            # bf16 boundaries thread through the arithmetic: gemm1 output is
            # cast, the activation reads the CAST values, and its result is
            # cast again BEFORE the stage-2 quantizer sees it.
            g_bf = (gate @ xg).bfloat16().float()
            u_bf = (up @ xg).bfloat16().float()
            act64 = torch.nn.functional.silu(g_bf) * u_bf
            act_bf = act64.bfloat16()
            aq = torch.empty((1, act_bf.numel()), dtype=torch.float8_e4m3fn, device=x.device)
            as_ = torch.empty((1, 1), dtype=torch.float32, device=x.device)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                aq, act_bf.float().reshape(1, -1), as_, None)
            act_q = (aq.reshape(-1).float() * as_.reshape(-1)[0]).to(dtype)
            row1.append(torch.cat([g_bf, u_bf]).to(dtype))
            row2.append(act_bf.to(dtype))
            d = down @ act_q
            downs.append(d.bfloat16().to(dtype))
            rowf = d if rowf is None else rowf + d
        s1s.append(torch.stack(row1).reshape(-1))
        acts.append(torch.stack(row2).reshape(-1))
        finals.append((rowf * weights[token, 0]).bfloat16().to(dtype))
    return (torch.stack(s1s), torch.stack(acts), torch.stack(downs).reshape(t_tokens, top_k, -1),
            torch.stack(finals))


def _loose_reference(reference, x, ids, weights, tp_rank=0, tp_size=1):
    """The superseded loose oracle: no stage-2 quant, a1 applied after silu.

    Kept ONLY as the defect discriminator: if the acceptance metric could not
    separate a real arithmetic error from accumulation order, this arm would
    land inside the same bound.
    """
    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        q = torch.empty((1, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, x[token].reshape(1, -1), s, None)
        for choice in range(ids.shape[1]):
            e = int(ids[token, choice])
            ref = reference[e]
            gate = ref["gate"]["weight"][lo:hi].float().to(x.device) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device)
            up = ref["up"]["weight"][lo:hi].float().to(x.device) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device)
            down = ref["down"]["weight"][:, lo:hi].float().to(x.device) \
                * ref["down"]["weight_scale"].to(x.device)
            g = gate @ q.reshape(-1).float()
            u = up @ q.reshape(-1).float()
            out[token] += weights[token, choice] * s.reshape(-1)[0] * (
                down @ (torch.nn.functional.silu(g) * u))
    return out.bfloat16()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-fixture", default=os.environ.get("TESSERA_CANONICAL_FIXTURE"))
    ap.add_argument("--canonical-layers", default="3,4")
    ap.add_argument("--canonical-experts", default="0,1")
    ap.add_argument("--stock-factory-check", action="store_true",
                    help="construct the stock backend/cfg selection on CPU (no CUDA)")
    ap.add_argument("--canonical-self-check", action="store_true",
                    help="CPU check of the canonical wiring facts (no vLLM needed)")
    ap.add_argument("--canonical-select-only", action="store_true",
                    help="parse+verify the fixture's own containers and stop (CPU)")
    args = ap.parse_args()
    if args.stock_factory_check:
        result = stock_factory_construction_check(
            args.canonical_fixture,
            layers=tuple(int(v) for v in args.canonical_layers.split(",")),
            experts=tuple(int(v) for v in args.canonical_experts.split(",")))
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result["ok"] else 1)

    if args.canonical_self_check:
        result = canonical_self_check(
            args.canonical_fixture,
            layers=tuple(int(v) for v in args.canonical_layers.split(",")),
            experts=tuple(int(v) for v in args.canonical_experts.split(",")))
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result["ok"] else 1)

    if args.canonical_fixture and not args.canonical_select_only:
        # A fixture request that silently fell through to the synthetic arms
        # would run SMALL wires at HIDDEN/INTER and print a pass: a false pass
        # at geometry nobody asked about.  The canonical path either runs or
        # the process stops here.
        try:
            import vllm  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"--canonical-fixture {args.canonical_fixture} needs the canonical "
                f"execution path, which needs the vLLM runtime; refusing rather "
                f"than running the synthetic arms at the wrong geometry ({exc})")
        report = canonical_execution(
            args.canonical_fixture,
            layers=tuple(int(v) for v in args.canonical_layers.split(",")),
            experts=tuple(int(v) for v in args.canonical_experts.split(",")))
        print(json.dumps(report, indent=1))
        raise SystemExit(0 if report["all_arms_active_and_bounded"] else 1)

    if args.canonical_select_only:
        units = canonical_units(args.canonical_fixture,
                                layers=tuple(int(v) for v in args.canonical_layers.split(",")),
                                experts=tuple(int(v) for v in args.canonical_experts.split(",")))
        summary = {}
        for layer, u in units.items():
            w13 = u["declared"]["groups"]["w13"]
            w2 = u["declared"]["groups"]["w2"]
            summary[layer] = {
                "target": u["target"], "family": u["family"],
                "experts": sorted(u["experts"]),
                "w13": {"rows": w13["rows"], "columns": w13["columns"],
                        "roles": [list(r) for r in w13["roles"]],
                        "q256": w13["q256"], "wire_stride": w13["wire_stride"]},
                "w2": {"rows": w2["rows"], "columns": w2["columns"],
                       "roles": [list(r) for r in w2["roles"]],
                       "wire_stride": w2["wire_stride"]},
                "containers": {str(e): {k: len(v["raw"][k])
                                        for k in ("w1", "w3", "w2")}
                               for e, v in u["experts"].items()},
                "parsed_roles": {str(e): {k: [r for r, _w in v["parsed"][k]]
                                          for k in ("w1", "w3", "w2")}
                                 for e, v in u["experts"].items()},
                "reference": canonical_reference_tensors(units, layer)["meta"],
            }
        print(json.dumps({"canonical_selection": summary, "bytes_verified": True}, indent=1))
        raise SystemExit(0)

    torch.manual_seed(11)
    report = {"device": torch.cuda.get_device_name(), "arms": []}
    ok = True

    # --- FP8 ordinary route: TP1 and both TP2 cuts -------------------------
    w13_blobs, w2_blobs, scheme, reference = _stack()
    declared = validate_tessera_moe_scheme(scheme, "m")
    blobs = {"w13": w13_blobs, "w2": [[b] for b in w2_blobs]}
    reference_cpu = moe_route.prepare_tessera_moe_experts(blobs, declared, "m", device="cpu")
    reference = [{"gate": {"weight": reference_cpu.w13_weight[e][:INTER].cpu(),
                           "weight_scale": reference_cpu.w13_weight_scale[e][:INTER].cpu()},
                  "up": {"weight": reference_cpu.w13_weight[e][INTER:].cpu(),
                         "weight_scale": reference_cpu.w13_weight_scale[e][INTER:].cpu()},
                  "down": {"weight": reference_cpu.w2_weight[e].cpu(),
                           "weight_scale": reference_cpu.w2_weight_scale[e].cpu()}}
                 for e in range(EXPERTS)]
    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, EXPERTS, (8, 2), dtype=torch.int32, device="cuda")
    weights = torch.rand(8, 2, device="cuda")
    quant = _fp8_quant_config()
    for tp_rank, tp_size in ((0, 1), (0, 2), (1, 2)):
        layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size)
        method = moe_route.build_tessera_moe_method(scheme, "m", "resident", layer)
        method.create_weights(layer, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
        _load_all(method, layer, w13_blobs, w2_blobs)
        method.process_weights_after_loading(layer)
        native = method.apply(layer, x, weights, ids, None, None)
        stock = _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size)
        ok &= _arm(f"fp8_tp{tp_size}_rank{tp_rank}", native, stock, report)

        if tp_size == 1:
            # --- router-weight-on-input placement: accepted only if actual
            # stock agrees; the same flag reaches stock's own kernel.
            layer.apply_router_weight_on_input = True
            native_in = method.apply(layer, x, weights, ids, None, None)
            stock_in = _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size,
                                  apply_router_weight_on_input=True)
            layer.apply_router_weight_on_input = False
            ok &= _arm("fp8_tp1_rank0_weight_on_input", native_in, stock_in, report)

            # --- stage-by-stage against ACTUAL stock ops on identical
            # quantized operands (no theoretical bound; the numbers are the
            # result, labelled in true bf16 ulps).
            # ONE expert for every staged token: the stock staged arms are
            # single-expert comparisons, so routing must not vary inside them.
            ids1 = torch.zeros(4, 1, dtype=torch.int32, device="cuda")
            x1 = x[:4].contiguous()
            weights1 = torch.ones(4, 1, device="cuda")
            ones = torch.ones_like(weights1)
            xq1 = torch.empty((4, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
            a1s = torch.empty((4, 1), dtype=torch.float32, device="cuda")
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(xq1, x1, a1s, None)
            s1_ref, act_ref, down_ref, _final = _staged(reference, x1, ids1, weights1)

            if method._native.gate_up is not None:
                native_gu = method._native.gate_up(x1, ids1, ones, preserve=True)
            else:
                native_gu = torch.cat([
                    method._native.gate(x1, ids1, ones, preserve=True),
                    method._native.up(x1, ids1, ones, preserve=True)], dim=-1)
            native_gu1 = native_gu[:, 0]                       # [T, 2I], k=1

            e0 = int(ids1[0, 0])
            w13 = torch.cat([reference[e0]["gate"]["weight"],
                             reference[e0]["up"]["weight"]]).cuda()
            s13 = torch.cat([reference[e0]["gate"]["weight_scale"],
                             reference[e0]["up"]["weight_scale"]]).cuda()
            stock_s1 = torch._scaled_mm(xq1, w13.t(), a1s,
                                        s13.reshape(1, -1).contiguous(), out_dtype=torch.bfloat16)
            ref_s1 = (xq1.float() * a1s) @ (w13.float() * s13.reshape(-1, 1)).T
            diag_s1 = {
                "scaled_mm_vs_dequant_matmul": float(
                    (stock_s1.float() - ref_s1).abs().max() / max(float(ref_s1.abs().max()), 1e-6)),
                "native_vs_dequant_matmul": float(
                    (native_gu1.float() - ref_s1).abs().max() / max(float(ref_s1.abs().max()), 1e-6)),
            }
            print("STAGE_S1_DIAG", json.dumps(diag_s1), flush=True)
            s1_diff = (native_gu1.float() - stock_s1.float()).abs()
            s1_mag = float(stock_s1.float().abs().max())
            s1_ulp = _bf16_ulp(s1_mag)

            # stock activation helper on the SAME bf16 gate/up input
            stock_act = torch.empty((4, INTER), dtype=torch.bfloat16, device="cuda")
            torch.ops._C.silu_and_mul(stock_act, native_gu1)
            mine_act = (torch.nn.functional.silu(native_gu1[:, :INTER].float())
                        * native_gu1[:, INTER:].float()).bfloat16()
            act_diff = (mine_act.float() - stock_act.float()).abs()
            act_mag = float(stock_act.float().abs().max())
            act_ulp = _bf16_ulp(act_mag)

            actq = torch.empty((4, INTER), dtype=torch.float8_e4m3fn, device="cuda")
            a2s = torch.empty((4, 1), dtype=torch.float32, device="cuda")
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(actq, stock_act, a2s, None)
            w2 = reference[e0]["down"]["weight"].cuda()
            s2 = reference[e0]["down"]["weight_scale"].cuda()
            stock_dn = torch._scaled_mm(actq, w2.t(), a2s,
                                        s2.reshape(1, -1).contiguous(), out_dtype=torch.bfloat16)
            native_dn = method._native.down(stock_act, ids1, ones, route_input=True,
                                            round_routes=True)
            ref_dn = (actq.float() * a2s) @ (w2.float() * s2.reshape(-1, 1)).T
            diag_dn = {
                "scaled_mm_vs_dequant_matmul": float(
                    (stock_dn.float() - ref_dn).abs().max() / max(float(ref_dn.abs().max()), 1e-6)),
                "native_vs_dequant_matmul": float(
                    (native_dn.float() - ref_dn).abs().max() / max(float(ref_dn.abs().max()), 1e-6)),
            }
            print("STAGE_DN_DIAG", json.dumps(diag_dn), flush=True)
            dn_diff = (native_dn.float() - stock_dn.float()).abs()
            dn_mag = float(stock_dn.float().abs().max())
            dn_ulp = _bf16_ulp(dn_mag)

            def _ulps(d, ulp):
                return None if ulp in (None, 0.0) else float(d.max() / ulp)

            stage = {"arm": "fp8_tp1_stages_vs_stock_ops",
                     "gate_up": {"max_abs": float(s1_diff.max()), "mag": s1_mag,
                                 "bf16_ulps": _ulps(s1_diff, s1_ulp)},
                     "silu_and_mul": {"max_abs": float(act_diff.max()), "mag": act_mag,
                                      "bf16_ulps": _ulps(act_diff, act_ulp)},
                     "down": {"max_abs": float(dn_diff.max()), "mag": dn_mag,
                              "bf16_ulps": _ulps(dn_diff, dn_ulp)}}
            report["stages"] = stage
            print("STAGE", json.dumps(stage), flush=True)
            # one bf16 cast boundary per stage: <= 1 ulp; the end-to-end arms carry
            # four boundaries and are bounded at 4.
            ok &= all(v is not None and v <= 1.0 for v in
                      (_ulps(s1_diff, s1_ulp), _ulps(act_diff, act_ulp), _ulps(dn_diff, dn_ulp)))

            # --- defect discriminator: the loose oracle must exceed the bound --
            loose = _loose_reference(reference, x, ids, weights)
            loose_err = float((loose.float() - stock.float()).abs().max())
            native_err = float((native.float() - stock.float()).abs().max())
            discriminator = {"arm": "loose_vs_native_vs_stock",
                             "native_vs_stock": native_err, "loose_vs_stock": loose_err,
                             "ratio": loose_err / max(native_err, 1e-12)}
            report["discriminator"] = discriminator
            print("DISCRIMINATOR", json.dumps(discriminator), flush=True)
            # A defect-sized disagreement must dwarf the accumulation-order one;
            # the loose oracle is the negative control.
            ok &= loose_err > 100 * native_err

    # --- folded BF16 research route, TP2 rank 0 ----------------------------
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    b13, b2, bscheme, expected = bf16_wires_native_data()
    layer = _native_layer(tp_rank=0, tp_size=2)
    import types
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(
            types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True))):
        method = moe_route.build_tessera_moe_method(
            bscheme, "m", "resident", layer,
            research_selected=moe_route.ResearchSelectedMoeConfig(
                max_experts_per_chunk=2, expected_tensor_parallel_size=2))
    method.create_weights(layer, 2, HIDDEN, INTER // 2, torch.bfloat16)
    _load_all(method, layer, b13, [pair[0] for pair in b2])
    method.process_weights_after_loading(layer)
    lo, hi = 0, INTER // 2
    w1 = torch.stack([
        torch.cat([expected[e][0][lo:hi], expected[e][0][INTER + lo:INTER + hi]])
        for e in range(2)]).cuda().contiguous()
    w2 = torch.stack([expected[e][1][:, lo:hi] for e in range(2)]).cuda().contiguous()
    # routing for 2 experts on the BF16 stacks
    ids2 = torch.randint(0, 2, (8, 2), dtype=torch.int32, device="cuda")
    native = method.apply(layer, x, weights, ids2, None, None)
    stock = fused_experts(x, w1.cuda(), w2.cuda(), weights, ids2,
                          activation=MoEActivation.SILU, global_num_experts=2)
    ok &= _arm("bf16_folded_tp2_rank0", native, stock, report)

    # --- the model's SwiGLU clamp, ACTIVE, on both families ----------------
    # Every arm above ran with no clamp, so none of them exercised the
    # arithmetic the derived fixture needs (GLM's ``swiglu_limit: 10.0``).
    # These arms drive inputs whose gemm1 accumulators cross the limit in BOTH
    # directions, on both TP2 rank cuts, under both weight-on-input semantics,
    # and grade the lane against the ACTUAL stock stage -- the op for the
    # activation, ``fused_experts`` for the routed result.
    CLAMP = 10.0
    x_clamped = (torch.randn(8, HIDDEN) * 8.0).bfloat16().cuda()
    quant_clamped = _fp8_quant_config()
    for tp_rank, tp_size in ((0, 2), (1, 2)):
        layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size)
        layer.swiglu_limit = CLAMP
        method = moe_route.build_tessera_moe_method(scheme, "m", "resident", layer)
        method.create_weights(layer, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
        _load_all(method, layer, w13_blobs, w2_blobs)
        method.process_weights_after_loading(layer)
        ones_ids = torch.zeros(4, 1, dtype=torch.int32, device="cuda")
        ones_w = torch.ones(4, 1, device="cuda")
        if method._native.gate_up is not None:
            gu = method._native.gate_up(x_clamped[:4], ones_ids, ones_w, preserve=True)
        else:
            gu = torch.cat([method._native.gate(x_clamped[:4], ones_ids, ones_w, preserve=True),
                            method._native.up(x_clamped[:4], ones_ids, ones_w, preserve=True)],
                           dim=-1)
        act = _activation_arm(
            f"fp8_tp2_rank{tp_rank}_activation_vs_stock_op",
            lambda g, u, lim: nwm._silu_and_mul(g, u, clamp_limit=lim),
            gu[:, 0, :INTER], gu[:, 0, INTER:], CLAMP, report)
        ok &= (act["activity"]["gate_over_limit"] > 0
               and act["activity"]["up_under_neg_limit"] > 0
               and act["activity"]["up_over_limit"] > 0), "clamp not active in both branches"
        # WITHDRAWN as a gate: ``_fp8_stock``/``fused_experts`` pass the clamp
        # only as far as a quant config that the non-modular activation path
        # never reads, so an arm built on it compares a CLAMPED native lane
        # against an UNCLAMPED stock stand-in.  This block is retained as
        # negative evidence -- it is what produced the original 115-134 ulp
        # residual -- and deliberately does NOT contribute to ``ok``.
        for weight_input in (False, True):
            layer.apply_router_weight_on_input = weight_input
            native_c = method.apply(layer, x_clamped, weights, ids, None, None)
            stock_c = _fp8_stock(reference, x_clamped, ids, weights, quant_clamped,
                                 tp_rank, tp_size,
                                 apply_router_weight_on_input=weight_input,
                                 swiglu_limit=CLAMP)
            withdrawn = _arm(f"WITHDRAWN_clamp_ignoring_fp8_tp2_rank{tp_rank}"
                             f"_weight_input={int(weight_input)}", native_c, stock_c, report)
            report.setdefault("withdrawn_gates", []).append(
                {"arm": f"fp8_tp2_rank{tp_rank}_weight_input={int(weight_input)}",
                 "reason": "stock side is fused_experts, which drops gemm1_clamp_limit",
                 "ulps_observed": report["arms"][-1].get("bf16_ulps_of_max"),
                 "counts_toward_acceptance": False})
            del withdrawn

    # folded BF16 research route (layer 4's family), both rank cuts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    for tp_rank, tp_size in ((0, 2), (1, 2)):
        layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size)
        layer.swiglu_limit = CLAMP
        with set_current_vllm_config(
                types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True))):
            method = moe_route.build_tessera_moe_method(
                bscheme, "m", "resident", layer,
                research_selected=moe_route.ResearchSelectedMoeConfig(
                    max_experts_per_chunk=2, expected_tensor_parallel_size=2))
        method.create_weights(layer, 2, HIDDEN, INTER // tp_size, torch.bfloat16)
        _load_all(method, layer, b13, [pair[tp_rank] if len(pair) > tp_rank else pair[0]
                                       for pair in b2])
        method.process_weights_after_loading(layer)
        local = INTER // tp_size
        lo, hi = tp_rank * local, (tp_rank + 1) * local
        w1c = torch.stack([
            torch.cat([expected[e][0][lo:hi], expected[e][0][INTER + lo:INTER + hi]])
            for e in range(2)]).cuda().contiguous()
        w2c = torch.stack([expected[e][1][:, lo:hi] for e in range(2)]).cuda().contiguous()
        # ALSO WITHDRAWN (same defect, BF16 family): ``fused_experts`` ignores
        # the clamp here too.  Kept as negative evidence, excluded from ``ok``.
        qc = FusedMoEQuantConfig.make(gemm1_clamp_limit=CLAMP)
        for weight_input in (False, True):
            layer.apply_router_weight_on_input = weight_input
            native_c = method.apply(layer, x_clamped, weights, ids2, None, None)
            stock_c = fused_experts(x_clamped, w1c, w2c, weights, ids2,
                                    activation=MoEActivation.SILU, global_num_experts=2,
                                    apply_router_weight_on_input=weight_input,
                                    quant_config=qc)
            _arm(f"WITHDRAWN_clamp_ignoring_bf16_tp2_rank{tp_rank}"
                 f"_weight_input={int(weight_input)}", native_c, stock_c, report)
            report.setdefault("withdrawn_gates", []).append(
                {"arm": f"bf16_folded_tp2_rank{tp_rank}_weight_input={int(weight_input)}",
                 "reason": "stock side is fused_experts, which drops gemm1_clamp_limit",
                 "ulps_observed": report["arms"][-1].get("bf16_ulps_of_max"),
                 "counts_toward_acceptance": False})

    report["passed"] = bool(ok)
    print(json.dumps(report))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
