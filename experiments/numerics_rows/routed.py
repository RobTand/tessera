"""Served routed loading and explicit mode-0, mode-2, and token-sum replay."""
from __future__ import annotations

import hashlib

from experiments.numerics_rows.common import Invalid, artifact_limit
from experiments.numerics_rows.row import dot_reference, finite_reference
from experiments.numerics_rows.reference import FiniteBound


def build_served(family, scheme, blobs, scales, experts, config, limit, prefix):
    """Give the served loader this rank's width and keep the full configuration."""
    from experiments import routed_pair_oracle as routed

    intermediate = routed.INTER
    try:
        routed.INTER = config.intermediate_size // config.moe_parallel_config.tp_size
        return routed.build_after(family, scheme, blobs, scales, experts, config, limit, prefix)
    finally:
        routed.INTER = intermediate


class Routed:
    def __init__(self, entry, cut, manifest, table):
        import torch
        from experiments import dense_fused_oracle as dense, routed_pair_oracle as routed
        from experiments.numerics_rows.cases import checkpoint_routed

        self.entry, self.cut, self.manifest, self.table = entry, cut, manifest, table
        self.top_k = 8
        self.device = "cuda"
        self.limit = artifact_limit(entry["root"])
        if self.limit != table.limit:
            raise Invalid("artifact swiglu_limit differs from the Q1 D table")
        store = dense.Store(entry["root"])
        scheme, blobs, scales = checkpoint_routed(store, entry)
        if {p: [hashlib.sha256(w).hexdigest() for w in wires] for p, wires in blobs.items()} != manifest["wire_sha256"]:
            raise Invalid("loaded wires differ from the C1 reader artifacts")
        self.family = {"TESSERA_FP8": "e4m3", "TESSERA_BF16": "value", "TESSERA_NVFP4": "e2m1"}[scheme["family"]]
        routed.HIDDEN = scheme["groups"]["w13"]["columns"]
        routed.INTER = scheme["groups"]["w13"]["rows"] // 2
        routed.TOP_K = 8
        config = routed.moe_config(scheme["experts"], self.limit)
        config.moe_parallel_config.tp_rank, config.moe_parallel_config.tp_size = cut["rank"], cut["size"]
        self.layer, self.method, self.info = build_served(
            {"family": scheme["family"]}, scheme, blobs, scales, scheme["experts"], config, self.limit, entry["name"])
        self.native = self.layer.tessera_routed_fused if self.family == "e2m1" else self.method._native
        if self.family != "e2m1" and self.native.uniform is None:
            raise Invalid("the routed row requires one uniform expert class")
        self.cols, self.inter, self.rows = self.native.gate.cols, self.native.down.cols, self.native.down.rows
        self.experts = scheme["experts"]
        self.gs13 = float(self.native.gs13) if self.family == "e2m1" else 1.0
        self.gs2 = float(self.native.gs2) if self.family == "e2m1" else 1.0
        self.reader = {(role["expert"], role["name"]): role for role in manifest["roles"]}

    def apply(self, x, ids, weights):
        return self.method.apply(self.layer, x, weights, ids, None, None)

    def pairs(self):
        from experiments.dense_fused_oracle import route_record

        record = route_record(self.layer)
        return [] if record is None else [(record["symbol"], record["decoder"])]

    def p4(self):
        import torch

        equal = True
        for projection, field in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            for expert in range(self.experts):
                _, scale = self.weight(expert, projection)
                if self.family == "e2m1":
                    gs = self.gs2 if projection == "down_proj" else self.gs13
                    expected = scale.float() / torch.tensor(gs, dtype=torch.float32)
                    loaded = getattr(self.native, "ratio_" + field)[expert:expert + 1]
                else:
                    expected = scale.float()
                    loaded = getattr(self.native, field).scale_all[expert]
                equal &= torch.equal(loaded.reshape(-1).view(torch.int32).cpu(), expected.reshape(-1).view(torch.int32).cpu())
        return equal

    def weight(self, expert, projection):
        import torch
        from experiments.numerics_rows.common import sha256

        role = self.reader[expert, projection]
        if not role["matches_attested_wire"]:
            raise Invalid("wire stamp differs from attested_wire")
        if sha256(role["reader_path"]) != role["reader_sha256"]:
            raise Invalid("C1 reader artifact changed")
        decoded = torch.load(role["reader_path"], map_location="cpu", mmap=True, weights_only=True)
        weight, scale = decoded["weight"], decoded["scale"]
        return weight, scale

    def quantized(self, x, gs, recorder):
        import torch
        from experiments.numerics_rows.quantizers import quantize
        from experiments.routed_pair_oracle import e2m1_table

        first = recorder.call("routed:quantizer", "quantizer", lambda: quantize(self.family, x, gs))
        again = recorder.call("routed:quantizer-repeat", "quantizer", lambda: quantize(self.family, x, gs))
        if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(first, again)):
            raise Invalid("two quantizer calls gave different bytes")
        q, sf = first
        if self.family != "e2m1":
            return q, sf if self.family == "e4m3" else None, q.double(), sf.reshape(-1) if self.family == "e4m3" else None
        cols = x.shape[1]
        packed = q.view(torch.uint8).reshape(x.shape[0], -1)[:, :cols // 2]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(x.shape[0], cols)
        blocks = sf.view(torch.uint8).reshape(x.shape[0], -1)[:, :cols // 16].contiguous().view(torch.float8_e4m3fn)
        a = e2m1_table(x.device)[codes.long()] * blocks.double().repeat_interleave(16, dim=1)
        return q.contiguous(), sf.view(torch.uint8).contiguous(), a, None

    def route(self, ids, weights):
        if ids.shape[1] != self.top_k:
            raise Invalid("routing top_k differs from the case manifest's eight-route plan")
        return self.native._routing(ids, weights)

    def launch(self, mode, q, sf, routes, out, *, row_mode=0, weighted=False):
        kwargs = {"a_row_mode": row_mode, "mul_weight": weighted, "limit": self.limit if mode == 0 else float("inf"), "out": out}
        if self.family == "e2m1":
            kwargs["counter"] = 1 if mode == 2 else 0
        self.native._launch(mode, q, sf, routes, **kwargs)

    def project(self, projection, a, a_scale, experts, source_rows, *, exact=True, router_weight=None):
        import torch

        outputs = torch.empty(experts.numel(), self.rows if projection == "down_proj" else self.inter,
                              device=a.device, dtype=torch.bfloat16)
        masks = torch.zeros_like(outputs, dtype=torch.bool)
        bounds = FiniteBound(torch.zeros_like(outputs, dtype=torch.float64))
        for expert in experts.unique().tolist():
            select = (experts == expert).nonzero().flatten()
            weight, scale = self.weight(expert, projection)
            if self.family == "e2m1":
                gs = self.gs2 if projection == "down_proj" else self.gs13
                scale = scale.float() / torch.tensor(gs, dtype=torch.float32)
            rows = source_rows[select]
            out, mask, bound = dot_reference(a[rows], weight, None if a_scale is None else a_scale[rows], scale.to(a.device), exact=exact, with_bounds=True, splits=self.splits, top_k=self.top_k,
                                             router_weight=None if router_weight is None else router_weight[select])
            outputs[select], masks[select] = out, mask
            bounds[select] = bound
        return outputs, masks, bounds

    def gate_reference(self, a, a_scale, ids, routes):
        import torch

        flat = routes.flat_sorted.long()
        experts = ids.flatten()[flat]
        tokens = flat // ids.shape[1]
        g, mg, gb = self.project("gate_proj", a, a_scale, experts, tokens)
        u, mu, ub = self.project("up_proj", a, a_scale, experts, tokens)
        finite_reference(g)
        finite_reference(u)
        gf, uf = g.float().clamp(max=self.limit), u.float().clamp(-self.limit, self.limit)
        d = torch.tensor([self.table(float(value)) for value in gf.flatten().cpu()], device=a.device, dtype=torch.float32).reshape(g.shape)
        bound = gb.activation(ub, self.limit, self.table.finite_certificate["held"])
        mode1 = FiniteBound(torch.zeros(flat.numel(), 2 * self.inter, dtype=torch.float64, device=a.device))
        mode1[flat] = FiniteBound.join((gb, ub), 1)
        return (d * uf).to(torch.bfloat16), mg & mu, bound, mode1.reshape(ids.shape[0], ids.shape[1], -1)

    def down_reference(self, a, a_scale, ids, weights, routes):
        import torch

        flat = routes.flat_sorted.long()
        experts = ids.flatten()[flat]
        out, mask, bound = self.project("down_proj", a, a_scale, experts, torch.arange(flat.numel(), device=a.device), router_weight=routes.rw_sorted)
        scattered, scattered_mask = torch.empty_like(out), torch.empty_like(mask)
        scattered_bound = FiniteBound(torch.zeros_like(out, dtype=torch.float64))
        scattered[flat], scattered_mask[flat], scattered_bound[flat] = out, mask, bound
        return scattered, scattered_mask, scattered_bound

    def preflight(self, x, ids, weights, recorder):
        from experiments.numerics_rows.run_routed import token_reference
        routes = self.route(ids, weights)
        _, _, a, a_scale = self.quantized(x, self.gs13, recorder)
        act, _, gate, _ = self.gate_reference(a, a_scale, ids, routes)
        finite_reference(act)
        _, _, a, a_scale = self.quantized(act, self.gs2, recorder)
        down, _, bound = self.down_reference(a, a_scale, ids, weights, routes)
        finite_reference(down)
        finite_reference(token_reference(down, ids.shape[1]))
        bound.held[routes.flat_sorted.long()] &= gate.held.all(1, keepdim=True)
        return bound.token_sum(ids.shape[1])

    def staged(self, x, ids, weights, recorder, *, unit=None):
        import torch
        from tessera import routed_fused as rf, routed_fused_e2m1 as fe
        from experiments.numerics_rows.run_routed import token_reference

        routes = self.route(ids, weights)
        from experiments.numerics_rows.row import finite_outputs

        q, sf, a, a_scale = self.quantized(x, self.gs13, recorder)
        act = torch.empty(routes.routes, self.inter, device=x.device, dtype=torch.bfloat16)
        reference, _, gate, _ = self.gate_reference(a, a_scale, ids, routes)
        finite_reference(reference)
        self.launch(0, q, sf, routes, act)
        if unit is not None:
            finite_outputs(unit, "composition:mode-0", act, gate)
        q, sf, a, a_scale = self.quantized(act, self.gs2, recorder)
        routed = torch.empty(routes.routes, self.rows, device=x.device, dtype=torch.bfloat16)
        reference, _, bound = self.down_reference(a, a_scale, ids, weights, routes)
        finite_reference(reference)
        self.launch(2, q, sf, routes, routed, row_mode=1, weighted=True)
        if unit is not None:
            finite_outputs(unit, "composition:mode-2", routed, bound)
        output = torch.empty(x.shape[0], self.rows, device=x.device, dtype=torch.bfloat16)
        library = fe._ext() if self.family == "e2m1" else rf._ext(self.native.library)
        finite_reference(token_reference(reference, ids.shape[1]))
        library.token_sum(routed, output, ids.shape[1])
        if unit is not None:
            bound.held[routes.flat_sorted.long()] &= gate.held.all(1, keepdim=True)
            bound = bound.token_sum(ids.shape[1])
            finite_outputs(unit, "composition:token_sum", output, bound)
            return output, bound
        return output
