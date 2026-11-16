"""Served routed loading and explicit mode-0, mode-2, and token-sum replay."""
from __future__ import annotations

from pathlib import Path

from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.row import dot_reference


class Routed:
    def __init__(self, entry, cut, manifest, table):
        import torch
        from experiments import dense_fused_oracle as dense, routed_pair_oracle as routed
        from experiments.numerics_rows.cases import checkpoint_routed

        self.entry, self.cut, self.manifest, self.table = entry, cut, manifest, table
        store = dense.Store(entry["root"])
        scheme, blobs, scales = checkpoint_routed(store, entry)
        self.family = {"TESSERA_FP8": "e4m3", "TESSERA_BF16": "value", "TESSERA_NVFP4": "e2m1"}[scheme["family"]]
        routed.HIDDEN = scheme["groups"]["w13"]["columns"]
        routed.INTER = scheme["groups"]["w13"]["rows"] // 2
        routed.TOP_K = 8
        config = routed.moe_config(scheme["experts"], 10.0)
        config.moe_parallel_config.tp_rank, config.moe_parallel_config.tp_size = cut["rank"], cut["size"]
        self.layer, self.method, self.info = routed.build_after(
            {"family": scheme["family"]}, scheme, blobs, scales, scheme["experts"], config, 10.0, entry["name"])
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
        return q, sf.view(torch.uint8), a, None

    def route(self, ids, weights):
        return self.native._routing(ids, weights)

    def launch(self, mode, q, sf, routes, out, *, row_mode=0, weighted=False):
        kwargs = {"a_row_mode": row_mode, "mul_weight": weighted, "limit": 10.0 if mode == 0 else float("inf"), "out": out}
        if self.family == "e2m1":
            kwargs["counter"] = 1 if mode == 2 else 0
        self.native._launch(mode, q, sf, routes, **kwargs)

    def project(self, projection, a, a_scale, experts, source_rows, *, exact=True):
        import torch

        outputs = torch.empty(experts.numel(), self.rows if projection == "down_proj" else self.inter,
                              device=a.device, dtype=torch.bfloat16)
        masks = torch.zeros_like(outputs, dtype=torch.bool)
        for expert in experts.unique().tolist():
            select = (experts == expert).nonzero().flatten()
            weight, scale = self.weight(expert, projection)
            if self.family == "e2m1":
                gs = self.gs2 if projection == "down_proj" else self.gs13
                scale = scale.float() / torch.tensor(gs, dtype=torch.float32)
            rows = source_rows[select]
            out, mask = dot_reference(a[rows], weight, None if a_scale is None else a_scale[rows], scale.to(a.device), exact=exact)
            outputs[select], masks[select] = out, mask
        return outputs, masks

    def gate_reference(self, a, a_scale, ids, routes):
        import torch
        from experiments.numerics_rows.cases import DTable

        flat = routes.flat_sorted.long()
        experts = ids.flatten()[flat]
        tokens = flat // ids.shape[1]
        g, mg = self.project("gate_proj", a, a_scale, experts, tokens)
        u, mu = self.project("up_proj", a, a_scale, experts, tokens)
        gf, uf = g.float().clamp(max=10.0), u.float().clamp(-10.0, 10.0)
        d = torch.tensor([self.table(float(value)) for value in gf.flatten().cpu()], device=a.device, dtype=torch.float32).reshape(g.shape)
        return (d * uf).to(torch.bfloat16), mg & mu

    def down_reference(self, a, a_scale, ids, weights, routes):
        import torch

        flat = routes.flat_sorted.long()
        experts = ids.flatten()[flat]
        out, mask = self.project("down_proj", a, a_scale, experts, torch.arange(flat.numel(), device=a.device))
        # Recompute the epilogue before BF16 conversion, with router weight in its specified position.
        for expert in experts.unique().tolist():
            select = (experts == expert).nonzero().flatten()
            weight, scale = self.weight(expert, "down_proj")
            if self.family == "e2m1":
                scale = scale.float() / torch.tensor(self.gs2, dtype=torch.float32)
            for row in select.tolist():
                cols = (a[row] != 0).nonzero().flatten()
                raw = (weight[:, cols.cpu()].to(a.device).double() * a[row, cols][None, :]).sum(1).float()
                if a_scale is not None:
                    raw = raw * a_scale[row]
                raw = raw * scale.to(a.device).float()
                raw = raw * routes.rw_sorted[row]
                out[row] = raw.to(torch.bfloat16)
        scattered, scattered_mask = torch.empty_like(out), torch.empty_like(mask)
        scattered[flat], scattered_mask[flat] = out, mask
        return scattered, scattered_mask

    def staged(self, x, ids, weights, recorder):
        import torch
        from tessera import routed_fused as rf, routed_fused_e2m1 as fe

        routes = self.route(ids, weights)
        q, sf, _, _ = self.quantized(x, self.gs13, recorder)
        act = torch.empty(routes.routes, self.inter, device=x.device, dtype=torch.bfloat16)
        self.launch(0, q, sf, routes, act)
        q, sf, _, _ = self.quantized(act, self.gs2, recorder)
        routed = torch.empty(routes.routes, self.rows, device=x.device, dtype=torch.bfloat16)
        self.launch(2, q, sf, routes, routed, row_mode=1, weighted=True)
        output = torch.empty(x.shape[0], self.rows, device=x.device, dtype=torch.bfloat16)
        library = fe._ext() if self.family == "e2m1" else rf._ext(self.native.library)
        library.token_sum(routed, output, ids.shape[1])
        return output
