"""CPU design controls for unchanged GLM wires and the existing TP slicer.

This does not enable a MoE TP route. Invoke explicitly through PrismaBuild;
GLM_TP2_REQUEST binds the wire/oracle sources and a unique evidence directory.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tessera.fused import parse_fused
from tessera.layout import can_shard
from tessera.serving.fp8_route import PreparedTesseraFp8Module, prepare_tessera_fp8_module
from tessera.serving.sharding import plan_shard, shard_parsed_roles
from tessera.serving.window import prepare_window
from tessera.unit_artifact import parse_unit_artifact

ROLES = ('gate_proj', 'up_proj', 'down_proj')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(row):
    path = Path(row['path'])
    assert sha(path) == row['sha256'], path
    return path


@pytest.fixture(scope='module')
def source():
    request_path = Path(os.environ['GLM_TP2_REQUEST'])
    request = json.loads(request_path.read_text())
    assert request['schema'] == 'tessera.glm_tp2_slice_design.v1'
    producer = json.loads(checked(request['producer_manifest']).read_text())
    result = {}
    for role in ROLES:
        row = producer['projections'][role]
        directory = Path(row['directory'])
        assert sha(directory/'receipt.json') == row['receipt_sha256']
        receipt = json.loads((directory/'receipt.json').read_text())
        assert receipt['q256'] == 512 and receipt['status'] == 'encoded_source_projection'
        assert sha(directory/'projection.wire') == receipt['wire_sha256']
        assert sha(directory/'independent-stock.safetensors') == receipt['stock_sha256']
        member, = parse_fused((directory/'projection.wire').read_bytes())
        assert member.name == role
        parsed = parse_unit_artifact(member.blob, device='cpu')
        stock = load_file(directory/'independent-stock.safetensors', device='cpu')
        result[role] = (parsed, stock, receipt)
    return request_path, request, result


def shard_plan(role, rank):
    # The copied stock GLM factory has E=288, H=4096, N=2048; TP2 gives N/2.
    # w13 roles are independently column-parallel; w2 is row-parallel.
    if role == 'down_proj':
        return plan_shard('glm.experts.w2', roles=[('down_proj',4096)],
            columns=2048, out_partitions=[4096], in_size=1024,
            tp_rank=rank, tp_size=2, input_size=2048, output_size=4096)
    return plan_shard('glm.experts.w13', roles=[('gate_proj',2048),('up_proj',2048)],
        columns=4096, out_partitions=[1024,1024], in_size=4096,
        tp_rank=rank, tp_size=2, input_size=4096, output_size=4096)


@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('rank', [0,1])
def test_original_wire_rank_slice_matches_independent_full_stock(source, role, rank):
    request_path, request, sources = source
    parsed, stock, receipt = sources[role]
    plan = shard_plan(role,rank)
    assert can_shard(parsed,2,plan.axis)
    name, sharded = shard_parsed_roles([(role,parsed)],plan)[0]
    assert name == role and sharded.manifest.encoder_fixture_id == parsed.manifest.encoder_fixture_id
    assert sharded.manifest.shard is not None
    lo,hi = rank*1024,(rank+1)*1024
    expected = stock['weight'][:,lo:hi] if role == 'down_proj' else stock['weight'][lo:hi]
    scales = stock['weight_scale'].reshape(-1)
    expected_scales = scales if role == 'down_proj' else scales[lo:hi]
    module = prepare_tessera_fp8_module([(role,sharded)],device='cpu')
    decoded = module.decode()
    assert torch.equal(decoded,expected.contiguous().view(torch.uint8))
    assert torch.equal(module.row_scale(),expected_scales)

    # The prepared batch's storage is affine in E: body/table/scales are per
    # expert; gather/shift/which metadata is shared. Check E=1,2,3 then derive
    # E=288 without allocating 288 experts on a CPU design job.
    counts = []
    for experts in (1,2,3):
        owner = PreparedTesseraFp8Module.stack([module]*experts)
        counts.append(owner.resident_bytes())
        if experts == 2:
            ids = torch.tensor([1,0,1],dtype=torch.int32)
            selected = owner.decode(ids,max_experts_per_chunk=1)
            assert all(torch.equal(x,decoded) for x in selected)
            assert owner.decode(torch.empty(0,dtype=torch.int32),max_experts_per_chunk=1).shape == (0,*decoded.shape)
            del selected
        del owner
    per_expert = counts[1]-counts[0]
    shared = counts[0]-per_expert
    assert counts[2] == 3*per_expert+shared and min(per_expert,shared)>0

    zero_pad_mismatches = None
    if rank == 1 and role != 'down_proj':
        assert sharded.unit.initial_state is not None
        wrong = prepare_window(sharded.unit.body_bits,sharded.unit.rates,
            sharded.unit.window_bits,sharded.unit.window_codes,'cpu',
            code_map=torch.tensor(sharded.grid.native,dtype=torch.uint8),initial_state=None).decode()
        zero_pad_mismatches = int(torch.count_nonzero(wrong != decoded))
        assert zero_pad_mismatches>0, 'rank1 zero-state negative control must fail'
    output = {'role':role,'rank':rank,'plan':dataclasses.asdict(plan),
        'source_wire_sha256':receipt['wire_sha256'],
        'shard_origin':str(sharded.manifest.shard),'window_bits':sharded.unit.window_bits,
        'rows':module.rows,'columns':module.columns,
        'exact_weight_bytes':True,'exact_scales':True,'selection_repeat_and_empty':True,
        'zero_pad_mismatches':zero_pad_mismatches,'owner_bytes_at_E123':counts,
        'owner_per_expert_bytes':per_expert,'owner_shared_metadata_bytes':shared,
        'derived_owner_bytes_at_E288':288*per_expert+shared,
        'request_sha256':sha(request_path),'GPU_execution':False,'MoE_route_enabled':False}
    Path(request['out'],f'{role}-rank{rank}.json').write_text(json.dumps(output,indent=2)+'\n')


def test_rolewise_w13_partition_and_partial_sum_oracle(source):
    request_path,request,sources = source
    weights = {}
    for role,(_,stock,_) in sources.items():
        weights[role] = stock['weight'].double()*stock['weight_scale'].reshape(-1,1).double()
    generator = torch.Generator().manual_seed(20260908)
    x = torch.randn(2,4096,generator=generator,dtype=torch.float64)
    g,u,d = (weights[role] for role in ROLES)
    def activation(gate,up):
        # GLM's configured gated SiLU clamp is elementwise, so it commutes
        # with slicing the intermediate axis. This is an FP64 math oracle,
        # not a native dynamic-activation-FP8 or collective-rounding oracle.
        return torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(min=-10,max=10)
    full = activation(x@g.T,x@u.T)@d.T
    partials,wrong = [],[]
    combined = torch.cat([g,u],0)
    scale_spans = []
    for rank in (0,1):
        lo,hi=rank*1024,(rank+1)*1024
        act = activation(x@g[lo:hi].T,x@u[lo:hi].T)
        partials.append(act@d[:,lo:hi].T)
        # This erroneous cut splits the fused aggregate rather than each role.
        wrongg,wrongu = combined[rank*2048:(rank+1)*2048].chunk(2,0)
        wrong.append(activation(x@wrongg.T,x@wrongu.T)@d[:,lo:hi].T)
        scale_spans.append(act.abs().amax(-1))
    got = sum(partials)
    torch.testing.assert_close(got,full,rtol=1e-12,atol=1e-12)
    wrong_max = float((sum(wrong)-full).abs().max())
    assert wrong_max>1e-6
    # Stock per-token A2 quantization observes each rank's LOCAL activation
    # width. Its scale can differ across TP ranks, so native TP2 must be
    # compared with stock TP2, not required to equal a TP1 quantized output.
    assert not torch.equal(scale_spans[0],scale_spans[1])
    Path(request['out'],'partial-sum-oracle.json').write_text(json.dumps({
        'FP64_math_oracle':True,'native_FP8_oracle':False,
        'max_abs':float((got-full).abs().max()),'wrong_fused_aggregate_slice_max_abs':wrong_max,
        'rank_local_activation_amax':[v.tolist() for v in scale_spans],
        'request_sha256':sha(request_path)},indent=2)+'\n')
