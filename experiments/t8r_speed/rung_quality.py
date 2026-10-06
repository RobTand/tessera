"""CPU weight-space screen of actual sampled GLM experts; not served KL."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from tessera.control import grid_for_name, unit_wire_bits
from tessera.export import encode_linear
from tessera.unit_artifact import read_unit_artifact
from tessera.dev_mode import seal_check


def quality_scope(grid,q,recipe):
    from tessera.export import served_recipe
    from tessera.structure import STRUCTURE_DENSE,STRUCTURE_ROUTED_MOE
    config=recipe.to_config()
    kinds=[kind for kind,structure in (('dense',STRUCTURE_DENSE),('routed',STRUCTURE_ROUTED_MOE))
           if served_recipe(grid,q,structure).to_config()==config]
    base=grid.name.split('x',1)[0]
    return {'format':f'TESSERA_{base}_K{grid.arity}','grid':grid.name,'arity':grid.arity,
            'rung':q,'recipe':config,'kernel_kinds':kinds,'owner':'tessera.export.encode_linear'}


def bind_existing_quality(path,action_key,out):
    """Recover scope from executed arguments and stored scores; paths stamp in dev."""
    from tessera.export import wire_recipe
    request=Path('/mnt/shared/prismabuild-fleet/cas/requests')/action_key[:2]/(action_key+'.json')
    sealed=json.loads(request.read_text())
    if sealed['action_key']!=action_key:raise ValueError('quality action key differs')
    command=sealed['params']['command']
    is_t4='experiments/t4_code/bench_geometry_e2m1.py' in command and '--quality' in command
    if not ('experiments/t8r_speed/rung_quality.py' in command or is_t4) or '--grid' not in command:
        raise ValueError('no explicit original quality producer/grid invocation')
    grid=grid_for_name(command[command.index('--grid')+1])
    original=Path(command[command.index('--out')+1])
    if is_t4:original=original/'quality.json'
    current = Path(path)
    seal_check("quality output pathname", str(original.resolve()), str(current.resolve()),
               where="D41 quality scope",
               refusal=ValueError('quality output differs from executed arguments'))
    raw = current.read_bytes()
    document = json.loads(raw)
    qs={int(s.removeprefix('q')) for s in command[command.index('--cases')+1].split(',')}
    if set(map(int,document['rungs']))!=qs:raise ValueError('quality rung census differs from execution')
    for qtext,row in document['rungs'].items():
        q=int(qtext)
        from tessera.export import served_recipe
        from tessera.structure import STRUCTURE_ROUTED_MOE
        recipe=served_recipe(grid,q,STRUCTURE_ROUTED_MOE) if is_t4 else wire_recipe(grid,q)
        if is_t4 and row.get('recipe')!=recipe.to_config():raise ValueError('recorded quality recipe differs from executed encoder owner')
        for sample in row.get('samples',[]):
            if is_t4:
                from fractions import Fraction
                from tessera.calculator import terminal_rate
                from tessera.manifest import body_rate_cap,scale_plane_terminal_flags,BodyKind
                base,refine,row_scale=scale_plane_terminal_flags(recipe.scale_plane)
                bits=Fraction(terminal_rate(q*grid.arity,32,256,arity=grid.arity,cap=body_rate_cap(recipe.body,grid),span=recipe.span,window_bits=recipe.window_bits,code_bytes=grid.code_bytes,with_scale_base=base,with_scale_refine=refine,with_row_scale=row_scale,with_diagonals=False,completion=0,with_forest=recipe.body is BodyKind.TCQ))*32*256
            else:bits=unit_wire_bits(grid,q,32,256)
            if sample['exact_bytes']*8!=bits or sample['accounted_bits']!={'numerator':bits.numerator,'denominator':bits.denominator}:
                raise ValueError('quality exact encoder bytes do not match executed grid/recipe')
        row['scope']=quality_scope(grid,q,recipe)
    document['format']=f"TESSERA_{grid.name.split('x',1)[0]}_K{grid.arity}"
    document['grid']=grid.name;document['arity']=grid.arity
    document['scope_provenance']={'action_key':action_key,'snapshot':sealed['params']['checkout_snapshot']['commit'],
        'original_file_sha256':hashlib.sha256(raw).hexdigest(),'binding':'executed explicit grid arguments plus exact encode_linear/unit_wire_bits identity'}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:json.dump(document,stream,indent=1,allow_nan=False)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--bind-existing',default='')
    ap.add_argument('--source-action',default='')
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(str(q) for q in range(768, 1153)))
    ap.add_argument("--grid", default="E4M3")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.bind_existing:
        if not args.source_action:raise ValueError('existing quality needs its original executed action')
        bind_existing_quality(args.bind_existing,args.source_action,args.out)
        print('Actual executed quality scope bound; scores unchanged',flush=True)
        return
    if not args.model:raise ValueError('quality model is required')
    torch.set_num_threads(1)
    model = Path(args.model)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    samples = []
    # A fixed actual expert tile for each projection. All rungs share these exact
    # BF16 source bytes and codec defaults; no random Gaussian quality surrogate.
    for role in ("gate", "up", "down"):
        name = f"model.language_model.layers.3.mlp.experts.0.{role}_proj.weight"
        with safe_open(str(model / mapping[name]), framework="pt", device="cpu") as h:
            source = h.get_slice(name)
            shape = source.get_shape()
            weight = source[:32, :256].contiguous()
        expected = (4096, 2048) if role == "down" else (2048, 4096)
        assert tuple(shape) == expected, (name, shape, expected)
        source_sha = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        samples.append((weight, {"tensor": name, "file": mapping[name], "source_shape": shape,
                                "slice": [[0,32],[0,256]], "source_sha256": source_sha,
                                "source_squared_norm": float(weight.double().square().sum())}))
    if args.preflight:
        print(json.dumps({"status":"passed", "samples":[m for _,m in samples]}), flush=True)
        return
    grid = grid_for_name(args.grid)
    result = {"schema":"tessera.rung_quality.v1", "source_kind":"actual_sampled_expert_weights",
              "device":"cpu", "model":str(model), "sample_selection":"layer 3, expert 0; first 32 rows and 256 columns of gate/up/down",
              "objective":"unweighted weight-space relative SSE; not served KL or promotion",
              "codec":"tessera.export.encode_linear defaults; read_unit_artifact; exact accountant identity",
              "rungs":{}, "start_unix":time.time()}
    result.update(format=f"TESSERA_{grid.name.split('x',1)[0]}_K{grid.arity}",grid=grid.name,arity=grid.arity)
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    for q in (int(s.removeprefix("q")) for s in args.cases.split(",")):
        start = time.time()
        row = {"measurement_status":"measured", "source_kind":result["source_kind"], "device":"cpu", "samples":[], "anomaly_flags":[]}
        from tessera.export import wire_recipe
        row['scope']=quality_scope(grid,q,wire_recipe(grid,q))
        try:
            for weight, meta in samples:
                unit = encode_linear(weight,grid=grid,q256=q)
                decoded = read_unit_artifact(unit.blob).double()
                assert bool(torch.isfinite(decoded).all()), "nonfinite decoded weights"
                bits = unit_wire_bits(grid,q,32,256)
                assert bits == unit.exact_bytes * 8, (q,bits,unit.exact_bytes)
                sse = float((decoded-weight.double()).square().sum())
                row["samples"].append({**meta,"relative_sse":sse/meta["source_squared_norm"],"squared_error":sse,
                                       "exact_bytes":unit.exact_bytes,"accounted_bits":{"numerator":bits.numerator,"denominator":bits.denominator}})
        except Exception as exc:
            row["measurement_status"]="failed"
            row["error"]=repr(exc)
            row["anomaly_flags"]=["quality_screen_failed"]
        result["rungs"][str(q)]=row
        row["seconds"]=time.time()-start
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result,indent=1,allow_nan=False))
        temporary.replace(path)
        print(json.dumps({"q256":q,"status":row["measurement_status"],"seconds":row["seconds"]}),flush=True)
    result["end_unix"]=time.time()
    # Raw adjacent-rung ratios are evidence, not an arbitrary anomaly threshold.
    for q in sorted(int(k) for k in result["rungs"]):
        row=result["rungs"][str(q)]
        high=result["rungs"].get(str(q+1))
        if row["measurement_status"]=="measured" and high and high["measurement_status"]=="measured":
            row["adjacent_higher_raw_error_ratios"]=[h["relative_sse"]/l["relative_sse"] if l["relative_sse"] else None for l,h in zip(row["samples"],high["samples"])]
    path.write_text(json.dumps(result,indent=1,allow_nan=False))


if __name__=="__main__":
    main()
