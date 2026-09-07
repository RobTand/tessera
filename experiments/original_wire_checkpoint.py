"""Bounded first-model handoff: original layer-2 cache units, all else source precision."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
import export_tessera_serving as exporter
from tessera.cached_unit import encoder_source_sha256, tensor_identity
from tessera.fused import parse_fused
from tessera.serving_parts import source_identity, sha256_file

ROOT = Path('/mnt/shared/tessera-clean-runtime-20260907/original-wire-layer2')
SRC = Path('/mnt/shared/models/LFM2.5-8B-A1B-BF16')
WIRES = Path('/mnt/shared/tessera-measurements/first-model-20260907/native-moe-wires-r1024')
CAPTURE = Path('/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/canonical')

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'artifact':str(path),'sha256':sha256_file(path),'bytes':path.stat().st_size}),flush=True)


def main():
    started=time.time()
    torch.set_num_threads(2)
    bundle=ROOT/'cached-units'; bundle.mkdir(exist_ok=False)
    complete_path=WIRES/'complete.json'
    assert sha256_file(complete_path)=='aa2d703f690c8478717ac7431e564340426d6a3c6defa05d78edc0c49ac58e82'
    complete=json.loads(complete_path.read_text())
    capture_path=CAPTURE/'capture_manifest.json'
    assert sha256_file(capture_path)=='db3cd996ee8a3ac82d62c6e7e2f23cdb995874b831adcf9360acdae682654823'
    capture=json.loads(capture_path.read_text())
    expected={f'model.layers.2.feed_forward.experts.{expert}.{proj}' for expert in range(32) for proj in ('w1','w3','w2')}
    assert set(complete['wires'])==expected
    assert encoder_source_sha256()=='57809bff862b880dc397e6d271a80c04d6c87d1af3bba12c076648fc5443355c'
    records={}; hessians={}; bridge={}
    for name in sorted(expected):
        entry=complete['wires'][name]
        record_path=Path(entry['record']); wire=Path(entry['wire'])
        assert sha256_file(record_path)==entry['record_sha256']
        record=json.loads(record_path.read_text())
        assert sha256_file(wire)==entry['wire_sha256']==record['blob_sha256']
        assert wire.stat().st_size==record['blob_bytes']
        shutil.copyfile(wire,bundle/record['file'])
        records[name]=record
        ce=capture['entries'][name]; cp=CAPTURE/ce['path']
        assert sha256_file(cp)==ce['sha256']
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        assert payload['name']==name
        H=payload['hessian']; assert tensor_identity(H)==record['identity']['calibration']['hessian']
        hessians[name]=H
        bridge[name]={'capture_file':str(cp),'capture_sha256':ce['sha256'],'hessian_identity':tensor_identity(H),'wire_sha256':entry['wire_sha256'],'record_sha256':entry['record_sha256']}
        del payload
    source=source_identity(SRC)
    write(bundle/'manifest.json',{'schema':'tessera.cached_units.v1','source':source,'units':records})
    provenance=dict(capture['identity']['calibration'],hessian_role='fit',capture_manifest_sha256=sha256_file(capture_path))
    torch.save({'H':hessians,'provenance':provenance},ROOT/'hessian.pt')
    write(ROOT/'hessian-bridge.json',{'capture_manifest_sha256':sha256_file(capture_path),'hessian_payload_sha256':sha256_file(ROOT/'hessian.pt'),'units':bridge})
    del hessians;gc.collect()
    _,dense,_,_=exporter.quantizable(SRC)
    # Routers are immutable automatically and the exporter forbids any plan entry for them.
    plan={name:'PASSTHROUGH' for name in dense if not exporter.MOE_ROUTER.match(name)}
    plan['model.layers.2.feed_forward.experts']={'grid':'E4M3','q256':1024,'source_layout':'unpacked_per_expert'}
    write(ROOT/'plan.json',plan)
    write(ROOT/'source-identity.json',source)
    cmd=[sys.executable,str(Path(exporter.__file__)),str(SRC),str(ROOT/'checkpoint'),'--grid','E4M3','--q256','1024','--device','cpu','--plan-json',str(ROOT/'plan.json'),'--cached-expert-units',str(bundle/'manifest.json'),'--hessian',str(ROOT/'hessian.pt')]
    write(ROOT/'export-command.json',{'argv':cmd})
    subprocess.run(cmd,check=True)
    output=ROOT/'checkpoint'
    config=json.loads((output/'config.json').read_text())
    assert config['quantization_config']['quant_method']=='tessera'
    wire_proofs={}; passthrough={}
    # Compare the actual emitted tensors, not the exporter summary.
    with safe_open(str(SRC/'model.safetensors'),framework='pt') as before, safe_open(str(output/'model.safetensors'),framework='pt') as after:
        source_names=set(before.keys()); output_names=set(after.keys())
        replaced={name+'.weight' for name in expected}
        assert output_names==(source_names-replaced)|{name+'.wire' for name in expected}
        for name in sorted(output_names):
            if name.endswith('.wire'):
                unit=name.removesuffix('.wire'); members=parse_fused(after.get_tensor(name).numpy().tobytes())
                rec=records[unit]; assert len(members)==1
                assert members[0].name==rec['identity']['projection']['projection']
                assert members[0].rows==rec['identity']['projection']['rows']
                digest=hashlib.sha256(members[0].blob).hexdigest(); assert digest==rec['blob_sha256']
                wire_proofs[unit]={'original_blob_sha256':digest,'exported_tensor':name,'framed_bytes':after.get_tensor(name).numel()}
            else:
                a=before.get_tensor(name); b=after.get_tensor(name)
                assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.view(torch.uint8),b.view(torch.uint8)),name
                passthrough[name]={'dtype':str(a.dtype),'shape':list(a.shape),'bytes':a.numel()*a.element_size()}
        del a,b
    proof={'schema':'tessera.original_wire_checkpoint.v1','status':'passed','scope':'Exact 96 original layer-2 wires in serving framing; every other tensor bit-exact source precision. No runtime qualification yet.','wire_complete_sha256':sha256_file(complete_path),'source_identity_sha256':sha256_file(ROOT/'source-identity.json'),'plan_sha256':sha256_file(ROOT/'plan.json'),'wires':wire_proofs,'passthrough':passthrough,'checkpoint_files':{p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()},'started_epoch':started,'finished_epoch':time.time()}
    write(ROOT/'export-proof.json',proof)

if __name__=='__main__':main()
