"""Raw-producer record freezing for the original-reference GLM operator acquisition.

``freeze`` validates and freezes the raw producer record; the standalone entry that read a
PQ local handoff is archived (archive/flux-migration/T-FINAL) because no producer of that
handoff remains. Raw panels name actual execution, never an unmeasured joint distortion table.
"""
import copy,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def freeze(inputs,prepared,source_sha256):
    from experiments import bench_native_moe_operator as moe
    dense=moe.dense;operator=prepared['operator']
    if inputs['schema']!='prismaquant.native_moe_inputs.v1':raise ValueError('native input schema differs')
    for key in ('shape','routing','profile_role_order','routing_capture_sha256','serving_config_sha256'):
        if operator[key]!=inputs[key]:raise ValueError('prepared '+key+' differs from independent inputs')
    if prepared['runtime']['image']!=inputs['runtime_image']:raise ValueError('prepared runtime image differs')
    members=inputs['members'];expected=[]
    for member in members:
        value={key:member[key] for key in ('unit','expert','role','format','shape','source_weight','rendered_weight')}
        value.update(wire_sha256=member['wire']['blob_sha256'],wire_record_sha256=dense.identity_sha256(member['wire']['record']))
        if member['activation'].get('input_global_scale') is not None:value['input_global_scale']=member['activation']['input_global_scale']
        expected.append(value)
    if operator['members']!=expected:raise ValueError('prepared complete member identities differ')
    route=operator['declared_route']
    if not route['decoder'].startswith('native_'):raise ValueError('raw acquisition requires actual compact native route')
    panel={'schema':moe.RAW_PANEL_SCHEMA,'unit':inputs['unit'],'format':inputs['format'],
        'shape':inputs['shape'],'members':members,'profile_role_order':inputs['profile_role_order'],
        'routing':inputs['routing'],'routing_capture_sha256':inputs['routing_capture_sha256'],
        'source_sha256':source_sha256,'calibration_sha256':inputs['calibration']['calibration_sha256'],
        'source_execution':inputs['routing_capture']['source_execution'],
        'source_execution_qualification_sha256':None,'probe_scope':None,
        'runtime_binding':{'member_formats':{m['unit']:m['format'] for m in members},
            'member_execution_identity_sha256':{m['unit']:dense.identity_sha256({'qname':m['unit'],
                **{key:m[key] for key in ('format','source_weight','rendered_weight','activation')}}) for m in members},
            'member_shapes':{m['unit']:moe._member_shape(inputs['shape'],m['role']) for m in members},
            'operator_route':json.dumps(route,sort_keys=True,separators=(',',':'),allow_nan=False)},
        'execution':inputs['execution'],'runtime':prepared['runtime'],
        'native_tensors_sha256':dense.identity_sha256(operator['native_tensors']),
        'scheme_sha256':operator['scheme_sha256'],'config_sha256':operator['config_sha256'],
        'serving_config_sha256':inputs['serving_config_sha256'],'workspace':prepared['workspace'],
        'workspace_sha256':dense.identity_sha256(prepared['workspace']),'numerics':inputs['numerics'],
        'phases':{phase:{**inputs['phases'][phase],'expected_route':route} for phase in moe.PHASES}}
    if 'reference_served_quantizer' in inputs:panel['reference_served_quantizer']=inputs['reference_served_quantizer']
    if 'source_acquisition' in inputs:panel['source_acquisition']=copy.deepcopy(inputs['source_acquisition'])
    return moe.validate_panel(copy.deepcopy(panel))
