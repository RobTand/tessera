"""Capture actual native quantizer operands; no timing or acceptance relaxation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
from tessera.serving.native_ops import require_native_fp8_quant, native_fp8_quant
p = argparse.ArgumentParser()
p.add_argument('--request', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
request = json.loads(a.request.read_text())
source = a.request.parent / request['tensors_path']
require_native_fp8_quant('original request activation diagnostic')
saved, phases = {}, {}
with safe_open(source, framework='pt', device='cpu') as f:
 for phase in ('prefill', 'decode'):
  x = f.get_tensor(phase + '.input').cuda()
  expected = f.get_tensor(phase + '.reference_qdq').cuda()
  codes, scales = native_fp8_quant(x)
  actual = (codes.float() * scales).to(x.dtype)
  rows = x.float()
  computed = (rows.abs().amax(-1, keepdim=True) / 448.).clamp_min(1./(448.*512.))
  formulas = {
   'compressed_tensors': quantize(rows, computed, torch.zeros_like(computed), FP8_DYNAMIC['input_activations'], dtype=torch.float8_e4m3fn),
   'divide_computed_scale': (rows / computed).clamp(-448.,448.).to(torch.float8_e4m3fn),
   'multiply_computed_reciprocal': (rows * computed.reciprocal()).clamp(-448.,448.).to(torch.float8_e4m3fn),
   'multiply_native_reciprocal': (rows * scales.reciprocal()).clamp(-448.,448.).to(torch.float8_e4m3fn),
   'divide_native_scale': (rows / scales).clamp(-448.,448.).to(torch.float8_e4m3fn),
  }
  mismatch = actual != expected
  indices = mismatch.nonzero()
  details = []
  for r,c in indices[:40].tolist():
   details.append({'row':r,'column':c,'input':float(x[r,c]),'expected_qdq':float(expected[r,c]),'actual_qdq':float(actual[r,c]),'native_code':float(codes[r,c]),'native_code_bits':int(codes.view(torch.uint8)[r,c]),'native_scale':float(scales[r,0]),'computed_scale':float(computed[r,0]),'formulas':{k:float(v[r,c]) for k,v in formulas.items()}})
  phases[phase] = {'shape':list(x.shape),'qdq_mismatch_count':int(mismatch.sum()),'scale_mismatch_count':int((scales != computed).sum()),'max_abs_qdq_error':float((actual.float()-expected.float()).abs().max()),'first_mismatches':details,'formulas':{k:{'native_code_mismatches':int((v.view(torch.uint8)!=codes.view(torch.uint8)).sum()),'expected_qdq_mismatches':int(((v.float()*computed).to(x.dtype)!=expected).sum())} for k,v in formulas.items()}}
  values = {'input':x,'expected_qdq':expected,'native_codes_bits':codes.view(torch.uint8),'native_scales':scales,'computed_scales':computed,'native_qdq':actual,**{k+'_bits':v.view(torch.uint8) for k,v in formulas.items()}}
  for key,value in values.items(): saved[phase+'.'+key]=value.detach().cpu().contiguous()
torch.cuda.synchronize()
data = a.out.with_suffix('.safetensors')
save_file(saved, str(data))
a.out.write_text(json.dumps({'schema':'tessera.native_quant_diagnostic.v1','request_sha256':hashlib.sha256(a.request.read_bytes()).hexdigest(),'data_path':str(data),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'phases':phases},indent=2)+'\n')
print(a.out.read_text(), flush=True)
