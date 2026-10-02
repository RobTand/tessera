#!/usr/bin/env bash
# dl380g10 pre-check: compile mla_prefill_mg.cu with FlashInfer's JIT flags for
# sparse_mla_sm120 (flashinfer/jit/core.py gen_jit_spec + jit/mla.py), nvcc
# 13.0.88 from the PyPI wheels, and dump SASS for the copy and L0 kernels.
set -uo pipefail
D=/home/rob/tmp/mla-toolchain
C=$(ls -d $D/venv/lib/python3*/site-packages/nvidia/cu13)
export PATH=$C/bin:$PATH
OUT=${1:?output_dir}
SRC=$(realpath "$(dirname "$0")/../../src/tessera/serving/csrc/mla_prefill_mg.cu")
mkdir -p "$OUT"
cd "$OUT"
FLAGS=(-std=c++17 --threads 1 -use_fast_math -DTESSERA_MLA_DECLARED_FAST_MATH=1 -Xfatbin=-compress-all --compress-mode=size
  -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3
  -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3
  -gencode=arch=compute_121a,code=sm_121a -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1)
# jit/cpp_ext.py build_common_cflags: vendored CCCL (FlashInfer 3rdparty/cccl at
# 16bd510c) with -I so it wins over the toolkit copy, then -isystem dirs.
COMMON=(-DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1
  -I$D/cccl/cub -I$D/cccl/libcudacxx/include -I$D/cccl/thrust
  -isystem $C/include -isystem $D/include)
CUDA=(--compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false
  -allow-unsupported-compiler)
echo "nvcc: $(nvcc --version | tail -2 | head -1)"
echo "host: $(g++ --version | head -1)"
sha256sum "$SRC"
t0=$(date +%s)
nvcc "${COMMON[@]}" "${CUDA[@]}" "${FLAGS[@]}" -Xptxas -v -c "$SRC" -o mla_prefill_mg.o > compile.log 2>&1
rc=$?
echo "compile rc=$rc seconds=$(( $(date +%s) - t0 ))"
grep -E 'error|warning|registers|spill|lmem|smem|Function properties|Compiling entry' compile.log | head -40
[ $rc -eq 0 ] || { tail -40 compile.log; exit 1; }
cuobjdump -sass mla_prefill_mg.o > ours.sass 2> cuobjdump.err
cuobjdump -res-usage mla_prefill_mg.o > ours.res 2>&1
cuobjdump -res-usage $D/served/sparse_mla_sm120.so 2>/dev/null | grep -A1 '_Z28sparse_mla_prefill_mg_kernelIL9ModelType3EL13QkComputeMode0ELi32ELi64ELi2EE' > served.res
cat ours.res served.res
MANGLED=_Z28sparse_mla_prefill_mg_kernelIL9ModelType3EL13QkComputeMode0ELi32ELi64ELi2EEvPK13__nv_bfloat16PKhPKiPS2_PfPKf17PrefillColdParams
cuobjdump -sass -fun "$MANGLED" $D/served/sparse_mla_sm120.so > served.sass 2> served.err
echo "served dump rc=$? lines=$(wc -l < served.sass) cuobjdump=$(cuobjdump --version | tail -1)"
python3 - served.sass ours.sass <<'PY'
import re, sys
served_path, ours_path = sys.argv[1], sys.argv[2]
def functions(text):
    out, name, body = {}, None, []
    for line in text.splitlines():
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            if name: out[name] = body
            name, body = m.group(1), []
        elif name is not None:
            body.append(line)
    if name: out[name] = body
    return out
def norm(lines):
    # One entry per instruction: its text and both 64-bit encoding words
    # (the second carries the scheduling control bits), so "identical" means
    # the same instruction stream, not only the same mnemonics.
    res, pending = [], None
    for l in lines:
        m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;?\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$', l)
        if m:
            pending = [re.sub(r'\s+', ' ', m.group(2)), m.group(3)]
            continue
        m = re.match(r'\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$', l)
        if m and pending is not None:
            pending.append(m.group(1))
            res.append(" | ".join(pending))
            pending = None
    return res
served = functions(open(served_path).read())
ours = functions(open(ours_path).read())
print("served functions:", list(served)[:3], "ours:", list(ours))
s = None
for k, v in served.items():
    if 'sparse_mla_prefill_mg_kernel' in k: s = norm(v); print("served", k, len(s), "instructions")
for k, v in ours.items():
    n = norm(v)
    print("ours", k, len(n), "instructions")
    if 'copy' in k and s is not None:
        same = n == s
        print("COPY_SASS_IDENTICAL", same)
        if not same:
            import difflib
            d = list(difflib.unified_diff(s, n, 'served', 'ours', n=1, lineterm=''))
            print("\n".join(d[:80]))
    if 'l0' in k:
        print("L0 LDL/STL:", sum(1 for x in n if x.split()[0].startswith(('LDL', 'STL')) or ' LDL' in x or ' STL' in x))
        print("L0 FADD with RZ:", sum(1 for x in n if 'FADD' in x and 'RZ' in x))
PY
