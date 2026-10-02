"""KDA raw-word/source/case admission, usable without scientific packages."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SOURCE_OWNERS = ("mhc/mhc_probe.py", "kda/conv_gate.py", "kda/conv_progress.py")


def harness_source_identity(root: Path | None = None) -> dict:
    """Bind the driver and both scientific-independent policy owners."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = {}
    for name in SOURCE_OWNERS:
        raw = (root / name).read_bytes()
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    body = {"schema": "tessera.kda_harness_source.v1", "files": files}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def harness_source_identity_matches(record, root: Path | None = None) -> bool:
    try:
        return record == harness_source_identity(root)
    except OSError:
        return False

KDA_P = 4096

KDA_HEADS = 32

KDA_WIDTH = 4

KDA_CONV_PTX_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

// Explicit rounding modifiers keep ptxas from re-fusing what the reference
// keeps separate (PTX mul/add without .rn may be contracted into FFMA).
__device__ __forceinline__ float p_fma(float a, float b, float c) {
  float d; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float p_mul(float a, float b) {
  float d; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float p_add(float a, float b) {
  float d; asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float p_sub(float a, float b) {
  float d; asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float p_ex2(float a) {
  float d; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float p_ex2_ftz(float a) {
  float d; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float p_div_full(float a, float b) {
  float d; asm volatile("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float p_div_rn(float a, float b) {
  float d; asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }

// MODE 0: the served SASS.  1: two roundings per tap.  2: ex2.approx.ftz.  3: div.rn.
template <int MODE>
__global__ void kda_conv_ref(const __nv_bfloat16* __restrict__ x, long sx_tok, long sx_dim,
                             const float* __restrict__ w, long sw_dim, long sw_w,
                             __nv_bfloat16* __restrict__ st, long ss_seq, long ss_dim, long ss_tok,
                             int state_len, const int* __restrict__ qsl, const int* __restrict__ idx,
                             const bool* __restrict__ has, int dim, __nv_bfloat16* __restrict__ out, long so_tok) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int s = blockIdx.y;
  if (c >= dim) return;
  const float w0 = w[c * sw_dim + 0 * sw_w], w1 = w[c * sw_dim + 1 * sw_w];
  const float w2 = w[c * sw_dim + 2 * sw_w], w3 = w[c * sw_dim + 3 * sw_w];
  float c0 = 0.f, c1 = 0.f, c2 = 0.f;
  __nv_bfloat16* b = st + (long)idx[s] * ss_seq + (long)c * ss_dim;
  if (has[s]) {
    // Stock prefill uses KERNEL_WIDTH-1 history columns, irrespective of
    // the physical cache length. MODE 4 preserves the old tail-index bug.
    const int first = MODE == 4 ? state_len - 3 : 0;
    c2 = __bfloat162float(b[(long)(first + 2) * ss_tok]);
    c1 = __bfloat162float(b[(long)(first + 1) * ss_tok]);
    c0 = __bfloat162float(b[(long)first * ss_tok]);
  }
  for (int t = qsl[s]; t < qsl[s + 1]; ++t) {
    const float xc = __bfloat162float(x[(long)t * sx_tok + (long)c * sx_dim]);
    float acc;
    if (MODE == 1) {
      acc = p_add(p_add(p_add(p_add(0.f, p_mul(c0, w0)), p_mul(c1, w1)), p_mul(c2, w2)), p_mul(xc, w3));
    } else {
      acc = p_fma(xc, w3, p_fma(c2, w2, p_fma(c1, w1, p_fma(c0, w0, 0.f))));
    }
    const float z = p_mul(p_sub(0.f, acc), 1.44269502162933349609375f);  // 0x3FB8AA3B
    const float e = MODE == 2 ? p_ex2_ftz(z) : p_ex2(z);
    const float den = p_add(e, 1.f);
    const float y = MODE == 3 ? p_div_rn(acc, den) : p_div_full(acc, den);
    out[(long)t * so_tok + c] = __float2bfloat16_rn(y);
    c0 = c1; c1 = c2; c2 = xc;
  }
  // One thread owns one sequence/channel, so the state write follows all
  // reads. Short fresh sequences retain leading +0; spare columns stay put.
  // MODE 5 makes the old physical-tail assumption observable on state writes.
  const int first = MODE == 5 ? state_len - 3 : 0;
  b[(long)first * ss_tok] = __float2bfloat16_rn(c0);
  b[(long)(first + 1) * ss_tok] = __float2bfloat16_rn(c1);
  b[(long)(first + 2) * ss_tok] = __float2bfloat16_rn(c2);
}

torch::Tensor conv_ref(torch::Tensor x, torch::Tensor w, torch::Tensor st, int64_t state_len,
                       torch::Tensor qsl, torch::Tensor idx, torch::Tensor has, int64_t mode) {
  const int dim = x.size(1), nseq = qsl.size(0) - 1;
  auto out = torch::empty({x.size(0), dim}, x.options());
  dim3 grid((dim + 127) / 128, nseq), block(128);
  auto stream = at::cuda::getCurrentCUDAStream();
#define KDA_LAUNCH(M) kda_conv_ref<M><<<grid, block, 0, stream>>>( \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), x.stride(1), \
    w.data_ptr<float>(), w.stride(0), w.stride(1), \
    reinterpret_cast<__nv_bfloat16*>(st.data_ptr()), st.stride(0), st.stride(1), st.stride(2), \
    (int)state_len, qsl.data_ptr<int>(), idx.data_ptr<int>(), has.data_ptr<bool>(), dim, \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0))
  switch (mode) {
    case 0: KDA_LAUNCH(0); break;
    case 1: KDA_LAUNCH(1); break;
    case 2: KDA_LAUNCH(2); break;
    case 3: KDA_LAUNCH(3); break;
    case 4: KDA_LAUNCH(4); break;
    case 5: KDA_LAUNCH(5); break;
    default: TORCH_CHECK(false, "unknown KDA PTX mode");
  }
#undef KDA_LAUNCH
  return out;
}

// Store both exponentials before +1. This makes the FTZ mutation observable
// even though that difference need not survive the served SiLU operation.
__global__ void kda_ex2_control(const float* accs, int n, float* raw, __nv_bfloat16* bf16) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float acc = accs[i];
  const float z = p_mul(p_sub(0.f, acc), 1.44269502162933349609375f);
  const float e0 = p_ex2(z), e2 = p_ex2_ftz(z);
  const float d0 = p_add(e0, 1.f), d2 = p_add(e2, 1.f);
  const float y0 = p_div_full(acc, d0), y2 = p_div_full(acc, d2);
  raw[8*i+0] = acc; raw[8*i+1] = z;
  raw[8*i+2] = e0; raw[8*i+3] = e2;
  raw[8*i+4] = d0; raw[8*i+5] = d2;
  raw[8*i+6] = y0; raw[8*i+7] = y2;
  bf16[2*i+0] = __float2bfloat16_rn(y0);
  bf16[2*i+1] = __float2bfloat16_rn(y2);
}

std::vector<torch::Tensor> ex2_control(torch::Tensor accs) {
  TORCH_CHECK(accs.is_cuda() && accs.scalar_type() == torch::kFloat32 && accs.is_contiguous(),
              "ex2 control needs contiguous CUDA fp32 accumulators");
  const int n = accs.numel();
  auto raw = torch::empty({n, 8}, accs.options());
  auto bf16 = torch::empty({n, 2}, accs.options().dtype(torch::kBFloat16));
  kda_ex2_control<<<(n+127)/128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
    accs.data_ptr<float>(), n, raw.data_ptr<float>(),
    reinterpret_cast<__nv_bfloat16*>(bf16.data_ptr()));
  return {raw, bf16};
}
"""

KDA_PTX_CASES = (("varlen", (5, 2, 9, 700), (True, False, True, False), 1.0, 0.0),
                 ("short", (2, 1, 3), (False, True, True), 1.0, 0.0),
                 ("served_2048", (2048,), (False,), 1.0, 0.0),
                 ("continued_2048", (2048,), (True,), 1.0, 0.0),
                 ("signed_zeros", (64, 300), (True, False), 1.0, 0.5),
                 ("large_x64", (512, 33), (True, False), 64.0, 0.0))

KDA_PTX_MODES = {0: "served_sass", 1: "mutant_two_roundings_per_tap", 2: "mutant_ex2_ftz", 3: "mutant_div_rn",
                 4: "mutant_physical_tail_read", 5: "mutant_physical_tail_write"}

KDA_GATE_CONTRACT = "tessera.kda_conv_screen.v2"

KDA_PTX_CFLAGS = ["-O3"]

def kdaex2_gate_errors(control: dict) -> list[str]:
    """Validate actual intermediate bits and their current source/compiled bindings."""
    if not isinstance(control, dict):
        return ["ex2 equivalence control absent or malformed"]
    try:
        if control["schema"] != "tessera.kda_ex2_control.v1":
            return ["ex2 equivalence control schema mismatch"]
        if (control["ptx_source_sha256"] != hashlib.sha256(KDA_CONV_PTX_SRC.encode()).hexdigest()
                or control["cuda_flags"] != KDA_PTX_CFLAGS):
            return ["ex2 equivalence control does not match current PTX/build flags"]
        for path, digest in ((control["compiled_module"], control["compiled_module_sha256"]),
                             (control["sass"]["path"], control["sass"]["sha256"])):
            artifact = Path(path).read_bytes()
            if not artifact or hashlib.sha256(artifact).hexdigest() != digest:
                return ["ex2 compiled identity or SASS digest mismatch"]
        rows, bf16 = control["raw_fp32_bits"], control["raw_bf16_bits"]
        n = control["finite_accumulator_count"]
        if type(n) is not int or n <= 0 or len(rows) != n or len(bf16) != n:
            return ["ex2 control has no complete finite-accumulator population"]
        if (any(len(row) != 8 or any(type(v) is not int or not -(1 << 31) <= v < (1 << 31) for v in row)
                for row in rows) or
                any(len(row) != 2 or any(type(v) is not int or not -(1 << 15) <= v < (1 << 15) for v in row)
                    for row in bf16)):
            return ["ex2 control raw words are malformed"]
        if not all((r[0] & 0x7fffffff) < 0x7f800000 for r in rows):
            return ["ex2 equivalence only covers finite FP32 accumulators"]
        changed = [r for r in rows if r[2] != r[3]]
        tiny_inputs = [r for r in rows if 0 < (r[1] & 0x7fffffff) < 0x00800000]
        errors = []
        if not changed:
            errors.append("ex2 mutation is vacuous: no raw exponential difference")
        if not all(0 < r[2] < 0x00800000 and r[3] == 0 and r[4] == r[5] == 0x3f800000 for r in changed):
            errors.append("ex2 difference is not subnormal erasure by rounded +1")
        if not tiny_inputs or not all(r[2] == r[3] == 0x3f800000 for r in tiny_inputs):
            errors.append("ex2 subnormal-input instruction behavior differs or is unobserved")
        for key, pairs in (("ex2", [(r[2], r[3]) for r in rows]),
                           ("denominator", [(r[4], r[5]) for r in rows]),
                           ("output_fp32", [(r[6], r[7]) for r in rows]),
                           ("output_bf16", bf16)):
            count = sum(a != b for a, b in pairs)
            if (type(control[key]["bits_differing"]) is not int or control[key]["bits_differing"] != count
                    or control[key]["bit_equal"] is not (count == 0)):
                errors.append(f"ex2 {key} summary disagrees with actual raw words")
            if key != "ex2" and count:
                errors.append(f"ex2 downstream {key} differs")
        if (type(control["input_subnormal_count"]) is not int or control["input_subnormal_count"] != len(tiny_inputs) or
                control["all_accumulators_finite"] is not True or
                control["input_subnormal_ex2_is_one_both"] is not True or
                control["changed_ex2_is_subnormal_flushed_to_positive_zero"] is not True):
            errors.append("ex2 control population/classification summary is malformed")
        return errors
    except (KeyError, TypeError, ValueError, OSError):
        return ["ex2 equivalence control absent, malformed, or artifacts unavailable"]

def kdaptx_case_gate_errors(screen: dict) -> list[str]:
    """Derive admission summaries from the complete fixed screen's raw-word comparisons."""
    try:
        if (type(screen["p"]) is not int or screen["p"] != KDA_P or
                type(screen["width"]) is not int or screen["width"] != KDA_WIDTH):
            return ["KDA case geometry differs from the current screen"]
        expected = {(layout, state_len, name): (list(lens), list(has))
                    for layout in ("SD", "DS") for state_len in (KDA_WIDTH - 1, KDA_WIDTH + 2)
                    for name, lens, has, *_ in KDA_PTX_CASES}
        rows = screen["cases"]
        if not isinstance(rows, list) or len(rows) != len(expected):
            return ["KDA case roster is absent or incomplete"]
        visited = set()
        exact_output, exact_state = True, True
        observed = {label: False for mode, label in KDA_PTX_MODES.items() if mode}

        def count(record, maximum):
            n = record["bits_differing"]
            if type(n) is not int or not 0 <= n <= maximum or record["bit_equal"] is not (n == 0):
                raise ValueError("KDA raw-word comparison count/flag is inconsistent")
            return n

        for row in rows:
            if type(row["state_len"]) is not int or type(row["layout"]) is not str or type(row["case"]) is not str:
                return ["KDA case identity is malformed"]
            key = (row["layout"], row["state_len"], row["case"])
            if key not in expected or key in visited:
                return ["KDA case roster contains an unexpected or duplicate cell"]
            visited.add(key)
            lens, has = expected[key]
            if (row["lens"] != lens or row["has"] != has or
                    any(type(v) is not int for v in row["lens"]) or
                    any(type(v) is not bool for v in row["has"])):
                return ["KDA case inputs differ from the current screen"]
            output_max = sum(lens) * 3 * KDA_P
            state_max = (len(lens) + 2) * 3 * KDA_P * row["state_len"]
            reference = row[KDA_PTX_MODES[0]]
            ref_output = count(reference, output_max)
            ref_state = count(reference["conv_state"], state_max)
            exact_output &= ref_output == 0
            exact_state &= ref_state == 0
            for mode, label in KDA_PTX_MODES.items():
                cell = row[label]
                output, state = count(cell, output_max), count(cell["conv_state"], state_max)
                if set(cell["qkv"]) != {"q", "k", "v"} or sum(count(cell["qkv"][k], output_max // 3)
                                                            for k in ("q", "k", "v")) != output:
                    raise ValueError("KDA merged q/k/v count disagrees with slice counts")
                if mode:
                    relative_output = count(cell["vs_candidate"], output_max)
                    relative_state = count(cell["state_vs_candidate"], state_max)
                    if ref_output == 0 and relative_output != output:
                        raise ValueError("KDA mutant output comparisons contradict exact stock/candidate output")
                    if ref_state == 0 and relative_state != state:
                        raise ValueError("KDA mutant state comparisons contradict exact stock/candidate state")
                    observed[label] |= relative_output > 0 or relative_state > 0
        if visited != set(expected):
            return ["KDA case roster is incomplete"]
        errors = []
        for field, derived in (("served_output_bit_equal_all", exact_output),
                               ("served_conv_state_bit_equal_all", exact_state),
                               ("served_bit_equal_all", exact_output and exact_state)):
            if screen[field] is not derived:
                errors.append(f"KDA {field} summary contradicts actual case comparisons")
        if not exact_output or not exact_state:
            errors.append("KDA actual output or convolution state differs from stock")
        if screen["mutants_seen"] != observed or any(type(v) is not bool for v in screen["mutants_seen"].values()):
            errors.append("KDA mutation summary contradicts actual case comparisons")
        return errors
    except (KeyError, TypeError, ValueError, AttributeError):
        return ["KDA case evidence is absent, malformed, or comparison counts are inconsistent"]

def kdaptx_gate_errors(screen: dict) -> list[str]:
    """v2: output/state mutation witnesses plus an active, erased ex2 intermediate."""
    errors = [] if screen.get("served_bit_equal_all") is True else ["served bitwise mismatch"]
    errors.extend(kdaptx_case_gate_errors(screen))
    if screen.get("gate_contract") != KDA_GATE_CONTRACT:
        errors.append("KDA numerical gate contract mismatch")
    seen = screen.get("mutants_seen", {})
    errors.extend(f"unobserved required mutant: {label}" for mode, label in KDA_PTX_MODES.items()
                  if mode not in (0, 2) and seen.get(label) is not True)
    if seen.get(KDA_PTX_MODES[2]) is not False:
        errors.append("ex2 output/state equivalence differs or is unobserved")
    errors.extend(kdaex2_gate_errors(screen.get("ex2_equivalence")))
    return errors
