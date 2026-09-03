# What the pinned build does with an NVFP4 MoE on a clamped config (2026-09-02)

`docs/tessera-serving-and-moe-contract.md` §9.1 recorded that
`--moe-backend flashinfer_b12x` raises on GLM-5.3-Flash-4layer's config, and
left the follow-on question open in as many words: "Which of those is backed on
sm121 is **not measured**." Issue #6 is that gap, and its consequence — an
NVFP4 MoE `requires_serve_flags` cell cannot be written while the backend name
in it would be a guess.

This closes the oracle half of that question by asking the runtime.

**Result.**

| question | answer, from the pinned build on this box |
|---|---|
| does the refusal reproduce? | **yes**, verbatim, at `fused_moe/oracle/nvfp4.py:263` |
| is there a supported flag? | **yes** — `--moe-backend flashinfer_cutlass` resolves. So does plain `auto`; no flag is needed |
| what does `auto` resolve to on sm121 for a clamped config? | **`FLASHINFER_CUTLASS` / `FlashInferExperts`**, W4A4 |
| does the clamp change what `auto` picks here? | **no** — clamped and unclamped both resolve to `FLASHINFER_CUTLASS` |
| would b12x serve here but for the clamp? | **yes** — unclamped, explicit b12x resolves to `FlashInferB12xExperts` |
| what would a fixed build need? | `FlashInferB12xExperts` to implement the SwiGLU clamp. `flashinfer_b12x_moe.py` contains no occurrence of `swiglu`, `clamp` or `limit` |

Harness: `experiments/nvfp4_moe_oracle_probe.py`, run through
`experiments/nvfp4_moe_oracle_probe.sh` (takes the serve lock; not a serve — no
model, no port). Result: `experiments/results/nvfp4_moe_oracle_probe.json`.
Build `prismaquant/glm53-mia-sm121:487ecf187` = `vllm 0.1.dev20051+g487ecf187`,
torch 2.13.0+cu130, `DeviceCapability(major=12, minor=1)`.

---

## 1. The scope is wider than the issue says: every GLM-5.3-Flash config is clamped

§9.1 scoped its finding to "build `487ecf187`, **this model's config**", and
noted that the general GLM NVFP4 route status "was measured on a config without
a `swiglu_limit`, and it does not generalise to GLM proper without
re-measuring". Re-measured — every GLM-5.3-Flash config on this box carries it:

```
$ python3 -c "
import json, pathlib
def find(o, path=''):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'swiglu_limit': out.append((path + '/' + k, v))
            out += find(v, path + '/' + k)
    return out
for f in sorted(pathlib.Path('/mnt/shared/models').glob('*/config.json')):
    print(f'{f.parent.name:<42s}', find(json.load(f.open())) or 'none')"

GLM-5.3-Flash                              [('/text_config/swiglu_limit', 10.0), ('/vision_config/swiglu_limit', 10.0)]
GLM-5.3-Flash-4layer                       [('/text_config/swiglu_limit', 10.0), ('/vision_config/swiglu_limit', 10.0)]
GLM-5.3-Flash-BF16                         [('/text_config/swiglu_limit', 10.0), ('/vision_config/swiglu_limit', 10.0)]
GLM-5.3-Flash-DFlash2                      none
GLM-5.3-Flash-EXL3-TR3-4bpw                [('/text_config/swiglu_limit', 10.0), ('/vision_config/swiglu_limit', 10.0)]
GLM-5.3-Flash-Tessera-E2M1K2-20260901      [('/text_config/swiglu_limit', 10.0), ('/vision_config/swiglu_limit', 10.0)]
Qwen3.8-Flash-Next                         none
```

So the refusal is a property of the **GLM-5.3-Flash family** on this build, not
of the four-layer surgical model. The 4-layer model inherited it from the
parent; it did not introduce it.

## 2. The refusal, from the runtime

`select_nvfp4_moe_backend` in the pinned build:

```python
    if (
        config.swiglu_limit is not None
        and requested_backend not in NVFP4_BACKENDS_WITH_CLAMP
    ):
        raise ValueError(
            f"Model sets swiglu_limit={config.swiglu_limit}, but the "
            f"explicitly requested moe_backend={runner_backend!r} does "
            f"not apply the SwiGLU clamp. Use 'flashinfer_trtllm', "
            f"'flashinfer_cutlass', 'flashinfer_cutedsl', 'cutlass', "
            f"'marlin', or 'humming' instead."
        )
```

Exercised, not read:

```json
"clamped_explicit_b12x": {
 "raised": "ValueError",
 "message": "Model sets swiglu_limit=10.0, but the explicitly requested moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp. Use 'flashinfer_trtllm', 'flashinfer_cutlass', 'flashinfer_cutedsl', 'cutlass', 'marlin', or 'humming' instead.",
 "where": ["  File \".../fused_moe/oracle/nvfp4.py\", line 263, in select_nvfp4_moe_backend", "    raise ValueError(", "ValueError: Model sets swiglu_limit=10.0, ..."]
}
```

**The filtering is deliberate, and the field says so.** `FusedMoEConfig`:

```python
    # SwiGLU clamp limit. When set, backends that do not implement the clamp
    # are filtered out by `FusedMoEExperts.is_supported_config` so the oracle
    # cannot silently select one and drop the clamp.
    swiglu_limit: float | None = None
```

That matters for how #6 is read: this is not a bug to work around. A build that
served GLM through b12x would be dropping a clamp the model's own config asks
for, and would be *quietly wrong* rather than loudly refused.

**A second exclusion the issue does not mention.** `FLASHINFER_B12X` is not in
`AVAILABLE_BACKENDS` at all on this build, clamp or no clamp:

```python
    # NOTE: the kernels are selected in the following order.
    # FLASHINFER_B12X is intentionally excluded from auto-selection until
    # the upstream CUTLASS SM121 MMA op guard is resolved; use
    # moe_backend="flashinfer_b12x" to opt in explicitly.
```

So on this build `auto` would never reach b12x even on an unclamped config, and
the only path to it is the explicit flag the clamp then refuses.

## 3. Which backends the device admits

Every NVFP4 expert class's own `_supports_current_device()`, called on this box.
No config is involved, so nothing here can be an artefact of a constructed one.

| backend | in `AVAILABLE_BACKENDS`? | in `NVFP4_BACKENDS_WITH_CLAMP`? | device admits on sm121 |
|---|:--:|:--:|:--:|
| FLASHINFER_TRTLLM | yes | yes | **no** (`is_device_capability_family(100)`) |
| FLASHINFER_CUTEDSL | yes | yes | **no** (family 100) |
| FLASHINFER_CUTEDSL_BATCHED | yes | **no** | no (family 100) |
| **FLASHINFER_CUTLASS** | yes | yes | **yes** |
| VLLM_CUTLASS | yes | yes | yes |
| MARLIN | yes | yes | yes |
| HUMMING | yes | yes | yes |
| EMULATION | yes | yes | yes |
| FLASHINFER_B12X | **no** | **no** | yes |

`is_device_capability_family_100: false`, `is_device_capability_family_120:
true`, and every `has_flashinfer_*` probe returns true, so nothing on this list
is missing for want of a package.

Two things fall out. First, the auto list §9.1 recorded —
`{FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTLASS, VLLM_CUTLASS,
MARLIN, EMULATION, HUMMING}` — is confirmed, and the runtime prints it itself
("out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL',
'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'HUMMING', 'EMULATION']").
Second, **five of those seven admit this device**, so the list is not empty and
the oracle has somewhere to land.

## 4. What the oracle resolves to

`select_nvfp4_moe_backend` with the W4A4 keys every NVFP4 MoE method passes
(`kNvfp4Static` × `kNvfp4Dynamic`) and a `FusedMoEConfig` carrying
GLM-5.3-Flash's MoE dimensions — 288 routed experts, top-8, hidden 4096,
`moe_intermediate_size` 2048, gated SiLU, DeepSeekV3 routing (`noaux_tc`),
bf16, no parallelism:

| case | result |
|---|---|
| `swiglu_limit=10.0`, `moe_backend='flashinfer_b12x'` | `ValueError` (§2) |
| `swiglu_limit=10.0`, `moe_backend='auto'` | **FLASHINFER_CUTLASS / FlashInferExperts** |
| `swiglu_limit=10.0`, `moe_backend='flashinfer_cutlass'` | FLASHINFER_CUTLASS / FlashInferExperts |
| `swiglu_limit=None`, `moe_backend='auto'` | FLASHINFER_CUTLASS / FlashInferExperts |
| `swiglu_limit=None`, `moe_backend='flashinfer_b12x'` | FLASHINFER_B12X / FlashInferB12xExperts |

Read rows 2 and 4 together: **the clamp does not change the answer on sm121.**
The two backends it removes from the auto list, TRTLLM and CUTEDSL, are
Blackwell-family-100 kernels that this device rejects anyway. Read rows 1 and 5
together: **b12x is fine on this device and the clamp is the only thing in the
way.**

**A methodological note that is part of the result.** The first run of this
probe picked `MoEActivation.SILU_NO_MUL` (its value string `silu_no_mul`
contains both "silu" and "mul") and every FlashInfer backend then reported "does
not support MoEActivation.SILU_NO_MUL activation", with `auto` falling through
to `VLLM_CUTLASS`. A plausible, wrong answer, from one wrong field in a
constructed config. The probe now asserts `activation.is_gated`, and this is
exactly why the §5 scope note is not boilerplate.

## 5. Scope: what this is not

The oracle leg's `FusedMoEConfig` is **constructed** from GLM-5.3-Flash's
`text_config`, not lifted off a live serve. What is established is the oracle's
own selection for those field values. What is not:

* weight loading and `process_weights_after_loading` for the chosen kernel;
* the "shape-specific fallbacks may still occur at runtime" the oracle's own
  docstring warns about;
* anything about generation quality, or about whether `FlashInferExperts`
  actually applies the clamp correctly on this shape;
* any other build. `487ecf187`, sm121, these fields.

So **#6 stays open and stays blocked**: `requires_serve_flags` for an NVFP4 MoE
cell still cannot be written, because the cell would claim a served route and no
NVFP4 MoE has been served here. What has changed is that the blocked-on-what is
now precise, and the flag a future serve should try is the runtime's own
recommendation rather than one copied from a note about a different config.

## 6. One prose claim this makes measurably false

`src/tessera/serving/config.py:284`, in the MoE refusal message, says NVFP4 W4A4
"needs `--moe-backend flashinfer_b12x` on GB10". §9.1 already flagged that as an
asserted runtime claim of the kind principle 14 forbids, and chose to correct it
"when the MoE route lands, not before, so that the correction and its evidence
travel together". The evidence now exists and points two ways at once: on any
GLM-5.3-Flash config that flag is **refused**, and on sm121 the route needs no
flag at all. The message is left as it is here — the route has not landed and
`tests/test_serving_dispatch.py:225` pins the string — and the correction is
tracked as **#31** so it cannot be lost between now and then.
