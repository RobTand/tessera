# #92: a uniformly-NVFP4 export declared `mixed-precision`

Branch `muse/ts-92-nvfp4fmt`, off master `82cdf51`.

## What the fix is

`tessera.stock.declared_format(config_groups)` derives the top-level
`quantization_config.format` from the groups that were actually written. Groups
that agree on one format make an artifact **of** that format and it says so;
groups that disagree make a mixed artifact and it still says `mixed-precision`,
which stays the honest label.

It keys on the set of distinct formats rather than on the number of `config_groups`
keys, because that is what both consumers in the runtime are asking: vLLM's
predicate asks what the weights are, not how many dictionary keys the exporter
chose to spend on them.

`tessera.stock.vllm_fp4_predicate(quant_method, format)` states what the
declaration resolves to — `vllm_is_nvfp4_quantized`, the reason, the consequence,
and the attestation of where the predicate was read. It lands on every manifest
the three call sites write. That is the second half of the issue's ask: a
genuinely mixed artifact gives up a fusion pass, and the loss is now a recorded
property of mixing rather than a silent side effect of a constant.

## The three call sites, which are not the same question

| Site | `quant_method` | What it now does |
|---|---|---|
| `export_stock_compressed.py` | `compressed-tensors` | derived, via `stock_quantization_config` |
| `export_tessera_serving.py` twin (was `:891`) | `compressed-tensors` | derived, via the same helper |
| `export_tessera_serving.py` route (was `:807`) | `tessera` | label kept, predicate recorded |

The issue described **both** `export_tessera_serving.py` sites as being under our
own plugin. Only the first is. `:891` writes the **stock twin** —
`"quant_method": "compressed-tensors"` — and that is the comparator artifact the
issue is about, so it carries the same defect and goes through the same helper.

The Tessera-route config keeps the generic label deliberately.
`is_nvfp4_quantized` requires `quantization == "compressed-tensors"`, which
`quant_method: "tessera"` is not, so its first conjunct fails whatever the string
says; and `tessera/serving/config.py` reads `config_groups` and never touches the
top-level field. Naming a format there would change nothing and claim something.
It records the predicate anyway, so the reason is on the artifact instead of in a
commit message.

Nothing quantized → no `quantization_config` at all, rather than one telling a
runtime to look for compressed tensors the checkpoint does not hold. The twin
already did this; the helper makes it the rule for both.

## What was read in the pinned runtime

Read in `vllm/vllm-openai:latest` on **sparky**, image id
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`,
reporting `vllm.__version__ == "0.28.0"`, at
`/usr/local/lib/python3.12/dist-packages/vllm`. Not taken from the issue text.

The image id is part of the reading, not decoration: the tag floats and the two
boxes are holding **different** images under it (filed as #100), so a record
naming only the tag would not say which build answered. Both are in the stamped
attestation.

**The predicate is where the issue says it is** —
`config/model.py:2096-2108`, verbatim:

```python
def is_nvfp4_quantized(self) -> bool:
    # ModelOpt NVFP4 checkpoints resolve to modelopt_fp4 quantization method
    if self.quantization in ("modelopt_fp4",):
        return True
    # For Compressed Tensors we look for `"format": "nvfp4-pack-quantized"`
    # in the quantization config
    quant_config = self.model_arch_config.quantization_config
    return (
        self.quantization == "compressed-tensors"
        and quant_config is not None
        and "nvfp4" in quant_config.get("format", "").lower()
    )
```

**It has exactly one consumer**, so the issue overstates the loss:

```
$ grep -rn "is_nvfp4_quantized" $V --include=*.py
config/vllm.py:143:        or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
config/model.py:2096:    def is_nvfp4_quantized(self) -> bool:
```

That is `enable_act_fusion` (`config/vllm.py:134-144`), i.e. the `fuse_act_quant`
pass-config entry at O1/O2/O3. The attention+quant half of the issue's claim does
**not** hold: `fuse_attn_quant` beside it is the module constant
`IS_QUANTIZED = False` (`config/vllm.py:113-120`, "currently set to False in all
cases", pending vllm#25689), so attention+quant fusion is off for every model in
0.28.0 — uniform-NVFP4 competitor included.

**On a default compiled serve the predicate is the only thing switching that
flag,** which is why the label mattered:

- `optimization_level` defaults to `O2` (`config/vllm.py:401`), whose pass config
  sets `fuse_act_quant: enable_act_fusion`.
- Under the default compiled backend, `custom_ops` resolves to `["none"]`
  (`config/vllm.py:1392-1399`), so `is_custom_op_enabled("silu_and_mul")` is
  False.
- `+quant_fp8` is appended only when `has_blocked_weights()`
  (`config/vllm.py:1368-1375`), which tests for `QuantizationStrategy.BLOCK`
  (`compressed_tensors.py:969-977`). The groups these exporters write declare
  `"strategy": "tensor_group"` (NVFP4, `export_stock_compressed.py:69,75`) and
  `"strategy": "channel"` (FP8, `:82`). Neither is BLOCK.

So both other disjuncts are False and the format string decides the flag.

**Scope: this is a compile-mode property, not a KL one.** `--enforce-eager` sets
compilation mode `NONE` (`config/vllm.py:1284-1290`) and no fusion pass runs on
either arm. The KL harnesses in `experiments/` serve eager by default
(`serve_and_dump_kl.sh:24`, `tessera_plugin_served.sh:50`), so this predicate
cannot have moved any KL number taken through those harnesses. (I did not sweep
for a compiled-mode stock-arm KL receipt; the compiled-mode lane receipts on
record are `quant_method: tessera`, where the predicate is False either way.) It
bites on a **speed**
comparison against a competitor checkpoint under a default compiled serve, which
is exactly the comparison the issue says is unpriced.

**The field is also read as a per-group default, and that leg is inert for us.**
`from_config` keeps it as `self.quant_format` (`compressed_tensors.py:250`), and
`get_scheme_dict` substitutes it only into a group that declared no format of its
own (`:963-964`). Every group both exporters write declares its own `format`, so
the substitution never fires: this change moves a compile flag, not a byte in
memory. Worth stating, because "derive the format" could otherwise read as a
change to what gets loaded.

**The pattern the flag would add is really built, so the difference is real.**
`SiluMulNvfp4QuantPattern` is registered only when
`silu_and_mul_nvfp4_quant_supported`
(`compilation/passes/fusion/act_quant_fusion.py:36-40, 298-299`), i.e. when
`torch.ops._C.silu_and_mul_nvfp4_quant` exists in the build. It does:

```
$ strings $V/_C_stable_libtorch.abi3.so | grep "silu_and_mul_nvfp4_quant("
silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, Tensor input, Tensor input_global_scale) -> ()
$ strings $V/_C_stable_libtorch.abi3.so | grep "cutlass_scaled_fp4_mm("   # positive control
cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b, ... ) -> ()
```

read out of the binary, with the op that serves NVFP4 on this image as the
positive control and a nonsense name (0 hits) as the negative.

### The `hasattr` reading is a trap, and it caught two of us

An earlier reading of **mine** reported the op absent, by
`hasattr(torch.ops._C, ...)` in a CPU container. Retracted. The coordinator then
independently attested the same "absent" on both boxes, by

    docker run --rm --entrypoint python3 vllm/vllm-openai:latest \
      -c "import torch; print(hasattr(torch.ops._C,'silu_and_mul_nvfp4_quant'))"

Run that command with a positive control and it answers itself:

```
silu_and_mul_nvfp4_quant False
cutlass_scaled_fp4_mm    False     <- the kernel that serves NVFP4 on this image
silu_and_mul             False
names registered under torch.ops._C: 7
```

`import torch` alone registers nothing of vLLM's; the ops come from
`vllm/_C_stable_libtorch.abi3.so`, and importing it without a GPU raises
`ImportError: libcuda.so.1`. So the method returns False for **every** name,
including ops we have served with. A probe that cannot distinguish a missing op
from a missing GPU is not evidence about the build.

This is why the binary reading above carries a positive and a negative control,
and it is the concrete reason principle 14 says attested, not asserted: the
runtime was never ambiguous, our instrument was.

**What is still not attested** is the per-SM guard. The same binary carries the
message `No compiled silu_and_mul nvfp4 quantization kernel for SM ` and an
`_sm1xxa` variant of the symbol, so whether the fused kernel exists for a given
target is a further question. The stamped record says so rather than implying the
pattern always lands.

That open question is not benign. `No compiled silu_and_mul nvfp4 quantization
kernel for SM ` is a `TORCH_CHECK` message, so the guard's failure mode is a
*raise at the first compiled forward*, not a silent slow path. This fix turns the
`fuse_act_quant` pass on for the stock twin, and every compiled-mode receipt of a
stock twin on record was taken while the twin declared `mixed-precision` — i.e.
with that pass off. No compiled serve has ever exercised a derived-format twin on
this image. **The twin's next compiled serve should go through
`experiments/serve_smoke_graph.sh` before any speed number is taken from it.** I
did not run that serve here (load; and one serve per box), so it is stated as the
gate it is, not as a completed check.

Sizing the effect is a throughput measurement I did not
take and was not asked for — and could not honestly have taken today, with sparky
at load 60 and 14 GB of 121 available.

## Tests

`tests/test_stock_declared_format.py`, 14 tests, offline, CPU-only.

- a one-group NVFP4 export declares `nvfp4-pack-quantized`, and a one-group FP8
  export declares `float-quantized`;
- two groups that **agree** are still uniform — the label describes the weights,
  not the exporter's bookkeeping;
- a genuinely mixed export declares `mixed-precision` **and** its record carries
  `vllm_is_nvfp4_quantized: False` with the `fuse_act_quant` consequence;
- the Tessera route fails the predicate on `quant_method`, and would fail it even
  if the format string said `nvfp4`;
- the recorded answer is re-derived by `eval`ing the **quoted vLLM expression**
  itself, so the implementation and the quote cannot drift apart silently;
- `stock_quantization_config` is exercised directly for the uniform, mixed and
  empty cases, loading `export_stock_compressed.py` by path the way
  `test_serving_export_gate.py` loads its exporter;
- a sweep over `experiments/**.py` fails on any reintroduced
  `"format": "mixed-precision"` literal — the rule, not today's call sites.

## Suite result

Run once, on **sparklina** (the quiet box), from a copy of this branch at
`/home/rob/tmp/ts92-suite`:

    PYTHONPATH=. python -m pytest -q -p no:randomly

RESULT_PLACEHOLDER

No full master baseline was computed: fifteen agents were running one
concurrently and that is what put sparky into swap. The question a baseline
answers — "was this already broken?" — is answered per failing file against a
pristine `82cdf51` checkout staged at `/home/rob/tmp/ts92-master`, which is
seconds rather than a second full suite.

## Off-task fixes on this branch

One, as its own commit so it reads and drops independently of the #92 work:

- `911f52f` — `git rm 63`, a zero-byte tracked file at the repo root swept into
  `705c040` by a whole-tree `git add`. Nothing references it.

## Issues filed

Both say why they were filed rather than fixed.

- **#100** — `vllm/vllm-openai:latest` is a floating tag and the two boxes hold
  **different images** under it (`61fc8a89…` on sparky, `89154ef0…` on
  sparklina), while the harnesses and docs call it the pinned runtime. Not
  fixed: which digest becomes the pin moves a default and could move served
  numbers, so it is Rob's call. Found while attesting this task's runtime claim,
  and it is the same shape as #92 one level down — an unrecorded difference
  between the arms of a comparison.
- **#98** — the stray `63`. Filed under the old brief, then **fixed** on this
  branch and the issue updated to say so.

Also commented on **#92** with the two corrections above (the `fuse_attn_quant`
overstatement, and `:891` being the stock twin rather than a plugin config).

## Artifacts on disk

None regenerated. The change is to what an export writes, not to any existing
export; no checkpoint was built and no bytes were moved. Scratch used:
`/home/rob/tmp/ts92/` (drafts) and `/home/rob/tmp/ts92-suite/` on sparklina (an
11 MB source copy for the test runs).
