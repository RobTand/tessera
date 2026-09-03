# Tessera serves itself: the wire's own vLLM plugin (2026-09-02)

**Result.** Tessera's serving machinery is now Tessera's.  `tessera.serving` is
an out-of-tree vLLM plugin -- an entry point in the `vllm.general_plugins`
group registering `quant_method: "tessera"` -- and it serves both routes the
Gridbook lane served: `TESSERA_NVFP4` (E2M1x2 wire -> NVFP4 tile, W4A4 on
`torch._scaled_mm` under `e2m1_group16_ue4m3_static`) and `TESSERA_FP8`
(E4M3/CHANNEL/window wire -> per-channel FP8 pair, W8A8 under
`fp8_per_token_dynamic`), in both residency modes, eager and compiled.  A serve
installs **one** package.  Nothing under `src/tessera` imports `gridbook`
(`tests/test_no_gridbook_import.py`, on the import graph, not the substring),
and the Gridbook lane is withdrawn at its contract v15.

The measurement is a **move**, not a new capability, so it is checked as one.
The three checkpoints served here are the **same inodes** the Gridbook lane
served -- `model.safetensors` hardlinked, not copied (e4m3 `11665664`, k2
`8523942`, mixed `11665707`, identical on both sides) -- and the only
difference between the two `config.json` files is
`quantization_config.quant_method: "gridbook" -> "tessera"` plus the new
`structure: "dense"` key on each scheme.  Every scheme value the decoders read
(`family`, `grid`, `body`, `plane`, `q256`, `rows`, `columns`, `roles`,
`wire_bytes`) is byte-for-byte the same dict.  Every arm is then compared
against that lane's own dump on those bytes.  The claim being tested is "the
arithmetic did not change".

It did not.  Eleven of the twelve served arms reproduce the Gridbook lane's
logits at **0.000000 mutual KL with 100% top-1**; the twelfth (NVFP4, resident,
compiled) reads 0.017591, and section 3 shows that to be inductor compiling one
graph twice into different kernels — the identical sources recompiled from an
empty cache land on **0.000000 / 100%** and differ from the chain's own build
by 0.017117.

---

## 1. What the plugin is

| piece | file | what it is |
|---|---|---|
| entry point | `pyproject.toml` `[project.entry-points."vllm.general_plugins"]` | `tessera = "tessera.serving:register"` |
| registration | `src/tessera/serving/__init__.py` | `register()` -> `register_quantization_config("tessera")(TesseraConfig)` |
| config / dispatch | `serving/config.py` | reads `config_groups` + `ignore`, refuses at parse, hands each module to a route |
| scheme grammar | `serving/scheme.py` | the per-module scheme dict: `family`, `grid`, `body`, `plane`, `q256`, `rows`, `columns`, `wire_bytes`, `roles` (+ `structure`) |
| route: NVFP4 | `serving/nvfp4_route.py` | span-2 TCQ wire -> NVFP4 tile + per-16 UE4M3 plane, `torch._scaled_mm` W4A4 |
| route: FP8 | `serving/fp8_route.py` | window wire over the CHANNEL plane -> per-channel FP8 pair, `torch._scaled_mm` W8A8 |
| resident decode | `serving/ops.py` | prepared planes, functional custom ops, the span-2 CUDA decode |
| streamed decode | `serving/window.py`, `serving/ops.py` | the packed planes held resident, decoded per forward |
| native extension | `serving/ext.py`, `serving/csrc/tessera_nvfp4.cu` | the JIT-built span-2 decoder, loaded at `process_weights_after_loading` |
| vLLM native ops | `serving/native_ops.py` | `scaled_fp4_quant` / `scaled_fp8_quant` through vLLM's own `torch.ops._C`, fail-closed when absent |
| residency flag | `serving/flags.py` | `TESSERA_SERVE_MODE=resident|streamed`, latched once per process |
| compile identity | `serving/compile_identity.py` | folds the mode into `additional_config` before any AOT hash |
| telemetry | `serving/telemetry.py` | `emit_route` / `route_shape`, the census's evidence |
| TP seam | `serving/sharding.py` | `plan_shard`, `_shard_unit_for_rank`, `shard_parsed_roles` (section 6) |
| the attested table | `serving/runtime_contract.json` | `tessera.runtime-contract.v1`, read via `importlib.resources` |
| census tool | `tools/tessera_route_census.py` | serves, walks the modules, reports what each one actually ran |

The streamed FP8 decode has **one** entry point --
`fp8_route.PreparedTesseraFp8Module.decode()`, which fans out to
`window.PreparedWindow.decode()` per role -- so the fused in-register GEMV
(`tessera.kernel_window`, in flight) swaps in at a single call site rather than
inside a forward.

The only flag is `TESSERA_SERVE_MODE`.  There is no enable switch: the bytes
select the plugin, because `quantization_config.quant_method` is `"tessera"`.

## 2. The plugin requirement is machine-readable

Principle 14 says a claim about a runtime is attested from a table, never from
prose.  A plugin requirement is exactly such a claim -- "this artifact needs
software vLLM does not ship" -- so it is a field, not a sentence.  Every
`lane_eligibility` cell in `serving/runtime_contract.json` carries

```
"route_status": "backed_with_serve_flag",
"requires_plugin": "tessera",
"requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"]
```

and both sides move together: `serving/contract.py` refuses a cell that omits
`requires_plugin` or names a different one, and PrismaQuant's preflight reads
the same field out of the installed Tessera (never a copy) before it will admit
a Tessera unit.  `tests/test_serving_contract.py` asserts the field exists on
every cell, that the value matches the registered `quant_method`, and that the
loader refuses a doctored cell.

Attested, from a GPU-less network-isolated container with **gridbook not
installed** and **no `GRIDBOOK_*` variable set**, replayable as
`/home/rob/tessera-runs/tsplugin/attestation.log`.  The contract is read
*before* anything imports vLLM, so the "torch imported: False" line is a fact
and not an ordering accident:

```
### entry points (vllm.general_plugins)
    lora_filesystem_resolver -> ...   lora_hf_hub_resolver -> ...
    tessera -> tessera.serving:register
### gridbook
WARNING: Package(s) not found: gridbook
    importable: False
### GRIDBOOK_* in env: 0
### packaged contract, read with no torch and no vllm
    tessera.runtime-contract.v1 contract_version 1 quant_method tessera
     format TESSERA_E2M1_K2 kind tessera_wire rungs [896]  modes ['resident', 'streamed']
     format TESSERA_E4M3_K1 kind tessera_wire rungs [1024] modes ['resident', 'streamed']
    tessera.lane-eligibility.v3 structures ['dense'] | cells 4
      tessera_e2m1_k2_dense_sm121_decode_scaled_mm_w4a4
         backed_with_serve_flag | plugin 'tessera' | ['TESSERA_SERVE_MODE=resident|streamed']
         | e2m1_group16_ue4m3_static | structure dense
      tessera_e2m1_k2_dense_sm121_batch_scaled_mm_w4a4    (same, batch regime)
      tessera_e4m3_k1_dense_sm121_decode_scaled_mm_w8a8   (fp8_per_token_dynamic)
      tessera_e4m3_k1_dense_sm121_batch_scaled_mm_w8a8    (same, batch regime)
    tensor_parallel: [('TESSERA_E2M1_K2', 1), ('TESSERA_E4M3_K1', 1)]
    expert_parallel units: [] (none: no served expert measurement)
    torch imported: False | vllm imported: False
### registration (imports vllm LAST, so the line above is honest)
    quant_method tessera -> tessera.serving.config.TesseraConfig
    gridbook in sys.modules: False
```

`structures: ["dense"]` is the honest scope, and it is a field so a gate can
read it: **there is no `routed_moe` cell, because no served MoE measurement
exists.**  A cell would be a claim about a runtime nobody has run.

## 3. Served: the census and the KL

**Route census** -- what each of the 112 modules actually ran, in both
regimes (`decode` M1, `prefill` M64; `M*` under the compiled forward),
read off `emit_route` telemetry rather than inferred:

| checkpoint | mode | regime | verdict | modules by family (decode) | contract(s) | decoder(s) | symbol |
|---|---|---|---|---|---|---|---|
| e4m3 | resident | eager | served | TESSERA_FP8=112 (other 0) | fp8_per_token_dynamic | torch_window | torch._scaled_mm |
| e4m3 | streamed | eager | served | TESSERA_FP8=112 (other 0) | fp8_per_token_dynamic | torch_window | torch._scaled_mm |
| e4m3 | resident | compiled | served | TESSERA_FP8=112 (other 0) | fp8_per_token_dynamic | torch_window | torch._scaled_mm |
| e4m3 | streamed | compiled | served | TESSERA_FP8=112 (other 0) | fp8_per_token_dynamic | torch_window | torch._scaled_mm |
| k2 | resident | eager | served | TESSERA_NVFP4=112 (other 0) | e2m1_group16_ue4m3_static | native_span2 | torch._scaled_mm |
| k2 | streamed | eager | served | TESSERA_NVFP4=112 (other 0) | e2m1_group16_ue4m3_static | native_span2 | torch._scaled_mm |
| k2 | resident | compiled | served | TESSERA_NVFP4=112 (other 0) | e2m1_group16_ue4m3_static | native_span2 | torch._scaled_mm |
| k2 | streamed | compiled | served | TESSERA_NVFP4=112 (other 0) | e2m1_group16_ue4m3_static | native_span2 | torch._scaled_mm |
| mixed | resident | eager | served | TESSERA_FP8=84, TESSERA_NVFP4=28 (other 0) | e2m1_group16_ue4m3_static, fp8_per_token_dynamic | native_span2, torch_window | torch._scaled_mm |
| mixed | streamed | eager | served | TESSERA_FP8=84, TESSERA_NVFP4=28 (other 0) | e2m1_group16_ue4m3_static, fp8_per_token_dynamic | native_span2, torch_window | torch._scaled_mm |
| mixed | resident | compiled | served | TESSERA_FP8=84, TESSERA_NVFP4=28 (other 0) | e2m1_group16_ue4m3_static, fp8_per_token_dynamic | native_span2, torch_window | torch._scaled_mm |
| mixed | streamed | compiled | served | TESSERA_FP8=84, TESSERA_NVFP4=28 (other 0) | e2m1_group16_ue4m3_static, fp8_per_token_dynamic | native_span2, torch_window | torch._scaled_mm |

**Mutual KL, plugin arm against the Gridbook-lane arm on the same bytes.**
`kl_tool compare` between the two dumps; 0.0000 with 100% top-1 means the
two runtimes produced the same logits.

| arm | reference dump (Gridbook lane) | mutual KL >= | confident | top-1 | plugin KL-vs-BF16 | gridbook KL-vs-BF16 |
|---|---|---|---|---|---|---|
| e8-resident | `qwen_gridbook_e8-resident` | 0.0000 | 0.0000 | 100.0% | 0.4660 | 0.4660 |
| e8-streamed | `qwen_gridbook_e8-streamed` | 0.0000 | 0.0000 | 100.0% | 0.4660 | 0.4660 |
| e8-resident-graph | `qwen_gridbook_e8-resident-graph` | 0.0000 | 0.0000 | 100.0% | 0.4669 | 0.4669 |
| e8-streamed-graph | `qwen_gridbook_e8-streamed-graph` | 0.0000 | 0.0000 | 100.0% | 0.4669 | 0.4669 |
| k2-resident | `qwen_gridbook_k2-resident-v14` | 0.0000 | 0.0000 | 100.0% | 0.6316 | 0.6316 |
| k2-streamed | `qwen_gridbook_k2-streamed` | 0.0000 | 0.0000 | 100.0% | 0.6316 | 0.6316 |
| k2-resident-graph | `qwen_gridbook_k2-resident-graph` | 0.0176 | 0.0129 | 95.6% | 0.6279 | 0.6271 |
| k2-streamed-graph | `qwen_gridbook_k2-streamed-graph` | 0.0000 | 0.0000 | 100.0% | 0.6271 | 0.6271 |
| mixed-resident | `qwen_gridbook_mixed-resident` | 0.0000 | 0.0000 | 100.0% | 0.6772 | 0.6772 |
| mixed-streamed | `qwen_gridbook_mixed-streamed` | 0.0000 | 0.0000 | 100.0% | 0.6772 | 0.6772 |
| mixed-resident-graph | `qwen_gridbook_mixed-resident-graph` | 0.0000 | 0.0000 | 100.0% | 0.6733 | 0.6733 |
| mixed-streamed-graph | `qwen_gridbook_mixed-streamed-graph` | 0.0000 | 0.0000 | 100.0% | 0.6733 | 0.6733 |

**The two residency modes against each other**, inside one runtime -- the
control that says whether a nonzero number is about the move or about the
compiled forward:

| checkpoint | regime | runtime | resident vs streamed KL >= | top-1 |
|---|---|---|---|---|
| E4M3 (FP8 route) | eager | Tessera plugin | 0.000000 | 100.00% |
| mixed | eager | Tessera plugin | 0.000000 | 100.00% |
| K2 (NVFP4 route) | eager | Tessera plugin | 0.000000 | 100.00% |
| K2 (NVFP4 route) | compiled | Tessera plugin | 0.017117 | 95.65% |
| K2 (NVFP4 route) | compiled | Gridbook lane | 0.000000 | 100.00% |

Both `qwen_gridbook_k2-resident` (07:01Z) and `qwen_gridbook_k2-resident-v14` (11:18Z) are dumps of the SAME Gridbook-lane artifact (`gbfam/qwen3-0.6b-tessera-k2-gridbook`) under the same corpus and tokenizer contract; the `-v14` one is the re-measure after Gridbook's contract bump added the K2 row, so it is the one this table compares against.  They agree to six decimals (0.631553 / 0.544852 / 58.41%), so the choice is immaterial -- it is named here only so a reviewer does not have to guess which of the two it was.

The reference for "is a nonzero difference a kernel or a bug" is the *lane's
own* spread between its eager and compiled forwards on identical weights,
measured before the move: **0.0269** (top-1 90.3%) on the E4M3 arm, **0.118**
(81.0%) on mixed, **0.2445** (70.0%) on the E2M1 arm
(`/home/rob/tessera-runs/gbfam/kl_*_eager-vs-graph.json`).  A mutual KL of 0
needs no such allowance and is what the move should produce, because the two
runtimes call the same `torch._scaled_mm` on the same decoded bytes.

### The one arm that is not zero, and what it turned out to be

Eleven of the twelve arms are **0.000000 with 100% top-1**.  One is not:
`k2-resident-graph`, the NVFP4 route in resident mode under the compiled
forward, at **0.017591 / 95.65%**.  That number is not accepted on the
"inside the floor" allowance; it was chased until it had a cause.

| comparison | KL >= | top-1 |
|---|---|---|
| plugin vs Gridbook, K2 **eager** resident | 0.000000 | 100.00% |
| plugin vs Gridbook, K2 **eager** streamed | 0.000000 | 100.00% |
| plugin vs Gridbook, K2 **compiled** streamed | 0.000000 | 100.00% |
| plugin vs Gridbook, K2 **compiled** resident *(the arm)* | 0.017591 | 95.65% |
| &nbsp;&nbsp;that arm's artifact, replayed by a second serve | 0.000000 | 100.00% |
| &nbsp;&nbsp;the replay vs Gridbook | 0.017591 | 95.65% |
| &nbsp;&nbsp;**same sources recompiled in an empty cache, vs Gridbook** | 0.000000 | 100.00% |
| &nbsp;&nbsp;**that rebuild vs the chain's build** | 0.017117 | 95.65% |

Read down that table.  The two runtimes agree exactly in **eager** mode in both
residencies, so the wire, the load path, the decode and the materialised bytes
are identical — the difference exists only under `torch.compile`.  Serving the
same artifact a second time reproduces the arm exactly (0.000000), but that
serve loaded the *same* AOT artifact (`15957ad9…`), so it proves only that one
compiled binary is deterministic.  The discriminating run is the last two rows:
the identical sources, recompiled from an **empty** compile cache into the same
content-addressed key, produce logits that match the Gridbook lane **exactly**
(0.000000 / 100%) and differ from the chain's own build by 0.017117.

That rebuild took two attempts and the second one is what served, so its
provenance is worth spelling out — the surviving serve log says
`fresh_compiles=0`, and a reader who greps only that will think the "fresh"
build was a cache load.  It was not.  `fresh_k2rg.sh` emptied
`vllm-cache-fresh/` at 15:43:19; **attempt 1 compiled into it** (`Using cache
directory: /root/.cache/vllm/torch_compile_cache/36d07e6697/rank_0_0/backbone`,
`Dynamo bytecode transform time: 4.83 s`, 19:44:04 UTC) and was then killed
mid-profile by another worker releasing GPU memory; attempt 2 served that
build.  The filesystem is the durable record: the cache dir was created
15:43:59 local, `torch_compile_cache/36d07e6697/` at **15:44:04**, and the
artifact `torch_aot_compile/15957ad9…/` was written at **15:47:30**, four
seconds before attempt 1's assertion.  (The script overwrote its per-attempt
log; it now keeps them.)  The key matches the chain's build because torch's AOT
key is derived from the graph and the config, not from the compiled output —
which is the point: **same key, same inputs, different kernels.**

So the residual is a property of **the build, not of the code**: inductor
compiled one graph twice and did not land on the same kernels.  This engine
config runs `benchmark_combo_kernel: True` (logged in every arm's
`compilation_config`), i.e. a compile-time benchmark whose outcome depends on
what else the box is doing — three other workers were loading this box during
that stage, and the rebuild's own first attempt was killed mid-profile by one
of them.  That is the likely mechanism and it is not proven here; what *is*
measured is that two builds of one graph differ and one of them is bit-equal to
the reference lane.

The magnitude corroborates it.  Compilation moves this arm by **0.2442**
(plugin) and **0.2445** (Gridbook) against its own eager dump — the two
runtimes reproduce *each other's* compile-induced deviation to three digits,
and the build-to-build residual is 14× smaller than that deviation.  On the
E4M3 arm both runtimes' compiled builds deviate from eager by **0.026861**, the
same number to six digits, and agree with each other at 0.000000: there the two
builds happened to make the same choice.

| the size of the effect compilation itself has on this arm | KL >= | top-1 |
|---|---|---|
| Gridbook lane, its own eager vs compiled (K2) | 0.244481 | 70.03% |
| Tessera plugin, its own eager vs compiled (K2) | 0.244223 | 70.57% |
| Gridbook lane, its own eager vs compiled (E4M3) | 0.026861 | 90.34% |
| Tessera plugin, its own eager vs compiled (E4M3) | 0.026861 | 90.34% |

The chain's number is reported as measured rather than replaced by the
rebuild's nicer one.  Picking the build that reads 0.000000 would be choosing
the measurement, and the honest statement is the pair: **the move is exact in
every arm; one compiled build of one arm is not reproducible to the kernel, and
the rebuild is exact.**

### The compile-cache identity, checked by construction

All twelve censuses ran against **one shared** `~/.cache/vllm` (the chain mounts
`$R/vllm-cache` into every container), and inside each container the order is
eager resident, eager streamed, **compiled streamed, then compiled resident** --
the second-mode-after-first case that crashed before the residency became part
of the compile identity.

vLLM keys its compile artifacts by `VllmConfig.compute_hash()` plus the contents
of the traced source files, and neither sees a residency mode: both modes trace
the same files into different forwards, so both would land on one AOT key and
the second process would load the first's bytecode with guards disabled.
`serving/compile_identity.py` folds the mode into `additional_config`, the one
hash input a plugin can reach.  Demonstrated directly rather than inferred from
a log line:

```
resident  additional_config = {"tessera": {"mode": "resident", "version": "0.1.0"}}
streamed  additional_config = {"tessera": {"mode": "streamed", "version": "0.1.0"}}
distinct: True
```

The version is in the record too, so a plugin bump re-keys both modes at once;
vLLM's own traced-file checksum still covers code changes within one mode.

The hook separates the **vLLM** cache directory, which is what it is for.  The
inner `torch_aot_compile` key is a different thing — torch derives it from the
FX graph and the compile config, and it does not see `additional_config` — so
the artifact keys below are evidence about the *graphs*, not about the hook:

| checkpoint | resident | streamed |
|---|---|---|
| E4M3 (FP8 route) | `7d3adebc…` | `f4feff5b…` |
| K2 (NVFP4 route) | `15957ad9…` | `dabbb7b6…` |
| mixed | `b89adc82…` | `d87cef52…` |

Six arms, six distinct keys: under this plugin the two residency modes never
present dynamo with the same graph, so they can never share a compiled forward
whatever the cache does.  The Gridbook lane's two K2 modes did the opposite —
**distinct** vLLM cache directories (`e9c6e8fc7c`, `f7e1257c69`: its own
mode-in-config mechanism working) but **one** AOT artifact (`ff23575d…`,
`gbfam/serve_qwen_gridbook_k2-*-graph.log`), i.e. one graph serving both
modes.  *Why* the two lanes trace differently is not established here and this
receipt does not guess; the fact is recorded because section 3's one non-zero
arm turns on it.

## 4. MoE: designed, refused, not served

Rob asked for MoE support "everywhere", and everywhere here means every place
that would otherwise pass an expert stack through in silence.  Three seams
exist and one cell deliberately does not.

**Dispatch.** `config.py::get_quant_method` matches vLLM's routed-experts layer
(`RoutedExperts`, or `FusedMoE` on older builds, by class *and* by name so a
rename cannot slip past) before it matches `LinearBase`.  For a prefix the
checkpoint named in `ignore` it returns `None` -- vLLM's own
`UnquantizedFusedMoEMethod`, which is the correct answer for experts the
producer declared BF16.  For anything else it **raises**, because `None` there
would serve uninitialised or BF16 expert memory while the artifact claimed to
be quantized.  The message names what the route will do when it lands.

**Export.** `experiments/export_tessera_serving.py::quantizable` returns
`(shards, shapes, expert_shapes)` -- 2-D `.weight` and rank-3+ `.weight`
separately -- because dropping the second kind silently is how an MoE
checkpoint exports as "fully quantized" with BF16 experts and nothing says so.
A plan that names a packed expert tensor is refused by name and shape.  A plan
that does not gets those modules written into `ignore`, printed, and the plugin
then serves them unquantized rather than refusing the layer.  When the expert
route lands the work is: one unit per expert per role, one `tessera.fused`
container per vLLM MoE module, `structure: "routed_moe"` on the scheme, decode
to the stock packed expert layouts (NVFP4 needs `--moe-backend
flashinfer_b12x` on GB10; FP8 is the compressed-tensors MoE path).  The rule
that a fused module's roles share ONE family holds for experts unchanged.

**Contract.** No `routed_moe` cell, and `structures: ["dense"]` says so in a
field.  PrismaQuant's routed-MoE admission cell is untouched by this change.

## 5. A third family costs one route module, one table entry, contract rows

`TESSERA_BF16` is being built (the same WINDOW body and CHANNEL plane as the
FP8 route, its 2^L alphabet snapped to bf16 instead of E4M3, decoded to a bf16
tile for the stock GEMM, W16A16 -- above ~6 bpp it replaces the 8-bit route).
The route is not built here; what is built is the boundary it plugs into, and
two places that would otherwise have blocked it are now closed:

* **The window table carries the family's dtype, not the decoder's.**
  `prepare_window` used to do `table.to(torch.uint8)` unconditionally, which
  would truncate a bf16 value table to zeros and ones -- silently, since the
  gather succeeds either way.  An **integral** table still narrows to uint8
  (grid codes; the E4M3 route folds its uint8 `code_map` of native bytes in and
  decodes to E4M3 bytes), a **floating** table is kept as it is, and a
  `code_map` on a value table is refused rather than applied, because a code map
  remaps codes and a value table has none.  `decode()` returns the table's
  dtype.  Byte-identical for both shipped families -- the FP8 route's own test
  decodes every role and compares against `tessera.decode.materialize_fp8` byte
  for byte, and 70 route tests pass with CUDA and the native extension.
* **FAMILY = ROUTE is a table, not an if-chain.**  `TESSERA_FAMILIES` is now
  `tuple(ROUTES)` rather than a hand-written tuple, and each `ROUTES` entry
  names its `builder` as `(module, function)` which `lane.build_tessera_method`
  imports lazily.  Adding a family no longer means editing a constant AND a
  dispatch branch AND the route table.  `tests/test_serving_dispatch.py`
  registers a fake family at runtime and asserts the dispatcher finds it with
  no other edit, and that an unknown family is still refused by name.

What still costs a line per family, by design: the contract's `formats[]` row,
its `lane_eligibility` cells and `_FAMILY_TO_ROUTE`.  Those are *claims about a
runtime* and must be written and reviewed one at a time (principle 14) -- there
is deliberately no code path that invents a cell for a family nobody has
served.

## 6. Tensor parallelism: the loaders are built for it, TP>1 refuses here

The artifact is **TP-agnostic** and stays so: a unit is encoded once, whole,
and the exporter never learns the TP degree.  A rank takes its slice at *load*.
That is the only design that keeps one checkpoint servable at any world size,
and it is why `serving/sharding.py` exists.

The cutter itself is built and verified on branch
`worktree-agent-a5d4cf4818e8e77ba` (design: `docs/design/tensor-parallel.md`
there), not merged at the time of this measurement.  This plugin targets its
names exactly:

* `layout.slice_unit(unit, rows=(r0, r1), cols=(c0, c1))` returns a
  **standalone** unit that decodes, through the same `tessera.decode` entry
  points, to precisely `decode(parent)[r0:r1, c0:c1]` -- bit for bit, no
  re-encoding.  Schema minor 4 adds the INITIAL_STATE plane; `row_offset = 0`
  is byte-identical to today's wire.
* `layout.shard_granularity(x)` and `layout.can_shard(x, tp, axis)` accept an
  `EncodedUnit`, a `ParsedUnit` or a bare `Manifest`, so a loader gates on the
  parse it already holds.  `can_shard` is asked **first** and is binding:
  granularity is necessary, not sufficient -- a column cut of a unit with a
  RELEASE plane or a mixed rate schedule is confined to whole 256-column
  superblocks, and only `can_shard` knows that.
* The axis vocabulary is `"row"` / `"column"`, `tessera.layout`'s own.  These
  strings are passed straight through, so `sharding.AXIS_ROWS == "row"` is
  asserted by test: a near-miss would be a `GrammarError` at load on the one
  configuration nobody serves yet.

What the plugin does with them:

* `plan_shard(prefix, rows, columns, out_size, in_size)` **derives the axis
  from the sizes vLLM asks for** -- `out_size * tp == rows` is a row split
  (ColumnParallel / MergedColumnParallel / QKVParallel), `in_size * tp ==
  columns` is a column split (RowParallel) -- never from a class name.  A
  Linear replicated inside a TP group asks for the whole shape and gets a whole
  plan.  Anything else refuses naming both shapes.
* `_shard_unit_for_rank(unit, tp_rank, tp_size, axis)` is the seam: identity at
  `tp_size == 1` (asserted by object identity, so a TP-capable build serves
  byte-identical bytes to one without it), and above one it calls `slice_unit`
  for **this rank's shard only**, which is what makes the cost O(1) in the TP
  degree.
* `runtime_contract.json` publishes `tensor_parallel.max_world_size: [1]` per
  family, and `config._require_tp1` refuses a larger world at method
  construction.  The refusal is a field and a gate, not prose.

  > **Superseded 2026-09-02 (#7), later the same day.**  The cutter merged, and
  > this gate went on refusing every world above one with a message saying the
  > cutter was absent — narrower than the code, and by then false.  It is now
  > `config._require_a_cutter` (refuses only where `layout.slice_unit` is
  > *missing*) plus `sharding.require_axis_supported` at `create_weights`
  > (refuses the one axis a route cannot start: `TESSERA_NVFP4` on rows).
  > `max_world_size` is **unchanged at 1** and keeps its meaning — the largest
  > world size a served receipt covers — because no multi-rank serve has been
  > run; what the contract adds beside it is `loader_axes`, the per-axis
  > statement of what the loader does, checked against
  > `sharding.ROUTE_TP_AXES`.  Everything measured in this document is at
  > `tp_size == 1` and is unaffected.

**The two families differ, and the plugin now says so correctly.**  This is the
one place an earlier draft of this work had it backwards:

* The **window body** -- the shipping E4M3 wire -- needs **no decoder change at
  all**.  `lane_planes.pack_window_planes` already prepends `L` pad bits per
  column and *the pad is* `state_{-1}`: writing the shard's stored state there
  instead of zeros makes the first window read, at bit `(t+1)*R` for `t = 0`,
  produce `(init << R | bits_0) mod 2^L`, which is the recursion's own first
  step.  So `window._pack` **threads** the state into the packing and
  `PreparedWindow.decode()` is untouched.  Refusing a start state there --
  which an earlier draft did -- would have refused the one lane the TP design
  says is already done.  What is refused instead is narrow and honest: an
  *installed* `pack_window_planes` that predates the `initial_state` parameter,
  because packing a shard against a zero pad decodes to plausible wrong weights
  in silence.  That refusal is a **pre-merge** condition only, and it was
  checked rather than assumed: on the branch the signature is
  `pack_window_planes(body_bits, rates, window_bits, initial_state=None)`, so
  once the slicer lands the thread succeeds and the refusal is unreachable
  (`git show worktree-agent-a5d4cf4818e8e77ba:src/tessera/lane_planes.py`).

**The pad-threading claim is measured, not argued.**  "The pad is `state_{-1}`"
is arithmetic about the wire, so it was run rather than reasoned.  The branch's
`src/tessera` was overlaid with this plugin's `serving/` package -- the
post-merge tree -- and a whole window unit's decode from row `t0` down was
compared against the decode of its rows `t0:` packed with the parent's state at
the cut, with an identity table so the decoded value *is* the raw `L`-bit
window and an error cannot hide inside an alphabet:

```
pack_window_planes signature: (body_bits, rates, window_bits, initial_state=None)
  t0=  1  shard rows=(23, 5)  == parent[1:] : True
  t0=  3  shard rows=(21, 5)  == parent[3:] : True
  t0=  6  shard rows=(18, 5)  == parent[6:] : True
  t0=  7  shard rows=(17, 5)  == parent[7:] : True
  t0= 12  shard rows=(12, 5)  == parent[12:] : True
  t0= 17  shard rows=(7, 5)  == parent[17:] : True
  control (zero pad at t0=7) differs from parent slice: True
THREADING OK
```

Six cut points, including `t0 < L` where the state is partly the pinned zero,
plus the negative control that says the match is not free.  The check ships as
`tests/test_serving_sharding.py::test_the_pad_really_is_state_minus_one_once_the_slicer_lands`,
which **skips** on a build whose `lane_planes` predates the parameter and
starts asserting the moment the slicer merges -- so the claim in this section
is guarded by the suite from then on rather than by this paragraph.
* The **span-2 trellis lane** (NVFP4) **refuses a shard**, matching upstream:
  its `SELECT_PAD` is the same opportunity, but the eight pad bits feed a
  window whose bit order `build_span2_luts` reverses, and threading a state
  through that reversal is unwritten and untested.  `layout.pack_unit_for_kernel`
  fails closed on the producer side naming the row offset;
  `ops.PreparedTesseraModule._require_no_initial_state` is the serving side of
  the same refusal.

## 7. What Gridbook gave up

`gridbook` on branch `tessera/family` (`6810060`, "Withdraw the Tessera lane:
it serves itself now") removes the lane modules, the CUDA source, the dispatch,
the `GRIDBOOK_TESSERA[_MODE]` flag pair, the `TESSERA_E2M1_K2` /
`TESSERA_E4M3_K1` format rows, their four `sm_121` `lane_eligibility` cells,
their TP units and their tests, and bumps the contract to **v15** with a
changelog entry.  v13 and v14 were never released, so this is a withdrawal, not
a break.  `TCQ_E2M1_R256` / `TCQ_E4M3_R256` and the `trellis_*` modules stay --
they are Gridbook's, not Tessera's.  The compile-cache, Dynamo-tracing and
data-pointer findings the lane produced stay in Gridbook's own modules,
attributed to the measurement rather than to the lane.

Verified independently of the report rather than accepted from it:
`grep -rniIl tessera` over `gridbook/`, `tests/` and `tools/` returns **zero
files**; the packaged contract reads `contract_version 15` with families
`['NVFP4_CB_K', 'FP8_CB_K', 'TCQ_E2M1_R256', 'TCQ_E4M3_R256']` and **no**
Tessera cell; and the only remaining mentions anywhere are `CHANGELOG.md`,
`docs/TESSERA-LANE.md`, and two history pointers *to* that file in
`docs/PLUGIN.md` and `docs/TRELLIS-R256-RESEARCH.md`.

Its suite, in the pinned image on a GB10 with the GPU free: **2510 passed, 513
skipped, 1 failed**.  The one failure is
`test_bench_serve.py::test_generated_report_is_the_only_checkout_dirty_check_exclusion`
raising `FileNotFoundError: 'git'` -- **the vLLM image ships no `git` binary**
(`which git` in the image returns nothing).  That whole file is **80 passed** on
the host, where `git` exists.  It is unrelated to the withdrawal.

## 8. PrismaQuant admits it, and admits nothing yet

`prismaquant` `d263f54` on branch `tessera/decouple-gridbook` ("Admit Tessera
through Tessera's own plugin pin, fail-closed until a release") adds a Tessera
runtime pin beside the Gridbook one
(`prismaquant/tessera_serving_runtime_pin.json`, no wheel digest -- Tessera
publishes no wheel, and asserting a digest for an archive that does not exist
is the hand-assertion principle 14 refuses), makes
`gridbook_lane_eligibility.py` a **two-vendor parser** rather than growing a
second copy of the v3 cell grammar, derives
`tessera_render.tessera_lane_attested()` from Tessera's packaged contract via
`importlib.resources`, adds `prismaquant/lane_specs/tessera.json`, and updates
`AGENTS.md` (four sanctioned lanes) and `docs/ARCHITECTURE.md` in the same
commit.

Verified here rather than accepted from a report:

```
$ PYTHONPATH=.:/home/rob/tessera/src python -c "from prismaquant import tessera_render as tr; ..."
contract path: /home/rob/tessera/src/tessera/serving/runtime_contract.json
families published: ['TESSERA_E2M1_K2', 'TESSERA_E4M3_K1']   cells: 4
TESSERA_E2M1_K2_R896  -> attested False
TESSERA_E4M3_K1_R1024 -> attested False
release pin satisfied: False
$ pytest tests/test_docs_staleness.py tests/test_architecture_doc.py
    20 passed
$ pytest tests/test_tessera_{lane_admission,footprint,formats,shape_dependent_recipe}.py \
         tests/test_gridbook_{serving_runtime_pin,runtime_pin,runtime_contract,attestation_interop}.py \
         tests/test_lane_spec_and_gguf_kl.py tests/test_serving_lane_metadata.py \
         tests/test_cb_lane_sharding.py
    220 passed, 12 skipped
```

Both re-run at commit time against the Tessera tree this commit carries, since
`tessera_lane_attested` reads the **installed** package: a PrismaQuant test
that passed against an older `runtime_contract.json` would certify nothing
about these bytes.

The agent's own run of PrismaQuant's full suite: 5719 passed, 144 skipped,
3 xfailed (log `/home/rob/tmp/pq_tessera_lane_fullsuite.log`).

The table parses and the lane is still **closed**: there is no Tessera release
tag, the pin carries `PENDING_TESSERA_RELEASE_*` sentinels, and
`require_exact_tessera_runtime_release` refuses them.  Cutting the tag is
outward-facing and is Rob's call.  PrismaQuant's routed-MoE admission cell is
untouched by the commit (the diff mentions `routed_moe` only in a docstring).

## 9. Tests

| where | how | result |
|---|---|---|
| GB10 (sparklina), GPU free, `nvcc` + `ninja`, native extension built | `pytest -q tests/` | **663 passed, 2 skipped, 0 failed** (13m23s) |
| GB10 under load, `CUDA_VISIBLE_DEVICES=""` | `pytest -q tests/` | 480 passed, 169 skipped, 4 failed |
| serving routes only, GPU free | `pytest tests/test_serving_{nvfp4_route,fp8_route,window,dispatch}.py` | 70 passed |
| Gridbook `tessera/family`, pinned image, GPU free | `pytest -q tests/` | 2510 passed, 513 skipped, 1 failed (no `git` in the image; that file is 80 passed on the host) |

The top row is the certifying run, on the tree this commit carries: it was
restarted from scratch after the pad-threading test was added, and the four
files that had changed since the earlier rsync were checked byte-identical
across the two boxes first, because a suite run on a stale copy certifies a
tree nobody is committing.  The count moved 660 -> 663 passed and 1 -> 2
skipped for a reason that adds up: one obsolete refusal test was replaced by
four (section 6), and the pad-threading test skips until the slicer merges.

The middle row's four failures are `tests/test_compensate.py` -- another
worker's file, which hardcodes `device="cuda"` with no skip guard, so hiding the
GPU fails it rather than skipping it.  Run with the GPU visible it is **7
passed**.  The one skip in the top row is the compile-identity test that
imports `vllm`, which is not in the host venv (it is exercised in the container
by all twelve censuses).

Two environment traps worth recording, because both LOOK like lane failures and
neither is:

* **`ninja` is in the venv's `bin`, not on a non-login ssh `PATH`.**  Without it
  `torch.utils.cpp_extension` cannot build `tessera_nvfp4.cu`, and three
  streamed-NVFP4 tests failed with `NativeKernelUnavailableError` on a box that
  has both a GPU and `nvcc`.  Exporting the venv's `bin` fixed all three with no
  code change.
* **A second GB10's `docker image inspect ... {{.Id}}` differs while the image
  is the same.**  Compare `{{.RepoDigests}}`: both boxes carry
  `vllm/vllm-openai@sha256:61fc8a89…`.  The local Id differs because docker 29
  stores images in containerd there.  Ranking two boxes as "different runtimes"
  off the local Id would have been wrong.



## 10. Reproduce

```bash
# route census, one arm
/home/rob/tessera/experiments/tessera_plugin_run.sh -- \
  "TESSERA_SERVE_MODE=resident python3 /work/tools/tessera_route_census.py \
     /home/rob/tessera-runs/tsplugin/qwen3-0.6b-tessera-e4m3 out.json \
     --expect-modules 112 --gpu-memory-utilization 0.30"

# the whole matrix (12 censuses, 12 KL serves, 14 comparisons)
/home/rob/tessera-runs/tsplugin/chain_tsplugin.sh

# the tables above
python3 /home/rob/tessera-runs/tsplugin/summarise.py

# the one non-zero arm: replay its artifact, then recompile it from an empty cache
/home/rob/tessera-runs/tsplugin/rep_k2rg.sh      # same artifact, second serve
/home/rob/tessera-runs/tsplugin/fresh_k2rg.sh    # same sources, VLLM_CACHE= empty dir
python3 /home/rob/tessera-runs/tsplugin/spread_table.py
```

One caveat the fingerprint does not cover.  `$R/vllm-cache` was **shared across
all three passes** and never wiped, so two compiled arms in the reported pass
(`e8-resident-graph`, key `7d3adebc…`, and `k2-streamed-graph`, `dabbb7b6…`)
loaded an artifact an earlier pass had compiled rather than compiling one.  A
hit means the traced sources hashed identically, so the *source* is the
reported tree either way — but section 3 has just shown that one key can hold
either of two builds, so the honest statement is that those two arms executed a
same-source build from an earlier pass.  Both matched the Gridbook lane at
0.000000 / 100%.

The measured tree is fingerprinted at
`/home/rob/tessera-runs/tsplugin/tree_at_measure.sha256` (22 files: the plugin
package, `tools/`, `pyproject.toml` and the two serve scripts), taken at chain
launch and re-verified before the commit -- the containers read the source
through a live read-only bind mount, so "measured at commit X" is otherwise
unprovable.  The census JSONs label themselves `3d419e7+plugin`: HEAD `3d419e7`
plus the then-uncommitted plugin tree whose hashes that file carries.

**Two earlier passes were discarded rather than reported.**  The first run of
the twelve censuses completed on an older tree; three changes then landed --
the window table's dtype and the route-table dispatch (section 5), and a fix to
the serve harness -- so it went to `preTP_censuses_superseded/`.  The second run
was stopped **mid-stage-B**, with five KL arms already on disk, when the TP
branch's real API turned out to differ from what this plugin had been written
against (the axis vocabulary, and the window lane threading a start state
rather than refusing it -- section 6); it went to
`preTP_censuses_superseded_v2/`, KL summaries included, and the whole matrix was
re-run on the tree the commit carries.  Neither was salvaged in part.  The
changes between run 2 and run 3 are provably inert at `tp_size == 1` -- the
seam returns the same object, and `_pack` with `initial_state=None` calls the
packer exactly as before -- but "provably inert" is an argument and the
fingerprint is a fact, so the arms were re-measured rather than relabelled.
Mixing them would have been a measurement of two code states wearing one
label.  The harness bug is worth naming because it is a shell trap,
not a lane fault: `--gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}"`
sat inside a single-quoted `bash -c` string, so it was expanded by the
*container's* shell where that variable was never passed, and the serve asked
for 0.85 of a box with 91 GiB free and refused to start while the host had
exported 0.30.  The value is now passed with `-e`, and the chain's retry
recognises that refusal as the box collision it is.

The retry then had a bug of its own, and it cost an arm.  It grepped
`serve_<arm>.out`, which holds only the last forty lines of the serve log,
while the `Error in memory profiling` assertion sits about a hundred lines up:
a box collision on `k2-streamed-graph` read as a lane failure and the arm was
dropped from the twelve.  It now greps the **full** serve log for both refusal
signatures.  The dropped arm was re-served and passed on its first attempt
(0.000000 / 100%), and the same retry correctly absorbed a real collision
during the rebuild in section 3 (`Initial free memory 107.51 GiB, current free
memory 111.07 GiB` — another worker releasing GPU memory mid-profile), which is
the shared-box condition this whole matrix ran under.
