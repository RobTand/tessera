# What the pinned build's expert loader does with a Tessera wire parameter (2026-09-03)

Issue #5 records, as a premise for the expert route, that
`RoutedExperts.build_expert_params_mapping` is suffix-agnostic
(`experts.w13_` / `w2_`) "so custom suffixes route fine". Half of that is
right, and the half that is wrong is the half that costs an export: the
**mapping** routes a custom suffix, and the **loader** then drops it, returning
`False` and writing nothing. No exception says so.

Asked of the runtime rather than read off it, because it is a claim about
another runtime (principle 14).

**Result.**

| question | answer, from the pinned build on this box |
|---|---|
| do the layer's parameter prefixes carry a suffix? | **yes** — `experts.w13_`, `experts.w2_`; the suffix is whatever the checkpoint tensor ends in |
| does `...experts.0.gate_proj.wire` reach a mapping row? | **yes** — `w13_wire`, shard `w1`, expert 0 |
| what `weight_name` does the loader see? | `model...mlp.experts.w13_wire` — contains **neither** `weight` **nor** `scale` |
| what does `RoutedExperts.weight_loader` do with it? | returns **`False`**, parameter **not written** — for `w13_wire`, `w2_wire` and `w13_wire_len` alike |
| control, same call, one substring different | `w13_weight` and `w2_weight` return **`True`** and **are written** |
| is there a path that does load it? | **yes** — a parameter carrying its own `weight_loader` is the one `load_weights` calls: it is invoked, the name is yielded, the parameter is written |

Harness: `experiments/moe_wire_loader_probe.py`, run through
`experiments/moe_wire_loader_probe.sh`. Result:
`experiments/results/moe_wire_loader_probe.json`. Build
`prismaquant/glm53-mia-sm121:487ecf187` = `vllm 0.1.dev20051+g487ecf187`,
torch 2.13.0+cu130. **No GPU and no serve lock**: the probe imports
`routed_experts` and calls two of its functions, so it creates no CUDA context
and takes nothing a serve wants.

---

## 1. Where the suffix's fate is decided

`load_weights` rewrites the checkpoint name before it calls the loader —
`weight_name = qual_name.replace(weight_name, param_name)` — so what
`weight_loader` receives is not `...experts.0.gate_proj.wire` but
`...experts.w13_wire`. Its dispatch is a chain of substring tests on that
string, and the last two are `"scale" in weight_name` and `"weight" in
weight_name`; a name matching neither falls off the end to
`return False if return_success else None`.

Measured, all five in one run against one stand-in for `self`:

```
w13_wire      shard=w1  returned False  parameter written: False
w2_wire       shard=w2  returned False  parameter written: False
w13_wire_len  shard=w1  returned False  parameter written: False
w13_weight    shard=w1  returned True   parameter written: True
w2_weight     shard=w2  returned True   parameter written: True
```

The controls are what make this a measurement rather than an anecdote. The
stand-in is minimal — it carries only the attributes the dispatch reads — so a
wire parameter that does not load could have been the stand-in's failure. It is
the same stand-in, the same call and the same shapes family for the two
`weight`-named parameters, and they load. The name is what separated them.

Two things this does **not** say. It is not a claim that the wire layout is
wrong: `[E, 2, S13]` with `w13_wire_len [E, 2]`
(`src/tessera/moe_layout.py`) is untouched by this, and its round trip is
tested. And it is not a claim about any other vLLM version: it is this build's
dispatch, and a build that renames the substrings moves the answer.

## 2. The remedy is a value the route already has to set

`load_weights` calls `param.weight_loader(...)` — the callable **on the
parameter** — not `self.weight_loader`. `create_weights` receives
`extra_weight_attrs["weight_loader"]` and every stock method passes it straight
to `set_weight_attrs`; a route that instead installs its own callable owns the
whole load for that parameter, and the substring dispatch above is never in the
path.

Measured by driving `load_weights` itself with such a parameter: the callable
was invoked once with `weight_name` `...experts.w13_wire`, `shard_id` `w1`,
`expert_id` 0 and the 40-byte tensor; `load_weights` yielded `w13_wire`; the
parameter was written.

That is not a workaround. A wire row is variable-length at a declared stride
(the manifest's `global_scale` rides as a varint whose width follows its value,
so a unit's blob length follows its data — #5's item 3, pinned by
`tests/test_fused.py`), and the stock helpers cannot express that: `_load_w13`
narrows by `expert_data.shape[shard_dim] // 2` and then `copy_`s shape-for-shape,
so a 40-byte blob into a 64-byte row is a size mismatch, not a short write. The
length companion has to be filled from the loaded tensor's own extent, which
only a loader that sees both can do.

## 3. What this settles for #5, and what it does not

Settles: `create_weights` for the expert route registers each wire parameter
**with its own `weight_loader`**, and the checkpoint carries one tensor per
(expert, projection) under a suffix the mapping routes. That is a design
constraint established before anything is written, which is the point — the
alternative was finding it at load, on a checkpoint that took hours to encode.

Does not settle: anything about the forward. There is no `ROUTES` entry, no
`apply`, no `routed_moe` in `scheme.STRUCTURES`, and no `lane_eligibility` cell
— and no served measurement, which is what a cell needs. `structures:
["dense"]` remains the true statement.
