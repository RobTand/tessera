# Serving a routed-MoE Tessera checkpoint: the cut, the sidecar, and what the census can and cannot say

*2026-09-04. Issue #5 item 5. Branch `wf/ts-5-serve`, off `wf/ts-5` at 77adfcc.*

The write half of the routed-MoE seam landed on `wf/ts-5` (884e841 through
77adfcc, not this branch's work). What it left open, in its own words, was
**"no served census and no KL"**. This is the attempt at that, and the first
thing to say about it is what it is measured on.

## The model is a cut, and every number below inherits that

The only routed-MoE model this box can serve is GLM-5.3-Flash-4layer. Its three
expert stacks are 21.74 G routed parameters and about **3.75 GPU-hours of
encode** — a serving receipt that costs four hours of a shared GPU before the
first load is a receipt nobody takes tonight, and the previous agent scoped it
out for exactly that reason.

So the expert *dimension* is narrowed instead of the ambition.
`experiments/moe_expert_cut.py` keeps experts `0..15` of each stack verbatim,
narrows both router tensors to match (`mlp.gate.weight` and
`mlp.gate.e_score_correction_bias` — a router left at 128 rows over 16 experts
routes tokens to weights that are not there), rebuilds the shard index from what
was actually written, edits `n_routed_experts`, and copies the tokenizer and
processor files byte for byte.

| | source | cut |
|---|---|---|
| routed experts / stack | 128 | 16 |
| `top_k` | 8 | 8 (unchanged) |
| tensors | — | 602 |
| size | — | 7.07 GiB |
| encode | ~3.75 GPU-h | 1012 s |

Same model class, same tokenizer, same real weights, same wire, same loader
path, same tile arithmetic. **Different routing.** Every KL below is
student-against-teacher *on the cut*, which measures the error the Tessera
expert route introduces on these experts. It is **not** a quality claim about
GLM-5.3-Flash: dropping 112 of 128 experts while leaving `top_k` at 8 changes
which experts a token reaches, so the cut's own BF16 teacher is the only honest
denominator, and that is what it is compared against.

## The export

`experiments/moe_export_glm_4layer.sh` on the cut, `--grid E4M3 --q256 1024
--layers 4 --passthrough-unrouted`, through the pool (action `b127654a9fb9`,
sparky, **1012 s**, 60.7-75.8 W against a ~140 W envelope; `gpu_utilization` read
96 % throughout and is non-diagnostic on GB10). Output
`/mnt/shared/tessera-runs/ts5/glm53-4layer-e16-tessera`, 4.9 GB, 598 tensors,
11 shards.

`experiments/ts5_sidecar_check.py` reads it back from safetensors headers and
JSON alone, before any GPU is spent on it:

```
quant_method='tessera' config_groups=11 ignore=159 tensors=598
routed_moe groups=3 other groups=8
  ...layers.{1,2,3}.mlp.experts: experts=16 grid=E4M3 body=WINDOW plane=CHANNEL
    w13: n=32 stride=4215633 max=4215633 spread=5 OK
    w2:  n=16 stride=4219649 max=4219649 spread=1 OK
wires=144  manifest requires_lanes=[]   NO PROBLEMS
```

Three things worth naming.

**The stride is derived three times and agrees three times.** The exporter takes
`max(lengths)` over the blobs it wrote; `moe_layout.unpack_moe_wires` re-derives
it from the loaded lengths and refuses a stride the bytes contradict; this check
recomputes it from the shard headers, sharing no code with either. A checker
that imports the writer it is checking checks nothing.

**The blob length really does follow the data, at GLM shape.** Within one stack,
at one shape and one rung: `gate_proj` 4215632/4215633, `up_proj`
4215628/4215629, `down_proj` 4219648/4219649. One byte, because
`ScalePlane.encode` writes the exact-`Fraction` `global_scale` as two varints
whose width follows their value. That one byte is why the group needs a stride
plus per-blob lengths at all, and it is now measured on 144 real GLM containers
rather than argued from the encoder.

**Eleven config groups, and the eight non-MoE ones are the control.** The
construction census (`docs/measurements/construction/glm53-flash-4layer.json`)
says exactly four Linear patterns are `offered_to_quant_config` on this model —
layer 0's `mlp.down_proj` and `mlp.gate_up_proj`, and layers 1-3's
`shared_experts.*` — which is eight modules, and eight dense groups is what the
exporter wrote. The other twenty patterns are never offered (vLLM builds them
with `quant_config=None`), so `--passthrough-unrouted` leaves them BF16 and the
export writes no wire for them. That is also why this export sidesteps issue
\#99's dense-GLM blocker: the module that fails there, `self_attn.f_b_proj`, is
one of the twenty.

## The census had to be fixed before it could say anything

`tools/tessera_route_census.py` joined its route records to the checkpoint's
`config_groups` targets **by name, in two different namespaces**. The records
come off `named_modules()`; the targets are written in the checkpoint's own
namespace. On GLM-5.3-Flash the model class's `hf_to_vllm_mapper` rewrites
`model.language_model.` to `language_model.model.`, so *nothing joined* — every
declared module would have been reported as reporting no route, on a healthy
serve, and every route as undeclared.

The fix replays the model's own unstacked mapper over the targets before
matching — the same translation `TesseraConfig.apply_vllm_mapper` makes at load
— and records both the mapping and the fact it was applied in the receipt. A
target the mapper drops maps to `None` and is reported, not silently identified
with itself. `tests/test_route_census_module_space.py` covers the five cases
(no mapper, a mapped path, a dropped target, a bare class name, a `re:` pattern);
27 tests pass on the pool (action `82eb43a4c855`).

This is a **defect the dense census never exposed** and would have made the first
MoE census unreadable.

## What the census can still not say, and why that is not a bug to fix tonight

Two things will make the MoE half of the census report problems even on a
working serve, and they are separable:

1. **`MoERunner` wraps `RoutedExperts` as `.routed_experts`.** The route record
   is stamped at `...mlp.experts.routed_experts`; the mapped declared target is
   `...mlp.experts`. Off by one module.
2. **`ROUTE_LAUNCHES` publishes no MoE entry**, so `_expected` cannot own the
   `(vllm.fused_moe.modular_kernel, torch_materialize_stock)` pair the expert
   route launches under.

Both are edits to the census's tables, not to the serving path, and neither
should be made blind: the right time to write a `routed_moe` launch pair is when
a served census has printed what the launch actually is. The eight dense groups
are the control that says the mapper fix works and these two are MoE-specific.


## Both predicted census defects are closed, and one had a shape I did not predict

The section above named two things that would make the MoE half of the census
report problems on a healthy serve. The first served census printed exactly
those and nothing else: **eight problems, three names, one cause each.**

**1. The off-by-one module, closed by containment.** `MoERunner` wraps
`RoutedExperts` as `.routed_experts`, so vLLM builds ONE quant method for the
declared stack prefix and attaches it to the child it constructs underneath.
The record lands at `...mlp.experts.routed_experts`; the mapped declaration is
`...mlp.experts`. An exact-name join reports the same served stack twice --
once as a route nothing declared, once as a declaration nothing served -- which
is six of the eight problems, and the other two are the summary line counting
the same three names again. `join_records_to_declared` joins a record whose
`kind` is `moe` to the single declared target that CONTAINS it, reports
ambiguity when two do rather than taking the longer prefix, and leaves a dense
record joining only to itself. A non-`moe` record under a declared target does
**not** join: containment is the expert stack's structure, not a general
fallback for a name that missed.

**2. The missing launch pair, closed -- but not where the section predicted.**
The prediction was a `ROUTE_LAUNCHES` entry. That table is the *contract's*
derivation source: `contract.validate_serving_contract` builds every cell's
`executes` from it, so a row added there changes what every dense cell is
checked against and belongs with the contract version bump, not with a census
fix. The expectation instead comes from the route that owns the dispatch --
`moe_route.census_expected`, the same ownership rule `fp8_gemv.census_expected`
follows -- and it is **one launch in both regimes**, because the stack is
materialised once at load and nothing branches on M.

The pair the serve actually reports is
`("vllm.fused_moe.modular_kernel:TRITON", "torch_materialize_stock")`, and the
suffix is why this is not a two-line table edit. `select_fp8_moe_backend` is
vLLM's own predicate over the kernels it finds on the box, so the backend is a
fact about the serve the record must keep and not a promise the route makes.
The census compares the entry point and keeps every exact string in its
histogram. Enumerating the backends we would accept would be a claim about
vLLM's kernel roster written in our prose (principle 14); pinning one would
refuse a box whose runtime picked another.

**3. The one I did not predict: a dense cell would have covered the stack.**
`census.cell_launch_agreement` keys a block to ONE structure and resolved a
record's cell from `(family, mode, regime, rung)`. The three stacks came back
`unattested` -- correctly -- but for the wrong reason: `rungs_by_module` is
keyed by module name, the record's name carries a `.routed_experts` suffix no
declaration has, so the rung lookup missed. That is an accident of the join,
and the join had just been taught to cross it. With rungs resolved, a
`structure: "dense"` cell whose `executes` is `torch._scaled_mm` over
`torch_window` would have started covering a stack that executes vLLM's
fused-MoE kernel -- reporting an agreement, or a disagreement, about a launch
no cell in contract v14 publishes. `STRUCTURE_BY_RECORD_KIND` maps a record's
`kind` to the structure whose cells could cover it, and it is checked **before**
the rung. An unknown kind is covered by nothing.

The stacks stay `unattested`, and that is the finished state for tonight, not a
gap papered over: contract v14 declares `structures: ["dense"]`, and a receipt
that invented coverage from that absence is the exact failure the block exists
to prevent.

## Status of the serve: three arms, one GPU, one serve lock

`experiments/ts5_moe_served.sh` takes three serves sequentially, because the box
has one serve lock and the two arms of a KL must not be resident at once:

1. the BF16 cut through the stock GLM image -> teacher logprobs;
2. the Tessera cut through **Tessera's own plugin** -> student logprobs + KL;
3. the Tessera cut a second time, on purpose, for the route census -- the route
   record lives on the layer objects inside the worker and an OpenAI-protocol
   serve cannot hand them out.

The corpus is the GLM-tokenized default `corpus_n8_s512.json` (n=8 x 512, 4088
scored positions), and the cut and the export both carry the source's
`tokenizer.json`/`tokenizer_config.json` byte for byte
(`19e77364...`/`98b12715...`), which is what `kl_tool`'s tokenizer-identity gate
compares -- so it passes without `--allow-tokenizer-mismatch`.

**Two memory numbers, because there are two consumers.** The serve arms take
`--gpu-memory-utilization 0.15` (18.24 GiB of 121.63): the teacher's own startup
accounting says 8.34 GiB of weights plus 4.42 GiB of peak activation, so 12.76
GiB is the floor and 0.15 clears it by 5.5 GiB of KV. The census takes
`TESSERA_CENSUS_MEM_UTIL=0.35`, because it drives `LLM(...)` rather than
`vllm serve --max-num-seqs 8`, and `LLM()`'s defaults (`max_num_seqs=256`, the
default chunked-prefill token budget) profile a far larger activation peak --
which is exactly how the first census attempt died, *after* a successful load,
on `Available KV cache memory: -2.02 GiB`. One knob was doing two jobs.

### It took four submissions, and three of the failures were mine

Honest accounting, because each cost GPU minutes on a shared box:

- **Attempts 2 and 3 of `0247025c68f2` were corrupted by me.** The pool worker
  executes the *live working tree*, so editing `ts5_moe_served.sh` while its
  action was running rewrote the script under a running bash: the published log
  shows `line 68: AFTER: command not found` and then
  `line 82: syntax error near unexpected token '('`. Commit before submit, and
  do not edit while an action runs.
- **All three attempts also failed for one real reason**, invisible until the
  action was withdrawn because pbrun buffers an action's stdout until it ends
  and a retried action never ends: `kl_tool.py dump` refuses
  `--role teacher` without `--teacher-label`, in 2 seconds, before any request.
  The driver passed three arguments to `serve_and_dump_kl.sh` where the fourth
  is the teacher label. Fixed in `6172e1e`; the driver now also tees to
  `served/driver.log` so an arm is readable while it runs (`99ac7b2`).
- **The withdraw itself left two orphans**, and they deadlocked the next
  submission for six minutes: a running `ts5-kl-teacher` container, and
  `serve.lock` as a directory with **no owner file** -- `serve_lock_release`
  ran far enough to `rm -f owner` before the TERM landed but not far enough to
  `rmdir`. `serve_lock_acquire` only sweeps a lock older than 3600 s on an
  otherwise idle box, so an ownerless lock is not stale by its own test and the
  next serve waits an hour for a lock nobody holds. Reaped by hand (my own
  container, my own lock); the loop is fixed in this branch, see below.

## The load hop is closed: vLLM read the routed-MoE checkpoint back

This is the result the branch's two docs named as unknown, and it is now
measured. From `served/census.log`, on the pinned Mia GLM image
(`prismaquant/glm53-mia-sm121:487ecf187`, digest `sha256:75ea13ed…`, vLLM
`0.1.dev20051+g487ecf187`):

```
[tsrun] ... plugin ['tessera', 'lora_filesystem_resolver', 'lora_hf_hub_resolver']
INFO ... Initializing a V1 LLM engine ... quantization=tessera, quantization_config=None
Loading safetensors checkpoint shards: 100% Completed | 11/11
INFO ... [model_runner.py:374] Model loading took 5.81 GiB and 29.296122 seconds
WARNING ... [fused_moe.py:1161] Using default MoE config ... E=16,N=2048,device_name=NVIDIA_GB10,dtype=fp8_w8a8
```

Four things in those five lines, none of which had a receipt before:

- **The checkpoint chose the plugin.** `quantization=tessera` with
  `quantization_config=None` on the command line: nothing enabled it, the
  sidecar did.
- **Every shard loaded.** 11/11, no `KeyError`. The model-level hop —
  `Glm5NextModel.load_weights` handing 144 `.wire` names through
  `expert_params_mapping` to the expert loader — is the hop both
  `tessera-moe-export-seam` and `tessera-moe-route-load` left open, and it
  carries weight.
- **5.81 GiB of weights**, against 4.9 GB of Tessera bytes on disk plus the
  BF16 remainder.
- **The expert method was built and configured**: vLLM looked for a fused-MoE
  tuning config at `E=16,N=2048,…,dtype=fp8_w8a8` — E=16 is the cut's expert
  count reaching the runtime's own MoE machinery, and `fp8_w8a8` is the
  `TESSERA_FP8` family's A-side.

**What it did not do is serve a token.** The load was followed by
`Available KV cache memory: -2.02 GiB` and vLLM refused to build a cache. That
is not a Tessera failure: the census drives `LLM(...)` under vLLM's default
chunked-prefill budget of 8192 batched tokens, so its profiling peak is several
times a `--max-num-seqs 8` serve's, and the 18.24 GiB the *serve* needed is not
enough for the *census*. One number was doing two jobs; the driver now has
`TESSERA_CENSUS_MEM_UTIL` (0.35) separate from `TESSERA_GPU_MEM_UTIL` (0.15).

## The serve happened. The measurement did not survive the model.

Both dumps completed. The plugin served the routed-MoE checkpoint, generated,
and answered 4088 logprob requests:

| arm | positions | top-K coverage (mean) | build |
|---|---:|---:|---|
| BF16 cut, stock image | 4088 | 0.2259 | eager, `95c38d65cc6d5194` |
| Tessera cut, **the plugin** | 4088 | 0.2265 | eager, `04250a9921a169e8` |

and `kl_tool` compared them:

```
metric=KL-vs-BF16  support=top-1024  partition=teacher-student-intersection
bound=lower bound (data-processing inequality)  regime=prefill  positions=4088
positions=4088  top1_agree=62.74%
  ALL   KL >= 0.005826   (<= 78.955220 at the declared floor 3.72e-44)
        teacher tail mass outside the compared support: mean 0.791350 max 0.838625
  legacy (v1) all=0.088701  confident=n/a (no confident position)
```

**None of that is a quality number, and the reason is not Tessera.** The
bound spans four orders of magnitude, 79% of the teacher's mass sits outside
the compared support, and there is no position anywhere in 4088 where the
teacher is confident enough to be worth comparing.

### The control, and it is decisive

The greedy smoke on the Tessera arm is gibberish —

```
completion: 'imersUnloadimers unload unload unload unload unload unloademyFan unload unloademyemyemy'
```

— and gibberish out of a quantized model is exactly the shape of a broken
expert route, so it needs a control rather than an explanation. The control is
the BF16 arm's own dump, and a third arm that predates tonight: the
**uncut** GLM-5.3-Flash-4layer, all 128 experts, BF16, dumped 2026-09-01 on the
same corpus contract (`/mnt/shared/tessera-kl/teacher_bf16.json.npz`). For each
arm, how often its top-1 is the corpus's actual next token, and where the true
token ranks in the 1025 returned:

| arm | next-token top-1 | median rank of the true token | mass in top-1025 | median top-1 prob | confident (>=0.5) |
|---|---:|---:|---:|---:|---:|
| GLM-5.3-Flash-4layer, **all 128 experts**, BF16 | **0.00%** | 1024 | 0.2716 | 0.0185 | 0 |
| the 16-expert cut, BF16 (the teacher) | **0.00%** | 1024 | 0.2259 | 0.0036 | 0 |
| the 16-expert cut, **Tessera wire** | **0.00%** | 1024 | 0.2265 | 0.0033 | 0 |
| *positive control:* Qwen3-0.6B BF16, its own corpus | **39.73%** | **1** | 0.9767 | 0.4300 | 1709 |

The last row is what a usable reference looks like, on the same tool and the
same code path: a trained model puts the true next token first at the median
position and holds 98% of its mass inside the compared support. It is the arm
`tessera-served-kl-2026-09-01` switched to when it hit this wall.

Not one correct next token in 4088, on any arm, including the one with no
quantization and every expert. The true token's median rank is 1024 of 1025 —
the model ranks it below essentially everything it was asked about. The
alignment is not in doubt: the true token is present in the returned support at
100% of positions (the serve appends it), while the off-by-one alignment
appears at 1.69%.

**So the 4-layer base is not a language model, my expert cut is not what broke
it, and the gibberish is inherited.** That is not a new discovery — it is
`docs/measurements/tessera-served-kl-2026-09-01.md`'s method note, reproduced:
*"A 4-layer cut is not a language model; its distribution is nearly flat and no
top-K comparison on it is informative. That run is retained as a plumbing
validation only."* The same ruling applies here, and this run is the same
category of evidence.

Two things tonight's numbers add to that ruling:

- **The cut is worse than the base, and the base was already unusable.** The
  uncut 128-expert base's median top-1 probability is 0.0185; the cut's is
  0.0036, five times flatter. But *neither* has a single confident position in
  4088. (The 1709 confident positions the 09-01 table quotes are **Qwen3-0.6B's**
  — the model that run switched to. The GLM 4-layer arm never had any. I wrote
  the opposite here first, from the table's column header, and the check below
  is what caught it.)
- **`top1_agree` is a trap on a model like this.** 62.74% looks respectable and
  is higher than the 23% the *uncut* base scored against its own quantization
  on 09-01 — which is the giveaway, not the reassurance. Two near-uniform
  distributions collapsed onto the same repeated junk token agree often and
  mean nothing by it. Agreement is only a quality statistic when the reference
  has an opinion.

### What this run does establish

Everything on the serving path, and nothing about quality:

1. the exporter's routed-MoE bytes **load** — 11/11 shards, no `KeyError`,
   `quantization=tessera` chosen by the checkpoint with `quantization_config=None`;
2. vLLM builds the expert method at the checkpoint's own expert count
   (`E=16,N=2048,device_name=NVIDIA_GB10,dtype=fp8_w8a8`);
3. the model **generates** through the plugin's expert route, eager, and again
   under `LLM(...)` in the census;
4. it answers 4088 prompt-logprob requests at k=1025 without a failure, and
   the dump's running clock (10.7 s at chunk 1 to 31.7 s at chunk 8, about
   3 s per 511-position chunk after the first) lands on top of the BF16 arm's
   (10.6 s to 32.2 s), so the route is not pathologically slow either. That is
   a wall-clock observation off the dump log, not a profiled latency claim.

The quality half of item 5 is **not** measured and cannot be measured on this
model. What would measure it: encode the full 128-expert stacks (21.74 G routed
parameters, ~3.75 GPU-hours) so the reference is the real GLM-5.3-Flash-4layer,
which is still a truncation and still flat — or, better, find a genuinely
trained routed-MoE model small enough to encode. Qwen3-0.6B is what the 09-01
run reached for when it hit this same wall, and it is dense.

### The census ran and lost its receipt to an NFS permission

The census loaded the checkpoint, generated twice, gathered every route record,
and then died:

```
File "/work/tools/tessera_route_census.py", line 526, in main
  with open(args.out, "w") as fh:
PermissionError: [Errno 13] Permission denied: '/mnt/shared/tessera-runs/ts5/served/census.json'
```

`/mnt/shared` is NFSv4 with `sec=sys` and the export squashes root; the census
runs as root inside the container and the output directory is `drwxrwxr-x
rob:rob`. Verified rather than inferred — container root gets `Permission
denied` on that directory and `OK` on a bind mount of `/home/rob/tmp`. The
wrapper's own default `RUNS` is a local path for exactly this reason; this
driver was the caller that pointed it at NFS. Fixed by writing to a local mount
and copying on the host, and the student arm now skips if its dump exists, so
the re-run costs one load instead of three.

### The check is now a script, because this is the second time

`experiments/kl_reference_usable.py` takes a teacher dump and its corpus
contract and answers, before a student serve is spent, whether the reference
can carry a comparison at all — next-token top-1, where the true token ranks,
how much mass the support holds, how many positions are confident. It refuses
(exit 2) when a reference cannot say anything, and it refuses (exit 1) a corpus
that is not the one the dump was taken on: passing the GLM contract against the
Qwen dump scored 1.54% and looked exactly like the failure it exists to raise,
which is how that gate came to be there.

Both cuts and the uncut base are refused by it. Qwen3-0.6B passes. Running it
on the teacher costs a second and would have said, before either serve tonight,
that the KL at the end was not going to be a number.

### The re-run died on the same permission one directory over

The fix above moved `census.json` to a local mount. The re-run failed anyway,
before loading anything, with vLLM reporting

```
Model architectures ['Glm5NextForConditionalGeneration'] failed to be inspected.
```

which reads like a model problem and is not one. Underneath it:
`PermissionError: '/ext/triton'`. `tessera_plugin_run.sh` forces
`TRITON_CACHE_DIR=/ext/triton` inside the container, `/ext` is whatever `$EXT`
the caller supplies, and this driver's default was `$OUT/ext` — NFS again, root
squash again, one directory over from the file I had just moved.

The first census run had survived it because *that* submission's pbrun command
line exported `EXT=/home/rob/tmp/ts5-ext` by hand, and the wrapper script I
re-submitted from did not. So the two runs differed in their submissions, not
in their code, which is the least visible way for two runs to differ. The
driver now defaults `EXT` to a local path, so the scratch directory is correct
for a caller who never thinks about it.

The recurring shape, twice in one night: **the container writes as root, and
every path handed to it must be one root can write.** `/mnt/shared` cannot be,
for any of them — output, `TMPDIR`, `TORCH_EXTENSIONS_DIR`, or the Triton
cache.

### A withdrawn action leaves its container running

Worth recording because it is a pool fact, not a Tessera one, and it cost GPU
minutes twice tonight.

`pbrun --withdraw` on action `975a7b593f73` reported
`withdrew ... from claimed; released 39 token(s); signalled TERM -2958554`. The
process group was gone a second later — and a `docker run` that group had
started at 08:04:52 was **still running**, holding roughly 45 GB of the box's
one GPU with no pool token behind it, because a container is a child of
`dockerd` and not of the process group the TERM reached. Nothing in the pool
knew it existed; the tokens it had been using were already back in the free
pool for another action to claim.

The same shape, earlier in the night, left a serve *lock* behind instead of a
container: the TERM landed inside `serve_lock_release`, between the `rm -f
owner` and the `rmdir`, and every later arm on the box blocked on a lock
directory with no owner. That one is fixed on our side —
`experiments/serve_lock.sh` now reaps an ownerless lock after a full poll and a
dead-owner lock immediately, with `tests/test_serve_lock.py` holding both rules
plus the case that matters more than either (a *live* owner still blocks).

The first version of that fix could not have run. `awk` exits 2 on a file that
is not there, an ownerless lock is exactly that file, and every wrapper that
sources the library runs under `set -euo pipefail` — so
`pid="$(_serve_lock_owner_pid)"` took awk's status and ended the script at exit
2 with an empty stderr, before the reaping rule it was written for. It would
have fired only in a shell without `-e`, which is to say only in a test that
forgot one. `tests/test_serve_lock.py` failed two of its six cases on first
run, both on the ownerless path, and `a35a5c1` is the one-token fix; 6 passed
after it. Recorded because the shape is the lesson: a rule added to a library
and never exercised by a caller is a rule nobody has run.

The container half is not ours to fix. Reported upward as a pool defect: a
withdraw that releases an action's tokens must also reap the containers the
action started, or the release is a lie.
