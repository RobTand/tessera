# The 16-bit route, served: TESSERA_BF16 on Tessera's own plugin (2026-09-02)

**Result.** `TESSERA_BF16` — the third `ROUTES` entry, W16A16, the window body
over the CHANNEL plane with its 2^L table snapped to bf16 — is **served** by
`tessera.serving` on vanilla vLLM 0.28, in both residency modes and under both
the eager and the compiled forward. Four route censuses record **all 112
declared modules** on the route, in both the prefill and the decode shape, with
zero problems. The served KL-vs-BF16 at the R = 7 rung (`q256 = 1792`, wire
7.129 bpp) is **0.004923** in both modes, **bit-identical between them**, against
**0.004929** for the folded stock twin vanilla vLLM serves from the same
encode. The fold this route exists to avoid is **below what this corpus
resolves at this rung**: the route is ahead on `all` by `+0.000006` nats
(1.0011x) and on top-1 by +0.073 pp, and behind on `confident` by `-0.000013`
nats (0.9961x). The signs disagree, so **no fold win is claimed here** — §3 has
the arithmetic for why that is the expected answer at R = 7, and #45 carries the
high-rung arm that would settle it.

This is the receipt `runtime_contract.json` v5 cites when it moves
`TESSERA_BF16_K1` from `attested_rungs_q256: []` to `[1792]` and publishes two
`sm_121` dense cells. Issue #9 asked for exactly this: a route module, one
`ROUTES` entry, contract rows, and *a served census plus a KL against the
plain-BF16 twin*. Nothing wider is attested — one rung, one platform, dense
only, no routed-MoE cell, `max_world_size 1`.

---

## 1. What was served, and that it is the exported bytes

| | |
|---|---|
| wire | `/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7` — Qwen3-0.6B, grid `BF16`, `q256 = 1792` (R = 7), body `WINDOW`, plane `CHANNEL` |
| bpp | body/wire **7.1292**, on disk **7.1317**, resident-mode **16.0** (112 modules, 196 units, 440 401 920 quantized params) |
| encoder | manifest `git: fc2c1c1`, written `2026-09-02T14:25:01` — **after** `795137c` (the reach-aware per-row start, 10:49), so these bytes carry it |
| served dir | `…-r7-plugin`, produced by `experiments/retarget_checkpoint_to_plugin.py`: `model.safetensors` is a **hardlink** (inode `88500`, 1 015 088 026 bytes on both), and the only `config.json` delta is `quant_method: "gridbook" → "tessera"` plus `structure: "dense"` on each scheme |
| twin | `…-r7-twin`, the exporter's `materialize_bf16_folded` rendering, verified unit for unit: `twincheck.json` — 196 units checked, **0 mismatched**, 311 tensors, no `quantization_config` |
| teacher | `qwen_teacher_bf16_v028` (the unquantized model on the same image) |
| corpus | `corpus_qwen_n8_s512.json` — Qwen-tokenized, n = 8 × 512, 4088 scored positions, `corpus_sha256 076d33ef…` |
| metric | KL-vs-BF16, top-1024 support, teacher–student intersection, **lower bound** — the same instrument as every row of `tessera-stock-lane-served-2026-09-02.md` |
| image | `vllm/vllm-openai:latest` = vLLM 0.28.0 / torch 2.13.0+cu130 / python 3.12.3 |
| device | `NVIDIA GB10`, compute capability `[12, 1]` (`sm_121`) |
| KL serve execution mode | **eager** — both KL serves ran `--enforce-eager` (`/home/rob/tessera-runs/bf16route/serve_qwen_tessera_bf16-r7-{resident,streamed}.log` line 9, `'enforce_eager': True`). The four censuses of §1 cover eager and compiled; the KL of §3 covers eager only. Noted 2026-09-04 for the contract v17 `evidence` field. |
| driver | `experiments/bf16_route_served.sh 1792` |

The served bytes are the exported bytes **by construction** (one inode), not by
a copy someone has to trust.

## 2. The census: what the serve recorded, in four combinations

`tools/tessera_route_census.py` compares what the serve **recorded** through
`emit_route` against what the checkpoint **declares**, module by module, in a
prefill shape (M = 64) and a decode shape (M = 1). Principle 9's leg: route
provenance is a gate input, not a log.

| mode | forward | verdict | decode modules | prefill modules | other-route | problems | s |
|---|---|---|---|---|---|---|---|
| resident | eager | **served** | 112 | 112 | 0 | 0 | 85.4 |
| resident | compiled | **served** | 112 | 112 | 0 | 0 | 135.6 |
| streamed | eager | **served** | 112 | 112 | 0 | 0 | 85.6 |
| streamed | compiled | **served** | 112 | 112 | 0 | 0 | 159.1 |

Every record in all four, in both phases:

```
contract  bf16_unquantized
decoder   torch_window
kind      dense
policy    TESSERA_BF16:<mode>
state     served
symbol    torch.mm
modules   112
```

`declared_families {"TESSERA_BF16": 112}`, `other_route_modules 0` — no module
fell through to a stock scheme, and no module of another family appeared. The
`decoder` value is the honest one: this route decodes the window body in **pure
torch** (`serving/window.py`), never through `kernel_window_gemv`, whose value
family is the shape this wire wants but whose `SUPPORTED_RATES = (1, 2, 4)`
excludes the rate-7 rung served here. That is filed as #47, not hidden.
Decode shapes are `M1:N1024:K2048 / M1:N1024:K3072 / M1:N4096:K1024 /
M1:N6144:K1024` eager and the same four with `M*` compiled (`route_shape()`
spells the token dim `M*` under `torch.compiler.is_compiling()`, because
`int()` on a symbolic dim is the graph break this lane has been bitten by).
Prefill is the same four at `M64`. All four arms emit the identical greedy
continuation, `' and the decode shape. The receipt names'`.

The compiled arms are not a formality. Every route bug this lane has had —
`int()` on the token dim, `data_ptr` fingerprints, a mutating op on an aliased
pool — was invisible to an eager census
(`tessera-gridbook-lane-served-2026-09-02.md`, "Graph mode: the lanes
could not start under vLLM's default serve").

Receipts: `/home/rob/tessera-runs/bf16route/census-bf16-r7-{resident,streamed}-{eager,compiled}.json`.

**On the commit stamp.** The four JSONs carry
`tessera_commit: 05cd046fd89d484172b2b72ca1aa6ce2d29b6ef5`, which is the
worktree's HEAD at the time of the run, not the commit this receipt lands in.
The tree that ran carried, on top of it, the `gemm_symbol` change (§5) — the
route table gained a `gemm_symbol` field, the three route modules read
`GEMM_SYMBOL` from it, and the census derives its expectation from the same
field — plus the driver's two operational fixes and a handful of comment and
docstring corrections. **No route arithmetic differs**, and that is checkable
rather than asserted: `git diff 05cd046 HEAD -- 'src/tessera/serving/*_route.py'`
touches **seven lines of code** — three `GEMM_SYMBOL` constants, the three
`symbol=` arguments they replace, and one `__all__` entry — plus one docstring
paragraph in `bf16_route.py` that this receipt is what made true.
`bf16_route.py` passed the literal `"torch.mm"` to `emit_route` at `05cd046`
and passes `GEMM_SYMBOL`, the same string, now. Every other delta under
`serving/` in `git diff 05cd046 HEAD --stat` is the `ROUTES` table's new
`gemm_symbol` field, the contract JSON, or a comment. At `05cd046` exactly
the census would have *refused* these same records, because its expectation was
the literal `"torch._scaled_mm"`; that refusal is what produced the change.

## 3. The served KL, and the fold

| arm | bpp (wire / resident) | KL all | KL confident | top-1 |
|---|---|---|---|---|
| **`TESSERA_BF16` route, resident** | 7.129 / 16.0 | **0.004923** | 0.003311 | 95.82% |
| **`TESSERA_BF16` route, streamed** | 7.129 / — | **0.004923** | 0.003311 | 95.82% |
| stock twin, folded scale, vanilla vLLM, no plugin | 16.0 / 16.0 | 0.004929 | 0.003298 | 95.74% |

**The two residency modes are bit-identical**, on every statistic the
instrument reports and not only on the headline: `all`, `confident` and the
top-1 agreement all match to the last digit the JSON carries
(`0.004923151123392585`, `0.003310922`, `95.81702544031312`). That is the
strongest thing this run says about the decode. `resident` materialises the
tile once at load and `streamed` decodes it inside every forward, from the same
packed planes; if either had a rounding the other does not, this is where it
would show, and there are 4088 positions of it.

**The fold, honestly.** The route is better on `all` by `+0.000006` nats
(1.0011x) and on top-1 by +0.073 pp, and *worse* on `confident` by `-0.000013`
nats (0.9961x). The signs disagree, so what this corpus establishes at this rung
is that **the fold's cost is below what it resolves** — not that the fold is
free, and not that the route won. The receipt does not claim a fold win, and
nothing downstream should quote one from it.

That is the arithmetic one should expect. The fold adds exactly one bf16
rounding of `s_i * t_ik`, relative error `<= 2^-9 ~ 0.2%`; the R = 7 body's own
error is far larger, and KL is quadratic in the weight perturbation, so the
fold's share goes as `(eps_fold / eps_wire)^2`. A ~0.1% share puts `eps_wire`
near 6%, which is the right order for a 7-bit body. The ratio is therefore a
function of the **rung**, and the region this family exists for — above ~6 bpp,
where the E4M3 alphabet floors and the trellis does not — is above the one
served here. Issue #45 carries the high-rung arm that would settle it.

**And a weight-space prediction, corrected on the way through.** The design
receipt priced the fold on six GLM experts at 15.4% of the output error at
R = 7 (`tessera-bf16-route-2026-09-02.md` §7b) and concluded from it that "the
route is ~15% better than its own twin". That does not follow from its own
definition: `fold = sqrt(out_bf16² - out²)` composes in quadrature, so a 15.4%
share is a 1.2% error gap and a **2.4%** squared-error gap. Carried to the
dense Qwen units of §7c (`out` 0.01214 at R = 7, the same 0.00134 fold
constant) the share is ~11% and the squared gap ~1.2%. Served, the gap is
0.11%, with the opposite sign on `confident`. So there are two corrections
here, not one: the design doc overstated the prediction by an order of
magnitude, and the corrected prediction still does not reach this instrument.
The design doc's §7b now says so; #45 carries the rung where it should.

**What stands regardless.** The cells attest *dispatch*, and the four censuses
establish that independently of the fold: 112 of 112 declared modules on the
route, in both the prefill and the decode shape, in both residency modes, under
both the eager and the compiled forward, zero problems, and the same greedy
continuation from all four. Separately, the route is not *worse* than the twin
by more than that same resolution on any statistic, which is the check that
would have caught an epilogue rounding twice.

**The twin is a control, not an equal arm.** It is a plain BF16 safetensors of
`materialize_bf16_folded` — the *same encode*, with the row scale multiplied
into the tile because a one-tensor checkpoint has no way to carry it
separately. Vanilla vLLM serves it with no plugin at all. So twin-vs-route
isolates exactly one thing: the fold. A route that came out **worse** than the
twin would be a defect — it would mean the epilogue rounds where it should not,
and it does not. Whether it comes out measurably *better* is a question about
the rung, and at this one the answer is no; see above and #45.

The driver's fixed acceptance block, run over those seven JSONs:

```
bf16-r7: resident=0.004923151123392585 streamed=0.004923151123392585 twin(folded)=0.0049286974863798514
  confident: resident=0.0033109218935382214 twin=0.0032981622257165022
  top1 agree: resident=95.817% twin=95.744%
  fold cost: all +0.000006 nats (1.0011x), confident -0.000013 nats (0.9961x)
  THE FOLD IS BELOW WHAT THIS CORPUS RESOLVES on at least one subset -- that is a statement about the rung, not a pass; do not cite a fold win from it
```

Comparison points on the same corpus, teacher and instrument (dense Qwen3-0.6B,
from `tessera-dense-reach-fix-2026-09-02.md` and
`tessera-stock-lane-served-2026-09-02.md`):

| arm | bpp | KL all |
|---|---|---|
| FP8 RTN per-channel, W8A8 | 8.0 | 0.0205 |
| Tessera-8 E4M3 window L=14, reach-aware | 4.07 | 0.1512 |
| production NVFP4 GPTQ+JSO, W4A4 | 4.5 | 0.5106 |
| Tessera-4 E2M1x2 TCQ, W4A4 | 4.0 | 0.6404 |

## 4. What this does and does not license

**Licensed** (contract v5): `TESSERA_BF16_K1`, `attested_rungs_q256: [1792]`,
two cells `tessera_bf16_k1_dense_sm121_{decode,batch}_mm_w16a16` —
`activation_contract: bf16_unquantized`, `route_status:
backed_with_serve_flag` (the residency is chosen by `TESSERA_SERVE_MODE` and
both values were served), `qualification: device_qualified` (the censuses ran
on this GB10, not on a table), `requires_plugin: tessera`,
`requires_serve_flags: ["TESSERA_SERVE_MODE=resident|streamed"]`.

**Not licensed.** One rung: the reader *range* is `[256, 4096]` (25 rungs
loaded, `experiments/bf16_reader_rate_range.py`) but only 1792 was **served**,
and a cell speaks for what was served. One platform (`sm_121`). Dense only —
there is no routed-MoE cell for any Tessera family and the loader refuses that
structure loudly. `tensor_parallel.units` still caps this family at
`max_world_size 1`. And nothing here makes the family *selectable* from
PrismaQuant: `ANCHOR_BUDGET_BITS` refuses `payload_bits >= 16` on a premise
that is TCQ's alone (a window body has no forest), which is a separate change
in a separate repo.

**Not a size claim.** `resident` holds a bf16 tile: 16 bits a weight is the
source precision, and the route does not offer that as compression. It is the
correctness path — the tile a stock GEMM consumes with no decoder in the serve.
`streamed` holds the packed planes at the artifact's own 7.13 bpp and decodes
each forward into a transient tile.

**Footprint, as the runtime reports it.** `Model loading took` reads **1.12
GiB** for `resident` and **0.71 GiB** for `streamed`, identically under the
eager and the compiled forward (four serves, four censuses). The 0.41 GiB gap
is the wire: 440 401 920 quantized params x (16 - 7.129) bits / 8 = 0.455 GiB
of weight bytes not held, less the CHANNEL plane and the 2^14-entry window
tables `streamed` keeps resident to decode with. The twin also loads 1.12 GiB,
as it must — same shapes, same bf16 tile, no plugin.

The `gpu_worker` line in the same logs reads the other way (2.65 / 2.41 GiB
resident eager / compiled, 2.93 / 2.44 GiB streamed) and is not a contradiction:
it is measured *after* profiling and reports weights **plus** non-torch memory,
so it includes the allocator's reserve for the transient tile the streamed
forward materialises each call — which is exactly the memory `streamed` trades
time to avoid holding. One number is the weights; the other is the weights plus
the working set. `Model loading took` is the one that answers "how big is the
model".

## 5. Two mechanism fixes this run forced

**The census had a hardcoded GEMM symbol.** Its first run refused all 112
modules per phase with `symbol='torch.mm'`, because it compared against the
literal `"torch._scaled_mm"` — correct for the two routes that existed when it
was written, wrong for the first route that calls anything else. The fix is at
the mechanism, not the checker: `gemm_symbol` is now a `ROUTES` field, each
route module derives `GEMM_SYMBOL` from it and stamps that on every record, and
the census derives its expectation from the same field. One place to remember.
`tests/test_serving_dispatch.py::test_every_route_declares_the_gemm_it_invokes`
fails on master with `AssertionError: TESSERA_NVFP4 does not declare the GEMM
it invokes`.

**The driver's acceptance checks were vacuous.** The summary block read a
top-level `mean_kl` from `kl_tool.py compare`'s output, which that tool has
never written — the key is `all.kl_lower_mean`. Every arm read `None`, so both
checks (`MODES DISAGREE`, `THE FOLD IS NOT COSTING ANYTHING`) were skipped and
the run would have printed `resident=None streamed=None twin=None` as though
that were a pass — and it did: **`served_r7.log`'s own summary line reads
`None`**, because the fix landed after the run it exposed. The numbers in §3
come from the JSONs; the verdict block quoted there is the fixed reader run over
those same files by hand.

The block now reads the real key and **refuses** a file it cannot find it in.
Only one of its two checks exits non-zero, and that asymmetry is the point: the
two residency modes decode the same wire and must agree exactly, so a difference
there is a decode defect and fails the run; the fold is a quality claim whose
size depends on the rung, so a sign flip there is *reported*, in the words the
run must not be allowed to launder into a pass. This is the same failure mode as
a guard that passes on both sides of a change: a check that cannot fail is worse
than no check, and a check that fails for the wrong reason is not much better.

**And one operational note.** The first attempt at the resident serve died with
`ValueError: Free memory on device cuda:0 (100.7/121.63 GiB) on startup is less
than desired GPU memory utilization (0.85, 103.38 GiB)` — both serve scripts
default to 0.85 of a **shared** 121.6 GiB unified pool, and another job on the
box held ~21 GiB. That is not a route fact; the driver now exports
`TESSERA_GPU_MEM_UTIL=0.45`, which is ample for a 0.6B model at a 4096 context.
The whole run was then repeated from the top, so every number above comes from
one uninterrupted invocation (`served_r7.log`; the refused attempt is kept as
`served_r7_oom.log`).

## 6. Where the numbers live

```
/home/rob/tessera-runs/bf16route/census-bf16-r7-resident-eager.json
/home/rob/tessera-runs/bf16route/census-bf16-r7-resident-compiled.json
/home/rob/tessera-runs/bf16route/census-bf16-r7-streamed-eager.json
/home/rob/tessera-runs/bf16route/census-bf16-r7-streamed-compiled.json
/home/rob/tessera-runs/bf16route/kl_tessera_bf16-r7-resident.json
/home/rob/tessera-runs/bf16route/kl_tessera_bf16-r7-streamed.json
/home/rob/tessera-runs/bf16route/kl_bf16-r7-twin.json
/home/rob/tessera-runs/bf16route/served_r7.log
/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7-twincheck.json
```

Design and quality background: `docs/measurements/tessera-bf16-route-2026-09-02.md`
(the wire, the alphabet floor, the fold's cost in weight space) and
`docs/measurements/tessera16-alphabet-floor-2026-09-02.md`.

Follow-ups this run filed rather than dropped: **#45** (the fold is below served
resolution at R = 7; the high-rung arm that would resolve it) and **#47** (the
route decodes in pure torch; the window GEMV's value family is the fast path it
never calls, and the provenance seam that dispatch opens is the one #42 sits
on).
