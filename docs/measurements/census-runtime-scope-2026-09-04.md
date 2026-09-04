# Census runtime scope — Tessera #126

The census matches a cell only under its exact runtime image and execution
mode. Missing image/mode, a mismatched value, and a historical cell with no
runtime scope remain unattested. The raw receipt records
`runtime={image,execution_mode}`; `--compiled` supplies the mode that also
controls `LLM(enforce_eager=...)`. `--runtime-image` requires a digest reference
before loading vLLM.

Compiled dense per-cell agreement remains unsupported because its route record
combines launches across shapes. Compiled routed-MoE agreement may compare a
single-launch record with a single-launch cell; combined observations remain
unattested. This is CPU validation of the comparison, not a new served receipt
or a promotion of any MoE cell.

The plugin wrapper injects its selected image after caller-supplied Docker
environment arguments, and all six shell census callers pass that value into
the census. A historical tag must be replaced by an explicit digest for a new
census; no digest is inferred from Docker's local image ID. Historical replay
reads only the receipt's explicit runtime context and reports unattested when
it is absent. It refuses a runtime mode contradicting `receipt.compiled`.

## Source and selected coverage

Measured core/CLI commit: `42b0cc3`; measured wrapper/callsite/replay commit:
`9ceeb7a`, both on producer contract commit `a4927fe`. Before publication,
architecture prose was included in each source commit under rule 10, producing
`4528e26` and `a25423b` respectively. Their executable code is unchanged from
the measured revisions. No wire bytes, defaults, or cells changed in these
census commits; the separate producer change owns v5 schema publication.

PrismaBuild selector action `073ca26cab7a028a2aed3a5ed77cfae3b398a6f000a130b026700c6c1f18f0d5`
compared the census changes with exact fetched base
`894305ee41b648e563bc1fa672948f5f6cf8c4af`. Its `narrowed` verdict selected:

- `tests/test_census_cell_agreement.py`
- `tests/test_census_engagement.py`
- `tests/test_census_runtime_scope.py`
- `tests/test_census_runtime_wiring.py`
- `tests/test_route_census_module_space.py`
- `tests/test_ts5_serving_gates.py`

`tests/test_runtime_image_pin.py` additionally checked the wrapper's existing
image guard. The selector's receipt is
`12e863b140d5ed9238d34e3a27d7ec24757b6552135849c0bdb5269ecf440327`.

## Red before the fix

Both red actions ran through PrismaBuild on dl380g10 with CPU=1, mem_gb=4,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. The deployed pool
retried failures; superseded populations remain beside the final population.

Action `dfd0436fd7e6` ran `tests/test_census_runtime_scope.py`: **16 failed**,
snapshot `6c0e94dc49092c0e4b31c8520e245fad5b1a8bac`. Population:
`/mnt/shared/tessera-runs/ts5/lfm25/astra-v5-census-red-r1.surface.json`.

| Added test | Pre-fix failure line |
| --- | --- |
| `test_missing_runtime_context_cannot_borrow_a_cell` | `E assert True is None` |
| `test_uncovered_runtime_image_or_mode_stays_unattested` (5 cases) | `E TypeError: cell_launch_agreement() got an unexpected keyword argument 'runtime_image'` |
| `test_matching_runtime_preserves_launch_refusal_and_context` | same `runtime_image` TypeError |
| `test_cell_without_runtime_scope_does_not_inherit_a_global_pin` | same `runtime_image` TypeError |
| `test_compiled_moe_single_launch_can_agree_but_dense_stays_unattested` | same `runtime_image` TypeError |
| `test_compiled_combined_moe_trace_does_not_claim_a_single_launch` | same `runtime_image` TypeError |
| `test_all_structure_agreement_threads_runtime_without_borrowing_dense_image` | `E TypeError: all_structure_agreement() got an unexpected keyword argument 'runtime_image'` |
| `test_cli_requires_an_exact_runtime_image_before_loading_vllm` (3 cases) | `E AttributeError: module 'runtime_scoped_census' has no attribute 'parse_args'` |
| `test_cli_records_mode_from_the_flag_that_controls_llm` (2 cases) | same `parse_args` AttributeError |

Action `0c990f8d3ae0` ran `tests/test_census_runtime_wiring.py`: **9 failed**,
snapshot `01f0cc8d1c1a45bf63a2b7f841358a99906e6302`. Population:
`/mnt/shared/tessera-runs/ts5/lfm25/astra-v5-census-wiring-red-r1.surface.json`.

| Added test | Pre-fix failure line |
| --- | --- |
| `test_plugin_wrapper_injects_selected_image_after_caller_environment` | `E AssertionError: assert 'forged-caller-value' == 'example/runt...1111111111111'` |
| `test_every_shell_census_invocation_passes_the_verified_wrapper_image` | `E AssertionError: experiments/allocated_serve_2026-09-02/chain_allocated.sh` at `assert '--runtime-image' in command` |
| `test_historical_replay_does_not_infer_runtime_from_global_pin_or_compiled` (2 cases) | `E AttributeError: module 'scoped_replay' has no attribute 'replay_runtime_context'` |
| `test_replay_uses_only_recorded_runtime_context` | same `replay_runtime_context` AttributeError |
| `test_replay_refuses_malformed_or_contradictory_runtime_context` (4 cases) | same `replay_runtime_context` AttributeError |

## Green on the implementation

PrismaBuild action
`193be37a4bfdcd0cf3bcb6793c5a2c4f71ebce48be800540834edaed0efd2f38`
ran all seven test files above: **78 passed, 0 failed**, 1.72 seconds, on
snapshot `1b4118e86adebe725930b9f16ef2a2e0ff836fbb` of source `9ceeb7a`.
Receipt: `2e278153027f2d53f2ec6b87a078b0f3766fd150eae87f446ed51886bafb41d4`.
Population: `/mnt/shared/tessera-runs/ts5/lfm25/astra-v5-census-green-r1.surface.json`.

Every red and green test population reported, verbatim:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Skip reasons are `{}` and uncollected modules are `[]` in all three tables.
No GPU or served performance claim is made.

PrismaBuild action
`05a8926e8b929087fa58b005bd998e7f17284ea8ede4a7a2b751e05a8266e6bc`
ran `bash -n` on all seven changed shell scripts, exit 0. Receipt:
`a2ddcb4a5578fde2900f197c938bbb869e5c54505e10fa11e05fe4bde0252f0a`.
