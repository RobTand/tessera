# Unnamed directory reads: issue 1014

## Census

The baseline is `83a1f38c4965fbf6beab9538c54bf0105617f705`.
PrismaBuild action `d194b658508126f77b00fabd025857c6ddbd0e2eaba18bd4640aa900bf3cf0a1` ran `select(root, [])` on its snapshot.
The selector reported 115 modules and 169 sites.
The table below classifies all 169 sites from that result, not from a separate `file_imports` pass.
The source expressions and lexical bindings came from the same snapshot.
Line numbers identify the baseline source.

Classes:

1. The source names the base. A small resolver correction can retain it.
2. A parameter or runtime value supplies the base. The accepted limit applies.
3. The base names external artifacts, installed dependencies, or box files, not this repository.
4. A temporary test fixture supplies the base.
5. The recognizer reports a call that is not a directory read.

This census has one class 1 site and no class 5 sites.
An imported module's `__file__` is a runtime value, not proof of this checkout's path.
External dependency packages have class 3 when the source identifies the external package.
Temporary fixture helpers have class 4 when their callers supply `tmp_path` or a fixture derived from it.

## Site classification

| Path | Line | Class | Source evidence |
| --- | ---: | ---: | --- |
| `_pb_native_moe_measure/per_job_install.py` | 35 | 2 | `root.rglob('*')`; `files` takes `root`. |
| `experiments/allocated_serve_2026-09-02/census_table.py` | 6 | 3 | `glob.glob('/home/rob/tessera-runs/allocated/census_*.json')`. |
| `experiments/allocated_serve_2026-09-02/kl_table.py` | 8 | 3 | `glob.glob('/mnt/shared/tessera-runs/allocated/qwen3-0.6b-*/tessera_serving_manifest.json')`. |
| `experiments/allocated_serve_2026-09-02/kl_table.py` | 23 | 3 | `glob.glob('/home/rob/tessera-runs/allocated/kl_tessera_*.json')`. |
| `experiments/allocated_serve_2026-09-02/kl_table2.py` | 5 | 3 | `glob.glob(f'{S}/qwen3-0.6b-*/tessera_serving_manifest.json')`; `S = '/mnt/shared/tessera-runs/allocated'`. |
| `experiments/allocated_serve_2026-09-02/kl_table2.py` | 27 | 3 | `glob.glob(f'{R}/mutual_*.json')`; `R = '/home/rob/tessera-runs/allocated'`. |
| `experiments/audit_byte_baseline.py` | 845 | 3 | `glob.glob(g)`; `ARTIFACT_GLOBS` contains `/home/rob/tessera-runs/*/*/cache/wire/*.tessera` and `/home/rob/tessera-runs/*/cache/wire/*.tessera`. |
| `experiments/bench_native_operator.py` | 468 | 3 | `(Path(torch.__file__).parent / "lib").glob("*cuda*.so*")` reads the installed Torch library directory. |
| `experiments/bench_routed_load.py` | 198 | 2 | `data.iterdir()`; `_checkpoint_index` takes `data`. |
| `experiments/bench_routed_load.py` | 219 | 2 | `data.iterdir()`; `_moe_layers` takes `data`. |
| `experiments/bench_routed_load.py` | 870 | 2 | `src.rglob("*.py")`; `_tree_identity` takes `src`. |
| `experiments/bf16_route_weight_space.py` | 234 | 2 | `glob.glob(directory + "/*.safetensors")`; `open_all` takes `directory`. |
| `experiments/bf16_twin_check.py` | 33 | 2 | `directory.glob("*.safetensors")`; `open_all` takes `directory`. |
| `experiments/capture_full_engine_resources.py` | 54 | 3 | `root.rglob("*")`; `root = Path(importlib.util.find_spec('vllm').origin).parent` identifies installed vLLM. |
| `experiments/checkout_runtime_identity.py` | 54 | 2 | `root.rglob('*')`; `files` takes `root` and converts it with `Path(root)`. |
| `experiments/compare_stock_checkpoints.py` | 22 | 2 | `path.glob("*.safetensors")`; `load` takes `path`. |
| `experiments/compile_build_forensics.py` | 117 | 2 | `root.rglob("*.best_config")`; `_best_configs` takes `root`. |
| `experiments/compile_build_forensics.py` | 212 | 2 | `base.glob(f"*/{args.prefix}/computation_graph.py")`; `base` derives from `args.a` or `args.b`. |
| `experiments/decode_back_to_bf16.py` | 85 | 2 | `src.glob(pattern)`; `decode_back` takes `src`. |
| `experiments/decode_back_to_bf16.py` | 117 | 2 | `args.out.iterdir()`; `ap.parse_args()` supplies `args`. |
| `experiments/dense4_census_out_check.py` | 46 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_perrow_legs.py` | 79 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_plane_census.py` | 75 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_plane_mechanism.py` | 64 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_qknorm_gauge.py` | 71 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_reach_sweep.py` | 63 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense4_read_bracket.py` | 41 | 2 | `glob.glob(f"{args.dir}/kl_*.json")`; `ap.parse_args()` supplies `args`. |
| `experiments/dense4_residual_census.py` | 78 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense_spread_census.py` | 31 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/dense_spread_fix.py` | 77 | 3 | `glob.glob(p + "/*.safetensors")`; `p = '/home/rob/dq-runs/fc45-0p6b-nvfp4/exported'`. |
| `experiments/export_checkpoint_driver.py` | 42 | 2 | `src.glob("*.safetensors")`; `build_plan` takes `src`. |
| `experiments/export_glm_4layer.py` | 54 | 2 | `src.glob("*.safetensors")`; `build_plan` takes `src`. |
| `experiments/export_stock_compressed.py` | 243 | 2 | `args.src.glob("*.safetensors")`; `ap.parse_args()` supplies `args`. |
| `experiments/export_stock_compressed.py` | 409 | 2 | `args.src.glob(pattern)`; `ap.parse_args()` supplies `args`. |
| `experiments/fp8_band.py` | 108 | 3 | The literal pattern is `/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model*.safetensors`. |
| `experiments/freegrid.py` | 29 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/full_engine_plugin_install.py` | 38 | 2 | `Path(root).rglob("*")`; `files` takes `root`. |
| `experiments/full_engine_plugin_install.py` | 48 | 2 | `(tree / "src").rglob("*")`; `source_tree_identity` takes `tree`. |
| `experiments/full_engine_reference.py` | 49 | 2 | `checkpoint.iterdir()`; the checkpoint derives from the `proof_path` parameter. |
| `experiments/full_engine_worker.py` | 242 | 2 | `package.rglob("*")`; `package = Path(tessera.__file__).resolve().parent` names the active runtime package. |
| `experiments/full_model_research_selected_checkpoint.py` | 199 | 2 | `out.iterdir()`; `out` derives from `args.out.resolve() / 'checkpoint'`. |
| `experiments/fused_member_rung_identity.py` | 86 | 2 | `args.src.glob("*.safetensors")`; `ap.parse_args()` supplies `args`. |
| `experiments/gemv_receipt_tables.py` | 29 | 2 | `glob.glob(os.path.join(d, pat))`; `_load` takes `d` and `pat`. |
| `experiments/glm53_508_graph_qual/compare-pad-508.py` | 7 | 2 | `recs.glob(f'{a}.pad*.json')`; `recs` derives from `sys.argv[1]`. |
| `experiments/glm53_508_graph_qual/matrix-508.py` | 36 | 2 | `recs.glob('engine-args-*.txt')`; `recs = pathlib.Path(sys.argv[1])`. |
| `experiments/glm53_508_graph_qual/matrix-508.py` | 37 | 2 | `recs.glob('engine-args-*.txt')`; `recs = pathlib.Path(sys.argv[1])`. |
| `experiments/glm53_508_graph_qual/memcheck-summary-508.py` | 19 | 2 | `recs.glob(f"{arm}.memcheck.*.log")`; `sys.argv` supplies `recs` and `arm`. |
| `experiments/glm53_508_graph_qual/prof-summary-508.py` | 12 | 2 | `root.rglob("*")`; `root = pathlib.Path(sys.argv[1])`. |
| `experiments/glm53_695_drafter_qual/build_stub_mtp.py` | 96 | 2 | `args.stub.iterdir()`; `ap.parse_args()` supplies `args`. |
| `experiments/glm53_695_drafter_qual/build_stub_mtp.py` | 179 | 2 | `args.out.iterdir()`; `ap.parse_args()` supplies `args`. |
| `experiments/glm53_695_drafter_qual/draftlog-695.py` | 62 | 2 | `glob.glob(f"{receipts}/{arm}.draft.*.jsonl")`; `load` takes `receipts` and `arm`. |
| `experiments/inductor_determinism_probe.py` | 97 | 2 | `root.rglob("*.py")`; `_scan_kernels` takes `root`. |
| `experiments/inductor_determinism_probe.py` | 189 | 2 | `root.rglob("*.best_config")`; `root = Path(cache_dir)` uses a `_child` parameter. |
| `experiments/kda/verify_conv_bank.py` | 25 | 2 | `(out / name).glob("*.cubin")`; `out = Path(sys.argv[1])`. |
| `experiments/ktuple.py` | 21 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/ladder_survey.py` | 59 | 3 | `glob.glob(f"/home/rob/.cache/huggingface/hub/{repo}/snapshots/*/model*.safetensors")` has a fixed external prefix. |
| `experiments/lane_numerics/gemm_real.py` | 12 | 3 | `glob.glob(f"{CK}/*.safetensors")`; `CK = '/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4'`. |
| `experiments/lane_numerics/hidden_kl.py` | 8 | 3 | `glob.glob(f"{CK}/*.safetensors")`; `CK = '/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4'`. |
| `experiments/lane_numerics/hidden_kl_band.py` | 20 | 2 | `glob.glob(f"{ck}/*.safetensors")`; `head` takes `ck`. |
| `experiments/ldlq_block_byte_check.py` | 68 | 2 | `Path(twin).glob("*.safetensors")`; a runtime manifest supplies `stock_twin`. |
| `experiments/loadcost.py` | 141 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/merge_tessera_parts.py` | 735 | 2 | `out.glob(pattern)`; `out = Path(args.out)`. |
| `experiments/mixed4.py` | 36 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/moe_plan_baseline.py` | 344 | 2 | `outdir.glob("*.safetensors")`; `_digest_export` takes `outdir`. |
| `experiments/original_wire_generation.py` | 45 | 2 | `package_root.rglob('*')`; `package_root = Path(tessera.__file__).resolve().parent`. |
| `experiments/pad_canonicality_census.py` | 38 | 3 | `glob.glob(pattern)`; `ARTIFACT_GLOBS` contains the two external wire patterns used by `audit_byte_baseline.py`. |
| `experiments/reach_predictor_check.py` | 60 | 2 | `Path(path).glob("*.safetensors")`; `open_all` takes `path`. |
| `experiments/refit_trailing_bytes.py` | 168 | 2 | `path.glob("*.safetensors")`; `load` takes `path`. |
| `experiments/refresh_native_resident_manifest.py` | 78 | 2 | `source.iterdir()`; `refresh` takes `source`. |
| `experiments/render_arm_to_bf16.py` | 59 | 2 | `args.src.glob("*.safetensors")`; `ap.parse_args()` supplies `args`. |
| `experiments/render_arm_to_bf16.py` | 89 | 2 | `args.src.glob(pattern)`; `ap.parse_args()` supplies `args`. |
| `experiments/retarget_checkpoint_to_plugin.py` | 66 | 2 | `args.src.iterdir()`; `ap.parse_args()` supplies `args`. |
| `experiments/rotate_checkpoint.py` | 93 | 2 | `src.glob("*.safetensors")`; `load_state_dict` takes `src`. |
| `experiments/rotate_checkpoint.py` | 241 | 2 | `args.src.glob("*.safetensors")`; `ap.parse_args()` supplies `args`. |
| `experiments/rotate_checkpoint.py` | 242 | 2 | `args.dst.glob("*.safetensors")`; `ap.parse_args()` supplies `args`. |
| `experiments/rotfull.py` | 16 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/run_glm_native_construction.py` | 145 | 2 | `root.iterdir()`; `root = args.out`. |
| `experiments/step4_capture_driver.py` | 279 | 2 | `capture_dir.glob("worker-*/runtime-observation.json")`; `qualify` takes `capture_dir`. |
| `experiments/t4_code/fp4_corrective_audit.py` | 37 | 2 | `source.glob("dense-*.pt")`; `source = Path(args.retained_outputs)` uses a runtime argument. |
| `experiments/t4_code/fp4_corrective_audit.py` | 63 | 2 | `source.glob("grouped-*.pt")`; `source = Path(args.retained_outputs)` uses a runtime argument. |
| `experiments/t4_code/summarize.py` | 38 | 2 | `Path(d).rglob("t4_code_*.json")`; `d` comes from `sys.argv[1:]`. |
| `experiments/t8.py` | 20 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/t8curve.py` | 14 | 3 | `glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors")`. |
| `experiments/t8r_speed/finalize_native_build.py` | 25 | 2 | `directory.glob('*.o')`; `finalize` takes `directory`. |
| `experiments/t8r_speed/finalize_native_build.py` | 26 | 2 | `directory.glob('*.so')`; `finalize` takes `directory`. |
| `experiments/t8r_speed/finalize_native_build.py` | 27 | 2 | `directory.glob('*.so')`; `finalize` takes `directory`. |
| `experiments/t8r_speed/finalize_native_build.py` | 28 | 2 | `directory.glob('*.o')`; `finalize` takes `directory`. |
| `experiments/t8r_speed/nccl_sweep/nccl_sweep.py` | 304 | 2 | `os.listdir(d)`; `d = os.path.join(a.out, f'rank{r}')` uses the `merge` parameter `a`. |
| `experiments/t8r_speed/scratch_bound.py` | 141 | 2 | `os.listdir(d)`; `d = os.path.join(args.routing, f'm{m}')`. |
| `experiments/t8r_speed/value_prefetch_numeric.py` | 66 | 2 | `sanitizer.parent.iterdir()`; `prepare` takes `sanitizer`. |
| `experiments/t8r_speed/value_prefetch_numeric.py` | 86 | 2 | `root.rglob("*")`; `root = runner / package` uses the `prepare` parameter `runner`. |
| `experiments/tessera16_alphabet_floor.py` | 363 | 2 | `glob.glob(d + "/*.safetensors")`; `open_all` takes `d`. |
| `experiments/tessera385_bench.py` | 62 | 2 | `root.rglob("*")`; `_source_identity` takes `package` and reads `package.__file__`. |
| `experiments/ts113_r6_compare.py` | 131 | 2 | `output.iterdir()`; `output = Path(sys.argv[1])`. |
| `experiments/ts5_sidecar_check.py` | 41 | 2 | `d.glob("*.safetensors")`; `headers` takes `d`. |
| `experiments/window_gemv_latency_ratio.py` | 128 | 2 | `glob.glob(os.path.join(path, "**", "*.pt.trace.json*"), recursive=True)`; `_trace_file` takes `path`. |
| `experiments/window_gemv_latency_ratio.py` | 130 | 2 | `glob.glob(os.path.join(path, "**", "*.json*"), recursive=True)`; `_trace_file` takes `path`. |
| `experiments/window_gemv_trace_summary.py` | 162 | 2 | `glob.glob(os.path.join(args.path, "**", "*.json*"), recursive=True)`; `ap.parse_args()` supplies `args`. |
| `src/tessera/cached_unit.py` | 168 | 2 | `root.rglob("*")`; `_encoder_source_profiles` takes `root`. |
| `src/tessera/export.py` | 2829 | 2 | `src.glob(pattern)`; `src = Path(source_dir)` uses an `export_checkpoint_streaming` parameter. |
| `src/tessera/kernel_window_gemv.py` | 455 | 2 | `glob.glob(os.path.join(build, "tessera_window_gemv*.so"))`; `_built_library` takes `build`. |
| `src/tessera/routed_fused.py` | 553 | 2 | `_glob.glob(os.path.join(build, f"{module}*.so"))`; `_built_library` takes `build` and `module`. |
| `src/tessera/serving/build_identity.py` | 286 | 2 | `slot.rglob("*.best_config")`; `_autotune_digest` takes `slot`. |
| `src/tessera/serving/build_identity.py` | 315 | 2 | `slot.glob("rank_*/backbone/computation_graph.py")`; `slot` derives from `read_cache_root` parameters. |
| `src/tessera/serving/build_identity.py` | 316 | 2 | `slot.glob("rank_*/backbone/cache_key_factors.json")`; `slot` derives from `read_cache_root` parameters. |
| `src/tessera/serving/ext.py` | 586 | 3 | `_glob.glob("/usr/local/cuda-*")` reads installed CUDA toolkits. |
| `src/tessera/serving/ext.py` | 655 | 3 | `_glob.glob(os.path.join(site, "nvidia", "*", "include"))`; `site` derives from `torch.__file__`. |
| `src/tessera/serving/ext.py` | 702 | 3 | `os.listdir(directory)`; `_vendored_cuda_includes()` supplies installed NVIDIA include directories. |
| `src/tessera/serving/mla_sparse_sm120.py` | 47 | 3 | `root.rglob("*")`; `root` uses `flashinfer.jit.env.FLASHINFER_INCLUDE_DIR`. |
| `src/tessera/serving/source_identity.py` | 66 | 2 | `package.rglob("*")`; `src` is a parameter with a runtime default. |
| `src/tessera/serving_parts.py` | 244 | 2 | `source.glob("*.safetensors")`; `source_identity` takes `source`. |
| `src/tessera/serving_parts.py` | 263 | 2 | `source.glob(pattern)`; `_auxiliary_sha256` takes `source`. |
| `src/tessera/serving_parts.py` | 394 | 2 | `root.joinpath("src").rglob("*")`; `export_identity` takes `root`. |
| `src/tessera/serving_parts.py` | 396 | 2 | `root.joinpath("experiments").glob("*.py")`; `export_identity` takes `root`. |
| `src/tessera/serving_parts.py` | 400 | 2 | `package.rglob("*")`; `package = root / 'tessera'` uses the `export_identity` parameter `root`. |
| `src/tessera/serving_parts.py` | 988 | 2 | `path.glob("*.safetensors")`; `_prove_part_contents` takes `path`. |
| `src/tessera/serving_parts.py` | 1194 | 2 | `source.glob(pattern)`; `merge_serving_parts` takes `source`. |
| `src/tessera/serving_parts.py` | 1234 | 2 | `source.glob("*.safetensors")`; `source_inventory` takes `source`. |
| `src/tessera/serving_parts.py` | 1291 | 2 | `source.glob(pattern)`; `source_part_identity` takes `source`. |
| `tests/conftest.py` | 190 | 1 | `base.iterdir()`; the loop uses `(repo, repo / 'src', repo / 'tests')`, with `repo` anchored by `__file__`. |
| `tests/test_audit_container_accounting.py` | 120 | 2 | `package.rglob("*.py")`; `package = pathlib.Path(container.__file__).parent` uses the active runtime module. |
| `tests/test_audit_sec2.py` | 225 | 2 | `package.rglob("*.py")`; `package = pathlib.Path(wire.__file__).parent` uses the active runtime module. |
| `tests/test_cuda_surface.py` | 225 | 4 | `trace.glob("*.jsonl")`; `trace = tmp_path / 'ordering'`. |
| `tests/test_cuda_surface.py` | 588 | 4 | `tmp_path.glob("*superseded*")`. |
| `tests/test_cuda_surface.py` | 596 | 4 | `tmp_path.glob("surface.x86.gw*.json")`. |
| `tests/test_cuda_surface.py` | 597 | 4 | `tmp_path.iterdir()`. |
| `tests/test_cuda_surface.py` | 829 | 4 | `tmp_path.glob("surface.superseded-*.json")`. |
| `tests/test_cuda_surface.py` | 830 | 4 | `tmp_path.iterdir()`. |
| `tests/test_cuda_surface.py` | 862 | 4 | `tmp_path.iterdir()`. |
| `tests/test_export_explicit_plan.py` | 431 | 4 | `out.iterdir()`; `out = tmp_path / 'out'`. |
| `tests/test_export_ignore_completeness.py` | 201 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_export_moe_layouts.py` | 244 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_export_moe_layouts.py` | 599 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_export_moe_layouts.py` | 725 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_export_moe_layouts.py` | 749 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_full_engine_reference.py` | 25 | 4 | `checkpoint.iterdir()`; `checkpoint = tmp_path / 'checkpoint'`. |
| `tests/test_full_engine_worker.py` | 294 | 4 | `tmp_path.glob("prefix-observer-worker-*.pstats")`. |
| `tests/test_full_engine_worker.py` | 470 | 4 | `tmp_path.glob("kv-worker-*.json")`. |
| `tests/test_full_engine_worker.py` | 495 | 4 | `package.iterdir()`; `package = tmp_path / 'tessera'`. |
| `tests/test_glm_routed_owner_inputs.py` | 227 | 4 | `tmp_path.glob("source.safetensors.partial-*")`. |
| `tests/test_glm_routed_owner_inputs.py` | 261 | 4 | `tmp_path.glob("receipt.json.tmp*")`. |
| `tests/test_issue56_column_groups.py` | 245 | 2 | `Path(inspect.getsourcefile(grammar_mod)).parent.glob("*.py")` uses the runtime module's source file. |
| `tests/test_merged_linear_partitions.py` | 320 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_merged_linear_partitions.py` | 372 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / 'out'`. |
| `tests/test_moe_greedy_smoke_pair_containers.py` | 214 | 4 | `state.glob("*.json")`; `_env(tmp_path)` supplies `FAKE_DOCKER_STATE`. |
| `tests/test_mtp_draft_shards.py` | 53 | 4 | `glob.glob(os.path.join(model_name_or_path, "*.safetensors"))`; the fake loader reads the `ckpt(tmp_path)` fixture. |
| `tests/test_parallel_merge_hash.py` | 75 | 4 | `out.iterdir()`; `_merged` callers supply `tmp_path / 'parallel'` or `tmp_path / 'serial'`. |
| `tests/test_producer_plan_public.py` | 90 | 4 | `out.glob("*.safetensors")`; `out = tmp_path / label`. |
| `tests/test_refit_trailing_serve.py` | 133 | 4 | `runs.glob(f"kl_{ARM}.json.attempt.*")`; the `harness` fixture uses `tmp_path / 'runs'`. |
| `tests/test_refit_trailing_serve.py` | 165 | 4 | `runs.glob(f"kl_{ARM}.json.attempt.*")`; the `harness` fixture uses `tmp_path / 'runs'`. |
| `tests/test_route_trace.py` | 73 | 4 | `tmp_path.iterdir()`. |
| `tests/test_s6b_row_groups.py` | 236 | 2 | `Path(inspect.getsourcefile(grammar_mod)).parent.glob("*.py")` uses the runtime module's source file. |
| `tests/test_serving_plan_schema.py` | 232 | 4 | `out.iterdir()`; `out = tmp_path / 'out'`. |
| `tests/test_source_digest_cache.py` | 165 | 4 | `cache.directory.glob("*.json")`; `cache = _cache(tmp_path)`. |
| `tests/test_source_digest_cache.py` | 173 | 4 | `cache.directory.glob("*.json")`; `cache = _cache(tmp_path, quiescent_seconds=300)`. |
| `tests/test_source_digest_cache.py` | 193 | 4 | `cache.directory.iterdir()`; `cache = _cache(tmp_path)`. |
| `tests/test_source_digest_cache.py` | 285 | 4 | `cache.directory.glob("*.json")`; `cache = _cache(tmp_path)`. |
| `tests/test_source_digest_cache.py` | 301 | 4 | `cache.directory.glob("*.json")`; `cache = _cache(tmp_path)`. |
| `tests/test_source_digest_cache.py` | 311 | 4 | `cache.directory.glob("*.json")`; `cache = _cache(tmp_path, quiescent_seconds=300)`. |
| `tests/test_source_digest_cache.py` | 326 | 4 | `cache.directory.iterdir()`; `cache = _cache(tmp_path)`. |
| `tests/test_source_profiles.py` | 129 | 3 | `root.rglob('*')`; the test requires a noneditable Git package and proves that `root` is below `sys.prefix`. |
| `tests/test_suite_source.py` | 54 | 4 | `tmp_path.glob('verifier-*')`; `_verifier` takes the temporary fixture directory. |
| `tools/comparison_input_intake.py` | 109 | 2 | `output.iterdir()`; `output = Path(args.output).resolve()` uses the `intake` parameter `args`. |
| `tools/glm_cpu_export_inventory.py` | 28 | 3 | `glob.glob(pattern)` iterates `/home/rob/venvs/*/bin/python`, `/home/rob/*venv*/bin/python`, and `/mnt/shared/venvs/*/bin/python`. |
| `tools/provision_producer_env.py` | 226 | 3 | `(work / "wheels").glob("tessera_quant-*.whl")`; `tempfile.mkdtemp` creates `work` below `/home/rob/tmp`. |
| `tools/provision_producer_env.py` | 245 | 2 | `(base / "lib").glob("python*/site-packages")`; `base = args.base.resolve()`. |
| `tools/provision_producer_env.py` | 249 | 2 | `(venv_root / "lib").glob("python*/site-packages")`; `venv_root = base.parent / args.name`. |
| `tools/tessera_attest.py` | 548 | 2 | `os.listdir(artifact)`; `_parsed_modules` takes `artifact`. |
| `tools/tessera_lane_preflight.py` | 76 | 2 | `os.listdir(path)`; `units_of` takes `path`. |

## Runtime bases with test consumers

The counts use the baseline graph from the same `select()` call.
The reverse walk excludes collection-probe edges, as the selector's scope walk does.
The module count excludes the seed module.
The test count includes the seed when it is a test file.
A reached conftest adds all tests in its scope.
These counts describe graph reach, not a claim that every test calls the directory read.

| Runtime-base module | Reverse modules | Test consumers |
| --- | ---: | ---: |
| `_pb_native_moe_measure/per_job_install.py` | 13 | 3 |
| `experiments/bench_routed_load.py` | 6 | 3 |
| `experiments/bf16_route_weight_space.py` | 7 | 1 |
| `experiments/compile_build_forensics.py` | 1 | 1 |
| `experiments/export_checkpoint_driver.py` | 4 | 3 |
| `experiments/export_stock_compressed.py` | 3 | 3 |
| `experiments/full_engine_plugin_install.py` | 6 | 4 |
| `experiments/full_engine_reference.py` | 19 | 12 |
| `experiments/inductor_determinism_probe.py` | 1 | 1 |
| `experiments/kda/verify_conv_bank.py` | 1 | 1 |
| `experiments/merge_tessera_parts.py` | 3 | 3 |
| `experiments/original_wire_generation.py` | 10 | 3 |
| `experiments/refit_trailing_bytes.py` | 1 | 1 |
| `experiments/refresh_native_resident_manifest.py` | 1 | 1 |
| `experiments/rotate_checkpoint.py` | 1 | 1 |
| `experiments/step4_capture_driver.py` | 4 | 3 |
| `experiments/t4_code/fp4_corrective_audit.py` | 2 | 1 |
| `experiments/t8r_speed/value_prefetch_numeric.py` | 2 | 2 |
| `experiments/ts5_sidecar_check.py` | 1 | 1 |
| `experiments/window_gemv_latency_ratio.py` | 1 | 1 |
| `experiments/window_gemv_trace_summary.py` | 2 | 1 |
| `src/tessera/cached_unit.py` | 713 | 464 |
| `src/tessera/export.py` | 713 | 464 |
| `src/tessera/kernel_window_gemv.py` | 713 | 464 |
| `src/tessera/routed_fused.py` | 713 | 464 |
| `src/tessera/serving/build_identity.py` | 9 | 4 |
| `src/tessera/serving/source_identity.py` | 714 | 464 |
| `src/tessera/serving_parts.py` | 713 | 464 |
| `tests/test_audit_container_accounting.py` | 0 | 1 |
| `tests/test_audit_sec2.py` | 0 | 1 |
| `tests/test_issue56_column_groups.py` | 0 | 1 |
| `tests/test_s6b_row_groups.py` | 0 | 1 |
| `tools/provision_producer_env.py` | 1 | 1 |
| `tools/tessera_attest.py` | 2 | 2 |
| `tools/tessera_lane_preflight.py` | 1 | 1 |

The two source-hash shapes named in the issue remain runtime bases: `cached_unit.py:168` and `serving_parts.py:394`.
Their 464 test consumers make a hidden source dependency important, but do not authorize the rejected full-suite fallback.
The runtime source-root default in `serving/source_identity.py` has the same limit.
The installed package controls do not prove a checkout dependency.

## Correction

The finite tuple at `tests/conftest.py:187` names all three bases from `__file__`.
The correction retains each literal tuple, list, or set element as a loop-variable alternative.
It does not evaluate a sequence as one scalar path argument.
It does not infer a parameter value, execute a helper, or add a dependency for temporary test fixtures.
No recognizer correction is required for this census.
The `os.path.join` and f-string patterns in class 2 still need runtime inputs, so more expression syntax cannot name their bases.

## Pre-fix regression evidence

PrismaBuild action `edcd16cc48512ed4da28114bff693d7a116e57872a836469a4f89627862a9a8e` ran the three new selection cases before the correction.
All three failed at `tests/test_impacted_tests.py:2634`:

```text
assert result["tests"] == ["tests/test_reader.py"], result
AssertionError: {'verdict': 'none', 'changed': 1, 'comparison': '', 'tests': [], ...}
```

The failed cases were `tuple`, `list`, and `set`.
The run used three xdist workers with `--dist worksteal` and `--durations=10` on `dl380g10`.
Torch `2.11.0+cpu` reported no CUDA device.
The population had zero skips and zero uncollected modules.
The run did not cover the CUDA surface.

## Measured correction and CLI smoke

PrismaBuild action `67a31871666e5984b7bafff70614475994f403f2292649a8802d4a2578af117a` completed with exit code zero on `dl380g10`.
Its checks matched all 169 report rows to the baseline census and verified all 35 runtime-module consumer rows.
It measured 114 remaining modules and 168 remaining sites through `select()`.
The only removed site was `tests/conftest.py:190`.
The class counts were 1, 103, 28, 37, and 0.

The actual selector CLI ran against the exact base and printed its receipt.
It correctly refused to exclude the snapshot's closure stamp without a declared source verifier.
The receipt reported `full` for that stamp, not for the directory correction.
The final test action declares the published `pbsnapshot.py verify` command before it uses the selector.

## Complete selected-file run

PrismaBuild action `0c41cd19592e99fcc6e387c089c0626d68aba1934af64b3ec660428a909632d4` ran the selector's 398 files.
The declared source verifier removed only its verified closure stamp.
The actual CLI reported `narrowed`; the census still contained 114 modules and 168 sites.
The run used eight xdist workers with `--dist worksteal` and `--durations=20` on `dl380g10`.
Torch `2.11.0+cpu` reported no CUDA device.

The result was 8,106 passed, three failed, 2,211 skipped, and zero uncollected modules.
No test allocated on a CUDA device.
This result does not cover the CUDA surface.
The action failed and has no success receipt in CAS.

The failures were:

- `tests/test_rung_allowability.py::GeometryHarvest::test_increment_reader_findings_preserve_correctness_holds`: the CPU environment has no `jsonschema`.
- `tests/test_stageprev_793_operative.py::test_actual_loaded_public_claim_identity_matches_published_manifest`: the shared helper requires SDK version 4.
- `tests/test_stageprev_793_prepare.py::test_actual_published_pbrun_class_scope_is_not_submitter_pin`: the same helper requires SDK version 4.

The unchanged helper asserts `client.SDK_VERSION == 4` at `tests/test_stageprev_793_prepare.py:34`.
The published owner states `SDK_VERSION = 5` at `src/prismabuild/client.py:78`.
These failures do not involve the loop resolver.
The follow-up baseline checks only the three failed files from the exact base.
It supplies `jsonschema` in a temporary environment, not in the shared CPU environment.

The population digest is `0fefa98def74171751b0f2500737ee3951b3a63649f44c2207d43f9c4cf68309`.
The retained stdout digest is `b8e29cb1a03e315a9baa5e6c68934706f592e2c253f2e51a55c834770189ce77`.
The complete population block is in stdout lines 350–433 of the action's first attempt.
The skip reasons below are verbatim:

```text
857  the lane is a CUDA kernel
204  the fused TCQ trellis is a CUDA path and needs triton
106  needs a CUDA device
97  the best-form step is a CUDA path
83  the fused LUT swap passes are a CUDA path and need triton
82  the encoder is a CUDA path
62  no CUDA device
60  CPU fixture control only; real vLLM loader/CUDA graph cases did not execute
52  the fused window Viterbi is an NVPTX path
43  encoder is a GPU job
42  the block-scaled FP4 instruction is sm_121a
34  the compact plane packers are CUDA paths
33  the lane is a CUDA GEMM
32  the native window MoE runs CUDA kernels
30  the kernel lane is a CUDA path
29  the captured TCQ trellis is a CUDA path
29  the Viterbi is CUDA
25  the encoder is a GPU job
24  the fused window Viterbi is a CUDA path and needs triton
24  native terminal CUDA coverage
22  the quantiser is a CUDA op
20  selected window kernel requires CUDA
19  the compact window repack is a CUDA path
18  native class dispatch needs CUDA
16  CUDA required
14  the native window lane is a CUDA path
14  the Tessera encoder is a CUDA path
13  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook/model.safetensors is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
10  requires the published PB SDK
9  the native method runs CUDA kernels
9  the trellises are CUDA
8  needs CUDA
8  the Tessera encoder and the native decoder are CUDA paths
8  the fused window Viterbi is CUDA
7  the loader staging paths are CUDA paths
7  could not import 'prismaquant.native_moe_panel': No module named 'prismaquant'
7  CUDA required for the native bitfield-boundary numerical oracle
6  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
5  the kernel lane runs on CUDA
4  harvest needs torch and jsonschema
4  could not import 'prismabuild': No module named 'prismabuild'
2  CUDA device required for the register-direct kernel
2  the native routed lane is a CUDA path
2  real vLLM is absent from the test interpreter
2  exact public manifest/phase control requires the published PB SDK
2  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-stock-twin/model.safetensors is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
2  needs two CUDA devices
2  the route prepares wires with CUDA packers
1  the fp4 activation quantizer is vLLM's operator
1  box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)
1  native public-reader callback controls require the published PB SDK
1  public schema/lease controls require the published PB SDK
1  could not import 'vllm.config': No module named 'vllm'
1  could not import 'vllm.v1.attention.backends.registry': No module named 'vllm'
1  could not import 'vllm.v1.attention.backends.mla.flashinfer_mla_sparse': No module named 'vllm'
1  could not import 'vllm': No module named 'vllm'
1  could not import 'vllm.model_executor.layers.quantization.base_config': No module named 'vllm'
1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
1  needs a device
1  E2M1 publishes no reader range
1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
1  explicit admitted synthetic bank proof only
1  the installed-source control requires a noneditable Git package
1  the rate streams are CUDA
1  graph output lifetime requires a CUDA device
1  native fused T-4 serving gate requires CUDA
1  needs a CUDA device: the positive arm builds the extension
```

## Selector contract receipt

PrismaBuild action `f05aa0ec9bc296e1c16c443f10543582656a4ec99c40e3cb408f54b8e936350d` completed with exit code zero.
The six selector contract files passed all 464 tests on `dl380g10`.
The run used six xdist workers with `--dist worksteal` and `--durations=20`.
Torch `2.11.0+cpu` reported no CUDA device.
The population had zero skips and zero uncollected modules.
No test allocated on a CUDA device.

The test files were:

```text
tests/test_impacted_tests.py
tests/test_source_dependencies.py
tests/test_source_execution_helpers.py
tests/test_source_execution_refusals.py
tests/test_guarded_reexports.py
tests/test_impacted_manual_gate.py
```

This receipt covers the three corrected loop cases, scalar-sequence refusal, the lowered ceiling, unknown loaders, outside bases, and collection probes.
The existing controls keep unnamed runtime bases unselected and external module globs unnamed.
The unchanged temporary fixture sites remain in the census.
This result does not cover the CUDA surface.

The CAS receipt digest is `2bacb1e3b6efea698fd237d90f0265d36a3bcb0a6c64e30951f88a2604fb8ff5`.
Its result payload digest is `482b383a373b6f23b7a9d455dc7e73f313d8a85638b36ea4a20e815447fb6ef7`.
The population digest is `9d69d782987e36de0411f01a80c64bb61d1ebf043ce63d05cd557be5cd9f58fc`.

## Baseline setup failure

PrismaBuild action `e0a517d0888080f5bed01fbe4d427b281df959acd6fc35ec817ae3073c3f3b11` failed before pytest started.
The temporary environment could read the system's `jsonschema`, but could not import pytest.
It produced no baseline population and proves no baseline result.
The corrected setup declares the admitted interpreter's dependency directories for the temporary child interpreter.

## Exact-base failed-file baseline

PrismaBuild action `40d813f9d451688c193ccea7ec0bd3247be854b2239146653583d728ad8196d2` checked an archive of base `83a1f38c4965fbf6beab9538c54bf0105617f705`.
It ran only the three files that failed in the selected-file run.
The temporary interpreter could read `jsonschema 4.19.2` and the admitted interpreter's pytest and Torch dependencies.
The shared CPU environment did not change.

The baseline pytest result was 90 passed, two failed, four skipped, and zero uncollected modules.
Torch `2.11.0+cpu` reported no CUDA device.
The run used three xdist workers with `--dist worksteal` and `--durations=10`.
Its only skip reason was `could not import 'prismabuild': No module named 'prismabuild'` for four tests.
No test allocated on a CUDA device.
This archive had no Git metadata, so the population's source identity was `unknown`; it is not a merge qualification.

Both SDK controls failed at the same unchanged version assertion.
The rung test passed when the interpreter could read `jsonschema`.
The diagnostic action returned zero only after it verified the exact two baseline failure names.
That action result does not state that baseline pytest passed.

The CAS receipt digest is `7d665faa5611c4bca769429f2636927a1edcc16ddbfc32560c096062c90bf643`.
The retained payload digest is `b67219fa983323c5087ea18955c66521438a999188be57fc76c18097993b229b`.
The population digest is `f48108c8113a03a5720ab5dc5be1142161905798d5a7b729c3d72c09de506f03`.

## Separate findings

The result file requests two nonblocking children.
No unrelated source changed on this branch.

- `[P3] Remove the stale SDK version assertion from published-owner controls`: the current SDK is version 5, but the shared helper requires version 4.
- `[P3] Supply jsonschema in the managed x86 CPU test environment`: the environment has Torch but lacks the dependency that the rung test requires.

The SDK assertion belongs to separate controls, not to directory dependency discovery.
The persistent environment belongs to the PrismaBuild test provider.
Both findings have direct failure evidence and the exact-base comparison above.
Both temporary evidence scripts were removed after the admitted actions captured their source.

## Merge-base census, 2026-10-09

The work branch now includes master `572539ad224c9ca3f36b9754051f73aed9ca99b4`.
The only merge conflict was in `CHANGELOG.md`; both entries remain.
The approved selector correction and all regression assertions remain unchanged.

PrismaBuild action `a75204c93bfc29835b8647ed4a2201ca143cdbfc85a2a9dd21566e15520de867` ran the census smoke on `dl380g10`.
The action completed with exit code zero.
Its `select(root, [])` result still contains 114 modules and 168 sites.
Every remaining module, site count, and line number matches the classified baseline after removal of the one corrected site.
The remaining classes contain 103 runtime bases, 28 external box reads, and 37 temporary test reads.
All 35 runtime-module consumer rows retain the reverse module and test counts in the table above.

The smoke also exercised `select(root, ["README.md"])`, which returned `narrowed`.
This CPU-only result does not qualify the CUDA surface.
The temporary smoke script was removed after the action captured its source.

## Merge-base selector receipt

PrismaBuild action `f42150c482e54e3cc3e85d05636934a23133fd5840bd524b33abf238106515e8` passed all 464 tests in the same six selector contract files.
The action tested the merged worktree and completed with exit code zero on `dl380g10`.
The run used four xdist workers, `--dist worksteal`, and `--durations=20`.
Each worker had one native math thread.
The controller reconciled all 464 collected tests with 464 passed call outcomes.
The retained payload had no setup, teardown, or collection failures.

Torch `2.11.0+cpu` reported no CUDA device.
The population had zero skips, zero uncollected modules, and zero device allocations.
This result does not qualify the CUDA surface.
The pre-fix regression evidence above remains valid; the merge changed none of those assertions.

The CAS receipt digest is `eb5b0dae579dbbfc207fd4102a6179b72392cb2a99d702d15c4160e205e74ed6`.
The retained payload digest is `e87a8ab051349acc167906bdf0999cfe01c0fef120e43214a01c8f0ba6ba21fe`.
The controller population digest is `427f8eaeab0294aaa7f7f317e454cd79286364023c56a1c1ebc048b5c9805425`.

