"""Declared-rule ownership derivation over one capture (tessera#399).

Every test here is a pure-dict regression on the rules in
``experiments/full_engine_ownership``: no torch, no vLLM, no CUDA, no capture
on disk. A passing suite establishes that each declared rule fires exactly
where its statement says it does and abstains -- by name, with a reason --
where it says it cannot. It establishes nothing about a real engine's
ownership, and no rule here is evidence that a class is assignment-invariant.
"""
import pytest

from experiments.full_engine_ownership import (
    EXTERNAL_RECORDS_SCHEMA, GEOMETRY_WITNESS_SCHEMA, OWNERSHIP_OBSERVATION_SCHEMA,
    OWNER_VIEWS_SCHEMA, RULES, TRANSIENT_WITNESS_SCHEMA,
    allocation_site, boundary_geometry_witness, dense_startup_check, derive_owner_views,
    external_record_views, ownership_observation, route_family_of, site_package,
    transient_gap_witness, views_by_allocation,
)

SITE = "/img/site-packages"
PLUGIN = SITE + "/tessera"
VLLM = SITE + "/vllm"
TORCH = SITE + "/torch"
OBSERVER = "/observer"

#: One roster row per family, so each route family holds exactly one unit and
#: the load-time route rule has something to resolve. ``members`` are the
#: manifest's HF-named member parameters of the fused unit.
ROSTER = [
    {"unit_id": "s:model.layers.0.self_attn.qkv_proj",
     "module": "model.layers.0.self_attn.qkv_proj", "family": "TESSERA_NVFP4",
     "members": ["model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_proj.weight"]},
    {"unit_id": "l:model.layers.0.self_attn.o_proj",
     "module": "model.layers.0.self_attn.o_proj", "family": "TESSERA_FP8",
     "members": ["model.layers.0.self_attn.o_proj.weight"]},
    {"unit_id": "l:model.layers.1.mlp.down_proj",
     "module": "model.layers.1.mlp.down_proj", "family": "TESSERA_BF16",
     "members": ["model.layers.1.mlp.down_proj.weight"]},
]

EVIDENCE = {
    "plugin_package_path": PLUGIN,
    "plugin_files": {"decode.py", "serving/nvfp4_route.py", "serving/fp8_route.py",
                     "serving/bf16_route.py"},
    "vllm_root": VLLM,
    "vllm_files": {"v1/worker/gpu_model_runner.py", "model_executor/layers/layernorm.py"},
    "observer_roots": [OBSERVER],
    "observer_libraries": [OBSERVER + "/libcollector.so"],
    "plugin_jit_prefix": "/ext/tessera_nvfp4/",
    "jit_cache_prefixes": ["/cache/triton"],
    "inventory_digests": {"plugin_source_sha256": "a" * 64},
    "roster": ROSTER,
}

#: ``before_model_load`` and ``ready_for_workload`` as history indices.
CHECKPOINTS = {"before_model_load": 5, "ready_for_workload": 20}


def _frames(*filenames):
    return [{"filename": name, "name": "fn", "line": 1} for name in filenames]


def _row(allocation_id="a", *, size=64, allocate_index=10, free_completed_index=None,
         categories=(), owners=(), stack=()):
    return {"allocation_id": allocation_id, "bytes": size,
            "allocate_index": allocate_index, "free_completed_index": free_completed_index,
            "observed_categories": list(categories), "observed_owners": list(owners),
            "scope_stack": list(stack)}


def _views(rows, frames_by_index=None, *, roster=None, evidence=None):
    return derive_owner_views(rows, frames_by_index or {}, checkpoint_index=CHECKPOINTS,
                              roster=ROSTER if roster is None else roster,
                              evidence=EVIDENCE if evidence is None else evidence)


# --- the allocation site -----------------------------------------------------


def test_the_site_is_the_innermost_python_frame_outside_the_torch_package():
    # Torch's own Python frames are the allocator's, not the caller's, and the
    # C++ unwind rows are not files at all.
    frames = _frames("??", TORCH + "/_tensor.py", VLLM + "/v1/worker/gpu_model_runner.py",
                     PLUGIN + "/decode.py")
    assert allocation_site(frames, TORCH)["file"] == VLLM + "/v1/worker/gpu_model_runner.py"
    assert allocation_site(frames, TORCH)["name"] == "fn"


def test_a_stack_with_no_frame_outside_torch_has_no_site():
    assert allocation_site(_frames(TORCH + "/_tensor.py", "??"), TORCH) is None
    assert allocation_site([], TORCH) is None
    assert allocation_site(None, TORCH) is None


@pytest.mark.parametrize("filename,expected", [
    (PLUGIN + "/decode.py", ("plugin", "decode.py")),
    (VLLM + "/v1/worker/gpu_model_runner.py", ("vllm", "v1/worker/gpu_model_runner.py")),
    (SITE + "/numpy/core/_methods.py", ("image", "numpy/core/_methods.py")),
    (OBSERVER + "/experiments/full_engine_worker.py",
     ("observer", "experiments/full_engine_worker.py")),
    ("/opt/elsewhere/tool.py", (None, None)),
])
def test_a_site_is_classified_by_the_inventory_that_lists_its_file(filename, expected):
    assert site_package({"file": filename}, EVIDENCE) == expected


@pytest.mark.parametrize("filename", [PLUGIN + "/serving/experimental_route.py",
                                      VLLM + "/v1/worker/unattested.py"])
def test_a_file_under_an_inventoried_root_that_the_roster_omits_is_not_classified(filename):
    # The run attested a roster, and a file outside it is a file nothing
    # attested -- being under the right directory is not the evidence.
    assert site_package({"file": filename}, EVIDENCE) == (None, None)


def test_no_site_classifies_to_no_package():
    assert site_package(None, EVIDENCE) == (None, None)


# --- the route family --------------------------------------------------------


def test_the_route_family_is_read_from_every_frame_not_only_the_innermost():
    frames = _frames(PLUGIN + "/decode.py", PLUGIN + "/serving/nvfp4_route.py")
    assert route_family_of(frames, EVIDENCE) == "TESSERA_NVFP4"


def test_frames_through_two_route_modules_resolve_no_family():
    frames = _frames(PLUGIN + "/serving/nvfp4_route.py", PLUGIN + "/serving/fp8_route.py")
    assert route_family_of(frames, EVIDENCE) is None
    assert route_family_of(_frames(PLUGIN + "/decode.py"), EVIDENCE) is None


# --- the owner views ---------------------------------------------------------


@pytest.mark.parametrize("category", ["fixed", "candidate", "kv"])
def test_a_census_category_is_repeated_never_rewritten(category):
    stack = ["u:scope"] if category == "candidate" else []
    views, summary = _views([_row(categories=[category], stack=stack)])
    assert (views[0]["class"], views[0]["rule"]) == (category, "census")
    assert summary["by_rule"] == {"census": 1}
    assert summary["by_class"] == {category: 1}


def test_a_candidate_takes_the_outermost_unit_on_its_scope_stack():
    views, _ = _views([_row(categories=["candidate"], stack=["outer", "inner"])])
    assert views[0]["unit"] == "outer"


def test_a_candidate_outside_every_unit_interval_resolves_through_the_roster():
    views, _ = _views([_row(categories=["candidate"],
                            owners=["model:parameter:model.layers.1.mlp.down_proj.weight"])])
    assert views[0]["unit"] == "l:model.layers.1.mlp.down_proj"


def test_a_member_module_of_a_fused_unit_resolves_to_that_unit():
    # q_proj is not a roster module; it is a member of the fused qkv_proj unit,
    # and the roster index is what says so.
    views, _ = _views([_row(categories=["candidate"],
                            owners=["model:buffer:model.layers.0.self_attn.q_proj.weight_scale"])])
    assert views[0]["unit"] == "s:model.layers.0.self_attn.qkv_proj"


def test_a_candidate_whose_owner_path_names_no_single_unit_keeps_a_null_unit():
    views, summary = _views([_row(categories=["candidate"],
                                  owners=["model:parameter:model.embed_tokens.weight"])])
    assert views[0]["class"] == "candidate" and views[0]["unit"] is None
    assert "no single roster unit" in views[0]["reason"]
    assert summary["candidate_without_unit"] == 1


def test_a_shared_row_allocated_before_the_model_is_fixed_by_history_order():
    # The capture's own event order proves it predates the model, its
    # assignment and every route; nothing else is asserted about it.
    views, summary = _views([_row(categories=["shared"], allocate_index=4,
                                  owners=["runner:uva"])])
    assert (views[0]["class"], views[0]["rule"]) == ("fixed", "history_order")
    assert summary["by_rule"] == {"history_order": 1}


@pytest.mark.parametrize("owner,kind", [
    ("native:l:model.layers.1.mlp.down_proj:1:input.x", "native boundary tensor"),
    ("runner:persistent_root", "persistent runtime root"),
    ("torch.cublas:workspace", "BLAS workspace"),
    ("cudnn:cache", "library cache buffer"),
])
def test_a_shared_row_after_the_model_load_abstains_and_names_its_kind(owner, kind):
    views, summary = _views([_row(size=256, categories=["shared"], allocate_index=9,
                                  owners=[owner])])
    assert views[0]["class"] is None and views[0]["rule"] == "pending_548"
    assert kind in views[0]["reason"] and "tessera#548" in views[0]["reason"]
    assert summary["null_views"] == 1 and summary["null_bytes"] == 256


def test_more_than_one_observed_category_abstains_rather_than_picking_one():
    views, _ = _views([_row(categories=["candidate", "fixed"])])
    assert views[0]["class"] is None and views[0]["rule"] is None
    assert "more than one ownership category" in views[0]["reason"]


@pytest.mark.parametrize("filename,expected_class,expected_rule", [
    (PLUGIN + "/decode.py", "candidate", "site:plugin"),
    (VLLM + "/v1/worker/gpu_model_runner.py", "fixed", "site:vllm"),
    (SITE + "/numpy/core/_methods.py", "fixed", "site:image"),
    (OBSERVER + "/experiments/full_engine_worker.py", "observer", "site:observer"),
])
def test_an_unowned_row_is_placed_by_the_package_its_site_belongs_to(
        filename, expected_class, expected_rule):
    stack = ["u:scope"] if expected_rule == "site:plugin" else []
    views, _ = _views([_row(stack=stack)], {10: _frames(filename)})
    assert (views[0]["class"], views[0]["rule"]) == (expected_class, expected_rule)
    assert views[0]["site"]["file"] == filename


def test_an_unowned_row_with_no_python_frame_outside_torch_is_named_not_bucketed():
    views, summary = _views([_row(size=128)], {10: _frames(TORCH + "/_tensor.py")})
    assert views[0]["class"] is None and views[0]["site"] is None
    assert "no Python frame outside Torch" in views[0]["reason"]
    assert summary["null_views"] == 1 and summary["null_bytes"] == 128


def test_a_site_no_inventory_attests_is_named_with_the_file_it_names():
    views, _ = _views([_row()], {10: _frames(PLUGIN + "/serving/experimental_route.py")})
    assert views[0]["class"] is None
    assert "no run inventory attests" in views[0]["reason"]
    assert views[0]["reason"].endswith("/serving/experimental_route.py")


def test_a_load_time_plugin_allocation_takes_the_single_unit_of_its_route_family():
    # Allocated between before_model_load and ready_for_workload and never
    # freed: a load-time table, and the route frame names the family.
    frames = _frames(PLUGIN + "/decode.py", PLUGIN + "/serving/nvfp4_route.py")
    views, _ = _views([_row(allocate_index=10)], {10: frames})
    assert views[0]["rule"] == "site:plugin"
    assert views[0]["unit"] == "s:model.layers.0.self_attn.qkv_proj"
    assert views[0]["reason"] is None


def test_a_route_family_holding_two_roster_units_leaves_the_unit_unresolved():
    roster = ROSTER + [{"unit_id": "l:model.layers.2.mlp.down_proj",
                        "module": "model.layers.2.mlp.down_proj",
                        "family": "TESSERA_NVFP4", "members": []}]
    views, summary = _views([_row()], {10: _frames(PLUGIN + "/serving/nvfp4_route.py")},
                            roster=roster)
    assert views[0]["class"] == "candidate" and views[0]["unit"] is None
    assert "has 2 roster units" in views[0]["reason"]
    assert summary["candidate_without_unit"] == 1


def test_a_load_time_plugin_allocation_through_two_routes_names_the_ambiguity():
    frames = _frames(PLUGIN + "/serving/nvfp4_route.py", PLUGIN + "/serving/fp8_route.py")
    views, _ = _views([_row()], {10: frames})
    assert views[0]["unit"] is None
    assert "no single route module" in views[0]["reason"]


def test_a_freed_plugin_allocation_with_no_scope_or_owner_path_is_named():
    views, _ = _views([_row(free_completed_index=12)], {10: _frames(PLUGIN + "/decode.py")})
    assert views[0]["class"] == "candidate" and views[0]["unit"] is None
    assert views[0]["reason"] == "plugin allocation with no unit scope and no owner path"


def test_the_summary_counts_every_rule_class_and_null_byte():
    rows = [_row("census-fixed", size=1, categories=["fixed"]),
            _row("shared-early", size=2, categories=["shared"], allocate_index=1),
            _row("shared-late", size=4, categories=["shared"], allocate_index=9),
            _row("vllm", size=8, allocate_index=11),
            _row("observer", size=16, allocate_index=12),
            _row("nowhere", size=32, allocate_index=13)]
    frames = {11: _frames(VLLM + "/v1/worker/gpu_model_runner.py"),
              12: _frames(OBSERVER + "/experiments/full_engine_worker.py"),
              13: _frames("/opt/elsewhere/tool.py")}
    views, summary = _views(rows, frames)
    assert summary["by_rule"] == {"census": 1, "history_order": 1, "none": 1,
                                  "pending_548": 1, "site:observer": 1, "site:vllm": 1}
    assert summary["by_class"] == {"fixed": 3, "null": 2, "observer": 1}
    assert summary["null_views"] == 2
    assert summary["null_bytes"] == 4 + 32
    assert summary["candidate_without_unit"] == 0
    assert [view["allocation_id"] for view in views] == [row["allocation_id"] for row in rows]


# --- the external CUDA records ----------------------------------------------


MARKERS = [(100, "before_model_load"), (200, "ready_for_workload")]
WINDOWS = [(300, 400, "l:model.layers.1.mlp.down_proj")]


def _record(kind, operation, address, size, timestamp, source):
    return {"memory_kind": kind, "operation": operation, "address": address,
            "bytes": size, "timestamp_ns": timestamp, "source": source,
            "device_id": 0, "context_id": 1}


def _external(records):
    return external_record_views(records, markers=MARKERS, unit_windows=WINDOWS,
                                 evidence=EVIDENCE)


@pytest.mark.parametrize("source,expected", [
    ("/ext/tessera_nvfp4/tessera_nvfp4_abc.so", "plugin_jit_static"),
    (OBSERVER + "/libcollector.so", "observer_static"),
    (SITE + "/nvidia/cublas/lib/libcublas.so", "image_static"),
    ("/cache/triton/abc/kernel.so", "image_jit_static"),
])
def test_a_device_static_is_classified_by_the_library_that_holds_it(source, expected):
    result = _external([_record(6, "allocate", 1000, 64, 150, source)])
    assert result["schema"] == EXTERNAL_RECORDS_SCHEMA
    assert result["records"][0]["class"] == expected
    assert result["records"][0]["phase"] == "before_model_load"
    assert result["startup_static"]["by_class"] == {expected: 1}
    assert result["startup_static"]["live_bytes"] == 64
    assert result["startup_static"]["sources"] == [
        {"source": source, "live_bytes": 64, "live_allocation_count": 1}]
    assert result["unresolved"] == []


@pytest.mark.parametrize("source,reason", [
    (None, "device static with no source library"),
    ("", "device static with no source library"),
    ("/opt/rogue/libmystery.so", "device static from a library outside"),
])
def test_a_device_static_no_run_evidence_places_is_named_and_keeps_the_closure_open(
        source, reason):
    result = _external([_record(6, "allocate", 1000, 64, 150, source),
                        _record(6, "free", 1000, 64, 160, source)])
    assert [view["class"] for view in result["records"]] == [None, None]
    assert len(result["unresolved"]) == 2
    assert reason in result["unresolved"][0]["reason"]
    assert result["startup_static"]["live_bytes"] == 0


def test_an_unresolved_static_live_at_the_end_is_listed_beside_the_resolved_ones():
    # Naming a record no evidence places must not cost the record that WAS
    # placed: both are live at capture end, so both are in the per-source
    # totals and the class histogram.
    result = _external([_record(6, "allocate", 1000, 64, 150, None),
                        _record(6, "allocate", 2000, 32, 160,
                                SITE + "/nvidia/cublas/lib/libcublas.so")])
    assert result["startup_static"]["live_bytes"] == 96
    assert result["startup_static"]["live_allocation_count"] == 2
    assert result["startup_static"]["by_class"]["image_static"] == 1
    assert len(result["unresolved"]) == 1


def test_two_overlapping_external_allocations_in_a_unit_window_sum_into_the_peak():
    result = _external([_record(3, "allocate", 5000, 100, 310, "/lib/libfoo.so"),
                        _record(3, "allocate", 6000, 40, 320, "/lib/libfoo.so"),
                        _record(3, "free", 5000, 100, 330, "/lib/libfoo.so"),
                        _record(3, "free", 6000, 40, 340, "/lib/libfoo.so")])
    assert result["external_native_peak_bytes"] == 140
    assert {view["class"] for view in result["records"]} == {"unit_window_external"}
    assert result["records"][0]["unit_window"] == "l:model.layers.1.mlp.down_proj"
    assert result["external_resident"]["live_bytes"] == 0


def test_an_allocate_and_free_pair_inside_a_window_does_not_double_count():
    # A simultaneous sweep, not a sum of sizes: the second allocation reuses
    # the window after the first is gone.
    result = _external([_record(3, "allocate", 5000, 100, 310, "/lib/libfoo.so"),
                        _record(3, "free", 5000, 100, 320, "/lib/libfoo.so"),
                        _record(3, "allocate", 7000, 100, 330, "/lib/libfoo.so"),
                        _record(3, "free", 7000, 100, 340, "/lib/libfoo.so")])
    assert result["external_native_peak_bytes"] == 100


def test_an_external_allocation_outside_every_window_is_a_library_external():
    result = _external([_record(3, "allocate", 8000, 256, 150, "/lib/libnccl.so")])
    assert result["records"][0]["class"] == "library_external"
    assert result["external_native_peak_bytes"] == 0
    assert result["external_resident"]["live_bytes"] == 256
    assert result["external_resident"]["records"] == [
        {"bytes": 256, "source": "/lib/libnccl.so", "unit_window": None}]
    assert result["external_resident"]["retained_from_unit_windows"] == []


def test_an_in_window_external_never_freed_is_reported_as_retained():
    result = _external([_record(3, "allocate", 5000, 100, 310, "/lib/libfoo.so")])
    assert result["external_native_peak_bytes"] == 100
    assert result["external_resident"]["retained_from_unit_windows"] == [
        {"bytes": 100, "source": "/lib/libfoo.so",
         "unit_window": "l:model.layers.1.mlp.down_proj"}]


def test_an_unsupported_memory_kind_is_named_rather_than_dropped():
    result = _external([_record(9, "allocate", 1000, 64, 150, "/lib/libfoo.so")])
    assert result["records"][0]["class"] is None
    assert "unsupported memory kind 9" in result["unresolved"][0]["reason"]
    assert result["record_count"] == 1


# --- the boundary geometry witness -------------------------------------------


WITNESS_ROSTER = [
    {"unit_id": "l:model.layers.0.mlp.down_proj", "module": "model.layers.0.mlp.down_proj",
     "family": "TESSERA_NVFP4", "members": []},
    {"unit_id": "l:model.layers.1.mlp.down_proj", "module": "model.layers.1.mlp.down_proj",
     "family": "TESSERA_BF16", "members": []},
]
WITNESS_STEPS = [("step:0", 0, 100)]


def _boundary_row(allocation_id, size, index, unit_id, kind):
    return _row(allocation_id, size=size, allocate_index=index,
                owners=[f"native:{unit_id}:1:{kind}"])


@pytest.mark.parametrize("second_bytes,agree", [(512, True), (768, False)])
def test_boundary_bytes_are_compared_across_the_families_of_one_role_and_step(
        second_bytes, agree):
    rows = [_boundary_row("a", 512, 3, "l:model.layers.0.mlp.down_proj", "input.x"),
            _boundary_row("b", second_bytes, 4, "l:model.layers.1.mlp.down_proj", "input.x")]
    witness = boundary_geometry_witness(rows, WITNESS_ROSTER, WITNESS_STEPS)
    assert witness["schema"] == GEOMETRY_WITNESS_SCHEMA
    cell, = witness["cells"]
    assert (cell["kind"], cell["role"], cell["step"]) == ("input.x", "down_proj", "step:0")
    assert set(cell["families"]) == {"TESSERA_NVFP4", "TESSERA_BF16"}
    assert cell["agree_across_families"] is agree
    assert witness["cells_with_several_families"] == 1
    assert witness["cells_agreeing_across_families"] == (1 if agree else 0)


def test_a_row_that_is_not_a_boundary_tensor_is_not_in_the_witness():
    rows = [_row("plain", owners=["model:parameter:model.layers.0.mlp.down_proj.weight"])]
    assert boundary_geometry_witness(rows, WITNESS_ROSTER, WITNESS_STEPS)["cells"] == []


# --- the transient gap witness -----------------------------------------------


TRANSIENT_ROSTER = [
    {"unit_id": "u0", "module": "model.layers.0.mlp.down_proj", "family": "F", "members": []},
    {"unit_id": "u1", "module": "model.layers.1.mlp.down_proj", "family": "F", "members": []},
    {"unit_id": "u2", "module": "model.layers.2.mlp.down_proj", "family": "F", "members": []},
]
TRANSIENT_STEPS = [("step:0", 0, 50), ("step:1", 50, 100)]
TRANSIENT_INTERVALS = [(10, 15, "i0", "u0"), (20, 25, "i1", "u1"), (30, 35, "i2", "u2"),
                       (60, 65, "i3", "u0"), (70, 75, "i4", "u1"), (80, 85, "i5", "u2")]


def _vllm_view(allocation_id, name="allocate"):
    return {"allocation_id": allocation_id, "class": "fixed", "unit": None,
            "rule": "site:vllm", "reason": None,
            "site": {"file": VLLM + "/model_executor/layers/layernorm.py", "name": name,
                     "line": 1, "package": "vllm",
                     "relative": "model_executor/layers/layernorm.py"}}


def _transient(rows):
    views = [_vllm_view(row["allocation_id"]) for row in rows]
    return transient_gap_witness(rows, views, TRANSIENT_INTERVALS, TRANSIENT_ROSTER,
                                 TRANSIENT_STEPS)


def test_per_layer_transient_signatures_that_agree_say_so():
    rows = [_row("l0", size=64, allocate_index=17), _row("l1", size=64, allocate_index=27)]
    witness = _transient(rows)
    assert witness["schema"] == TRANSIENT_WITNESS_SCHEMA
    step = witness["steps"]["step:0"]
    assert step["layers"] == 2 and step["distinct_signatures"] == 1
    assert step["all_layers_agree"] is True
    assert step["signature_groups"] == [{"layers": [0, 1], "rows": 1, "bytes": 64}]


def test_one_layer_whose_transients_differ_is_a_second_signature():
    rows = [_row("l0", size=64, allocate_index=17), _row("l1", size=96, allocate_index=27)]
    step = _transient(rows)["steps"]["step:0"]
    assert step["distinct_signatures"] == 2 and step["all_layers_agree"] is False
    assert sorted(group["layers"] for group in step["signature_groups"]) == [[0], [1]]


def test_rows_after_the_last_unit_of_a_step_are_bucketed_apart_from_every_layer():
    rows = [_row("l0", size=64, allocate_index=17), _row("tail", size=8, allocate_index=40)]
    step = _transient(rows)["steps"]["step:0"]
    assert step["layers"] == 1
    assert step["outside_layers"]["after-last-unit"] == {"rows": 1, "bytes": 8}
    assert "pre-first-unit" not in step["outside_layers"]


def test_rows_before_the_first_unit_of_a_step_belong_to_that_step_not_the_last_layer():
    # The previous unit is resolved WITHIN the step, so the rows opening the
    # second step are its own bucket rather than the first step's last layer.
    rows = [_row("pre0", size=4, allocate_index=5), _row("pre1", size=6, allocate_index=55)]
    witness = _transient(rows)
    assert witness["steps"]["step:0"]["outside_layers"] == {
        "pre-first-unit": {"rows": 1, "bytes": 4}}
    assert witness["steps"]["step:1"]["outside_layers"] == {
        "pre-first-unit": {"rows": 1, "bytes": 6}}
    assert witness["steps"]["step:1"]["layers"] == 0


def test_only_stock_engine_rows_are_in_the_transient_witness():
    rows = [_row("l0", size=64, allocate_index=17)]
    views = [dict(_vllm_view("l0"), rule="site:plugin")]
    assert transient_gap_witness(rows, views, TRANSIENT_INTERVALS, TRANSIENT_ROSTER,
                                 TRANSIENT_STEPS)["steps"] == {}


# --- the dense startup check -------------------------------------------------


DENSE_SCHEMA = "tessera.full_engine_dense_startup_observation.v1"


def _dense(units=None, memory_allocated_bytes=4096, schema=DENSE_SCHEMA):
    return {"schema": schema, "memory_allocated_bytes": memory_allocated_bytes,
            "units": units if units is not None else {
                "u0": {"family": "TESSERA_NVFP4", "manifest_resident_bytes_resident_mode": 512}}}


def _resident_candidate(allocation_id, size, unit):
    row = _row(allocation_id, size=size, allocate_index=10)
    view = {"allocation_id": allocation_id, "class": "candidate", "unit": unit,
            "rule": "census", "reason": None, "site": None}
    return row, view


def test_per_unit_exact_agreement_closes_the_dense_startup_check():
    row, view = _resident_candidate("a", 512, "u0")
    check = dense_startup_check([row], [view], _dense(), ready_index=20)
    assert check["closed"] is True
    assert check["units"]["u0"]["difference_bytes"] == 0
    assert check["units_disagreeing"] == [] and check["candidate_units_outside_manifest"] == []
    assert check["allocator_sample_bounds_ledger"] is True
    assert check["ledger_live_bytes_at_ready_for_workload"] == 512
    # An agreeing unit has nothing to enumerate and prices nothing unpriced.
    assert check["units"]["u0"]["resident_rows"] is None
    assert check["manifest_unpriced_resident_bytes"] == 0


def test_a_one_byte_per_unit_disagreement_refuses_the_dense_startup_check():
    row, view = _resident_candidate("a", 513, "u0")
    check = dense_startup_check([row], [view], _dense(), ready_index=20)
    assert check["closed"] is False
    assert check["units_disagreeing"] == ["u0"]
    assert check["units"]["u0"]["difference_bytes"] == 1


def test_a_candidate_unit_outside_the_manifest_is_listed_and_refuses():
    rows_views = [_resident_candidate("a", 512, "u0"), _resident_candidate("b", 8, "u9")]
    check = dense_startup_check([row for row, _ in rows_views],
                                [view for _, view in rows_views], _dense(), ready_index=20)
    assert check["candidate_units_outside_manifest"] == ["u9"]
    assert check["closed"] is False


def test_an_allocator_sample_below_the_ledger_live_bytes_refuses():
    row, view = _resident_candidate("a", 512, "u0")
    check = dense_startup_check([row], [view], _dense(memory_allocated_bytes=100),
                                ready_index=20)
    assert check["allocator_sample_bounds_ledger"] is False
    assert check["closed"] is False


@pytest.mark.parametrize("dense", [None, {}, _dense(schema="tessera.other.v1")])
def test_an_observation_of_another_schema_is_no_dense_startup_check(dense):
    row, view = _resident_candidate("a", 512, "u0")
    assert dense_startup_check([row], [view], dense, ready_index=20) is None


# --- the observation ---------------------------------------------------------


def test_the_observation_carries_one_view_per_allocation_under_one_schema():
    rows = [_row("a", categories=["fixed"])]
    views, summary = _views(rows)
    observation = ownership_observation(
        views=views, summary=summary, evidence=EVIDENCE,
        external_records=_external([]), geometry_witness=boundary_geometry_witness(
            rows, ROSTER, [("step:0", 0, 100)]),
        transient_witness=transient_gap_witness(rows, views, [], ROSTER, [("step:0", 0, 100)]),
        dense_startup_check=None)
    assert observation["schema"] == OWNERSHIP_OBSERVATION_SCHEMA
    assert observation["views"]["schema"] == OWNER_VIEWS_SCHEMA
    assert set(observation["views"]["rules"]) == set(RULES)
    assert observation["views"]["evidence"]["plugin_package_path"] == PLUGIN
    assert observation["dense_startup_check"] is None
    assert views_by_allocation({"owner_views": observation}) == {"a": views[0]}


@pytest.mark.parametrize("ledger", [{}, {"owner_views": None},
                                    {"owner_views": {"schema": "tessera.other.v1"}}])
def test_a_ledger_with_no_ownership_observation_yields_no_views(ledger):
    assert views_by_allocation(ledger) == {}


def test_a_manifest_below_the_ledger_names_every_row_it_does_not_price():
    # The excess is not absorbed into a residual: it is a byte count and a
    # list of the rows that carry it, by census owner or allocation site.
    priced = _row("priced", size=512, allocate_index=10,
                  owners=["model.layers.0.self_attn.o_proj.weight"])
    priced_view = {"allocation_id": "priced", "class": "candidate", "unit": "u0",
                   "rule": "census", "reason": None, "site": None}
    unpriced = _row("unpriced", size=64, allocate_index=11)
    unpriced_view = {"allocation_id": "unpriced", "class": "candidate", "unit": "u0",
                     "rule": "site:plugin", "reason": None,
                     "site": {"file": PLUGIN + "/decode.py", "name": "build_table", "line": 3,
                              "package": "plugin", "relative": "decode.py"}}
    check = dense_startup_check([priced, unpriced], [priced_view, unpriced_view],
                                _dense(), ready_index=20)
    assert check["closed"] is False
    assert check["units"]["u0"]["agree"] is False
    assert check["units"]["u0"]["difference_bytes"] == 64
    assert check["manifest_unpriced_resident_bytes"] == 64
    assert check["units"]["u0"]["resident_rows"] == [
        {"bytes": 512, "allocation_id": "priced",
         "owner": "model.layers.0.self_attn.o_proj.weight", "site": None, "rule": "census"},
        {"bytes": 64, "allocation_id": "unpriced", "owner": None,
         "site": "decode.py:build_table", "rule": "site:plugin"}]


def test_a_ledger_below_the_manifest_disagrees_and_prices_no_unpriced_bytes():
    # The other direction: the manifest claims residency the ledger never saw.
    # It refuses just as loudly, and there are no unpriced bytes to name.
    row, view = _resident_candidate("a", 256, "u0")
    check = dense_startup_check([row], [view], _dense(), ready_index=20)
    assert check["closed"] is False
    assert check["units"]["u0"]["difference_bytes"] == -256
    assert check["manifest_unpriced_resident_bytes"] == 0
    assert [entry["allocation_id"] for entry in check["units"]["u0"]["resident_rows"]] == ["a"]


def test_a_resident_row_with_several_census_owners_carries_them_all():
    row = _row("aliased", size=600, allocate_index=10,
               owners=["embedding.weight", "lm_head.weight"])
    view = {"allocation_id": "aliased", "class": "candidate", "unit": "u0",
            "rule": "census", "reason": None, "site": None}
    check = dense_startup_check([row], [view], _dense(), ready_index=20)
    assert check["units"]["u0"]["resident_rows"][0]["owner"] == ["embedding.weight",
                                                                 "lm_head.weight"]
