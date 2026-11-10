"""Tests that cannot run hermetically, each with its reason.

The remote platform turns networking off and runs each test in a container
that has a few CPUs of a larger host, as an unprivileged user, on a runfiles
tree with every input executable and no .git. A test listed here gets back
what its tags name. Keep this table short and the reasons true; the PR that
adds an entry says why the test can't use a declared input instead. Entries
are keyed by the test file's target; its `.bytes` lane shares them.

- `runner`: the test runs on the Remote Bazel runner itself, with pytest in the
  runner's git checkout, as on a hosted gate shard. Its result is never cached,
  because it reads that checkout, which must hold full history (a shallow
  clone makes git fetch inside the test). The runner is amd64, so the arm64 pass
  (`--config=remote-arm64`) leaves these out.
- `requires-network`: the remote executor gets a network.
- `gpu`: every node in the file needs a CUDA device and skips on a CPU
  executor. The target is `manual`, so the cloud gate's `//...` leaves it
  out; a GPU lane selects it by the tag. Its CUDA run stays where it is today
  (tools/merge_suite.py through Flux, --strict-cuda); that lane is not yet qualified.
"""

_GPU = "every node needs a CUDA device; on a CPU executor the whole file skips."
_GIT = "it reads the git repository (rev-parse, ls-files or ls-tree), which a remote executor's runfiles don't have."

NONHERMETIC = {
    "//tests:test_dense_prepared_census": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_encode_host_sync_contract": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_glm53_shared_fold_cuda": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_kernel": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_lane_planes": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_mhc_fusion_cuda": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_native_a4_loader_staging": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_native_a4_serving": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_native_fp8_quant": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_native_window_moe_method": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_regdirect_class_binding_cuda": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_regdirect_op_cuda": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_routed_terminal_cuda": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_selected_window_kernel": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_serving_head_route": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_serving_mla_mask_runtime_gpu": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_serving_moe_tp2": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_serving_native_window": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_shipped_checkpoint_minor7": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_stock": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_tcq_fused": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_tcq_graph": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_gemm": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_gemm_grouped": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_graph": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_lut_axis": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_lut_compact": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_viterbi_best_form": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_window_viterbi_fast": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_wire_device_unpack": struct(tags = ["gpu"], reason = _GPU),
    "//tests:test_cuda_surface": struct(tags = ["runner"], reason = "the source identity it verifies is measured with git, which a remote executor's runfiles don't have."),
    "//tests:test_merge_suite": struct(tags = ["runner"], reason = "its real producer publication must bind a Git-measured checkout identity to the immutable native launch; partial runfiles cannot attest that source."),
    "//tests:test_bazel_build": struct(tags = ["runner"], reason = "it regenerates every BUILD file, which lists the repository's files with git ls-files."),
    "//tests:test_gate_selection": struct(tags = ["runner"], reason = _GIT),
    "//tests:test_directory_read_census": struct(tags = ["runner"], reason = "it measures every tracked Python source for the repository directory-read ceiling and verifies that complete Git population; partial runfiles can silently omit readers."),
    "//tests:test_graph_impact": struct(tags = ["runner"], reason = "it measures the tracked checkout with git ls-files and compares the merge-base graph; a declared file subset cannot represent that repository-wide contract."),
    "//tests:test_runtime_image_pin": struct(tags = ["runner"], reason = "it lists every tracked file with git ls-files to find copies of the pin."),
    "//tests:test_serving_identity_1095": struct(tags = ["runner"], reason = "it reads a historical commit's tree with git ls-tree."),
    "//tests:test_serving_manifest_json": struct(tags = ["runner"], reason = _GIT),
    "//tests:test_t4_fused_qualify": struct(tags = ["runner"], reason = _GIT),
    "//tests:test_t4_prepare_inputs_release_gate": struct(tags = ["runner"], reason = _GIT),
    "//tests:test_test_imports": struct(tags = ["runner"], reason = "it compares the actual checkout's test imports with its Git merge base using ls-tree and archive, which remote runfiles cannot supply."),
}

def nonhermetic(name):
    """(tags, exec_properties, env) for one test target."""
    entry = NONHERMETIC.get("//%s:%s" % (native.package_name(), name.removesuffix(".bytes")))
    if not entry:
        return [], {}, {}
    tags, exec_properties, env = list(entry.tags), {}, {}
    if "requires-network" in entry.tags:
        exec_properties["dockerNetwork"] = "bridge"
    if "gpu" in entry.tags:
        tags.append("manual")
    if "runner" in entry.tags:
        # It reads the checkout, which Bazel does not hash: never reuse a result.
        tags += ["no-remote-exec", "no-sandbox", "no-cache", "external"]
        env["TS_BAZEL_LANE"] = "runner"
    return tags, exec_properties, env
