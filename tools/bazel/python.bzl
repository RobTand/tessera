"""Macros the generated BUILD files use: one library per module, one test per file and lane.

tools/bazel/build_files.py writes the targets. These macros add what every
module or test shares, so the generated files hold only per-module evidence.
"""

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@ts_pip//:requirements.bzl", "requirement")
load("//tools/bazel:lanes.bzl", "BYTES_PACKAGES", "CONFTEST", "TORCH_PACKAGES")
load("//tools/bazel:nonhermetic.bzl", "nonhermetic")
load("//tools/bazel:test_inputs.bzl", "TEST_INPUTS")

# What every hosted gate shard has: tests/conftest.py, pyproject.toml's pytest
# options and the gate's collection plugin. Each test's budgets are an argument.
_TEST_DATA = [
    "//:pyproject.toml",
    "//tests:gate_flaky_inputs",
]
_TEST_DEPS = [
    CONFTEST,
    "//tools/bazel:runner",
    "//tools:ci/gate_collect",
    "//tools:ci/gate_run",
]
_LANE_PACKAGES = {"bytes": BYTES_PACKAGES, "torch": TORCH_PACKAGES}

def _unique(labels):
    """A generated label and a shared one may name the same target once each."""
    unique = {}
    for label in labels:
        unique[str(native.package_relative_label(label))] = label
    return list(unique.values())

def ts_py_library(name, srcs, deps = [], data = [], **kwargs):
    py_library(name = name, srcs = srcs, deps = deps, data = data, **kwargs)

def ts_py_test(name, path, library, lane, args = [], tags = [], deps = [], data = [], **kwargs):
    """Run the test file at repository path `path`, whose module is `library`, in one gate lane.

    The torch lane has the CPU torch wheel; the bytes lane has only the
    no-torch pins, as the hosted gate's no-torch lane does. Every test gets
    the cc toolchain's compiler: torch builds C++ extensions and Inductor
    kernels with it at run time.
    """
    extra = TEST_INPUTS.get("//%s:%s" % (native.package_name(), name.removesuffix(".bytes")))
    extra_tags, exec_properties, env = nonhermetic(name)
    py_test(
        name = name,
        main = "//tools/bazel:python_runner.py",
        srcs = ["//tools/bazel:python_runner.py"],
        args = [path, lane] + args,
        deps = _unique([library] + deps + _TEST_DEPS + [requirement(package) for package in _LANE_PACKAGES[lane]]),
        data = _unique(_TEST_DATA + data + (extra.data if extra else [])),
        tags = tags + extra_tags + ["gate-lane-" + lane],
        exec_properties = exec_properties,
        env = dict({"TS_BAZEL_CC": "$(CC)"}, **env),
        toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
        **kwargs
    )
