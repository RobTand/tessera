"""FlashKDA's prefill kernels, built by Tessera with the serving image's own flags.

GLM-5.3's KDA layers run their prefill through FlashKDA (vllm-project/FlashKDA
at ``17a037d9``, MIT), which the pinned vLLM builds into its ``_flashkda_C``
extension.  The KDA prefill fusions change FlashKDA's first kernel, so Tessera
carries FlashKDA's device sources verbatim under ``serving/csrc/flashkda/``
(with FlashKDA's MIT license) and builds them here.

THE BUILD IS THE IMAGE'S BUILD.  A fused kernel is only comparable with stock
when everything it does not change compiles to stock's machine code.  vLLM
af5b4857 builds FlashKDA with the flags in
``cmake/external_projects/flashkda.cmake``; step 1 of the KDA fusion work
(PrismaBuild 46d6b23a) showed that nvcc 13.0.88 with those flags reproduces the
image's sm_120f cubin kernel for kernel.  :func:`build_stock` compiles
FlashKDA's own ``fwd_launch.cu`` through this module's build path, and
``experiments/kda/build_gate.py`` compares its cubin with the image's inside the
image.  No variant is trusted before that comparison passes.

FLAGS.  ``torch.utils.cpp_extension`` adds four ``-D__CUDA_NO_*`` defines to
every CUDA compile, ahead of ``extra_cuda_cflags``.  FlashKDA builds without
them, so :data:`UNDO_TORCH_DEFINES` undoes each with ``-U``, which the
preprocessor applies in command-line order.  An explicit ``-gencode`` in
``extra_cuda_cflags`` makes torch skip its own architecture flags.  There is
no ``-lineinfo``: the image's build has none, and it changes the cubin.

CUTLASS.  Not vendored: FlashKDA needs about 900 of its headers.
``TESSERA_FLASHKDA_CUTLASS`` names a directory that holds CUTLASS
``5c149f52``'s ``include``, ``examples/common`` and ``tools/util/include`` and a
``COMMIT`` file naming that commit.  The build refuses any other commit and an
unset variable; it never guesses a path.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = [
    "CUTLASS_COMMIT",
    "CUTLASS_ENV",
    "FLASHKDA_COMMIT",
    "NVCC_FLAGS",
    "STOCK_MODULE_NAME",
    "UNDO_TORCH_DEFINES",
    "VENDORED_SHA256",
    "FlashKdaBuildError",
    "build_stock",
    "cuda_cflags",
    "cutlass_root",
    "include_dirs",
    "vendored_dir",
    "verify_vendored",
]

#: The FlashKDA commit vLLM af5b4857 pins, and the one ``serving/csrc/flashkda`` holds.
FLASHKDA_COMMIT = "17a037d98da546deb4591e967cf961a43c034d8b"
#: FlashKDA 17a037d9's CUTLASS submodule commit.
CUTLASS_COMMIT = "5c149f52a436782210263fb2f19b354443a61c6a"
#: Names the CUTLASS header directory (see the module docstring).
CUTLASS_ENV = "TESSERA_FLASHKDA_CUTLASS"

#: sha256 of each vendored file, relative to ``serving/csrc/flashkda``: the bytes at
#: FlashKDA 17a037d9 (``csrc/fwd.h``, ``csrc/smxx/*``, ``LICENSE``).  A build refuses
#: a tree whose copy differs, so an edit to a vendored file is a test failure here
#: rather than a silent change to "stock".
VENDORED_SHA256 = {
    "LICENSE": "05f1750624d6ab5f6dd59dea79e3156f4fec6b3065c94aeeaf265df677f9e6e8",
    "fwd.h": "e6602af11632257ef6427d7ce3c3a2389a18626ec3f5b6c73420cf8fe19e8a59",
    "smxx/fwd_kernel1.cuh": "dd25a959999ed49539a76b9f49f6505f9e4143d21c42271488d978830a1694ad",
    "smxx/fwd_kernel2.cuh": "9532bb9a3df2e14cd601ebd38151660fc3440abd748a16c16429f4483d35f094",
    "smxx/fwd_launch.cu": "3861a455c193c89b1f392fe37cb254475bdef81e2c64efcf227afa7ab9110aab",
    "smxx/utils.cuh": "54320ce0e163308133a490bc6e782bbc5c1ab3c7b608b0ce9619fefdf25d63d7",
}

#: vLLM af5b4857's FlashKDA nvcc flags for CUDA 13, as step 1 reproduced them, minus
#: the output mode (``-cubin``) and the include paths (:func:`include_dirs`).
NVCC_FLAGS = (
    "-gencode", "arch=compute_120f,code=sm_120f",
    "-std=c++17",
    "-DNDEBUG", "-DENABLE_FP8", "-DUSE_CUDA", "-UPy_LIMITED_API",
    "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "--use_fast_math", "-O3",
)

#: The defines ``torch.utils.cpp_extension`` adds to every CUDA compile, undone.
UNDO_TORCH_DEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)

#: The build gate's library: FlashKDA's own ``fwd_launch.cu`` and nothing else.
STOCK_MODULE_NAME = "tessera_flashkda_stock"


class FlashKdaBuildError(RuntimeError):
    """The FlashKDA sources, CUTLASS or the toolchain are not the pinned ones."""


def vendored_dir() -> Path:
    """``serving/csrc/flashkda``, resolved from this package (never repo-root arithmetic)."""
    return Path(__file__).resolve().parent / "serving" / "csrc" / "flashkda"


def verify_vendored(root: Path | None = None) -> None:
    """Refuse a vendored tree whose bytes are not FlashKDA 17a037d9's."""
    root = vendored_dir() if root is None else Path(root)
    bad = []
    for rel, want in VENDORED_SHA256.items():
        path = root / rel
        try:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            bad.append(f"{rel}: {exc}")
            continue
        if got != want:
            bad.append(f"{rel}: sha256 {got}, FlashKDA {FLASHKDA_COMMIT[:8]} has {want}")
    if bad:
        raise FlashKdaBuildError("vendored FlashKDA sources differ: " + "; ".join(bad))


def cutlass_root() -> Path:
    """The CUTLASS header directory ``TESSERA_FLASHKDA_CUTLASS`` names, checked."""
    value = os.environ.get(CUTLASS_ENV, "").strip()
    if not value:
        raise FlashKdaBuildError(
            f"{CUTLASS_ENV} is not set; it names a directory holding CUTLASS "
            f"{CUTLASS_COMMIT[:8]}'s include, examples/common and tools/util/include "
            "plus a COMMIT file")
    root = Path(value)
    try:
        commit = (root / "COMMIT").read_text().strip()
    except OSError as exc:
        raise FlashKdaBuildError(f"{CUTLASS_ENV}={root}: no readable COMMIT file ({exc})") from exc
    if commit != CUTLASS_COMMIT:
        raise FlashKdaBuildError(
            f"{CUTLASS_ENV}={root} holds CUTLASS {commit!r}; FlashKDA "
            f"{FLASHKDA_COMMIT[:8]} builds with {CUTLASS_COMMIT}")
    missing = [d for d in ("include", "examples/common", "tools/util/include")
               if not (root / d).is_dir()]
    if missing:
        raise FlashKdaBuildError(f"{CUTLASS_ENV}={root} lacks {missing}")
    return root


def include_dirs(cutlass: Path | None = None) -> list[str]:
    """FlashKDA's include path, in vLLM's order: its csrc, then CUTLASS's three."""
    cutlass = cutlass_root() if cutlass is None else Path(cutlass)
    return [str(vendored_dir()), str(cutlass / "include"), str(cutlass / "examples" / "common"),
            str(cutlass / "tools" / "util" / "include")]


def cuda_cflags() -> list[str]:
    """``extra_cuda_cflags`` for every FlashKDA translation unit Tessera builds."""
    return [*NVCC_FLAGS, *UNDO_TORCH_DEFINES]


def _build_directory(module: str) -> str:
    from tessera.jit_build_lock import GUARDED_BUILD_SUFFIX
    from tessera.serving.backend import platform_token

    import torch

    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-flashkda")
    build = os.path.join(root, f"{module}_{platform_token(torch=torch)}") + GUARDED_BUILD_SUFFIX
    os.makedirs(build, exist_ok=True)
    return build


def build_stock(*, verbose: bool = False) -> tuple[str, str]:
    """Compile FlashKDA's own ``fwd_launch.cu`` through this build path.

    Returns ``(library, build_directory)``.  The library holds every kernel the
    image's ``_flashkda_C`` holds and no binding: it is the build gate's subject,
    loaded with ``torch.ops.load_library`` and never called.  Needs no GPU (set
    ``TESSERA_PLATFORM_TOKEN`` on a box without one).
    """
    import torch
    from torch.utils.cpp_extension import load

    from tessera.jit_build_lock import jit_build_lock
    from tessera.serving.backend import ensure_toolchain_on_path

    verify_vendored()
    includes = include_dirs()
    ensure_toolchain_on_path(torch)
    build = _build_directory(STOCK_MODULE_NAME)
    with jit_build_lock(build):
        library = load(
            name="tessera_flashkda_stock",  # literal: the contract scanner reads it
            sources=[str(vendored_dir() / "smxx" / "fwd_launch.cu")],
            extra_cuda_cflags=cuda_cflags(), extra_include_paths=includes,
            build_directory=build, verbose=verbose, is_python_module=False)
    return str(library), build
