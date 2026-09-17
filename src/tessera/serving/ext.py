"""Toolchain resolution and the published extension table (no loader of its own).

The package builds **one** JIT extension today: the window-body GEMV
(``csrc/window_gemv.cu``), asked for by ``tessera.kernel_window_gemv`` and
published here as the ``tessera_window_gemv`` entry the runtime contract and
the window lanes' predicates read.  This module owns the parts every JIT build
in the package shares -- finding an ``nvcc``, a ``ninja`` and the CUDA include
directories without guessing, and the vocabulary the contract publishes
(``NATIVE_EXTENSIONS``, the fallback statuses, the lane field lists) -- plus
the CUDA-resolve helpers ``backend.py`` and the window loader call.

The span-2 NVFP4 decoder and its loader (``tessera_nvfp4.cu``,
``get_tessera_ext``) were RETIRED with the A4 whole-weight expansion: the A4
lanes now decode the compact loader's packed planes in-kernel
(``tessera.kernel_a4``), and the independent correctness oracle --
``tessera.stock.materialize_stock`` on a parsed unit -- lives in the tests, not
in a serving path.  Historical receipts that name ``native_span2`` remain
history; nothing here publishes that decoder as loadable.

TOOLCHAIN.  Finding ``nvcc``/``ninja`` is not always the operator's job: torch
freezes ``cpp_extension.CUDA_HOME`` at import and resolves it to
``/usr/local/cuda`` whenever that path merely EXISTS, so a box whose
``update-alternatives`` link points at a PARTIAL install fails every build with
``nvcc: not found`` while a complete toolkit sits one directory away.
:func:`_resolve_cuda_home` finishes the search torch starts, prefers the
toolkit whose version matches the one torch was built against, and ADOPTS what
it selects into the frozen global (an operator's own
``CUDA_HOME``/``CUDA_PATH`` always wins).  ``_resolve_ninja`` looks beside
``sys.executable`` when ``PATH`` does not carry one, because a venv invoked by
absolute path (every non-login ssh) has its own ``bin`` off ``PATH``.
:func:`toolchain_report` is the public probe: a test that skips on a missing
toolchain has to tell "no compiler on this box" from "the compiler is here and
the build broke".
"""
from __future__ import annotations

import os
import shutil
import sys

from ..kernel_roster import SUPPORTED_RATES, WINDOW_BITS_SUPPORTED

__all__ = [
    "NATIVE_EXTENSIONS",
    "WINDOW_GEMV_MODULE_NAME",
    "WINDOW_GEMV_SOURCE",
    "FALLBACK_REFUSED",
    "FALLBACK_STATUSES",
    "FALLBACK_SUBSTITUTED",
    "MATCH_BASENAME_FNMATCH",
    "LANE_FIELDS",
    "LANE_REQUIREMENT_FIELDS",
    "WINDOW_GEMV_LANE",
    "NativeKernelUnavailableError",
    "IncompleteInstallError",
    "csrc_dir",
    "native_source_path",
    "substitutes_when_unavailable",
    "toolchain_report",
]

#: The window-body GEMV's JIT module name, asked for by
#: ``tessera.kernel_window_gemv`` (``load(name="tessera_window_gemv", ...)``).
#: A full NAME, not a prefix: that load path carries no build-identity hash,
#: so the library on disk is ``tessera_window_gemv.so`` exactly.  One constant
#: for the contract table and for ``fp8_gemv``'s reachability test, so a rename
#: on either side is a test failure rather than a quietly short table.
WINDOW_GEMV_MODULE_NAME = "tessera_window_gemv"

#: The source, relative to this package -- the form the contract publishes, so
#: a consumer resolves it the same way ``csrc_dir`` does.  It is the ONE
#: file of that name in the package: the loader that JIT-builds it resolves
#: the same entry through :func:`native_source_path`, so the published path
#: and the built path are one inode, not two copies kept equal by a test
#: (#134: the window GEMV was built from ``tessera/csrc/`` and published from
#: ``tessera/serving/csrc/``).
WINDOW_GEMV_SOURCE = "csrc/window_gemv.cu"

#: How a consumer turns ``filename_glob`` into a decision: ``fnmatch`` it
#: against the BASENAME of a mapped ``.so``.  A value, not prose, because the
#: whole point of publishing the name is that a fingerprint can match it
#: without knowing whether the string is a stem, a prefix or a pattern.
MATCH_BASENAME_FNMATCH = "basename_fnmatch"

#: What a serve does when a native extension cannot build.  ``substituted``
#: means a named substitute decoder ran and the serve is a DIFFERENT numeric
#: object than the native one; ``refused`` means no serve exists at all, so an
#: absent ``.so`` and a route record in that mode is an impossible pair.
FALLBACK_SUBSTITUTED = "substituted"
FALLBACK_REFUSED = "refused"
FALLBACK_STATUSES = (FALLBACK_SUBSTITUTED, FALLBACK_REFUSED)

#: The fields of a ``lane`` block: the decoder a serve on this extension's own
#: lane stamps, and (optionally) what a unit's WIRE must be before the lane can
#: read it at all.
LANE_FIELDS = ("decoder", "requires")
#: The wire predicates a ``lane.requires`` block may state.  The first four
#: are functions of a ``(grid, q256)`` pair; the rest are the DECORATION
#: classes a unit can carry beyond its rung -- RELEASE overrides, diagonals,
#: rotation, a TP shard's start state, the grid's arity -- which the loader
#: refuses by name and the published predicate therefore states too (#264):
#: a predicate narrower than the loader made the byte-time preflight publish
#: READABLE, exit 0, for wire every module refuses at load.  The plan-time
#: gate decides them from the plan's own statement (the exporter writes
#: whole, undecorated units; a caller planning otherwise says so); the
#: byte-time gate reads them off the wire, which is where a start state --
#: a property of a LATER slice, ``layout.slice_unit`` -- is caught.
LANE_REQUIREMENT_FIELDS = ("column_rates", "window_bits", "body", "plane",
                           "release_overrides", "diagonals", "rotation",
                           "start_state", "grid_arities")

#: The window GEMV's lane, and the reason this block exists.
#:
#: ``kernel_window_gemv`` repacks each column's code stream at that column's
#: OWN rate, and the kernel has a lane only where 16 rows of R-bit codes are
#: one 64-, 32- or 16-bit chunk (``chunk_width_supported`` in the ``.cu``) --
#: R = 3 would need 6-byte lanes.  So a unit is readable by this lane only
#: when EVERY column rate is in ``kernel_roster.SUPPORTED_RATES`` and its
#: window is in ``WINDOW_BITS_SUPPORTED``; the enforcement point is
#: ``kernel_window_gemv.lane_refusal_for_parsed``, which decides THIS block
#: through ``scheme.decide_lane_requirements`` (``repack_window_body`` keeps
#: the rate check as the packing backstop).
#:
#: Published because the constraint is a PRODUCER's problem.  A rung is a root
#: rate, and ``grammar.bresenham_rate_schedule`` mixes the two rates
#: bracketing it -- so q256 1006 (root 3.93) is columns at rate 3 and 4, and
#: EVERY unit of such a checkpoint refuses this lane at load, silently, module
#: by module, while the census that was meant to measure the lane records a
#: full house of ``torch_window`` and an empty problem list (issue #104: all
#: six allocated checkpoints carried a rate outside the set, so no evidence we
#: held could exercise the lane at all).  With the predicate in the contract a
#: producer refuses the plan instead (``scheme.refuse_unreachable_lane``).
#:
#: The two numeric values are READ OFF THE KERNEL (``tessera.kernel_roster``
#: parses ``csrc/window_gemv.cu``'s own ``TESSERA_GEMV_RATES`` /
#: ``TESSERA_GEMV_WINDOW_BITS`` declaration, which is what the file's dispatch
#: is generated from).  They used to be literals, on the ground that this
#: module is read by a producer with no torch and so cannot import
#: ``kernel_window_gemv`` -- true, and it made this block a restatement tied to
#: another restatement by a test, with the kernel tied to neither (issue #145).
#: ``kernel_roster`` is torch-free precisely so that this module can derive
#: them: a lane predicate published here now cannot outlive the rates the
#: kernel instantiates.
#:
#: THE BLOCK IS THE LANE'S WHOLE PREDICATE (#264).  Four conditions here
#: against nine the loader refuses was the defect: ``prepare_from_parsed``
#: also refuses, by name, RELEASE overrides, diagonals, rotation, a TP
#: shard's start state (``layout.slice_unit`` stamps ``initial_state``; the
#: kernel supplies ``state_{-1} = 0`` itself) and a grid of the wrong arity
#: -- and none of that was published, so the plan-time and byte-time gates,
#: which read THIS block, declared READABLE wire the loader refuses module
#: by module.  The five decoration classes are published as what the lane
#: does not read (``false`` / ``["none"]`` / ``[1]``), and every gate --
#: ``scheme.refuse_unreachable_lane``, ``scheme.lane_wire_report``, the
#: loader itself and ``bf16_route.gemv_refusal_for_unit`` -- decides them
#: through one function, ``scheme.decide_lane_requirements``, over this
#: block, so the predicate and the loader cannot drift.
#:
#: ONE loader clause is deliberately NOT here: ``prepare_from_parsed``'s
#: scalar-256-``native`` grid check.  That is an entry-point fact of the
#: E4M3 table build, not a lane fact -- the SAME extension reads BF16 window
#: wire through ``prepare_value_unit``, whose grid is scalar with 65536
#: codes -- so publishing size/native would declare wire unreadable
#: that the lane serves.  ``grid_arities`` is the lane-wide part: the kernel
#: decodes one code per position, whatever the alphabet.
WINDOW_GEMV_LANE = {
    "decoder": "window_gemv",
    "requires": {
        "column_rates": list(SUPPORTED_RATES),
        "window_bits": list(WINDOW_BITS_SUPPORTED),
        "body": "window",
        "plane": "channel",
        "release_overrides": False,
        "diagonals": False,
        "rotation": ["none"],
        "start_state": False,
        "grid_arities": [1],
    },
}

#: The native code this package can load INTO A SERVING PROCESS, as the
#: runtime contract publishes it.
#:
#: WHY IT IS PUBLISHED.  ``runtime_contract.json`` says what the plugin
#: EXECUTES; this says what it LOADS.  A consumer that keys reproducibility on
#: extension residency (PrismaQuant's serve fingerprint, its architecture
#: doc's section 7.4: KL is bit-identical inside one container session and
#: drifts across them, keyed on whether a lane's ``.so`` was resident) had to
#: mirror the name in its own repository, which is principle 14 read backwards
#: -- a claim about THIS runtime, maintained in another one.
#:
#: WHY NOT ``optional``.  "This build may not have compiled it" and "the route
#: runs correctly without it" are two facts, and the second one differs by
#: RESIDENCY: a lane that prepares at load may substitute its named torch
#: decoder, while a lane that decodes inside a traced forward refuses.  A
#: fingerprint whose job is to tell a native serve from a fallback serve needs
#: the substitute's NAME -- the value the route stamps in ``telemetry``'s
#: ``decoder`` field -- not a boolean that says only "absence is survivable
#: somewhere".
#:
#: WHAT BELONGS HERE.  An entry iff the ``.so`` can be resident in a SERVING
#: process, i.e. some module reachable from ``tessera.serving`` loads it.
#: ``tessera.kernel_window_gemv`` builds ``tessera_window_gemv`` and the
#: streamed window lanes reach it (issue #10).
#: ``tests/test_serving_native_extensions.py`` decides that by walking the
#: import graph, so the day a route loads another one the table goes red
#: rather than staying quietly short.  The span-2 NVFP4 decoder's entry was
#: removed at retirement: nothing under ``tessera.serving`` loads that ``.so``
#: any more, and a table that kept it would publish a library no serve can
#: map.
#:
#: THE GEMV ENTRY'S FALLBACK.  Both modes
#: substitute the torch window decode (``torch_window``) without the library:
#: streamed serves the same bytes through decode + ``_scaled_mm`` (slower, the
#: bytes the load-time cross-check verified), and resident never needed the
#: library at all (it decodes once at load through the same torch decoder).
#: "Substituted" is therefore the honest status in both modes even though the
#: resident serve is numerically untouched by the absence -- "refused" would
#: claim no serve exists, which is false.  The fallback is VISIBLE anyway: a
#: GEMV serve stamps ``window_gemv`` on its route record (``fp8_gemv``), so a
#: census histogram tells a substituted serve from a native one without the
#: decoder field having to.
NATIVE_EXTENSIONS = [
    {
        # The window GEMV's module name is EXACT (no identity hash), so the
        # glob is the name with a ``*`` the validator's meaning-check
        # requires, matching the one file the load path writes.
        "module_name_prefix": WINDOW_GEMV_MODULE_NAME,
        "filename_glob": WINDOW_GEMV_MODULE_NAME + "*.so",
        "match": MATCH_BASENAME_FNMATCH,
        "source": WINDOW_GEMV_SOURCE,
        "loaded_by": "tessera.serving.fp8_gemv",
        # BOTH window routes.  ``bf16_route.prepare_bf16_gemv`` repacks through
        # ``kernel_window_gemv.prepare_value_unit`` exactly as ``fp8_gemv`` does
        # and branches its ``apply`` on the same ``layer.tessera_gemv``, so the
        # 16-bit route's streamed dispatch turns on whether this extension
        # mapped.  Publishing one route said the other's serve is unaffected by
        # it, which is the claim a consumer keys a fingerprint on.
        # ``loaded_by`` still names ``fp8_gemv`` alone: the field is one string
        # by schema, and widening it is a second contract shape change.
        "routes": ["TESSERA_FP8", "TESSERA_BF16"],
        "lane": WINDOW_GEMV_LANE,
        "when_unavailable": {
            "resident": {"status": FALLBACK_SUBSTITUTED,
                         "decoder": "torch_window"},
            "streamed": {"status": FALLBACK_SUBSTITUTED,
                         "decoder": "torch_window"},
        },
    },
]


def substitutes_when_unavailable(mode: str, module_name_prefix: str) -> bool:
    """May a serve in residency ``mode`` decode without this extension?

    The routes gate on this rather than on a mode comparison of their own, so
    the published ``when_unavailable`` block IS what the serve does -- the same
    shape as ``sharding.ROUTE_TP_AXES`` behind ``loader_axes``.  A table that
    said a mode substitutes where the route refuses would be a claim about a
    runtime that does not exist.
    """
    for entry in NATIVE_EXTENSIONS:
        if entry["module_name_prefix"] != module_name_prefix:
            continue
        behaviour = entry["when_unavailable"].get(mode)
        if behaviour is None:
            raise ValueError(
                f"{module_name_prefix!r} publishes no behaviour for residency {mode!r}; "
                f"it declares {sorted(entry['when_unavailable'])}")
        return behaviour["status"] == FALLBACK_SUBSTITUTED
    raise ValueError(f"no native extension is declared with prefix {module_name_prefix!r}")

def _nvcc_root(nvcc: str) -> str:
    """The toolkit root holding ``bin/nvcc``."""
    return os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))


def _has_nvcc(root: str | None) -> bool:
    return bool(root) and os.access(os.path.join(root, "bin", "nvcc"), os.X_OK)


def _resolve_cuda_home(torch) -> str | None:
    """Point ``CUDA_HOME`` at a toolkit that actually contains ``nvcc``.

    Returns the root in use, or ``None`` when no complete toolkit was found.
    This COMPLETES torch's search rather than replacing it -- the candidates
    are torch's own answer first, then the ``which nvcc`` fallback torch would
    have reached had ``/usr/local/cuda`` not existed, then the versioned roots.
    Nothing here is a threshold or a preference: the test is whether a path
    holds an executable ``bin/nvcc``.

    Whatever it answers, the root the build compiles under when this returns
    is the root this function last adopted -- returning one while
    ``cpp_extension.CUDA_HOME`` names another is the defect in issue #298, and
    ``_nvcc_for_build()`` is the reading that has to agree.
    """
    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit:
        # The choice is ADOPTED, not merely reported.  A selection made after
        # torch's import reaches the build only through
        # ``cpp_extension.CUDA_HOME`` (issue #298), and it is adopted whether
        # or not the root holds a compiler: a refused selection that left the
        # displaced toolkit in that global would build with a compiler nobody
        # named.  Only the RETURN value is gated on completeness, so an
        # incomplete explicit root still reports "no toolkit" -- and the build
        # that then fails, fails under the root the operator named.
        displaced = _torch_cuda_home()
        _adopt_cuda_home(explicit)
        if _has_nvcc(explicit):
            return explicit
        if displaced and displaced != explicit and _has_nvcc(displaced):
            print(f"[tessera-serving] ERROR: the toolkit named by CUDA_HOME/CUDA_PATH "
                  f"({explicit}) holds no bin/nvcc, so no native build is selected; "
                  f"{displaced} was already loaded by torch and is NOT used in its "
                  f"place. Name a complete toolkit, or unset both variables to let "
                  f"the resolver search.", file=sys.stderr, flush=True)
        return None

    candidates: list[str] = []
    TORCH_CUDA_HOME = _torch_cuda_home()
    if TORCH_CUDA_HOME:
        candidates.append(TORCH_CUDA_HOME)
    found = shutil.which("nvcc")
    if found:
        candidates.append(_nvcc_root(found))
    # The toolkit torch was built against is the RIGHT one when several are
    # installed, so it is tried before the merely-newest.
    version = getattr(getattr(torch, "version", None), "cuda", None)
    if version:
        candidates.append(f"/usr/local/cuda-{version}")
    import glob as _glob
    candidates.extend(sorted(_glob.glob("/usr/local/cuda-*"), reverse=True))

    for root in candidates:
        if _has_nvcc(root):
            _adopt_cuda_home(root)
            return root
    return None


def _torch_cuda_home() -> str | None:
    """The toolkit root ``cpp_extension.load`` builds with RIGHT NOW.

    One reader for one fact, because three spellings of "ask torch which
    toolkit it froze at import" is three answers waiting to disagree: the
    search's first candidate, the root an explicit selection displaces and
    :func:`_nvcc_for_build`'s compiler are all this value.
    """
    try:
        from torch.utils import cpp_extension
    except Exception:  # noqa: BLE001 -- no torch, no build, nothing to name
        return None
    return getattr(cpp_extension, "CUDA_HOME", None)


def _adopt_cuda_home(root: str) -> None:
    """Make ``root`` the toolkit this process builds with.

    Both halves are needed and neither is redundant: ``load()`` builds its
    nvcc path from ``torch.utils.cpp_extension.CUDA_HOME``, a module global
    frozen at IMPORT time, so the environment variable alone arrives too late
    for a torch that is already imported; and the environment variable is what
    the compiler's own subprocesses and any later import will read.

    The module global is assigned unconditionally.  It used to be left alone
    whenever it already named a root holding an ``nvcc``, which is a no-op on
    the search path -- that root is the search's own first candidate, so the
    two agree whenever it is complete -- and wrong on the explicit path, where
    it left an operator's named toolkit reported as selected while the
    previous one did the compiling (issue #298).
    """
    os.environ["CUDA_HOME"] = root
    try:
        from torch.utils import cpp_extension
    except Exception:  # noqa: BLE001 -- nothing to correct without torch
        return
    cpp_extension.CUDA_HOME = root


#: Headers ``ATen/cuda/CUDAContextLight.h`` includes unconditionally.  If the
#: toolkit's own include tree is missing one of these, no torch CUDA extension
#: can compile against it, whatever else is installed.
_ATEN_CUDA_HEADERS = ("cusparse.h", "cublas_v2.h", "cusolverDn.h")


def _vendored_cuda_includes() -> list[str]:
    """Include dirs from the pip CUDA packages installed BESIDE this torch.

    The wheels (``nvidia-cu13`` and friends) ship the headers under
    ``<site-packages>/nvidia/*/include``, and they are installed as torch's own
    dependency, so they match the CUDA torch was built against by
    construction -- which is exactly the property that makes them safe to
    compile ATen's headers with.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    import glob as _glob
    return sorted(d for d in _glob.glob(os.path.join(site, "nvidia", "*", "include"))
                  if os.path.isdir(d))


def _include_shim_root() -> str:
    """A writable directory to hang the header shim off, beside the builds."""
    root = (os.environ.get("TESSERA_EXT_DIR")
            or os.environ.get("TORCH_EXTENSIONS_DIR")
            or os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"))
    return os.path.join(root, "tessera_include_shim")


def _repair_include_path(cuda_home: str | None) -> list[str]:
    """One include dir holding ONLY the headers the toolkit is missing.

    The shim holds exactly the headers the toolkit LACKS, which makes
    shadowing impossible by construction rather than by care.  That is the
    whole design: ``load`` places user includes BEFORE its system includes, so
    adding ``nvidia/cu13/include`` wholesale does not "add cusparse.h" -- it
    shadows the toolkit's entire runtime header tree with the pip one, and the
    build fails further in, on ``__cudaLaunch`` (measured, not feared).  Nor
    is a hand-listed set enough: ``cusolverDn.h`` includes
    ``cusolver_common.h``, so the gap has to be closed by the rule "absent
    from the toolkit", not by naming headers.  Everything the toolkit does
    have still resolves from the toolkit.

    Gated on the fact that provokes it: a header ATen includes is genuinely
    absent from ``$CUDA_HOME/include``.  A complete toolkit gets an empty list
    and compiles exactly as it did before.

    Measured cause: the pinned GLM serving image (``glm53-mia-sm121``) ships
    ``nvcc`` and ``ninja`` but no ``cusparse.h`` under ``/usr/local/cuda``,
    while a matching one sits in ``nvidia/cu13/include`` -- so the streamed
    JIT lanes could not build on the very image that serves them.
    """
    if not cuda_home:
        return []
    include = os.path.join(cuda_home, "include")
    missing = [h for h in _ATEN_CUDA_HEADERS if not os.path.isfile(os.path.join(include, h))]
    if not missing:
        return []
    # Every header the pip packages have and the toolkit does not.  Scoped to
    # the top level of each include dir: a vendored SUBDIRECTORY would shadow
    # a toolkit subdirectory of the same name, which is the failure above.
    sources: dict[str, str] = {}
    for directory in _vendored_cuda_includes():
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            continue
        for header in entries:
            if not header.endswith((".h", ".hpp", ".inl")):
                continue
            if header in sources or os.path.exists(os.path.join(include, header)):
                continue
            candidate = os.path.join(directory, header)
            if os.path.isfile(candidate):
                sources[header] = candidate
    if not any(h in sources for h in missing):
        return []
    shim = _include_shim_root()
    try:
        os.makedirs(shim, exist_ok=True)
        for header, src in sources.items():
            link = os.path.join(shim, header)
            if os.path.realpath(link) != os.path.realpath(src):
                if os.path.lexists(link):
                    os.unlink(link)
                os.symlink(src, link)
    except OSError as exc:
        print(f"[tessera-serving] WARNING: cannot build a header shim at {shim} ({exc}); "
              f"{include} is missing {missing}", file=sys.stderr, flush=True)
        return []
    print(f"[tessera-serving] {include} is missing {missing}; filling {len(sources)} absent "
          f"headers from the pip CUDA packages beside torch, via {shim}",
          file=sys.stderr, flush=True)
    return [shim]


def _nvcc_for_build() -> "str | None":
    """The CUDA compiler command torch's JIT loader will actually invoke.

    Resolved by the loader's own rule (``torch/utils/cpp_extension``,
    ``_write_ninja_file``): ``PYTORCH_NVCC`` when set, else
    ``CUDA_HOME/bin/nvcc`` from the ``cpp_extension.CUDA_HOME`` module global
    that ``load()`` reads.  NEVER ``$NVCC`` or a bare ``nvcc`` from PATH --
    the loader consults neither, so hashing them named a compiler the build
    did not run and missed the one it did (issue #242).  ``_resolve_cuda_home``
    ADOPTS the toolkit it selects into that same frozen global, which is what
    makes this answer and the build's compiler one thing.
    """
    if "PYTORCH_NVCC" in os.environ:
        return os.environ.get("PYTORCH_NVCC")
    home = _torch_cuda_home()
    return os.path.join(home, "bin", "nvcc") if home else None


def _resolve_ninja() -> str | None:
    """``ninja`` on PATH, or the one beside this interpreter (put on PATH)."""
    found = shutil.which("ninja")
    if found:
        return found
    beside = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "ninja")
    if os.access(beside, os.X_OK):
        os.environ["PATH"] = os.path.dirname(beside) + os.pathsep + os.environ.get("PATH", "")
        return beside
    return None


def toolchain_report(torch=None) -> dict[str, object]:
    """What the build would find right now: ``{"nvcc":…, "ninja":…, "cuda_home":…}``.

    Public because a test that SKIPS on a missing toolchain has to be able to
    tell "no compiler on this box" from "the compiler is here and the build
    broke".  Only the first is a skip; the second is a failure.

    NOT a pure probe, despite the name: asking the question is how the answer
    becomes true.  It ADOPTS what it finds -- ``os.environ["CUDA_HOME"]``,
    ``cpp_extension.CUDA_HOME`` and ``PATH`` -- because a report that a
    compiler exists somewhere the build will not look is worth nothing.  Call
    it before a JIT lane prepares, which is where that matters.
    """
    if torch is None:
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            torch = None
    cuda_home = _resolve_cuda_home(torch) if torch is not None else None
    ninja = _resolve_ninja()
    return {
        "cuda_home": cuda_home,
        "nvcc": os.path.join(cuda_home, "bin", "nvcc") if cuda_home else None,
        "ninja": ninja,
        "extra_includes": _repair_include_path(cuda_home),
        "complete": bool(cuda_home and ninja),
    }

class IncompleteInstallError(FileNotFoundError):
    """The package is installed without its CUDA sources (a packaging defect)."""


class NativeKernelUnavailableError(RuntimeError):
    """The native operator is unavailable and no substitute may be selected."""


def csrc_dir() -> str:
    """The packaged CUDA sources, resolved relative to THIS module.

    Never repo-root arithmetic: under a non-editable install only the package
    lands in site-packages, so a repo-relative path does not exist and every
    build fails.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def native_source_path(module_name_prefix: str) -> str:
    """The absolute path of the ``.cu`` the contract publishes for an extension.

    Resolved from :data:`NATIVE_EXTENSIONS` by ``module_name_prefix`` -- the
    one table the contract reads -- and refused (``IncompleteInstallError``) if
    the file is not in this install.  Every JIT loader in the package takes its
    source from here, so what the contract says is built is what is built.
    """
    for entry in NATIVE_EXTENSIONS:
        if entry["module_name_prefix"] == module_name_prefix:
            source = entry["source"]
            break
    else:
        raise KeyError(f"no native extension publishes module prefix {module_name_prefix!r}")
    head, _, rel = source.partition("/")
    if head != "csrc" or not rel or "/" in rel:
        raise ValueError(f"native extension source {source!r} is not a file under csrc/")
    return os.path.join(_require_csrc(rel), rel)


def _require_csrc(*names: str) -> str:
    d = csrc_dir()
    missing = [n for n in names if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise IncompleteInstallError(
            f"tessera is installed without its CUDA sources: {missing} not found under {d}. "
            "This is a packaging defect, not a missing CUDA toolchain -- reinstall tessera or "
            "install from a checkout.")
    return d
