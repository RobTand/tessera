"""Device crosscheck: the dense A8/A16 routes at PREFILL shapes, TP1 and TP2.

WHAT THIS DRIVES.  The production load path of the two dense window routes --
``lane.build_tessera_method`` -> ``create_weights`` -> ``process_weights_after_
loading`` -> ``apply`` -- on a real container and a real rank geometry, and it
compares the served output against the RETAINED reference: the materialising
preparation (``prepare_tessera_fp8_module`` / ``prepare_tessera_bf16_module``,
which decodes through ``tessera.decode.materialize_fp8`` /
``materialize_bf16``) whose tile and row scale are multiplied by a stock
matmul.  That reference is the arm the native lane replaced, so agreement is
what makes the replacement a substitution rather than a second renderer.

WHY PREFILL IS THE SHAPE.  The dense routes serve one packed GEMM at every M,
but the arm that was retired for this regime was the prefill fallback: past
``GEMV_MAX_M`` (8) the streamed lane decoded a whole ``[rows, columns]`` tile
per forward and called ``torch._scaled_mm`` / ``torch.mm``.  So the M set walks
the boundaries rather than the middle: the empty batch, one row, the GEMV
limit and the row past it (8, 9), the ``tl.dot`` tile (17), a ``block_m``
multiple (64), a shape that divides nothing (129) and a prefill-sized 512.

WHAT THE DISPATCH PROBE IS FOR.  Numerical agreement alone cannot tell "the
native GEMM ran" from "a materialiser ran and agreed", and before the
substitution the streamed route DID materialise.  So around every ``apply``
the probe counts the packed bundle's launches (``PreparedWindowGemm.__call__``,
at least one per role) and makes every materialising entry point raise if it is
entered: ``PreparedWindow.decode``, ``materialize_fp8``/``materialize_bf16``,
``torch._scaled_mm`` and ``torch.mm``.  An arm fails if anything in the second
set is called, whatever its numbers.

SHAPE VALIDITY.  Two refusal arms hand a real wire to a declaration that does
not match it (the other family, and a wrong rung).  Both must refuse by name at
the load, in this build: an unsupported case fails closed rather than being
served through an expansion nobody declared.

RUN IT.  vLLM work is exempt from PrismaBuild, and this drives the real vLLM
route on a device, so it runs directly in the pinned image (the image is the
one the mixed-fixture serve and the A16 TP1 run used):

  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests \
    -v <source snapshot>:/work:ro -v /mnt/shared:/mnt/shared:ro \
    --entrypoint bash <image a5424378…> -c '
      python3 -m torch.distributed.run --nproc_per_node=1 --standalone \
        /work/tests/native_dense_prefill_cross_check.py --tp 1 --json /tmp/tp1.json
      python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
        /work/tests/native_dense_prefill_cross_check.py --tp 2 --json /tmp/tp2-$RANK.json
      python3 /work/tests/native_dense_prefill_cross_check.py --merge /tmp/report.json \
        /tmp/tp1.json /tmp/tp2-0.json /tmp/tp2-1.json'

WHICH FIXTURES.  ``--fixture`` names a tree and ``--family`` names a payload
family; with neither, every fixture in the table runs.  The GLM pair is the
artifact the dense lane is held to now -- ``--fixture GLM_A8 --fixture GLM_A16``
is 96 arms (two families, two modules each, eight M, and the three rank cuts
TP1 / TP2 rank 0 / TP2 rank 1) -- while the small pair the lane landed against
stays selectable beside it, because one key is one tree and the two families
are no longer the same thing as the two artifacts.  A key crossed with the
other family's name intersects nothing, and that is REFUSED by name rather
than reported as a green table of zero arms.  ``--preflight`` resolves
whichever set is selected and needs no device.

RANKS ARE REAL, so one process is ONE rank of the world it declares.  The
production load path registers a ``BasevLLMParameter``, whose constructor asks
the process-global tensor-parallel group for this rank; a harness that never
brought a group up dies in ``create_weights`` before it reaches a single arm.
Each process therefore does what a serve does -- ``init_distributed_environment``
then ``initialize_model_parallel(world, 1)``, inside a real ``VllmConfig`` --
and then drives ONLY ``tp_size == world``, ``tp_rank == this rank``.  ``--merge``
joins the per-rank reports and refuses an incomplete set: every process
declares the arms it is about to drive, so a rank that died leaves its
declaration unmet and the merge fails by name instead of quietly reporting a
smaller table.  The refusal arms are requested exactly the same way, one per
declared module and label: the merge names a refusal arm that is missing or
undeclared, because a roster that lost its arms still satisfies ``all()``
over what is left.

The config is built with ``distributed_executor_backend="external_launcher"``
because torchrun, not vLLM, launches these ranks -- the mode vLLM provides for
exactly that, and the one that does not refuse a world larger than the node's
device count.  A box with fewer devices than ranks runs every rank on device 0
(``local_rank``), and the TP group is built WITHOUT a device communicator there
-- recorded per rank, never inferred: the device communicator wants one device
per rank (pynccl + custom all-reduce), and every arm here is rank-local
arithmetic with no collective to place.

The bring-up's model config is a stock stand-in written to a temporary
directory.  It supplies a rank geometry and nothing else (``load_format="dummy"``,
no weight is read, no tokenizer), because the fixtures declare
``quant_method: tessera`` and vLLM only validates that with the plugin
installed -- which this harness does not need, since it calls
``lane.build_tessera_method`` directly.  The arms' declarations still come from
each fixture's own ``config_groups``.

``--preflight`` needs no device and no vLLM: it resolves the fixtures, checks
each declaration with ``validate_tessera_scheme`` and builds every rank's plan,
which is the plumbing a device run would otherwise fail on first.  That mode
is CPU work and goes through PrismaBuild like every other CPU action.

BOUNDS.  The small fixtures are two dense modules (the widest is 4096x4096) and
the GLM fixtures are layer 0 of the sharded canonical-census export, read
through the single index-selected shard each ``wire_bytes`` tensor maps to, so
the arms are bounded by construction; ``--mset`` caps the batch and the whole
run is one bounded process.  Tolerances are the chosen screens the route tests
use, not composed bounds, and they are printed with every arm.

WHY THE BATCH FLAG IS NOT ``--m``.  A device run goes through
``python3 -m torch.distributed.run``, whose parser sits in FRONT of this
harness's, and ``--m`` is an ambiguous abbreviation of its own options
(``--max-restarts``, ``--monitor-interval``, ``--module``, ``--master-addr``,
``--master-port``): measured on the pinned image, ``--m 0`` after the script
path is refused with ``error: ambiguous option: --m`` before this module's
parser ever sees it, while ``--mset`` passes through.  The flag is spelled
``--mset`` so the documented device command works.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

#: The dense fixtures, by KEY -- a key is a tree, and the ``family`` field is
#: the route that serves it.  The two are separate because one family has more
#: than one artifact: the small modules the lane landed against and the real
#: GLM layer-0 export the lane is held to now, side by side, so the old
#: comparison stays runnable without standing in for the new one.
#:
#: Each entry names a config group and the container tensor beside it, and
#: locates the tree through ``tests/box_artifacts.py`` -- the one home for the
#: roots this repository reads but does not own -- with this harness's own
#: override on top.  A box that keeps them elsewhere says so; nothing here
#: spells a path.
#:
#: ``GLM_A8``/``GLM_A16`` are layer 0 of the canonical-census first-artifact
#: exports: one merged container per arm, and the harness reads each
#: ``wire_bytes`` tensor through the single shard the index maps it to.  Both
#: carry the same geometry -- gate_up ``24576x4096`` with roles
#: ``gate_proj``/``up_proj`` at 12288 rows each, down ``4096x12288`` -- at
#: ``q256`` 1024, and differ in the payload family (E4M3 vs BF16) exactly as
#: the small pair did.
FIXTURES = {
    "TESSERA_FP8": {
        "family": "TESSERA_FP8",
        "env": "TESSERA_A8_DENSE_FIXTURE",
        "artifact": ("shared_runs", "derivatives", "mixedA4A8A16-layers0-4-20260916"),
        "modules": (
            {"group": "tessera_model_language_model_layers_3_mlp_shared_experts_gate_up_proj",
             "tensor": "model.language_model.layers.3.mlp.shared_experts.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_language_model_layers_3_mlp_shared_experts_down_proj",
             "tensor": "model.language_model.layers.3.mlp.shared_experts.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
    "TESSERA_BF16": {
        "family": "TESSERA_BF16",
        "env": "TESSERA_A16_DENSE_FIXTURE",
        "artifact": ("shared_runs", "bf16", "qwen0.6b-bf16-r7-plugin"),
        "modules": (
            {"group": "tessera_model_layers_0_mlp_gate_up_proj",
             "tensor": "model.layers.0.mlp.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_layers_0_mlp_down_proj",
             "tensor": "model.layers.0.mlp.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
    "GLM_A8": {
        "family": "TESSERA_FP8",
        "env": "TESSERA_GLM_A8_DENSE_FIXTURE",
        "artifact": ("measurements", "glm-canonical-census-20260908",
                     "first-artifact-exports", "a8", "merged-b426d18893"),
        "modules": (
            {"group": "tessera_model_language_model_layers_0_mlp_gate_up_proj",
             "tensor": "model.language_model.layers.0.mlp.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_language_model_layers_0_mlp_down_proj",
             "tensor": "model.language_model.layers.0.mlp.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
    "GLM_A16": {
        "family": "TESSERA_BF16",
        "env": "TESSERA_GLM_A16_DENSE_FIXTURE",
        "artifact": ("measurements", "glm-canonical-census-20260908",
                     "first-artifact-exports", "a16", "merged-b426d18893"),
        "modules": (
            {"group": "tessera_model_language_model_layers_0_mlp_gate_up_proj",
             "tensor": "model.language_model.layers.0.mlp.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_language_model_layers_0_mlp_down_proj",
             "tensor": "model.language_model.layers.0.mlp.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
}

#: The payload families the fixtures cover, derived from the fixtures
#: themselves so a new artifact of an existing family needs no edit here.
FAMILIES = tuple(sorted({spec["family"] for spec in FIXTURES.values()}))

#: The M set the prefill arms walk.  See the module docstring for the shape
#: each value is at.
DEFAULT_M = (0, 1, 8, 9, 17, 64, 129, 512)

#: One refusal arm per declared module and label.  ``_refusal_arms`` builds
#: exactly these for every module it drives, and ``_merged`` requests exactly
#: these of every per-rank report -- the roster has ONE home so the request and
#: the run cannot drift apart.
REFUSAL_LABELS = ("family", "rung")

#: The chosen screens the route tests use, per family, against the reference
#: product's own magnitude: the FP8 route's served error screen, and the
#: module screen ``test_serving_native_window`` holds the BF16 lane to.  They
#: are SCREENS, not composed bounds, and every arm prints which one it used.
BF16_SCREEN = {"fixed": 5.0e-3, "of_max_abs": 1.0e-2}
FP8_SCREEN = {"fixed": 0.0, "of_max_abs": 0.0, "relative": 8.0e-3}


def _failure(message: str):
    raise SystemExit(f"native_dense_prefill_cross_check: {message}")


def _fixture_root(spec) -> Path:
    """This box's tree for one family, the same way every other gate asks.

    The root comes from ``box_artifacts`` (env variable or documented default);
    this harness's own variable overrides it for a tree kept somewhere else.
    An absent tree fails with the sentence ``box_artifacts`` writes, which
    names the root and the variable that moves it.
    """
    import os

    import box_artifacts

    override = os.environ.get(spec["env"], "")
    if override:
        root = Path(override)
    else:
        key, *parts = spec["artifact"]
        base = box_artifacts.root(key)
        if base is None:
            _failure(box_artifacts.reason(key))
        root = base.joinpath(*parts)
    if not (root / "config.json").is_file():
        _failure(
            f"{box_artifacts.reason(spec['artifact'][0], root)}"
            f" -- and {root} has no config.json; the dense fixture this crosscheck "
            f"prices against is missing (this harness's override is {spec['env']})"
        )
    return root


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _torch_env() -> tuple[int, int, int]:
    """This process's coordinates, as torchrun sets them (1/0/0 without it)."""
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return world, rank, local_rank


def _stand_in_model_dir() -> Path:
    """A models directory for the bring-up alone: a rank geometry needs a config.

    The fixtures carry ``quant_method: tessera``, which vLLM refuses unless the
    plugin is installed and registered.  This harness calls
    ``lane.build_tessera_method`` directly and never builds a served model, so
    the bring-up's ``ModelConfig`` is a stock architecture's and nothing is
    read from it (``load_format="dummy"``, tokenizer skipped).
    """
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="dense-prefill-bringup-"))
    (path / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 1024,
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
        "torch_dtype": "bfloat16", "tie_word_embeddings": False,
    }, indent=1) + "\n")
    return path


def _install_cpu_only_tp_group(world_size: int, local_rank: int) -> None:
    """vLLM's TP group WITHOUT a device communicator, for a one-device box.

    ``initialize_model_parallel`` builds the TP group with a device
    communicator (pynccl + custom all-reduce), which wants one CUDA device per
    rank; a box with one device cannot give it one.  Every arm in this harness
    is rank-local arithmetic -- there is no collective in the replaced lane or
    the native one -- so what the arms need from the group is its rank
    arithmetic, and that is what this group provides.  ``_TP`` has no setter in
    vLLM 0.28, so this assigns the same global ``initialize_model_parallel``
    assigns; if that name ever moves, this raises rather than silently
    serving a different group.
    """
    from vllm.distributed import parallel_state as ps

    group = ps.init_model_parallel_group([list(range(world_size))], local_rank,
                                        "gloo", use_device_communicator=False,
                                        group_name="tp")
    ps._TP = group


def _bring_up(world_size: int, rank: int):
    """The bring-up a real serve does, so a vLLM parameter can answer.

    Returns ``(VllmConfig, ExitStack, record)``: the load path reads the current
    config (``BasevLLMParameter``, the routes' CustomOps), so the stack stays
    open for the whole driving phase.  ``record`` says which TP group this rank
    actually got and why -- a one-device box shares device 0 between ranks, and
    the record names the fallback rather than leaving it to be inferred.
    """
    import contextlib
    import os

    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.engine.arg_utils import EngineArgs

    devices = int(torch.cuda.device_count())
    # Ranks are PROCESSES: on a box with fewer devices than ranks they share
    # device 0, exactly as the research TP2 tests drive two cuts of one module.
    local_rank = rank % max(1, devices)
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT") or str(_free_port())
    engine_args = EngineArgs(model=str(_stand_in_model_dir()), load_format="dummy",
                             enforce_eager=True, max_model_len=2048,
                             skip_tokenizer_init=True,
                             tensor_parallel_size=world_size,
                             # THIS PROCESS LAUNCHES THE RANKS (torchrun does), which
                             # is what ``external_launcher`` means to vLLM: it skips
                             # vLLM's own rank-to-device placement -- including its
                             # refusal to plan a world larger than the node's device
                             # count -- because the launcher owns that mapping.  A
                             # one-device box runs the ranks on device 0, and
                             # ``local_rank`` below says so in the report.
                             distributed_executor_backend="external_launcher")
    vllm_config = engine_args.create_engine_config()
    stack = contextlib.ExitStack()
    stack.enter_context(set_current_vllm_config(vllm_config, check_compile=False))
    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank,
        distributed_init_method=f"tcp://{master}:{port}", backend="gloo")
    record = {"world_size": world_size, "tp_rank": rank, "local_rank": local_rank,
              "device_count": devices, "tp_group": "vllm-default",
              "executor_backend": "external_launcher"}
    try:
        initialize_model_parallel(world_size, 1)
    except Exception as exc:  # noqa: BLE001 -- recorded, and narrowed below
        if devices >= world_size:
            raise
        record["tp_group"] = "cpu-only-no-device-communicator"
        record["tp_group_fallback"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        _install_cpu_only_tp_group(world_size, local_rank)
    return vllm_config, stack, record


def _read_wire(root: Path, tensor: str) -> bytes:
    """The container bytes for ``tensor``, from an index or a single shard."""
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        matches = [key for key in weight_map if key == tensor or key.endswith(tensor)]
        if len(matches) != 1:
            _failure(f"{tensor} matches {len(matches)} entries in {index}")
        key = matches[0]
        with safe_open(str(root / weight_map[key]), framework="pt") as handle:
            return bytes(handle.get_tensor(key).detach().cpu().numpy().tobytes())
    shard = root / "model.safetensors"
    if not shard.is_file():
        _failure(f"{root} carries neither {index.name} nor {shard.name}")
    with safe_open(str(shard), framework="pt") as handle:
        matches = [key for key in handle.keys() if key == tensor or key.endswith(tensor)]
        if len(matches) != 1:
            _failure(f"{tensor} matches {len(matches)} tensors in {shard}")
        key = matches[0]
        return bytes(handle.get_tensor(key).detach().cpu().numpy().tobytes())


def _declared(root: Path, group: str, blob: bytes, prefix: str):
    """The config group's scheme, validated, with the blob's own byte count."""
    from tessera.serving.scheme import validate_tessera_scheme

    config = json.loads((root / "config.json").read_text())
    groups = config["quantization_config"]["config_groups"]
    if group not in groups:
        _failure(f"{group} is not a config group of {root}")
    scheme = dict(groups[group]["scheme"])
    if int(scheme.get("wire_bytes", len(blob))) != len(blob):
        _failure(
            f"{group}: the sidecar declares {scheme.get('wire_bytes')} bytes and the "
            f"container carries {len(blob)}"
        )
    scheme["wire_bytes"] = len(blob)
    declared = validate_tessera_scheme(scheme, prefix)
    return scheme, declared


def _plan(declared, parallel: str, tp_rank: int, tp_size: int):
    """This rank's ``ShardPlan``, in the numbering ``create_weights`` uses."""
    from tessera.serving.sharding import plan_shard

    roles = [(str(name), int(rows)) for name, rows in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    if tp_size == 1:
        return plan_shard("crosscheck", roles=roles, columns=columns,
                          out_partitions=[r for _, r in roles], in_size=columns,
                          tp_rank=0, tp_size=1, input_size=columns, output_size=rows)
    if parallel == "column":
        # Column-parallel: the OUTPUT row axis is cut, the input width is whole.
        return plan_shard("crosscheck", roles=roles, columns=columns,
                          out_partitions=[r // tp_size for _, r in roles],
                          in_size=columns, tp_rank=tp_rank, tp_size=tp_size,
                          input_size=columns, output_size=rows)
    # Row-parallel: the INPUT width is cut, the output row axis is whole.
    return plan_shard("crosscheck", roles=roles, columns=columns,
                      out_partitions=[r for _, r in roles],
                      in_size=columns // tp_size, tp_rank=tp_rank, tp_size=tp_size,
                      input_size=columns, output_size=rows)


def _create_weight_args(declared, parallel: str, tp_size: int):
    """The four numbers a vLLM ``LinearBase`` hands ``create_weights``."""
    roles = [(str(name), int(rows)) for name, rows in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    if tp_size == 1:
        return dict(input_size_per_partition=columns,
                    output_partition_sizes=[r for _, r in roles],
                    input_size=columns, output_size=rows)
    if parallel == "column":
        return dict(input_size_per_partition=columns,
                    output_partition_sizes=[r // tp_size for _, r in roles],
                    input_size=columns, output_size=rows)
    return dict(input_size_per_partition=columns // tp_size,
                output_partition_sizes=[r for _, r in roles],
                input_size=columns, output_size=rows)


def _layer(tp_rank: int, tp_size: int):
    """A ``LinearBase`` stand-in: the rank's own TP coordinates and nothing else.

    The same seam ``tests/test_serving_fp8_route.py`` drives the route through,
    and the same one the window-MoE stock crosscheck uses: the route reads
    ``tp_rank``/``tp_size``, registers one parameter and two buffers, and
    deletes the parameter when it is spent.
    """
    import torch

    class _Layer(torch.nn.Module):
        def __init__(self, rank: int, size: int):
            super().__init__()
            self.tp_rank = int(rank)
            self.tp_size = int(size)

    return _Layer(tp_rank, tp_size)


@contextlib.contextmanager
def _dispatch_probe():
    """Count the packed lane; make every materialising entry point raise."""
    import torch

    import tessera.decode as decode
    from tessera import window_gemm as wg
    from tessera.serving import window as reference_window

    record = {"native_calls": 0, "materialiser_calls": []}
    patches = []

    def _patch(owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        patches.append((owner, name, original))

    original_call = wg.PreparedWindowGemm.__call__

    def counted(self, *args, **kwargs):
        record["native_calls"] += 1
        return original_call(self, *args, **kwargs)

    def refuse(what):
        def _refuse(*_args, **_kwargs):
            record["materialiser_calls"].append(what)
            raise AssertionError(f"the route entered the materialising path: {what}")
        return _refuse

    try:
        _patch(wg.PreparedWindowGemm, "__call__", counted)
        _patch(reference_window.PreparedWindow, "decode", refuse("PreparedWindow.decode"))
        _patch(decode, "materialize_fp8", refuse("materialize_fp8"))
        _patch(decode, "materialize_bf16", refuse("materialize_bf16"))
        if hasattr(torch, "_scaled_mm"):
            _patch(torch, "_scaled_mm", refuse("torch._scaled_mm"))
        _patch(torch, "mm", refuse("torch.mm"))
        yield record
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def _reference_product(family: str, blob: bytes, scheme, plan, x):
    """The retired arm's product: the reference decoder's tile and row scale."""
    import torch

    from tessera.serving import bf16_route, fp8_route
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.serving.sharding import shard_parsed_roles

    parsed = parse_tessera_blob_for_scheme(blob, scheme, "crosscheck", device="cuda")
    roles = shard_parsed_roles(parsed, plan)
    if family == "TESSERA_FP8":
        from tessera.serving import native_ops

        module = fp8_route.prepare_tessera_fp8_module(roles, device="cuda")
        # The FP8 reference's ``decode`` is the BYTE tile (its docstring's
        # claim is byte identity with ``materialize_stock``), so the values are
        # the E4M3 view of it, exactly as the route's own ``_scaled_mm`` takes.
        tile = module.decode().view(torch.float8_e4m3fn).to(torch.float32)
        a_q, a_scale = native_ops.native_fp8_quant(x.contiguous())
        left = a_q.to(torch.float32) * a_scale.to(torch.float32)
    else:
        module = bf16_route.prepare_tessera_bf16_module(roles, device="cuda")
        tile = module.decode().to(torch.float32)
        left = x.to(torch.float32)
    scale = module.row_scale().to(torch.float32).reshape(-1, 1)
    return (left @ (tile * scale).t()).to(torch.bfloat16)


def _drive_route(family: str, scheme, declared, blob: bytes, parallel: str,
                 tp_rank: int, tp_size: int, x, mode: str):
    """The production load path, then one forward, under the dispatch probe."""
    import torch

    from tessera.serving import lane as serving_lane

    serving_lane.reset_for_tests()
    prefix = f"crosscheck.{family}.tp{tp_size}.rank{tp_rank}"
    method = serving_lane.build_tessera_method(scheme, prefix, mode)
    layer = _layer(tp_rank, tp_size)
    method.create_weights(layer, params_dtype=torch.bfloat16,
                          **_create_weight_args(declared, parallel, tp_size))
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    method.process_weights_after_loading(layer)
    with _dispatch_probe() as probe:
        got = method.apply(layer, x)
    return got, layer, probe


def _arm(fixture: str, family: str, module: dict, declared, scheme, blob: bytes, parallel: str,
         tp_rank: int, tp_size: int, m: int, mode: str, report: dict):
    import torch

    plan = _plan(declared, parallel, tp_rank, tp_size)
    columns = int(declared["columns"])
    x_full = torch.randn(max(m, 1), columns, dtype=torch.bfloat16, device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(1000 + m))
    if parallel == "row" and tp_size > 1:
        role = next(iter(declared["roles"]))[0]
        lo, hi = plan.role(str(role)).lo, plan.role(str(role)).hi
        x = x_full[:, lo:hi].contiguous()
    else:
        x = x_full
    x = x[:m].contiguous()
    want = _reference_product(family, blob, scheme, plan, x)
    got, layer, probe = _drive_route(family, scheme, declared, blob, parallel,
                                     tp_rank, tp_size, x, mode)
    entry = {
        "fixture": fixture, "family": family, "group": module["group"], "mode": mode,
        "tp_rank": tp_rank, "tp_size": tp_size, "parallel": parallel, "m": m,
        "shape": [int(v) for v in got.shape],
        "native_calls": probe["native_calls"],
        "materialiser_calls": probe["materialiser_calls"],
    }
    if m == 0:
        entry["passed"] = (
            tuple(got.shape) == tuple(want.shape)
            and not probe["materialiser_calls"]
            and probe["native_calls"] > 0
        )
        report["arms"].append(entry)
        return entry["passed"]
    diff = (got.float() - want.float()).abs()
    magnitude = want.float().abs().max().clamp_min(1e-9)
    screen = (FP8_SCREEN if family == "TESSERA_FP8" else BF16_SCREEN)
    limit = screen["fixed"] + screen["of_max_abs"] * float(magnitude)
    if screen.get("relative"):
        limit = max(limit, screen["relative"] * float(magnitude))
    error = float(diff.max())
    entry.update({
        "max_abs": error, "max_over_mag": float(error / float(magnitude)),
        "tolerance": limit, "tolerance_derivation": "chosen screen, not a composed bound",
    })
    # The load path's own claims, read off the prepared layer: the weights stay
    # packed, no decoded tile is registered, and the record names the native op.
    native = getattr(layer, "tessera_native", None)
    packed_ok = native is not None and native.packed_bytes() < (
        int(declared["rows"]) * int(declared["columns"]) * 2)
    absent = [name for name in ("weight_fp8", "weight_bf16", "tessera_prepared",
                               "tessera_gemv")
              if hasattr(layer, name)]
    entry.update({
        "packed_bytes": None if native is None else int(native.packed_bytes()),
        "materialising_attributes": absent,
        "decoder": getattr(layer, "tessera_decoder", None),
    })
    passed = (
        error <= limit
        and not probe["materialiser_calls"]
        and probe["native_calls"] > 0
        and packed_ok
        and not absent
    )
    entry["passed"] = passed
    report["arms"].append(entry)
    return passed


def _refusal_arms(fixture: str, family: str, module: dict, declared, blob: bytes, report: dict):
    """A real wire under a declaration that does not match it must refuse."""
    from tessera.serving.scheme import parse_compact_blob_for_scheme

    other = "TESSERA_BF16" if family == "TESSERA_FP8" else "TESSERA_FP8"
    # Both wrong declarations are DERIVED from the wire's own facts.  The rung
    # arm used to hard-code the wrong rung as ``896`` for the FP8 family and
    # ``1024`` for the BF16 one -- which is a mismatch for the small Qwen BF16
    # fixture (q256 1792) and NOT a mismatch for a BF16 wire that is itself at
    # q256 1024, where the arm then accepted the wire it was built to refuse.
    # A rung offset cannot collide with the wire's own rung; the family arm
    # carries the other family's grid and leaves the rung alone.
    wrong_rung = int(declared["q256"]) - 128
    cases = {
        "family": dict(declared, family=other,
                       grid=("BF16" if other == "TESSERA_BF16" else "E4M3")),
        "rung": dict(declared, q256=wrong_rung),
    }
    if set(cases) != set(REFUSAL_LABELS):
        _failure(f"{module['group']}: the refusal arms are {sorted(cases)}, but the "
                 f"requested roster is {sorted(REFUSAL_LABELS)}")
    for label, wrong in cases.items():
        if all(wrong.get(field) == declared.get(field)
               for field in ("family", "grid", "q256")):
            _failure(f"{module['group']}: the {label} refusal arm matches the wire it "
                     f"is built to refuse")
    ok = True
    for label in REFUSAL_LABELS:
        wrong = cases[label]
        entry = {"fixture": fixture, "family": family, "group": module["group"],
                 "refusal": label}
        try:
            parse_compact_blob_for_scheme(blob, wrong, "crosscheck", device="cpu")
        except ValueError as exc:
            message = str(exc)
            entry["refused"] = True
            entry["message"] = message[:400]
            # A refusal for the RIGHT reason: the reader compares the wire's
            # own sidecar facts against the declaration and says which differ.
            # A bare "invalid scheme" would pass an existence check and still
            # leave the mismatch unstated.
            entry["named_the_mismatch"] = "sidecar scheme declares" in message
            ok = ok and entry["named_the_mismatch"]
        else:
            entry["refused"] = False
            entry["message"] = "accepted a wire the declaration does not match"
            ok = False
        report["refusals"].append(entry)
    return ok


def _arm_id(fixture: str, group: str, tp_size: int, tp_rank: int, m: int) -> tuple:
    """One arm's identity: the tuple a report declares and a merge matches."""
    return (str(fixture), str(group), int(tp_size), int(tp_rank), int(m))


def _entry_id(entry: dict) -> tuple:
    return _arm_id(entry["fixture"], entry["group"], entry["tp_size"],
                   entry["tp_rank"], entry["m"])


def _merged(selected, paths, m_set) -> dict:
    """Join the per-rank reports, and REFUSE an incomplete set.

    Each rank declares the arms it is about to drive before it drives them, so
    the merge can tell "this rank passed everything it ran" from "this rank did
    not run everything": a process that died mid-table leaves a declared arm
    unobserved, and a world that was never launched leaves its whole row
    missing.  Both are named here rather than shrinking the table quietly --
    the failure mode a per-rank report cannot show on its own.

    The refusal arms are held to that same shape: every rank carries one per
    selected module and label, so a rank that dropped them is named here
    instead of quietly shrinking the roster ``all()`` then read.
    """
    selected = _require_selected(selected, "the merge's request")
    reports = []
    for path in paths:
        try:
            reports.append(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as exc:
            _failure(f"merge input {path} is not readable JSON ({type(exc).__name__}: {exc})")
    if not reports:
        _failure("merge needs at least one per-rank report")
    for report in reports:
        if ("expected_arms" not in report or "arms" not in report
                or "refusals" not in report):
            _failure("merge input is not a per-rank device report (no expected/observed "
                     "arms or no refusals); --preflight output is not a device table")

    # What the request names, exactly like the numerical arms below: one refusal
    # arm per selected module and label.  A set that lost an arm -- or all of
    # them -- still satisfies ``all()`` over what is left, so the roster is
    # COMPARED, never merely deduplicated.
    wanted_refusals = {(fixture, module["group"], label)
                       for fixture, spec in selected.items()
                       for module in spec["modules"]
                       for label in REFUSAL_LABELS}

    observed: dict = {}
    for report, path in zip(reports, paths):
        here = {_entry_id(entry) for entry in report["arms"]}
        declared = {tuple(row) for row in report["expected_arms"]}
        missing = sorted(declared - here)
        unexpected = sorted(here - declared)
        if missing or unexpected:
            _failure(f"{path} ran {len(here)} of {len(declared)} arms it declared; "
                     f"missing {missing[:4]}{'...' if len(missing) > 4 else ''}, "
                     f"undeclared {unexpected[:4]}")
        for entry in report["arms"]:
            key = _entry_id(entry)
            if key in observed:
                _failure(f"arm {key} appears in more than one input report")
            observed[key] = entry
        carried = {(entry["fixture"], entry["group"], entry["refusal"])
                   for entry in report["refusals"]}
        missing_refusals = sorted(wanted_refusals - carried)
        undeclared_refusals = sorted(carried - wanted_refusals)
        if missing_refusals or undeclared_refusals:
            _failure(
                f"{path} carries {len(carried)} of {len(wanted_refusals)} refusal arms "
                f"the request names; missing {missing_refusals[:4]}"
                f"{'...' if len(missing_refusals) > 4 else ''}, "
                f"undeclared {undeclared_refusals[:4]}")

    worlds = sorted({entry["tp_size"] for entry in observed.values()})
    groups = {fixture: [module["group"] for module in spec["modules"]]
              for fixture, spec in selected.items()}
    wanted = {(fixture, group, world, rank, m)
              for fixture, module_groups in groups.items()
              for group in module_groups
              for world in worlds
              for rank in range(world)
              for m in m_set}
    absent = sorted(wanted - set(observed))
    if absent:
        _failure(f"the merged table is incomplete: {len(absent)} arms were never run, "
                 f"starting with {absent[:4]}")

    first = reports[0]
    conflicts = {key: sorted({str(report.get(key)) for report in reports})
                 for key in ("device", "torch", "vllm", "mode")
                 if len({str(report.get(key)) for report in reports}) > 1}
    if conflicts:
        _failure(f"the per-rank reports disagree about {sorted(conflicts)}: {conflicts}")

    refusals, seen = [], set()
    for report in reports:
        for entry in report["refusals"]:
            key = (entry["fixture"], entry["group"], entry["refusal"])
            if key not in seen:
                seen.add(key)
                refusals.append(entry)
    # Every rank carried the whole roster, so the joined table IS that roster;
    # the verdict is read off it rather than off whatever the lists held.  This
    # mirror of the arms' own ``absent`` check names a refusal that no rank ran
    # before the verdict reads a key that is not there.
    by_roster = {(entry["fixture"], entry["group"], entry["refusal"]): entry
                 for entry in refusals}
    absent_refusals = sorted(wanted_refusals - set(by_roster))
    if absent_refusals:
        _failure(f"the merged table is missing {len(absent_refusals)} refusal arms the "
                 f"request names, starting with {absent_refusals[:4]}")
    ranks = [{"world_size": report.get("world_size"), "tp_rank": report.get("tp_rank"),
              "local_rank": (report.get("bringup") or {}).get("local_rank"),
              "tp_group": (report.get("bringup") or {}).get("tp_group"),
              "arms": len(report["arms"]),
              "passed": sum(1 for entry in report["arms"] if entry.get("passed")),
              "all_passed": all(entry.get("passed") for entry in report["arms"])}
             for report in reports]
    tp_groups = sorted({rank["tp_group"] for rank in ranks if rank["tp_group"]})
    device_counts = sorted({(report.get("bringup") or {}).get("device_count")
                            for report in reports
                            if (report.get("bringup") or {}).get("device_count") is not None})
    fallback = [f"rank {rank['tp_rank']} of {rank['world_size']}: "
                f"{(report.get('bringup') or {}).get('tp_group_fallback')}"
                for rank, report in zip(ranks, reports)
                if (report.get("bringup") or {}).get("tp_group_fallback")]
    return {
        "device": first.get("device"), "torch": first.get("torch"),
        "vllm": first.get("vllm"), "mode": first.get("mode"),
        "worlds": worlds, "ranks": ranks,
        "tp_groups": tp_groups, "device_counts": device_counts,
        "tp_group_fallbacks": fallback,
        "arms": [observed[key] for key in sorted(observed)],
        "refusals": sorted(refusals, key=lambda entry: (entry["fixture"], entry["group"],
                                                        entry["refusal"])),
        "arms_total": len(observed), "arms_expected_total": len(wanted),
        "refusals_total": len(by_roster),
        "refusals_expected_total": len(wanted_refusals),
        "all_arms_passed": all(entry.get("passed") for entry in observed.values()),
        "all_refusals_passed": all(
            by_roster[key].get("refused") and by_roster[key].get("named_the_mismatch")
            for key in sorted(wanted_refusals)),
    }


def _select(fixtures, families) -> dict:
    """The fixture keys a request names, without refusing the empty answer.

    A key is a tree and its ``family`` is the route that serves it, so the two
    filters are not independent: ``--fixture GLM_A8 --family TESSERA_BF16``
    intersects nothing, because GLM_A8 carries the FP8 family.  This function
    is the pure selector; ``_require_selected`` is the refusal in front of it.
    """
    return {fixture: spec for fixture, spec in FIXTURES.items()
            if (not fixtures or fixture in fixtures)
            and (not families or spec["family"] in families)}


def _require_selected(selected, request: str) -> dict:
    """The selection, or a refusal when the request names no tree.

    An empty intersection is not a smaller table: it is a request for a table
    of zero arms, and every ``all()`` over that table is vacuously true.  So
    each mode that turns a selection into a table refuses it HERE, named,
    instead of reporting a green run that covered nothing.
    """
    if not selected:
        _failure(
            "the selection is empty: "
            f"{request} names no tree of {', '.join(sorted(FIXTURES))} -- one family "
            "per key, so a family crossed with the other family's key selects "
            "nothing, and a table of zero arms is not a green one")
    return selected


def _preflight(selected) -> dict:
    """Resolve and check everything a device run would fail on first.

    The refusal arms are CPU work -- the reader compares the wire's own sidecar
    against the declaration and needs no device -- so they run here as well:
    this mode is the receipt for "an unsupported declaration fails closed by
    name", and the device run is what remains for the numerics.
    """
    selected = _require_selected(selected, "the preflight's selection")
    out = {"fixtures": [], "refusals": [], "ok": True}
    for fixture, spec in selected.items():
        family = spec["family"]
        root = _fixture_root(spec)
        for module in spec["modules"]:
            blob = _read_wire(root, module["tensor"])
            _, declared = _declared(root, module["group"], blob, module["group"])
            if declared["family"] != family:
                _failure(f"{module['group']} declares {declared['family']}, not {family}")
            out["ok"] = _refusal_arms(fixture, family, module, declared, blob,
                                      out) and out["ok"]
            plans = []
            for parallel, tp_rank, tp_size in (
                    (module["parallel"], 0, 1),
                    (module["parallel"], 0, 2),
                    (module["parallel"], 1, 2)):
                plan = _plan(declared, parallel, tp_rank, tp_size)
                plans.append({"tp_rank": tp_rank, "tp_size": tp_size,
                              "shard_rows": int(plan.shard_rows),
                              "shard_columns": int(plan.shard_columns)})
            out["fixtures"].append({
                "fixture": fixture, "family": family, "root": str(root),
                "group": module["group"],
                "tensor": module["tensor"], "bytes": len(blob),
                "rows": int(declared["rows"]), "columns": int(declared["columns"]),
                "q256": int(declared["q256"]), "parallel": module["parallel"],
                "plans": plans,
            })
    return out


def _parser() -> argparse.ArgumentParser:
    """This harness's own CLI, kept separate so a test can read its spellings.

    A device run puts ``torch.distributed.run`` in front of this parser, and
    torchrun resolves some option-like tokens before this module's ``argv``
    exists (see the module docstring on ``--mset``).  That is a property of the
    SPELLINGS, so the spellings are reachable without running ``main``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", action="append", choices=sorted(FIXTURES),
                        help="restrict to a fixture key; repeatable (default: all)")
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES),
                        help="restrict to a payload family; repeatable "
                             "(default: every family)")
    # NOT ``--m``: see the module docstring.  torchrun in front of this parser
    # refuses it as an ambiguous abbreviation of its own options.
    parser.add_argument("--mset", type=int, action="append", dest="mset",
                        help=f"batch sizes (default: {' '.join(map(str, DEFAULT_M))})")
    parser.add_argument("--mode", default="streamed", choices=("streamed", "resident"),
                        help="the declared residency (default: streamed)")
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2),
                        help="the widest world size to drive (default: 2)")
    parser.add_argument("--preflight", action="store_true",
                        help="resolve the fixtures and build every plan; no device needed")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="join per-rank device reports into one table and check it")
    parser.add_argument("--json", help="write the report here as well as to stdout; "
                                       "a '{rank}' in the path is filled per process")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    # WHAT THIS RUN WAS ASKED FOR.  A key is a tree and a family is a route, so
    # the two filters can intersect empty -- ``--fixture GLM_A8 --family
    # TESSERA_BF16`` is one -- and every mode below would then report a green
    # table of zero arms.  The refusal names the request that selected nothing.
    selected = _require_selected(
        _select(args.fixture, args.family),
        f"--fixture {' '.join(args.fixture) if args.fixture else '(all)'} crossed with "
        f"--family {' '.join(args.family) if args.family else '(all)'}")

    if args.merge:
        report = _merged(selected, args.merge, args.mset or DEFAULT_M)
    elif args.preflight:
        report = _preflight(selected)
    else:
        import torch

        if not torch.cuda.is_available():
            _failure("no CUDA device; this crosscheck drives the route on a device")
        world, rank, _local_rank = _torch_env()
        if world != args.tp:
            _failure(
                f"this process is rank {rank} of a world of {world}, but --tp {args.tp} "
                f"was asked for; each world size runs under its own torchrun: "
                f"python3 -m torch.distributed.run --nproc_per_node={args.tp} "
                f"--standalone {Path(__file__).name} --tp {args.tp}"
            )
        _config, _stack, bringup = _bring_up(world, rank)
        report = {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "mode": args.mode,
            "world_size": world,
            "tp_rank": rank,
            "bringup": bringup,
            "arms": [],
            # What this process is ABOUT to drive.  A rank that dies leaves
            # these unmet, and --merge refuses the incomplete set by name.
            "expected_arms": [],
            "refusals": [],
        }
        try:
            import vllm

            report["vllm"] = getattr(vllm, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            report["vllm"] = f"unavailable: {type(exc).__name__}: {exc}"
        ok = True
        for fixture, spec in selected.items():
            family = spec["family"]
            root = _fixture_root(spec)
            for module in spec["modules"]:
                blob = _read_wire(root, module["tensor"])
                scheme, declared = _declared(root, module["group"], blob, module["group"])
                ok = _refusal_arms(fixture, family, module, declared, blob, report) and ok
                for m in (args.mset or DEFAULT_M):
                    report["expected_arms"].append(
                        _arm_id(fixture, module["group"], world, rank, m))
                    ok = _arm(fixture, family, module, declared, scheme, blob,
                              module["parallel"], rank, world, m,
                              args.mode, report) and ok
        report["all_arms_passed"] = bool(ok)
    report["preflight"] = bool(args.preflight)
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json:
        # One process per rank writes its own file, so the same command serves
        # every rank of a torchrun phase.
        Path(str(args.json).replace("{rank}", str(report.get("tp_rank", 0)))
             ).write_text(text + "\n")
    if args.preflight:
        return 0 if report["ok"] else 1
    if args.merge:
        return 0 if (report["all_arms_passed"] and report["all_refusals_passed"]) else 1
    return 0 if report.get("all_arms_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
