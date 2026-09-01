"""Write a Tessera checkpoint, and read one back.

``build_unit_artifact`` already turns one encoded Linear into bytes that
``read_unit_artifact`` inverts exactly.  This module is the model-level walk
around that seam: encode the Linears a plan names, copy everything else
verbatim, and record the identity a reader needs to accept the result.

**Rungs are addressed by ``(grid, q256)``, not by name.**  A rung *name* like
``TESSERA_E2M1_K2_R896`` is PrismaQuant's label for an allocator candidate; the
thing the wire commits to is ``encoder_profile_id``, which hashes the code, the
forest construction, the rate set and the grid digest.  Keeping the parser on
the producer side and the identity on the wire means a mislabelled artifact is
still *unambiguous* -- the reader rebuilds the grid from the profile id and
refuses anything it cannot reproduce.  Two spellings of one spec is the failure
that identity discipline exists to prevent, so there is deliberately only one
place the grid is decided.

**Rendering identity is asserted, not assumed** (principle 8).  Every unit is
read back off its own bytes and compared to the encoder's reconstruction before
it is written.  The surrogate that priced the Linear, the KL that validated it,
and the bytes that ship are then the same tensor by construction rather than by
three code paths agreeing.

**The artifact declares itself unbacked** (principle 9).  No serving runtime
decodes this container today.  ``route_status`` says so in a field a gate can
read, so nothing downstream can mistake "exportable" for "servable".
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from functools import lru_cache
from pathlib import Path

import torch

from .alphabet import PayloadGrid, build_forest, grid_digest
from .decode import reconstruct_unit
from .encode import encode_unit
from .errors import GrammarError
from .grammar import bresenham_rate_schedule
from .manifest import RotationState
from .trellis import ConvCode
from .unit_artifact import build_unit_artifact, read_unit_artifact

__all__ = [
    "CONTAINER_VERSION",
    "ExportReport",
    "ExportedUnit",
    "encode_linear",
    "export_checkpoint",
    "export_checkpoint_streaming",
    "load_tessera_weight",
    "read_checkpoint_config",
]

#: Bumped when the on-disk *container* changes shape.  The per-unit wire has
#: its own identity (``encoder_profile_id``); this versions the checkpoint
#: layout around it -- the tensor suffix, the config schema, the plan encoding.
CONTAINER_VERSION = 1

#: The suffix a quantized Linear's bytes are stored under.  A reader that does
#: not know Tessera sees an opaque uint8 tensor under a name that is not the
#: original, so it cannot silently load the blob *as* a weight.
BLOB_SUFFIX = ".tessera"

DEFAULT_CODE = ConvCode(memory=6)
DEFAULT_GROUP = 32
DEFAULT_HALF = 16
#: Scale-plane refits per unit (``encode_unit``).  An encoder setting, not
#: wire: the bytes decode identically at any value.  Recorded in the config so
#: a merge can refuse parts built at different settings.
DEFAULT_SCALE_REFIT = 4


@dataclass(frozen=True)
class ExportedUnit:
    """One serialised Linear, with the bytes it actually cost."""

    name: str
    blob: bytes
    rows: int
    columns: int
    q256: int
    exact_bytes: int

    @property
    def params(self) -> int:
        return self.rows * self.columns

    @property
    def bpp(self) -> Fraction:
        """Bits per quantizable parameter, exact -- counted, never estimated."""
        return Fraction(self.exact_bytes * 8, self.params)


@dataclass(frozen=True)
class ExportReport:
    """What was written, and what it weighs."""

    units: "tuple[ExportedUnit, ...]"
    passthrough_bytes: int
    quantized_bytes: int
    quantized_params: int
    grid_digest: str

    @property
    def body_bpp(self) -> Fraction:
        """bpp over *quantizable* parameters only (principle 12).

        Passthrough tensors -- embeddings, norms, anything the plan left alone
        -- are excluded from the denominator and reported separately, because
        a bpp that silently averages them in is not comparable to anyone's.
        """
        if not self.quantized_params:
            return Fraction(0)
        return Fraction(self.quantized_bytes * 8, self.quantized_params)

    @property
    def total_bytes(self) -> int:
        return self.passthrough_bytes + self.quantized_bytes


@lru_cache(maxsize=256)
def _plan_for(grid: PayloadGrid, q256: int, columns: int):
    """Rate schedule and forests for one (grid, rung, width).

    Cached because the forests are an exhaustive per-rate optimisation and are
    identical for every Linear of the same width at the same rung -- on a
    288-expert MoE layer that is hundreds of units sharing one plan, and
    rebuilding it per tensor is the export's largest avoidable cost.
    """
    root = Fraction(q256 * grid.arity, 256)
    rates = bresenham_rate_schedule(root, columns, cap=grid.rate_cap)
    forests = {rate: build_forest(rate, grid=grid) for rate in sorted(set(rates))}
    return rates, forests


def encode_linear(
    weight: torch.Tensor,
    *,
    grid: PayloadGrid,
    q256: int,
    name: str = "unit",
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    completion: "int | None" = 0,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
) -> ExportedUnit:
    """Encode one ``[out_features, in_features]`` weight to artifact bytes.

    ``completion`` is the second rate axis and it was previously nailed shut at
    zero here.  A column at body rate ``R`` may spend up to ``cap - R`` further
    bits selecting among the descendants its trellis subset reaches; ``None``
    spends every one of them, an integer spends at most that many, and ``0``
    spends none.  It is a real rate: the artifact pays for what it spends, so
    ``(q256, completion)`` is a two-dimensional rate grid, not a rung and a
    switch.  The default stays ``0`` so the exporter's rung names keep meaning
    the rate they have always meant -- ``q256`` alone -- and a caller that wants
    the other axis asks for it.

    ``verify`` reads the bytes back and compares to the encoder's own
    reconstruction.  It is on by default and costs one decode: the guarantee
    that the shipped bytes mean what the surrogate priced is worth more than
    the milliseconds, and an exporter that only *believes* it round-trips is
    how a rendering confound gets into an artifact.
    """
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D weight, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if rows % grid.arity:
        raise GrammarError(
            f"{name}: {rows} rows is not divisible by the grid arity {grid.arity}"
        )
    rates, forests = _plan_for(grid, q256, columns)
    unit = encode_unit(
        weight, forests, rates, code,
        rotation=rotation, with_diagonals=with_diagonals,
        completion=completion, group=group, half=half,
        scale_refit=scale_refit,
    )
    # ``q256`` here is the rung's PER-POSITION rate (the R-number in a rung
    # name, and what ``artifact_bpp`` prices).  ``build_unit_artifact`` declares
    # the per-CODE rate, and a code spans ``arity`` positions.  Passing the
    # per-position number straight through produces a legal artifact whose
    # manifest states half the rate it carries -- silent, and exactly the
    # confusion ``build_unit_artifact``'s own comment flags.
    _, region, blob = build_unit_artifact(
        unit, name, forests, q256 * grid.arity, code
    )
    if verify:
        recovered = read_unit_artifact(blob, device=weight.device)
        reference = reconstruct_unit(unit, forests, code)
        if not torch.equal(recovered, reference):
            raise GrammarError(
                f"{name}: the bytes do not decode to the encoder's own "
                "reconstruction -- refusing to write a unit whose surrogate "
                "and payload disagree"
            )
    return ExportedUnit(
        name=name, blob=blob, rows=rows, columns=columns,
        q256=q256, exact_bytes=len(region),
    )


def export_checkpoint(
    tensors: "dict[str, torch.Tensor]",
    plan: "dict[str, int]",
    out_dir: "str | Path",
    *,
    grid: PayloadGrid,
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    extra_config: "dict | None" = None,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
) -> ExportReport:
    """Write ``tensors`` to ``out_dir``, encoding every name ``plan`` rates.

    ``plan`` maps tensor name -> per-position body rate in q256 units.  A name
    in ``plan`` that is absent from ``tensors`` is an error rather than a
    no-op: a plan that silently fails to apply is how an artifact ends up
    heavier than the allocation that justified it.
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(plan) - set(tensors))
    if missing:
        raise KeyError(
            f"plan names {len(missing)} tensor(s) not present: {missing[:5]}"
        )

    payload: "dict[str, torch.Tensor]" = {}
    units: "list[ExportedUnit]" = []
    passthrough_bytes = 0
    for name, tensor in tensors.items():
        if name in plan:
            unit = encode_linear(
                tensor, grid=grid, q256=plan[name], name=name, code=code,
                group=group, half=half, rotation=rotation,
                with_diagonals=with_diagonals, verify=verify,
                scale_refit=scale_refit,
            )
            units.append(unit)
            payload[name + BLOB_SUFFIX] = torch.frombuffer(
                bytearray(unit.blob), dtype=torch.uint8
            )
        else:
            payload[name] = tensor.contiguous().cpu()
            passthrough_bytes += tensor.numel() * tensor.element_size()

    save_file(payload, str(out / "model.safetensors"))

    quantized_bytes = sum(u.exact_bytes for u in units)
    quantized_params = sum(u.params for u in units)
    report = ExportReport(
        units=tuple(units),
        passthrough_bytes=passthrough_bytes,
        quantized_bytes=quantized_bytes,
        quantized_params=quantized_params,
        grid_digest=grid_digest(grid),
    )

    _write_config(out, grid, code, group, half, rotation, with_diagonals,
                  report, plan, extra_config, scale_refit)
    return report


def _write_config(out: Path, grid, code, group, half, rotation, with_diagonals,
                  report: "ExportReport", plan: "dict[str, int]",
                  extra_config: "dict | None", scale_refit: int = 0) -> None:
    config = {
        "quant_method": "tessera",
        "container_version": CONTAINER_VERSION,
        "blob_suffix": BLOB_SUFFIX,
        "grid": {
            # The digest is the wire identity and the only field a reader may
            # trust.  The name and base are recorded so the config can be
            # *audited* -- a config that cannot say which grid it used forces
            # every reader to reverse a hash to answer "what format is this?".
            "digest": grid_digest(grid),
            "name": grid.name,
            "base": grid.name.split("x")[0],
            "partition": grid.partition,
            "arity": grid.arity,
            "size": grid.size,
            "rate_cap": grid.rate_cap,
        },
        "conv_memory": code.memory,
        # ``refit`` counts trellis passes (= refits); ``schedule`` says how they
        # interleave, because the same count meant a different encoder before
        # 61df165 (k refits BETWEEN k+1 passes) -- the merge guard compares both.
        "scale": {"group": group, "half": half, "refit": scale_refit,
                  "schedule": "amax" if scale_refit == 0 else "trailing-refit"},
        "rotation": rotation.name,
        "with_diagonals": bool(with_diagonals),
        "route_status": "unbacked",
        "requires_serve_flags": [],
        # A unit is one trellis blob, not a sliceable tensor: the path runs down
        # rows within a column, so a row-parallel split cuts the trellis along
        # its own state. EXL3 narrows tensor dims and is TP-agnostic; Tessera
        # must be *re-encoded* per rank, which makes an artifact TP-specific.
        # Declared so a loader cannot quietly use it at the wrong degree.
        "tp_size": 1,
        "accounting": {
            "quantized_params": report.quantized_params,
            "quantized_bytes": report.quantized_bytes,
            "passthrough_bytes": report.passthrough_bytes,
            "body_bpp": float(report.body_bpp),
            "body_bpp_exact": [report.body_bpp.numerator,
                               report.body_bpp.denominator],
        },
        "plan": dict(plan),
        "rungs_q256": sorted({u.q256 for u in report.units}),
    }
    if extra_config:
        config.update(extra_config)
    (out / "tessera_config.json").write_text(json.dumps(config, indent=2))


def export_checkpoint_streaming(
    source_dir: "str | Path",
    out_dir: "str | Path",
    plan: "dict[str, int]",
    *,
    grid: PayloadGrid,
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    device: "str | torch.device" = "cuda",
    extra_config: "dict | None" = None,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
    copy_aux: bool = True,
    progress=None,
    shard_filter: "set[str] | None" = None,
) -> ExportReport:
    """Export shard-by-shard, holding one shard in memory at a time.

    The in-memory ``export_checkpoint`` cannot touch the models this format
    exists for: the target is a 100B-plus checkpoint whose weights never fit
    beside their own encoding.  One output shard is written per input shard, so
    the mapping stays 1:1 and a partial run is inspectable rather than opaque.

    Encoding runs on ``device``; the trellis is the whole cost of an export and
    it is a GPU job (principle 7).

    ``shard_filter`` restricts the run to a subset of input shards.  The 1:1
    shard mapping is what makes this safe: shards share no state -- the plan is
    per-tensor, the forests are rebuilt per (grid, rung, width) and cached, and
    nothing accumulates across shards except the report -- so N boxes each
    taking a disjoint subset produce exactly the files one box would have
    written, and the run becomes embarrassingly parallel across a fleet.  The
    index and config a filtered run writes cover **only its own shards**; the
    caller merges them.  This exists because a 320B-parameter export is nine
    hours on one GB10 and the second one was idle at 4 W.
    """
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: "dict[str, list[str]]" = {}
        for tensor_name, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor_name)
    else:
        shards = {}
        for shard_path in sorted(src.glob("*.safetensors")):
            with safe_open(str(shard_path), framework="pt") as handle:
                shards[shard_path.name] = list(handle.keys())

    if shard_filter is not None:
        unknown = sorted(set(shard_filter) - set(shards))
        if unknown:
            raise KeyError(f"shard_filter names absent shards: {unknown[:5]}")
        shards = {k: v for k, v in shards.items() if k in shard_filter}
        if not shards:
            raise ValueError("shard_filter selected no shards")

    known = {name for names in shards.values() for name in names}
    missing = sorted(set(plan) - known)
    if missing and shard_filter is None:
        raise KeyError(
            f"plan names {len(missing)} tensor(s) not present: {missing[:5]}"
        )

    units: "list[ExportedUnit]" = []
    passthrough_bytes = 0
    new_weight_map: "dict[str, str]" = {}

    for position, (shard, names) in enumerate(sorted(shards.items()), start=1):
        payload: "dict[str, torch.Tensor]" = {}
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    unit = encode_linear(
                        tensor.to(device), grid=grid, q256=plan[name], name=name,
                        code=code, group=group, half=half, rotation=rotation,
                        with_diagonals=with_diagonals, verify=verify,
                        scale_refit=scale_refit,
                    )
                    units.append(unit)
                    key = name + BLOB_SUFFIX
                    payload[key] = torch.frombuffer(
                        bytearray(unit.blob), dtype=torch.uint8
                    )
                else:
                    payload[name] = tensor.contiguous()
                    passthrough_bytes += tensor.numel() * tensor.element_size()
        for key in payload:
            new_weight_map[key] = shard
        save_file(payload, str(out / shard), metadata={"format": "pt"})
        del payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress is not None:
            progress(position, len(shards), shard, len(units))

    report = ExportReport(
        units=tuple(units),
        passthrough_bytes=passthrough_bytes,
        quantized_bytes=sum(u.exact_bytes for u in units),
        quantized_params=sum(u.params for u in units),
        grid_digest=grid_digest(grid),
    )
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": report.total_bytes},
         "weight_map": new_weight_map}, indent=2))
    _write_config(out, grid, code, group, half, rotation, with_diagonals,
                  report, plan, extra_config, scale_refit)
    if copy_aux:
        for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
            for aux in src.glob(pattern):
                if aux.name in {"model.safetensors.index.json"}:
                    continue
                shutil.copy2(aux, out / aux.name)
    return report


def read_checkpoint_config(out_dir: "str | Path") -> dict:
    return json.loads((Path(out_dir) / "tessera_config.json").read_text())


def _shard_holding(out: Path, key: str) -> Path:
    """Locate the shard holding ``key``, honouring a written index.

    ``export_checkpoint_streaming`` writes one shard per input shard plus an
    index; only the in-memory ``export_checkpoint`` writes a lone
    ``model.safetensors``.  A reader that assumes the single-file layout can
    read back nothing this format actually exports at scale, so the index is
    consulted first and the single file is the fallback, not the rule.
    """
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if key not in weight_map:
            raise KeyError(f"{key!r} is not in this checkpoint's index")
        return out / weight_map[key]
    return out / "model.safetensors"


def load_tessera_weight(
    out_dir: "str | Path", name: str, device: "str | torch.device" = "cpu"
) -> torch.Tensor:
    """Decode one Linear back out of a written checkpoint."""
    from safetensors import safe_open

    out = Path(out_dir)
    config = read_checkpoint_config(out)
    key = name + config.get("blob_suffix", BLOB_SUFFIX)
    with safe_open(str(_shard_holding(out, key)), framework="pt") as handle:
        if key not in handle.keys():
            raise KeyError(f"{name!r} is not a quantized unit in this checkpoint")
        blob = handle.get_tensor(key)
    return read_unit_artifact(bytes(blob.numpy().tobytes()), device=device)
