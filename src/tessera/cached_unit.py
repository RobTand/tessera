"""Producer receipts for reusing exact unit bytes, without importing serving.

A unit's wire does not record the source weight or calibration Hessian. Those
inputs belong to this receipt; the wire still owns geometry, recipe and encoder
identity. Acceptance compares both against freshly supplied producer inputs.
This is an intake gate, not evidence that a serving runtime supports the unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

from .container import parse
from .encoder_identity import encoder_fixture_id, resumable
from .export import (ActivationSource, DEFAULT_CODE, DEFAULT_GROUP, DEFAULT_HALF,
                     HESSIAN_IDENTITY, wire_recipe)
from .grammar import bresenham_rate_schedule
from .manifest import BodyKind, ContainerClass, RotationState
from .unit_artifact import _reach_attrs, build_unit_artifact, encoder_profile_id

CACHE_SCHEMA = "tessera.cached_units.v1"
INPUT_SCHEMA = "tessera.cached_unit_inputs.v1"
ENCODING_INPUT_SCHEMA = "tessera.encoding_inputs.v1"


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def tensor_identity(tensor) -> dict:
    """Hash actual contiguous values, dtype and shape; never a filename."""
    import torch

    value = tensor.detach().cpu().contiguous()
    shape = list(value.shape)
    dtype = str(value.dtype)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": dtype, "shape": shape}, sort_keys=True).encode())
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {"algorithm": "sha256.dtype_shape_contiguous.v1", "dtype": dtype,
            "shape": shape, "sha256": digest.hexdigest()}


@lru_cache(maxsize=1)
def encoder_source_sha256() -> str:
    """Conservatively bind the producer package, including unmeasured branches.

    The behavior fixture owns numerical compatibility; this extra source seal
    refuses reuse across edits outside its finite witnesses as well. It may
    reject a harmless source edit, but never relabels the encoder fixture.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*")
                       if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def encoding_input_identity(weight, unit_name: str, grid, q256: int, *,
                            activation: ActivationSource | None = None) -> dict:
    """Source/H/settings identity shared by dense and projected campaign units.

    This function imposes no source-layout or runtime topology. A caller using
    the projected cache/export boundary adds the producer projection through
    ``unit_input_identity``. No invented expert fields are needed for a dense
    campaign's resume check.
    """
    if not isinstance(unit_name, str) or not unit_name:
        raise ValueError("encoding input identity requires a unit name")
    if len(weight.shape) != 2 or min(weight.shape) <= 0:
        raise ValueError("encoding input identity requires a nonempty 2-D source weight")
    if type(q256) is not int or q256 <= 0:
        raise ValueError("cached unit rung must be a positive integer")
    name = ActivationSource.unit_name(unit_name)
    calibration = None
    if activation is not None:
        if name not in activation.hessians:
            raise ValueError(f"{name}: cached unit has no exact Hessian key")
        hessian = activation.hessians[name]
        if list(hessian.shape) != [weight.shape[1], weight.shape[1]]:
            raise ValueError(f"{name}: cached unit Hessian shape disagrees with columns")
        settings = activation.config_block()
        # Paths and prose are not calibration identity. Every numerical setting
        # in the owner's config remains, including trailing objectives/sweeps.
        settings.pop("note", None)
        settings["hessian"] = {key: activation.provenance[key] for key in HESSIAN_IDENTITY}
        calibration = {"settings": settings, "hessian": tensor_identity(hessian)}
    return _json_copy({"schema": ENCODING_INPUT_SCHEMA, "unit": name,
                       "source": tensor_identity(weight), "calibration": calibration,
                       "recipe": {"grid": grid.name, "q256": q256,
                                  **wire_recipe(grid, q256).to_config()},
                       "encoder_source_sha256": encoder_source_sha256(),
                       "encoder_fixture_id": encoder_fixture_id().hex()})


def unit_input_identity(weight, projection: dict, grid, q256: int, *,
                        activation: ActivationSource | None = None) -> dict:
    """Add an explicit producer projection to the common encoding inputs.

    ``projection.tensor`` is the logical producer tensor name WITH ``.weight``;
    cache keys use ``ActivationSource.unit_name(tensor)`` WITHOUT that suffix.
    ``source_tensor`` remains the exact physical checkpoint key, with whatever
    suffix that checkpoint owns. The projection comes from the producer plan,
    not from an intake-side guess about tensor rank or model architecture.
    """
    required = {"tensor", "source_tensor", "source_layout", "source_slice",
                "expert", "projection", "group", "rows", "cols"}
    absent = required - projection.keys()
    if absent:
        raise ValueError(f"cached unit projection missing {sorted(absent)}")
    if not isinstance(projection["tensor"], str) or not projection["tensor"].endswith(".weight"):
        raise ValueError("cached unit projection.tensor must include .weight")
    if list(weight.shape) != [projection["rows"], projection["cols"]]:
        raise ValueError("cached unit source shape disagrees with producer projection")
    identity = encoding_input_identity(weight, projection["tensor"], grid, q256,
                                        activation=activation)
    return _json_copy({**identity, "schema": INPUT_SCHEMA,
                       "projection": {key: projection[key] for key in sorted(required)}})


def _local_filename(name: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"cached unit filename must be a local leaf: {name!r}")
    return name


@dataclass(frozen=True)
class AcceptedUnit:
    blob: bytes
    manifest: object
    wire_bytes: int


def _check_wire(blob: bytes, identity: dict):
    from .control import grid_for_name
    from .planes import PlaneKind

    schema = identity.get("schema")
    if schema not in (INPUT_SCHEMA, ENCODING_INPUT_SCHEMA):
        raise ValueError("cached unit input schema is unsupported")
    projected = schema == INPUT_SCHEMA
    if projected != ("projection" in identity):
        raise ValueError("cached unit input schema/projection fields disagree")
    shape = identity["source"]["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("cached unit source identity must carry an exact 2-D shape")
    rows, columns = shape
    if projected and [identity["projection"]["rows"], identity["projection"]["cols"]] != shape:
        raise ValueError("cached unit projection geometry disagrees with source identity")
    artifact = parse(blob)
    manifest = artifact.manifest
    recipe_spec = identity["recipe"]
    grid = grid_for_name(recipe_spec["grid"])
    q256 = recipe_spec["q256"]
    recipe = wire_recipe(grid, q256)
    if recipe_spec != {"grid": grid.name, "q256": q256, **recipe.to_config()}:
        raise ValueError("cached unit recipe differs from the producer recipe")
    geometry = manifest.geometry
    if manifest.shard is not None or len(manifest.terminals) != 1:
        raise ValueError("cached unit must be one complete, unsharded terminal")
    order = {kind: index for index, kind in enumerate(manifest.plane_order)}
    if any(artifact.terminal.plane_elements[order[plane.kind]] != plane.element_count
           for plane in manifest.planes):
        raise ValueError("cached unit must carry complete planes, not a terminal prefix")
    if (geometry.rows, geometry.columns, geometry.quantizable_params) != (
            rows, columns, rows * columns):
        raise ValueError("cached unit wire geometry disagrees with source projection")
    superblock = inspect.signature(build_unit_artifact).parameters["superblock"].default
    if (geometry.group_weights, geometry.half_weights, geometry.superblock_columns) != (
            DEFAULT_GROUP, DEFAULT_HALF, superblock):
        raise ValueError("cached unit wire group geometry differs from the encoder defaults")
    if manifest.branch.root_q256 != q256 * grid.arity:
        raise ValueError("cached unit wire rung differs from the requested rung")
    if manifest.branch.rotation != RotationState.NONE or manifest.branch.container != ContainerClass.GRIDBOOK:
        raise ValueError("cached unit wire rotation/container differs from the encoder defaults")
    cap = grid.payload_bits if recipe.body is BodyKind.WINDOW else grid.rate_cap
    rates = bresenham_rate_schedule(Fraction(q256 * grid.arity, 256), columns, cap=cap)
    code = None if recipe.body is BodyKind.WINDOW else DEFAULT_CODE
    profile = encoder_profile_id(code, rates, grid, recipe.span, recipe.scale_plane,
                                 recipe.body, recipe.window_bits, recipe.window_seed,
                                 recipe.window_sigma, recipe.channel_sigma)
    if manifest.encoder_profile_id != profile or manifest.rates != rates:
        raise ValueError("cached unit wire encoder profile/rate schedule differs from recipe")
    wire_profile = encoder_profile_id(
        code, manifest.rates, grid, manifest.span, manifest.scale_plane.kind,
        manifest.body, manifest.window_bits, *_reach_attrs(manifest))
    if manifest.encoder_profile_id != wire_profile:
        raise ValueError("cached unit wire recipe/reach fields disagree with encoder profile")
    if (manifest.body, manifest.span, manifest.scale_plane.kind, manifest.window_bits) != (
            recipe.body, recipe.span, recipe.scale_plane, recipe.window_bits):
        raise ValueError("cached unit wire recipe fields differ from requested recipe")
    # The ordinary campaign/export seam spends no completion or release bits.
    for kind in (PlaneKind.RELEASE, PlaneKind.COMPLETION):
        index = manifest.plane_order.index(kind)
        if artifact.terminal.plane_elements[index]:
            raise ValueError(f"cached unit wire carries non-default {kind.name} elements")
    if not resumable(manifest):
        raise ValueError("cached unit wire encoder fixture is not resumable by this encoder")
    return artifact


def make_unit_record(blob: bytes, identity: dict, *, filename: str) -> dict:
    """Record a just-produced unit using the same validator as export intake."""
    _check_wire(blob, identity)
    return {"file": _local_filename(filename), "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "blob_bytes": len(blob), "identity": _json_copy(identity)}


def verify_cached_unit(blob: bytes, record: dict, expected_identity: dict) -> AcceptedUnit:
    if set(record) != {"file", "blob_sha256", "blob_bytes", "identity"}:
        raise ValueError("cached unit record has missing or unknown fields")
    _local_filename(record["file"])
    if record["blob_bytes"] != len(blob) or record["blob_sha256"] != hashlib.sha256(blob).hexdigest():
        raise ValueError("cached unit blob size/sha256 mismatch")
    observed = record["identity"]
    if not isinstance(observed, dict) or set(observed) != set(expected_identity):
        raise ValueError("cached unit input identity fields differ")
    for key, value in expected_identity.items():
        if observed[key] != value:
            raise ValueError(f"cached unit {key} identity mismatch")
    artifact = _check_wire(blob, expected_identity)
    return AcceptedUnit(blob, artifact.manifest, artifact.terminal.exact_bytes)


class CachedUnitBundle:
    """Closed unit roster; all filenames/source bindings checked before reads."""

    def __init__(self, manifest: dict, directory: Path, expected_units: set[str], source: dict):
        if set(manifest) != {"schema", "source", "units"} or manifest["schema"] != CACHE_SCHEMA:
            raise ValueError("cached unit bundle has an unsupported schema or fields")
        if manifest["source"] != source:
            raise ValueError("cached unit bundle source checkpoint identity mismatch")
        units = manifest["units"]
        if not isinstance(units, dict) or set(units) != set(expected_units):
            raise ValueError("cached unit bundle coverage differs from the complete producer plan")
        files = set()
        for key, record in units.items():
            name = _local_filename(record["file"])
            if name in files:
                raise ValueError(f"duplicate cached unit filename: {name}")
            files.add(name)
            if record["identity"]["unit"] != key:
                raise ValueError(f"cached unit coverage key {key} disagrees with receipt")
        self.directory = Path(directory).resolve()
        self.units = _json_copy(units)
        self.manifest_sha256 = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def read(self, key: str) -> tuple[bytes, dict]:
        record = self.units[key]
        path = self.directory / record["file"]
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError(f"cached unit filename escapes bundle: {path}")
        return path.read_bytes(), record


def read_manifest(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate cached unit JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)
