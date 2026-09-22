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
ROOTED_CACHE_SCHEMA = "tessera.cached_units.v2"
INPUT_SCHEMA = "tessera.cached_unit_inputs.v1"
ENCODING_INPUT_SCHEMA = "tessera.encoding_inputs.v1"


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest_host_tensor(value) -> str:
    """The ``sha256.dtype_shape_contiguous.v1`` digest of a contiguous host tensor.

    The construction ``tensor_identity`` stamps on every cached unit and every
    sealed capture, in one place: dtype and shape as JSON, a NUL, then the
    bytes.  ``value`` must already be a contiguous CPU tensor -- a pinned
    staging buffer the seal prefetch filled counts, which is why this is
    split out -- and the bytes are fed to the hash through the buffer
    protocol rather than ``tobytes()``: the digest is the same, the 64 MiB
    copy under the GIL is not made.
    """
    import torch

    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("digest_host_tensor needs a contiguous CPU tensor")
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                             sort_keys=True).encode())
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def host_fingerprint(value) -> int:
    """An exact, order-free integer over a contiguous host tensor's bytes.

    The sum of the bytes read as int32 words (int64 accumulator, so it is
    exact and independent of reduction order), or of the raw bytes when the
    length is not a whole number of words.  ``export.device_fingerprint``
    computes the same integer from a device tensor without staging it: the
    seal prefetch takes this one from the bytes it digested, the consumer
    takes that one from the tensor it is about to encode, and they agree
    exactly when the bytes do.  It is a change detector for the seal's memo,
    not a digest -- the digest is the sha256 beside it.
    """
    import numpy as np
    import torch

    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("host_fingerprint needs a contiguous CPU tensor")
    raw = value.view(torch.uint8).numpy()
    if raw.nbytes % 4 == 0:
        return int(raw.view(np.int32).sum(dtype=np.int64))
    return int(raw.sum(dtype=np.int64))


def tensor_identity(tensor) -> dict:
    """Hash actual contiguous values, dtype and shape; never a filename."""
    value = tensor.detach().cpu().contiguous()
    return {"algorithm": "sha256.dtype_shape_contiguous.v1", "dtype": str(value.dtype),
            "shape": list(value.shape), "sha256": digest_host_tensor(value)}


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


HESSIAN_IDENTITY_MODES = ("committed", "digested")


class CachedUnitIdentity:
    """Derive cached-unit input identities, establishing H identity one of two ways.

    ``derive(weight, unit_name, unit, grid, q256, *, activation)`` is the
    producer's own identity factory -- the historical producer's when a
    receipt was written by one, this package's otherwise -- and every field
    of every identity still comes from it.  What this class decides is
    where ``calibration.hessian`` comes from:

    ``digested``
        the producer consumes H through ``activation.hessians[name]`` and
        digests it, the path every cached export took before this class.
        On a reference document that is one whole canonical ``.pt`` read
        per unit, for bytes the cached path never otherwise touches.
    ``committed``
        the sealed commitment the reference document holds for the unit,
        served by ``ReferenceHessians.commitment``.  Exact by construction:
        ``ReferenceHessians.__getitem__`` refuses any payload whose identity
        differs from that commitment, so the digested value on an accepting
        run is the committed value.  The first unit is still derived in full
        -- H read, digested -- and the spliced form is required to equal it
        before any other unit is served; that witness also fixes
        ``calibration.settings``, which is unit-independent (it is the
        owner's ``config_block`` less prose, the same for every unit).

    Only a ``ReferenceHessians`` owner has commitments; a plain mapping is
    digested whatever was asked, and no activation means no calibration
    block at all.  ``record()`` says which happened, so a receipt never
    implies bytes were compared when they were not.  Safe to call from
    several threads once witnessed; the witness itself is serialised.
    """

    def __init__(self, derive, activation, *, mode: str = "committed"):
        import threading

        if mode not in HESSIAN_IDENTITY_MODES:
            raise ValueError(f"cached unit Hessian identity mode must be one of {HESSIAN_IDENTITY_MODES}")
        self._derive = derive
        self.activation = activation
        self._lock = threading.Lock()
        self._settings = None
        self._witness = None
        self._reference = None
        if activation is None:
            self.established = None
        else:
            from .hessian_capture import ReferenceHessians
            reference = isinstance(activation.hessians, ReferenceHessians)
            self.established = mode if reference else "digested"
            if reference:
                owner = activation.hessians
                self._reference = {
                    "path": activation.provenance.get("path"),
                    "document_sha256": owner.document_sha256,
                    **{k: v for k, v in owner.binding().items() if k != "schema"}}

    def __call__(self, weight, unit_name: str, unit, grid, q256: int) -> dict:
        if self.established != "committed":
            return self._derive(weight, unit_name, unit, grid, q256, activation=self.activation)
        with self._lock:
            if self._settings is None:
                return self._witness_unit(weight, unit_name, unit, grid, q256)
        return self._committed(weight, unit_name, unit, grid, q256)

    def _witness_unit(self, weight, unit_name, unit, grid, q256):
        from .errors import GrammarError

        full = self._derive(weight, unit_name, unit, grid, q256, activation=self.activation)
        self._settings = _json_copy(full["calibration"]["settings"])
        spliced = self._committed(weight, unit_name, unit, grid, q256)
        if spliced != full:
            self._settings = None
            raise GrammarError(f"{full['unit']}: committed Hessian identity disagrees with "
                               "the consumed derivation; refusing to serve commitments")
        self._witness = {"unit": full["unit"], "agreed": True}
        return full

    def _committed(self, weight, unit_name, unit, grid, q256):
        identity = self._derive(weight, unit_name, unit, grid, q256, activation=None)
        name = identity["unit"]
        hessians = self.activation.hessians
        if name not in hessians:
            raise ValueError(f"{name}: cached unit has no exact Hessian key")
        commitment = hessians.commitment(name)
        if commitment["shape"] != [weight.shape[1], weight.shape[1]]:
            raise ValueError(f"{name}: cached unit Hessian shape disagrees with columns")
        identity["calibration"] = {"settings": self._settings, "hessian": commitment}
        return _json_copy(identity)

    def record(self) -> dict:
        """How calibration H identity was established, for the export receipt."""
        served = reference = None
        if self._reference is not None:
            reference = dict(self._reference, capture_sha256=self.activation.capture_sha256())
        if self.established == "committed":
            served = len(self.activation.hessians.receipt()["committed_units_served"])
        return _json_copy({"schema": "tessera.cached_unit_hessian_identity.v1",
                           "established": self.established,
                           "reference": reference,
                           "witness": self._witness,
                           "committed_units_served": served})


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
        rooted = manifest.get("schema") == ROOTED_CACHE_SCHEMA
        fields = {"schema", "source", "units"}
        if rooted:
            fields |= {"wire_roots", "unit_roots", "producer_packages",
                       "reuse_authority", "encoder_adoptions", "served_activation_policy", "served_activations"}
        if set(manifest) != fields or manifest["schema"] not in (CACHE_SCHEMA, ROOTED_CACHE_SCHEMA):
            raise ValueError("cached unit bundle has an unsupported schema or fields")
        from .serving_parts import SOURCE_PART_SCHEMA, prove_source_part
        if isinstance(source, dict) and source.get("schema") == SOURCE_PART_SCHEMA:
            # A serving part hashed only the shards it reads (tessera#495);
            # the bundle's whole-checkpoint identity must vouch for each.
            whole = manifest["source"]
            if (not isinstance(whole, dict)
                    or set(whole) != {"config_sha256", "auxiliary_sha256", "files", "tensors"}
                    or not isinstance(whole["files"], dict) or not isinstance(whole["tensors"], dict)
                    or set(whole["files"]) != set(whole["tensors"].values())):
                raise ValueError("cached unit bundle source is not a whole-checkpoint identity")
            prove_source_part(source, whole, "cached unit bundle")
        elif manifest["source"] != source:
            raise ValueError("cached unit bundle source checkpoint identity mismatch")
        units = manifest["units"]
        if not isinstance(units, dict) or set(units) != set(expected_units):
            raise ValueError("cached unit bundle coverage differs from the complete producer plan")
        self.directory = Path(directory).resolve()
        self.roots = {"legacy": self.directory}
        self.unit_roots = dict.fromkeys(units, "legacy")
        self.producer_packages = {}
        self.reuse_authority = None
        self.encoder_adoptions = {}
        self.served_activation_policy, self.served_activations = None, {}
        if rooted:
            self._bind_rooted(manifest, units)
        files = set()
        for key, record in units.items():
            name = _local_filename(record["file"])
            location = (str(self.roots[self.unit_roots[key]]), name)
            if location in files:
                raise ValueError(f"duplicate cached unit filename: {name}")
            files.add(location)
            if record["identity"]["unit"] != key:
                raise ValueError(f"cached unit coverage key {key} disagrees with receipt")
        self.units = _json_copy(units)
        self.manifest_sha256 = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def read(self, key: str) -> tuple[bytes, dict]:
        record = self.units[key]
        root = self.roots[self.unit_roots[key]]
        path = root / record["file"]
        if root.resolve() != root or path.is_symlink() or path.resolve().parent != root:
            raise ValueError(f"cached unit filename escapes bundle: {path}")
        return path.read_bytes(), record

    def _bind_rooted(self, manifest, units):
        roots, owners = manifest["wire_roots"], manifest["unit_roots"]
        if (not isinstance(roots, dict) or not roots or not isinstance(owners, dict)
                or set(owners) != set(units) or set(owners.values()) != set(roots)):
            raise ValueError("cached unit root coverage differs from selected units")
        resolved = {}
        for name, spelling in roots.items():
            path = Path(spelling)
            if (not isinstance(name, str) or not name or not path.is_absolute()
                    or str(path) != spelling or path.resolve() != path or not path.is_dir()):
                raise ValueError("cached unit root must be a canonical existing directory")
            resolved[name] = path
        if len(set(resolved.values())) != len(resolved):
            raise ValueError("cached unit roots alias the same directory")
        packages = manifest["producer_packages"]
        seals = {record["identity"]["encoder_source_sha256"] for record in units.values()}
        if not isinstance(packages, dict) or set(packages) != seals:
            raise ValueError("cached unit producer coverage differs from selected encoder seals")
        for seal, package in packages.items():
            if (not isinstance(seal, str) or len(seal) != 64
                    or any(c not in '0123456789abcdef' for c in seal)
                    or not isinstance(package, dict) or set(package) != {"path", "sha256"}
                    or package["sha256"] != seal or not Path(package["path"]).is_absolute()):
                raise ValueError("cached unit producer package lacks an exact source binding")
        authority = manifest["reuse_authority"]
        if not isinstance(authority, dict) or set(authority) != {
                "catalog_extension", "candidate_overlay", "encoder_source_proofs",
                "checkpoint_encoder_source_sha256"}:
            raise ValueError("rooted cached units need explicit catalog extension authority")
        for name, schema in (("catalog_extension", "prismaquant.joint_catalog_extension.v1"),
                             ("candidate_overlay", "prismaquant.t4_adopted_catalog.v1")):
            if _bound_document(authority[name]).get("schema") != schema:
                raise ValueError("rooted cached unit authority schema differs")
        proofs = authority["encoder_source_proofs"]
        if not isinstance(proofs, list):
            raise ValueError("rooted cached unit encoder proofs must be explicit bindings")
        proof_documents = {json.dumps(bound, sort_keys=True): _bound_document(bound) for bound in proofs}
        if len(proof_documents) != len(proofs):
            raise ValueError("rooted cached unit encoder proof is duplicated")
        original = authority["checkpoint_encoder_source_sha256"]
        adoptions = manifest["encoder_adoptions"]
        changed = {name for name, record in units.items()
                   if record["identity"]["encoder_source_sha256"] != original}
        if not isinstance(adoptions, dict) or set(adoptions) != changed:
            raise ValueError("rooted cached unit adoption coverage differs")
        used_proofs = set()
        for name, adoption in adoptions.items():
            if (not isinstance(adoption, dict) or adoption.get("schema") !=
                    "prismaquant.joint_catalog_source_adoption.v1"):
                raise ValueError("cached unit source adoption schema differs")
            candidate, reference = adoption["candidate_encoding_identity"], adoption["reference_encoding_identity"]
            if (candidate != units[name]["identity"] or reference.get("unit") != name
                    or adoption.get("reference_pair", [None])[0] != name
                    or reference.get("encoder_source_sha256") != original):
                raise ValueError("cached unit source adoption identities differ")
            for field in ("unit", "source", "calibration", "encoder_fixture_id"):
                if field not in reference or reference[field] != candidate.get(field):
                    raise ValueError("cached unit source adoption changed " + field)
            if reference.get("projection") != candidate.get("projection"):
                raise ValueError("cached unit source adoption changed projection")
            key = json.dumps(adoption["encoder_source_proof"], sort_keys=True)
            proof = proof_documents.get(key)
            if (not proof or proof.get("schema") != "prismaquant.reseal_proof_bundle.v1"
                    or proof.get("ok") is not True or proof.get("encoder_fixture_id_equal") is not True
                    or proof.get("pins", {}).get("old", {}).get("encoder_source_sha256") != original
                    or proof.get("pins", {}).get("new", {}).get("encoder_source_sha256") != candidate["encoder_source_sha256"]
                    or set((proof.get("fixture_id", {}).get("ids") or {}).values()) != {candidate["encoder_fixture_id"]}):
                raise ValueError("cached unit encoder source proof does not authorize this adoption")
            used_proofs.add(key)
        if used_proofs != set(proof_documents):
            raise ValueError("rooted cached unit proof roster contains unused authority")
        policy_bound, served = manifest["served_activation_policy"], manifest["served_activations"]
        if not isinstance(served, dict) or not set(served) <= set(units):
            raise ValueError("rooted cached unit served activation coverage differs")
        if policy_bound is None:
            added_a4 = any(units[name]["identity"].get("recipe", {}).get("grid") == "E2M1x2"
                           and units[name]["identity"]["recipe"].get("q256") == 896 for name in adoptions)
            if served or added_a4:
                raise ValueError("served activation values lack their bound policy")
        else:
            policy = _bound_document(policy_bound)
            if (policy.get("schema") != "prismaquant.joint_served_activation_policy.v1"
                    or policy.get("format") != "TESSERA_E2M1_K2_R896"):
                raise ValueError("rooted cached unit served activation policy schema differs")
            groups = policy["executed_grouping"]["groups"]
            index = {name: (key, group) for key, group in groups.items() for name in group["members"]}
            expected = {}
            for name in adoptions:
                recipe = units[name]["identity"].get("recipe", {})
                if recipe.get("grid") == "E2M1x2" and recipe.get("q256") == 896:
                    if name not in index:
                        raise ValueError("selected A4 unit absent from served activation policy")
                    key, group = index[name]
                    expected[name] = {"group": key, "input_global_scale": group["input_global_scale"]}
            if served != expected:
                raise ValueError("selected served activations differ from bound executed groups")
        self.served_activation_policy, self.served_activations = policy_bound, _json_copy(served)
        self.roots, self.unit_roots = resolved, dict(owners)
        self.producer_packages = _json_copy(packages)
        self.reuse_authority, self.encoder_adoptions = _json_copy(authority), _json_copy(adoptions)

    def require_served_scales(self, scales):
        """Check the actual serialized fp32 inputs against priced group values."""
        import struct
        for name, value in self.served_activations.items():
            key = name + ".input_global_scale"
            expected = struct.unpack("f", struct.pack("f", value["input_global_scale"]))[0]
            if scales.get(key) != expected:
                raise ValueError(f"{name}: exported activation scale differs from the bound served policy")


def _bound_document(bound):
    if not isinstance(bound, dict) or set(bound) != {"path", "sha256"}:
        raise ValueError("cached unit authority needs an exact path/SHA256 binding")
    path = Path(bound["path"])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("cached unit authority must name an absolute regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("cached unit authority SHA256 differs")
    return json.loads(raw, object_pairs_hook=_unique_json_pairs)


class ProducerCachedUnitIdentities:
    """Keep each historical factory and its H commitment witness separate."""

    def __init__(self, bundle, producers, derive, activation, *, mode="committed"):
        self.bundle, self.producers = bundle, producers
        if set(producers) != set(bundle.producer_packages):
            raise ValueError("cached unit identity producers differ from the bound bundle")
        self.identities = {
            seal: CachedUnitIdentity(
                lambda *args, _producer=producer, **kwargs: derive(_producer, *args, **kwargs),
                activation, mode=mode)
            for seal, producer in producers.items()}
        self.established = "per_producer"

    def producer_for(self, key):
        seal = self.bundle.units[key]["identity"]["encoder_source_sha256"]
        return self.producers[seal]

    def __call__(self, weight, unit_name, unit, grid, q256):
        key = ActivationSource.unit_name(unit_name)
        seal = self.bundle.units[key]["identity"]["encoder_source_sha256"]
        return self.identities[seal](weight, unit_name, unit, grid, q256)

    def record(self):
        return {"schema": "tessera.cached_unit_hessian_identity.by_producer.v1",
                "producers": {seal: identity.record()
                              for seal, identity in sorted(self.identities.items())}}


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate cached unit JSON key: {key}")
        result[key] = value
    return result


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_json_pairs)
