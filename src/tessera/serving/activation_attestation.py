"""What the pinned runtime's activation quantizer actually emits (#484).

``runtime_contract.json`` publishes an activation contract as a NAME --
``e2m1_group16_ue4m3_static`` -- and a name says the grid, the group and the
scale dtype.  It does not say how a value becomes a code.  The fp4 route hands
the artifact's static global scale to vLLM's compiled ``scaled_fp4_quant``
(``nvfp4_route.py`` ``apply`` -> ``native_ops.native_fp4_quant``), so the
ROUNDING DECISION belongs to the runtime, and a producer that re-implements it
is asserting a runtime behaviour rather than reading one.  That is what this
module exists to end: the contract carries a table of inputs and the codes the
kernel emitted for them, and a consumer checks its own quantizer against the
table instead of against its own belief.

Two halves, deliberately separated:

* the PROBE SPEC below is the repository's -- the inputs, and only the inputs.
  It is constructed here, in bf16 bit patterns, so a published vector's input
  can be checked against it with no torch and no GPU.  A table whose inputs
  are not these is not this table.
* the OUTPUTS (``stored_scale``, ``codes``) come from running the kernel, in
  ``experiments/attest_activation_quantizer.py``, and from nowhere else.  A
  hand-authored output is another assertion, which is the thing being fixed.

The probes are chosen at the points where two honest implementations of "E2M1,
group 16, UE4M3 block scale, static global" diverge, not at random: the seven
E2M1 midpoints (where a tie-break rule shows), the same midpoints one bf16 ulp
either side (which separates a tie-break difference from a reduced-precision
intermediate, because a value a ulp off a midpoint lands ON it only if the
intermediate is coarse), every code, element saturation, and the block scale's
underflow tie and overflow boundary.  Each probe says which boundary it exists
for, and :func:`probe_coverage` derives the covered set from the spec rather
than from a roster that would go stale.

Stdlib only, on purpose: the ``pure`` CI job has no torch, and the table's
well-formedness -- and its agreement with the E2M1 and E4M3 formats, which is
the half neither side disputes -- must be checkable there.
"""
from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping

from ..alphabet import E2M1_VALUES

__all__ = [
    "ACTIVATION_QUANTIZER_SCHEMA",
    "E2M1_MAX",
    "E2M1_MIDPOINTS",
    "E4M3_MAX_FINITE",
    "E4M3_NAN_BYTES",
    "BOUNDARIES",
    "Probe",
    "PROBES",
    "bf16_bits",
    "bf16_value",
    "bf16_step",
    "e4m3_byte_value",
    "nearest_e4m3_bytes",
    "probe_coverage",
    "nearest_e2m1_codes",
    "validate_activation_quantizers",
]

#: The block's own schema string.  It moves when the GRAMMAR moves, which is
#: the same rule ``lane_eligibility.schema`` follows: a reader that cannot read
#: this grammar must fail closed on the name rather than half-read the table.
ACTIVATION_QUANTIZER_SCHEMA = "tessera.activation-quantizer.v1"

#: The largest E2M1 magnitude, and the divisor NVFP4's block scale is defined
#: by (``block_scale = amax / 6``).  Both are the format's, not a tuning knob.
E2M1_MAX = 6.0

#: The largest finite E4M3FN value, and the two NaN encodings.  ``E4M3_VALUES``
#: in ``tessera.alphabet`` maps NaN to 448 so that a payload grid stays total;
#: here the distinction is the point, because whether the kernel saturates an
#: over-range block scale to 448 or emits NaN is exactly what is unpublished.
E4M3_MAX_FINITE = 448.0
E4M3_NAN_BYTES = frozenset({0x7F, 0xFF})

#: The seven midpoints of the positive E2M1 lattice, derived from the lattice.
E2M1_MIDPOINTS: tuple[float, ...] = tuple(
    (E2M1_VALUES[i] + E2M1_VALUES[i + 1]) / 2.0 for i in range(7)
)

#: The boundary kinds a probe may cover.  A published table's coverage is
#: derived from the probes it carries, so this is a vocabulary, not a claim.
BOUNDARIES = (
    "e2m1_midpoint_dyadic",
    "e2m1_midpoint_reciprocal",
    "e2m1_midpoint_global_scale",
    "e2m1_midpoint_ulp_below",
    "e2m1_midpoint_ulp_above",
    "e2m1_code_identity",
    "e2m1_saturation",
    "block_scale_underflow_tie",
    "block_scale_underflow_above",
    "block_scale_overflow",
)

_UNITS = {"group"}
_GRIDS = {"E2M1"}
_BLOCK_SCALES = {"UE4M3"}
_GLOBAL_SCALES = {"static_per_module"}

_GENERATED_KEYS = frozenset({
    "image", "vllm", "torch", "device", "compute_capability", "driver",
    "generator_sha256",
})
_CONTRACT_KEYS = frozenset({
    "op", "unit", "unit_length", "grid", "block_scale", "global_scale",
    "vectors",
})
_VECTOR_KEYS = frozenset({"id", "boundary", "global_scale", "input",
                          "stored_scale", "codes"})


# -- number formats ---------------------------------------------------------
def bf16_bits(value: float) -> int:
    """A Python float to its BF16 encoding, round-to-nearest-even.

    BF16 is F32 truncated to its top 16 bits, so the rounding is on the 16 low
    bits of the F32 encoding and nothing else needs a float library.
    """
    word = struct.unpack(">I", struct.pack(">f", value))[0]
    low = word & 0xFFFF
    rounded = word >> 16
    if low > 0x8000 or (low == 0x8000 and (rounded & 1)):
        rounded += 1
    return rounded & 0xFFFF


def bf16_value(bits: int) -> float:
    """A BF16 encoding to its exact value."""
    if not 0 <= int(bits) <= 0xFFFF:
        raise ValueError(f"bf16 encoding out of range: {bits!r}")
    return struct.unpack(">f", struct.pack(">I", int(bits) << 16))[0]


def bf16_step(bits: int, direction: int) -> int:
    """One BF16 ulp away from ``bits``, toward larger (+1) or smaller (-1) magnitude.

    Encoding space, not value space: consecutive magnitudes are consecutive
    encodings within one sign, which is what makes "one ulp off a midpoint" a
    fact about the format rather than about a float library.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    magnitude = bits & 0x7FFF
    if magnitude == 0 and direction == -1:
        raise ValueError("no BF16 magnitude below zero")
    return (bits & 0x8000) | (magnitude + direction)


def _exact_bf16(value: float) -> int:
    """The BF16 encoding of a value that must be exactly representable."""
    bits = bf16_bits(value)
    if bf16_value(bits) != value:
        raise ValueError(f"{value!r} is not exactly representable in BF16")
    return bits


def e4m3_byte_value(byte: int) -> "float | None":
    """One E4M3FN byte to its value; ``None`` for the two NaN encodings."""
    if not 0 <= int(byte) <= 0xFF:
        raise ValueError(f"e4m3 byte out of range: {byte!r}")
    byte = int(byte)
    if byte in E4M3_NAN_BYTES:
        return None
    sign = -1.0 if byte >> 7 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0 ** -6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


#: Every finite non-negative E4M3FN encoding, ascending by value.  Consecutive
#: values are consecutive encodings across the subnormal/normal boundary too,
#: so "nearest, ties to even ENCODING" is round-to-nearest-even for this format.
_E4M3_FINITE = tuple(
    (byte, e4m3_byte_value(byte)) for byte in range(0x7F) if byte not in E4M3_NAN_BYTES
)


def nearest_e4m3_bytes(value: float) -> tuple[int, ...]:
    """Every finite E4M3FN encoding at minimum distance from ``value``.

    The SET, for the same reason :func:`nearest_e2m1_codes` returns one: the
    block scale has its own tie, at ``amax/6 = 2**-10`` where zero and the
    smallest subnormal are equidistant, and which side the kernel takes is a
    fact to READ from this table rather than to assume in the reader that
    checks it.  Refuses out of range rather than saturating, because whether
    the runtime saturates is another such fact.
    """
    if not value >= 0.0:
        raise ValueError(f"nearest_e4m3_bytes takes a non-negative value, got {value!r}")
    if value > E4M3_MAX_FINITE:
        raise ValueError(
            f"{value!r} is above the largest finite E4M3FN value; whether the "
            "runtime saturates or emits NaN is published, not computed here"
        )
    best = min(abs(value - candidate) for _byte, candidate in _E4M3_FINITE)
    return tuple(byte for byte, candidate in _E4M3_FINITE
                 if abs(value - candidate) == best)


def nearest_e2m1_codes(normalized: float) -> tuple[int, ...]:
    """Every E2M1 code at minimum distance from ``normalized`` -- one, or two on a tie.

    Returning the SET is the point.  A tie has two nearest codes and the
    format does not say which one a kernel picks; asserting one here would
    re-assert the thing the contract is being asked to publish.
    """
    best = min(abs(normalized - value) for value in E2M1_VALUES)
    codes = [code for code, value in enumerate(E2M1_VALUES)
             if abs(normalized - value) == best]
    # +0 and -0 are distinct encodings of one value; a signed zero's sign is
    # the kernel's to publish, so both codes stay in the set.
    return tuple(codes)


# -- the probe spec ---------------------------------------------------------
class Probe:
    """One group of 16 BF16 inputs at one global scale, and why it exists."""

    __slots__ = ("identifier", "boundary", "global_scale", "input_bits")

    def __init__(self, identifier: str, boundary: str, global_scale: float,
                 input_bits: Iterable[int]) -> None:
        if boundary not in BOUNDARIES:
            raise ValueError(f"{identifier}: unknown boundary {boundary!r}")
        bits = tuple(int(b) for b in input_bits)
        if len(bits) != 16:
            raise ValueError(f"{identifier}: an NVFP4 group is 16 values, got {len(bits)}")
        self.identifier = identifier
        self.boundary = boundary
        self.global_scale = float(global_scale)
        self.input_bits = bits

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(bf16_value(b) for b in self.input_bits)

    @property
    def amax(self) -> float:
        return max(abs(v) for v in self.values)

    def implied_block_scale(self) -> float:
        """``amax / 6 * G`` -- the value the format says the block scale encodes."""
        return self.amax / E2M1_MAX * self.global_scale

    def as_json(self) -> dict:
        return {
            "id": self.identifier,
            "boundary": self.boundary,
            "global_scale": "0x%08x" % struct.unpack(
                ">I", struct.pack(">f", self.global_scale))[0],
            "input": ["0x%04x" % b for b in self.input_bits],
        }


def _pad(values: list, filler: float = 0.0) -> list:
    return values + [filler] * (16 - len(values))


def _signed(values: list) -> list:
    """A magnitude list and its negatives, which pins the sign convention."""
    return values + [-v for v in values]


def _scaled_midpoints(used_scale: float) -> list:
    return [m * used_scale for m in E2M1_MIDPOINTS]


def _build_probes() -> tuple[Probe, ...]:
    probes: list[Probe] = []

    # G = 1, amax = 6 -> block scale 6/6 = 1, used scale 1.  The seven E2M1
    # midpoints are then literally the input values, and every one of them is
    # exactly representable in BF16, so the tie-break is probed with no
    # rounding of the input anywhere in the construction.
    probes.append(Probe(
        "midpoint_dyadic", "e2m1_midpoint_dyadic", 1.0,
        [_exact_bf16(v) for v in
         [E2M1_MAX] + _signed(list(E2M1_MIDPOINTS)) + [0.0]],
    ))
    # Every code, both signs, and zero, at the same used scale.
    probes.append(Probe(
        "code_identity", "e2m1_code_identity", 1.0,
        [_exact_bf16(v) for v in
         [0.0, -0.0] + [E2M1_VALUES[c] for c in range(1, 8)]
         + [-E2M1_VALUES[c] for c in range(1, 8)]],
    ))

    # G = 1, amax = 9 -> block scale 9/6 = 1.5, used scale 1.5.  1.5 is exact
    # in E4M3 and its reciprocal is NOT exact in F32, so this separates a
    # tie-break difference from the reciprocal the installed kernels use.
    # Every midpoint x 1.5 is still exact in BF16 (at most five significand
    # bits), so the input is exact and only the arithmetic is under test.
    reciprocal = _scaled_midpoints(1.5)
    probes.append(Probe(
        "midpoint_reciprocal", "e2m1_midpoint_reciprocal", 1.0,
        [_exact_bf16(v) for v in [9.0] + _signed(reciprocal) + [0.0]],
    ))
    # The same group at G = 1.5: block scale 9/6*1.5 = 2.25 (exact in E4M3),
    # used scale 2.25/1.5 = 1.5 again.  Same lattice, so a difference between
    # this row and the one above is the global scale's own arithmetic.
    probes.append(Probe(
        "midpoint_global_scale", "e2m1_midpoint_global_scale", 1.5,
        [_exact_bf16(v) for v in [9.0] + _signed(reciprocal) + [0.0]],
    ))

    # A second inexact reciprocal, so "the kernel resolves an exact tie by the
    # format's own rule" is not a claim about one scale.  amax 10.5 -> block
    # scale 1.75, exact in E4M3, and 1/1.75 is not exact in F32 either; every
    # midpoint times 1.75 is still exact in BF16 (at most seven significand
    # bits), so the inputs stay exact while the arithmetic does not.
    probes.append(Probe(
        "midpoint_seven_fourths", "e2m1_midpoint_reciprocal", 1.0,
        [_exact_bf16(v) for v in
         [10.5] + _signed(_scaled_midpoints(1.75)) + [0.0]],
    ))

    # The same midpoints one BF16 ulp either side.  A value one ulp off a
    # midpoint is NOT a tie in F32 and rounds unambiguously; it becomes a tie
    # only if the intermediate is coarser than F32.  The amax element stays
    # 9.0 so the block scale does not move between these and the row above.
    for direction, boundary in ((-1, "e2m1_midpoint_ulp_below"),
                                (1, "e2m1_midpoint_ulp_above")):
        nudged = [bf16_step(_exact_bf16(v), direction) for v in reciprocal]
        probes.append(Probe(
            "midpoint_reciprocal_ulp_%s" % ("below" if direction < 0 else "above"),
            boundary, 1.0,
            [_exact_bf16(9.0)] + nudged + [b | 0x8000 for b in nudged]
            + [_exact_bf16(0.0)],
        ))

    # amax 6.25 -> 6.25/6 = 1.0416..., which rounds DOWN to E4M3 1.0, so the
    # group's own maximum normalises to 6.25 and must saturate at the top code.
    probes.append(Probe(
        "element_saturation", "e2m1_saturation", 1.0,
        [_exact_bf16(v) for v in
         _pad([6.25, -6.25, 6.125, -6.125, 6.0, -6.0, 5.9375, -5.9375])],
    ))

    # The block scale's underflow tie: amax = 6 * 2**-10 puts amax/6 exactly
    # half way between E4M3 zero and its smallest subnormal, so the tie-break
    # decides whether the whole group is zero.  And the next BF16 amax above
    # it, which is unambiguously the subnormal.
    tie_amax = E2M1_MAX * 2.0 ** -10
    for identifier, boundary, amax_bits in (
        ("block_scale_underflow_tie", "block_scale_underflow_tie",
         _exact_bf16(tie_amax)),
        ("block_scale_underflow_above", "block_scale_underflow_above",
         bf16_step(_exact_bf16(tie_amax), 1)),
    ):
        amax = bf16_value(amax_bits)
        probes.append(Probe(identifier, boundary, 1.0, [amax_bits] + [
            _exact_bf16(v) for v in _pad(
                [-amax, tie_amax / 2.0, -tie_amax / 2.0, tie_amax / 4.0], 0.0)[:15]
        ]))

    # The block scale's overflow: amax/6 = 512, above the largest finite E4M3
    # value.  PrismaQuant's oracle clamps to 448; whether the kernel clamps or
    # emits an E4M3 NaN is unpublished, and the difference is a whole group.
    probes.append(Probe(
        "block_scale_overflow", "block_scale_overflow", 1.0,
        [_exact_bf16(v) for v in
         _pad([3072.0, -3072.0, 1536.0, -1536.0, 768.0, 384.0, 0.5])],
    ))
    return tuple(probes)


PROBES: tuple[Probe, ...] = _build_probes()


def probe_coverage(identifiers: Iterable[str]) -> frozenset[str]:
    """The boundaries a set of published vector ids reaches, per the spec."""
    wanted = set(identifiers)
    return frozenset(p.boundary for p in PROBES if p.identifier in wanted)


# -- validation -------------------------------------------------------------
def _keys(payload: Any, where: str, required: frozenset, optional=frozenset()) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{where} is missing {missing}")
    unknown = sorted(set(payload) - required - set(optional))
    if unknown:
        raise ValueError(f"{where} carries unknown field(s) {unknown}")


def _hex_word(value: Any, digits: int, where: str) -> int:
    if (not isinstance(value, str) or not value.startswith("0x")
            or len(value) != digits + 2
            or any(c not in "0123456789abcdef" for c in value[2:])):
        raise ValueError(
            f"{where} must be a lowercase 0x-prefixed {digits}-digit hex "
            f"encoding, got {value!r}"
        )
    return int(value, 16)


def validate_activation_quantizers(block: Any, *, platforms: Iterable[str],
                                   cell_contracts: Mapping[str, Iterable[str]],
                                   require_image,
                                   where: str = "runtime_contract.activation_quantizers",
                                   ) -> None:
    """Refuse a table this package would not itself stand behind.

    ``platforms`` is the lane table's platform set and ``cell_contracts`` maps
    each platform to the activation contracts its own cells name: an
    attestation for a platform with no cells, or for a contract nothing on
    that platform executes, is a claim about a runtime that does not serve
    here.  ``require_image`` is the contract's own digest-reference rule,
    passed in rather than imported, so this module stays importable with no
    torch and with no cycle back into the validator.

    ABSENCE IS NOT A DEFAULT.  A platform or a contract with no entry
    publishes no attestation at all, and a consumer gate must refuse the
    route rather than fall back to its own model of the arithmetic -- which
    is the failure this table exists to end.
    """
    _keys(block, where, frozenset({"schema", "generator", "platforms"}))
    if block["schema"] != ACTIVATION_QUANTIZER_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {ACTIVATION_QUANTIZER_SCHEMA!r}, got "
            f"{block['schema']!r}"
        )
    generator = block["generator"]
    if not isinstance(generator, str) or not generator.startswith("experiments/"):
        raise ValueError(
            f"{where}.generator must be the repository path of the script that "
            f"produced this table, under experiments/; got {generator!r}"
        )
    entries = block["platforms"]
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError(f"{where}.platforms must be a non-empty object")
    known = set(platforms)
    for platform, entry in entries.items():
        at = f"{where}.platforms[{platform!r}]"
        if platform not in known:
            raise ValueError(
                f"{at} attests a platform lane_eligibility does not publish")
        _keys(entry, at, frozenset({"generated", "contracts"}))
        _keys(entry["generated"], at + ".generated", _GENERATED_KEYS)
        require_image(entry["generated"]["image"], at + ".generated.image")
        for field in sorted(_GENERATED_KEYS - {"image"}):
            value = entry["generated"][field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"{at}.generated.{field} must be a non-empty string")
        served = set(cell_contracts.get(platform, ()))
        contracts = entry["contracts"]
        if not isinstance(contracts, Mapping) or not contracts:
            raise ValueError(f"{at}.contracts must be a non-empty object")
        for name, contract in contracts.items():
            _validate_contract(contract, name, served, f"{at}.contracts[{name!r}]")


def _validate_contract(contract: Any, name: str, served: set, at: str) -> None:
    if name not in served:
        raise ValueError(
            f"{at} attests an activation contract no cell on this platform "
            f"executes; the platform's cells name {sorted(served)}"
        )
    _keys(contract, at, _CONTRACT_KEYS)
    for field, allowed in (("unit", _UNITS), ("grid", _GRIDS),
                           ("block_scale", _BLOCK_SCALES),
                           ("global_scale", _GLOBAL_SCALES)):
        if contract[field] not in allowed:
            raise ValueError(
                f"{at}.{field} is {contract[field]!r}; this reader knows only "
                f"{sorted(allowed)} and will not guess at another one's meaning"
            )
    if contract["unit_length"] != 16:
        raise ValueError(f"{at}.unit_length must be 16, the NVFP4 group")
    if not isinstance(contract["op"], str) or not contract["op"]:
        raise ValueError(f"{at}.op must name the operator that emitted the table")
    vectors = contract["vectors"]
    if not isinstance(vectors, list) or not vectors:
        raise ValueError(f"{at}.vectors must be a non-empty array")
    by_id = {p.identifier: p for p in PROBES}
    seen: set = set()
    for index, vector in enumerate(vectors):
        _validate_vector(vector, by_id, seen, f"{at}.vectors[{index}]")
    missing = sorted(set(BOUNDARIES) - probe_coverage(seen))
    if missing:
        raise ValueError(
            f"{at}.vectors reaches no probe for {missing}; a table that skips a "
            "boundary two implementations diverge at is not an attestation of "
            "this contract"
        )


def _validate_vector(vector: Any, by_id: Mapping[str, Probe], seen: set, at: str) -> None:
    _keys(vector, at, _VECTOR_KEYS)
    identifier = vector["id"]
    probe = by_id.get(identifier)
    if probe is None:
        raise ValueError(
            f"{at}.id is {identifier!r}, which is not a probe this package "
            f"defines; the INPUTS are the repository's and only the outputs "
            "are the runtime's answer"
        )
    if identifier in seen:
        raise ValueError(f"{at}.id {identifier!r} is published twice")
    seen.add(identifier)
    if vector["boundary"] != probe.boundary:
        raise ValueError(
            f"{at}.boundary is {vector['boundary']!r}; the spec says "
            f"{probe.boundary!r}")
    expected = probe.as_json()
    if vector["global_scale"] != expected["global_scale"]:
        raise ValueError(
            f"{at}.global_scale is {vector['global_scale']!r}; the spec's probe "
            f"uses {expected['global_scale']!r}")
    if vector["input"] != expected["input"]:
        raise ValueError(
            f"{at}.input is not the input this package's probe {identifier!r} "
            "constructs; an edited input makes the row an assertion again")
    stored = vector["stored_scale"]
    if type(stored) is not int or not 0 <= stored <= 0xFF:
        raise ValueError(f"{at}.stored_scale must be one E4M3FN byte, 0..255")
    codes = vector["codes"]
    if (not isinstance(codes, list) or len(codes) != 16
            or any(type(c) is not int or not 0 <= c <= 15 for c in codes)):
        raise ValueError(f"{at}.codes must be 16 E2M1 nibbles, 0..15")
    _check_arithmetic(probe, stored, codes, at)


def _check_arithmetic(probe: Probe, stored: int, codes: list, at: str) -> None:
    """The half of the table the format itself decides, and no more.

    On both levels the check is "a nearest encoding, or one of two tied
    nearest encodings", never "this encoding".  Which side of a tie the kernel
    takes is precisely what the table publishes, so testing it against a rule
    chosen here would delete the reason the table exists; a published value
    that is not even nearest is a corrupt table rather than a runtime fact.
    Outside E4M3's finite range -- the overflow probe -- the block scale's
    encoding IS the runtime fact and nothing about it is asserted.
    """
    implied = probe.implied_block_scale()
    if implied <= E4M3_MAX_FINITE:
        candidates = nearest_e4m3_bytes(implied)
        if stored not in candidates:
            raise ValueError(
                f"{at}.stored_scale is 0x{stored:02x}; the group's amax "
                f"{probe.amax!r} over 6 times G {probe.global_scale!r} is "
                f"{implied!r}, whose nearest E4M3FN encoding(s) are "
                + ", ".join("0x%02x" % c for c in candidates)
            )
    scale = e4m3_byte_value(stored)
    if scale is None or scale == 0.0:
        # The kernel published a NaN or zero block scale.  That is the fact
        # this probe exists for; the group's codes carry no separate claim.
        return
    used = scale / probe.global_scale
    for index, (value, code) in enumerate(zip(probe.values, codes)):
        candidates = nearest_e2m1_codes(value / used)
        if code not in candidates:
            raise ValueError(
                f"{at}.codes[{index}] is {code} ({E2M1_VALUES[code]!r}); "
                f"{value!r} over the used scale {used!r} is "
                f"{value / used!r}, whose nearest E2M1 code(s) are "
                f"{[E2M1_VALUES[c] for c in candidates]!r}"
            )
