import sys, hashlib, struct
sys.path.insert(0,'src')
from fractions import Fraction
from tessera.manifest import *
from tessera.planes import *
from tessera.layout import build_planes
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera import container
from tessera.canonical import decode_uint, encode_uint

geo = Geometry(rows=8, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=64)
rates = bresenham_rate_schedule(root_from_q256(512), 8)
planes = build_planes(geo, rates, b"AB", b"CD")
tot = sum(p.byte_length() for p in planes)
region = bytes(range(tot%256))*0 + bytes([i%251 for i in range(tot)])
pay = hashlib.sha256(region).digest()
br = BranchIdentity("u", 512, RotationState.NONE, ContainerClass.GRIDBOOK)
tr = TerminalRecord("full", 0, tuple(p.element_count for p in planes), tot, Fraction(8*tot,64), pay)
m = Manifest(bytes(32), br, geo, ArrangementMode.BRESENHAM, rates, planes, (tr,), pay)
art = container.serialize(m, region)
print("P2 baseline parse ok:", container.parse(art).terminal.slot_id)
# corrupt the payload
bad = bytearray(art); bad[-1] ^= 0xFF
try:
    container.parse(bytes(bad)); print("P2 corrupt ACCEPTED (bad)")
except Exception as e: print("P2 corrupt rejected:", type(e).__name__)
# now inflate the header's region_bytes by 1 and corrupt
bad2 = bytearray(art)
hdr = list(struct.unpack("<8sHHIII", bytes(bad2[:24])))
hdr[5] += 1
bad2[:24] = struct.pack("<8sHHIII", *hdr)
bad2[-1] ^= 0xFF
try:
    p = container.parse(bytes(bad2)); print("P2 INFLATED+corrupt ACCEPTED as", p.terminal.slot_id, "-- digest check skipped")
except Exception as e: print("P2 inflated rejected:", type(e).__name__, e)

# P3 superblock granule split
geo2 = Geometry(rows=1, columns=10, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=10)
r2 = (2,)*10
pl2 = build_planes(geo2, r2, b"", b"")
body = [p for p in pl2 if p.kind is PlaneKind.BODY][0]
print("P3 columns=10 sb=4 body counts:", body.counts, "offsets:", body.restart_offsets,
      "| true per-superblock:", (8,8,4))

# P4 quantizable_params free
geo3 = Geometry(rows=8, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=10**9)
print("P4 geometry with quantizable_params=1e9 for a 64-param unit: ACCEPTED ->", geo3.quantizable_params, "positions", geo3.positions)

# P5 LSB_FIRST descriptor accepted
d = PlaneDescriptor(PlaneKind.BODY, IndexDomain.POSITION, Storage.INLINE, 1, BitOrder.LSB_FIRST, 1,
                    CountGranularity.WHOLE_PLANE, (8,), (0,), PayloadDtype.RAW_BITS, bytes(32))
print("P5 LSB_FIRST BODY descriptor accepted:", d.bit_order.name)
# P5b wrong index domain / dtype
d2 = PlaneDescriptor(PlaneKind.DIAG_SU, IndexDomain.AXIS_OUT, Storage.INLINE, 16, BitOrder.MSB_FIRST, 1,
                    CountGranularity.WHOLE_PLANE, (8,), (0,), PayloadDtype.E8M0, bytes(32))
print("P5b DIAG_SU with AXIS_OUT/E8M0 accepted:", d2.index_domain.name, d2.payload_dtype.name)

# P6 decode_uint beyond 64 bits
blob = bytes([0xFF]*9 + [0x7F])
v,_ = decode_uint(blob)
print("P6 decode_uint accepted value:", v, "> 2^64-1:", v > (1<<64)-1, "| encode_uint would reject")
