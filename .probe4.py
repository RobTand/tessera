import sys, hashlib, dataclasses
sys.path.insert(0,'src')
from fractions import Fraction
from tessera.manifest import *
from tessera.planes import *
from tessera.layout import build_planes
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera import container

# P13: BODY element_count unbound from rates x rows
geo = Geometry(rows=8, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=64)
rates = bresenham_rate_schedule(root_from_q256(512), 8)   # sum=16, rows=8 -> 128 body bits
planes = list(build_planes(geo, rates, b"", b""))
i = [p.kind for p in planes].index(PlaneKind.BODY)
planes[i] = dataclasses.replace(planes[i], counts=(3,3), restart_offsets=(0,3))
planes = tuple(planes)
els=[p.element_count for p in planes]; tot=sum(p.byte_length() for p in planes)
tr = TerminalRecord("t",0,tuple(els),tot,Fraction(8*tot,64),bytes(32))
try:
    m = Manifest(bytes(32), BranchIdentity("u",512,RotationState.NONE,ContainerClass.GRIDBOOK), geo,
                 ArrangementMode.BRESENHAM, rates, planes, (tr,), bytes(32))
    print("P13 BODY declared 6 bits where rates*rows require 128: ACCEPTED")
except Exception as e:
    print("P13 rejected:", type(e).__name__, e)

# P14: superblock_columns wildly larger than columns
try:
    g = Geometry(rows=1, columns=8, superblock_columns=100000, group_weights=32, half_weights=16, quantizable_params=8)
    from tessera.grammar import superblock_quota_ok
    print("P14 superblock_columns=100000 > columns=8 accepted; quota check vacuous:",
          superblock_quota_ok((2,)*8, 100000, Fraction(2)))
except Exception as e:
    print("P14 rejected:", e)

# P15: truncation at a non-byte-aligned quota boundary is not a byte prefix
geo2 = Geometry(rows=3, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=24)
r2 = (2,)*8
pl2 = build_planes(geo2, r2, b"", b"")
b = [p for p in pl2 if p.kind is PlaneKind.BODY][0]
print("P15 body counts", b.counts, "full bytes", b.byte_length(), "| first-superblock-only bytes", b.byte_length(b.counts[0]),
      "-> cut at bit", b.counts[0], "which is byte-aligned:", b.counts[0] % 8 == 0)
# force a non-aligned cut
print("     cut at 20 bits -> declared bytes", b.byte_length(20), "; full plane's byte 2 holds real body bits 16..23,")
print("     but a re-emitted truncation must zero bits 20..23 (D4). Two byte strings for one terminal.")
