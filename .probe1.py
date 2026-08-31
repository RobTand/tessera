import sys, hashlib
sys.path.insert(0,'src')
from fractions import Fraction
from tessera.manifest import *
from tessera.planes import *
from tessera.layout import build_planes
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera import container

geo = Geometry(rows=8, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=64)
rates = bresenham_rate_schedule(root_from_q256(512), 8)
print("rates", rates)
planes = build_planes(geo, rates, b"AB", b"CD")
for p in planes:
    print(p.kind.name, p.count_granularity.name, p.counts, p.restart_offsets, p.byte_length())

# --- P1: STORED vs BRESENHAM non-canonicality ---
prof = bytes(32); pay = bytes(32)
br = BranchIdentity("u", 512, RotationState.NONE, ContainerClass.GRIDBOOK)
tr = TerminalRecord("t", 0, tuple(p.element_count for p in planes), sum(p.byte_length() for p in planes), Fraction(8*sum(p.byte_length() for p in planes),64), pay)
m1 = Manifest(prof, br, geo, ArrangementMode.BRESENHAM, rates, planes, (tr,), pay)
m2 = Manifest(prof, br, geo, ArrangementMode.STORED,    rates, planes, (tr,), pay)
print("P1 same rates:", m1.rates == m2.rates)
print("P1 encoded equal:", m1.encode() == m2.encode(), len(m1.encode()), len(m2.encode()))
print("P1 digests:", m1.manifest_digest().hex()[:16], m2.manifest_digest().hex()[:16])
