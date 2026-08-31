import sys, hashlib
sys.path.insert(0,'src')
from fractions import Fraction
from tessera.scale_codec import *
from tessera.manifest import *
from tessera.planes import *
from tessera.layout import build_planes
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.bitio import BitWriter

print("P7 digest:", legal_set_digest(0))
print("P7 census:", classification_census(0))
print("P7 doc digest matches:", legal_set_digest(0) == "da39862453b9670fbe71e1e71880a0e995b960f383248bf4dc4acf9aa880a1b3")
# clip sensitivity
for clip in (1,2,3,-1):
    c = classification_census(clip)
    print("   clip", clip, c, legal_set_digest(clip)[:16])

# P10 D2 vs D4: bitio MSB-first packing of (half0,half1) vs pack_refinement_byte
h0 = HalfWord(delta=1, mantissa=5)   # nibble 0xD
h1 = HalfWord(delta=0, mantissa=2)   # nibble 0x2
w = BitWriter(); w.write(h0.nibble,4); w.write(h1.nibble,4)
print("P10 D4 MSB-first plane byte: 0x%02X | D2 pack_refinement_byte: 0x%02X" % (w.bytes[0], pack_refinement_byte(h0,h1)))
print("    -> decoding 0x%02X under D2 gives half0=%s half1=%s (swapped)" % (w.bytes[0],)+ tuple() if False else "")
a,b = unpack_refinement_byte(w.bytes[0])
print("    D2 read of the D4-packed byte: half0 d=%d m=%d, half1 d=%d m=%d (truth was d=%d m=%d / d=%d m=%d)" % (a.delta,a.mantissa,b.delta,b.mantissa,h0.delta,h0.mantissa,h1.delta,h1.mantissa))

# P8 REFERENCE-storage planes are byte-invisible
geo = Geometry(rows=8, columns=8, superblock_columns=4, group_weights=32, half_weights=16, quantizable_params=64)
rates = bresenham_rate_schedule(root_from_q256(512), 8)
planes = list(build_planes(geo, rates, b"AB", b"CD"))
import dataclasses
planes[0] = dataclasses.replace(planes[0], storage=Storage.REFERENCE)
planes = tuple(planes)
print("P8 ALPHABET REFERENCE byte_length:", planes[0].byte_length(), "element_count:", planes[0].element_count)
pay=bytes(32)
def mk(alpha_count):
    els = [p.element_count for p in planes]; els[0]=alpha_count
    tot = sum(p.byte_length(e) for p,e in zip(planes,els))
    return TerminalRecord("a%d"%alpha_count, 0, tuple(els), tot, Fraction(8*tot,64), pay), tot
t0,b0 = mk(0); t2,b2 = mk(2)
print("P8 two logically different terminals (alphabet 0 vs 2 elements) -> bytes", b0, b2)
br = BranchIdentity("u",512,RotationState.NONE,ContainerClass.GRIDBOOK)
try:
    Manifest(bytes(32), br, geo, ArrangementMode.BRESENHAM, rates, planes, (t0,t2), pay)
    print("P8 both accepted in one manifest")
except Exception as e:
    print("P8 manifest refuses:", type(e).__name__, e)

# P12 geometry not a multiple of group_weights
g = Geometry(rows=10, columns=10, superblock_columns=5, group_weights=32, half_weights=16, quantizable_params=100)
print("P12 positions=100, group_weights=32 accepted; SCALE_BASE count = %d groups covering %d of 100 weights" % (100//32, (100//32)*32))
