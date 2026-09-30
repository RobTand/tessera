"""The fused frame's size is public API: consumers price it without private names."""
from tessera import fused
from tessera.fused import HEADER_BYTES, MEMBER_ROW_BYTES, frame_bytes, pack_fused
from tessera.unit_artifact import TERMINAL_SLOT_ID


def test_frame_bytes_matches_packed_length():
    members = [("q", 4, b"\x01" * 7), ("k", 2, b"\x03"), ("v_é", 2, b"\x02" * 33)]
    names = [m[0] for m in members]
    blobs = sum(len(m[2]) for m in members)
    assert len(pack_fused(members)) == frame_bytes(names) + blobs


def test_frame_constants_are_the_struct_sizes():
    assert HEADER_BYTES == fused._HEADER.size == 10
    assert MEMBER_ROW_BYTES == fused._MEMBER.size == 14
    assert frame_bytes(["ab"]) == HEADER_BYTES + MEMBER_ROW_BYTES + 2


def test_terminal_slot_id_is_public():
    assert TERMINAL_SLOT_ID == "t-nvfp4"
