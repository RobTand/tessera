"""The packet must cover the complete authorized wire geometry."""
from itertools import product
import json
from pathlib import Path


PACKET = Path(__file__).resolve().parents[1] / "docs/measurements/pq2459-fixture-inputs.json"
SUFFIX = "_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e"

EXPECTED = {
    ("TESSERA_BF16_K1", "dense", "batch", (832, 880, 960, 1024, 1088)),
    ("TESSERA_BF16_K1", "dense", "decode", (832, 880, 960, 1024, 1088)),
    ("TESSERA_BF16_K1", "routed_moe", "batch", (1024,)),
    ("TESSERA_BF16_K1", "routed_moe", "decode", (1024,)),
    ("TESSERA_E4M3_K1", "dense", "batch", (832, 960, 1024, 1088)),
    ("TESSERA_E4M3_K1", "dense", "decode", (832, 960, 1024, 1088)),
    ("TESSERA_E4M3_K1", "routed_moe", "batch", (896, 928, 1024, 1088)),
    ("TESSERA_E4M3_K1", "routed_moe", "decode", (896, 928, 1024, 1088)),
}


def test_authorized_cells_have_actual_wire_geometry():
    packet = json.loads(PACKET.read_text())
    assert {(cell["family"], cell["structure"], cell["regime"], tuple(cell["q256"]))
            for cell in packet["cells"]} == EXPECTED
    for cell in packet["cells"]:
        name = cell["family"].lower()
        assert cell["id"] == f"{name}_{cell['structure']}_sm121_{cell['regime']}_resident" + SUFFIX
        assert cell["runtime"]["tessera_commit"] == packet["serving"]["commit"]
        assert cell["runtime"]["serving_source_sha256"] == packet["serving"]["source_sha256"]
        shapes = packet["profiles"][cell["profiles"][0]][cell["structure"] + "_nk"]
        required = set(product(cell["q256"], map(tuple, shapes)))
        observed = {(wire["q256"], tuple(wire["rank_local_nk"]))
                    for wire in cell["wires"]}
        assert required <= observed, (cell["id"], required - observed)
        for wire in cell["wires"]:
            assert wire["bytes"] > 0
            assert len(bytes.fromhex(wire["sha256"])) == 32
            assert wire["artifact"] in packet["artifacts"]
            assert ".45." not in wire["target"]
    assert set(packet["entrypoints"]) == {"tr3_batch", "speed_batch", "speed_decode"}
    assert len(packet["token_inputs"]["receipt"]["arrays"]) == 25
    assert packet["qualification"]["qualified_cells"] == 0
    assert packet["authorization"]["decision"] == "dec-1009-160015-14aa"
