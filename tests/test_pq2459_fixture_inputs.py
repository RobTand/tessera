"""The packet must cover the complete authorized wire geometry."""
from itertools import product
import json
from pathlib import Path


PACKET = Path(__file__).resolve().parents[1] / "docs/measurements/pq2459-fixture-inputs.json"


def test_authorized_cells_have_actual_wire_geometry():
    packet = json.loads(PACKET.read_text())
    expected = {
        (family, structure, regime)
        for family, structure, regime in product(
            ("TESSERA_BF16_K1", "TESSERA_E4M3_K1"),
            ("dense", "routed_moe"), ("batch", "decode"))
    }
    assert {(cell["family"], cell["structure"], cell["regime"])
            for cell in packet["cells"]} == expected
    for cell in packet["cells"]:
        shapes = packet["profiles"][cell["profiles"][0]][cell["structure"] + "_nk"]
        required = set(product(cell["q256"], map(tuple, shapes)))
        observed = {(wire["q256"], tuple(wire["rank_local_nk"]))
                    for wire in cell["wires"]}
        assert required <= observed, (cell["id"], required - observed)
        for wire in cell["wires"]:
            assert wire["artifact"] in packet["artifacts"]
            assert ".45." not in wire["target"]
    assert packet["qualification"]["qualified_cells"] == 0
