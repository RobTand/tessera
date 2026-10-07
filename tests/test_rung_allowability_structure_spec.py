"""Unit tests for --structure-spec in rung_allowability_table.

CPU only. This module imports with stdlib alone and never touches torch.
"""
import copy
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEED = os.path.join(HERE, "..", "experiments", "t8r_speed")
FIXTURE = os.path.join(SPEED, "structure_spec_glm53_tp2.json")
sys.path.insert(0, os.path.normpath(SPEED))

import rung_allowability_table as tab


def _write_spec(payload):
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(payload, handle)
    handle.close()
    return handle.name


def _glm_payload():
    with open(FIXTURE) as handle:
        return json.load(handle)


class ParseSpec(unittest.TestCase):
    def test_glm_fixture_matches_compiled_defaults(self):
        parsed = tab.parse_structure_spec(FIXTURE)
        self.assertEqual(parsed["shapes"], tab.SHAPES)
        self.assertEqual(parsed["ms"], tab.MS)
        self.assertEqual(
            parsed["meta"],
            {"experts": 288, "top_k": 8, "hidden": 4096, "inter": 1024},
        )
        self.assertEqual(parsed["spec_id"], "glm53-tp2")

    def test_missing_path_raises(self):
        with self.assertRaises((OSError, ValueError)):
            tab.parse_structure_spec(
                os.path.join(SPEED, "does-not-exist.json")
            )

    def test_bad_schema_raises(self):
        payload = _glm_payload()
        payload["schema"] = "other.v1"
        name = _write_spec(payload)
        try:
            with self.assertRaises(ValueError):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_bad_shapes_raise(self):
        base = _glm_payload()
        cases = [
            {"shapes": []},
            {"shapes": [{"bad": "object"}]},
            {"shapes": [dict(base["shapes"][0], kernel_kind="sparse")]},
            {"shapes": [dict(base["shapes"][0], shape_id="")]},
            {"shapes": [dict(base["shapes"][0], rows=0)]},
            {"shapes": [dict(base["shapes"][0], columns=-1)]},
            {"shapes": [dict(base["shapes"][0], mode=1)]},
        ]
        for override in cases:
            payload = copy.deepcopy(base)
            payload.update(override)
            name = _write_spec(payload)
            try:
                with self.assertRaises(
                    ValueError, msg=json.dumps(override)
                ):
                    tab.parse_structure_spec(name)
            finally:
                os.unlink(name)

    def test_bad_ms_and_meta_raise(self):
        base = _glm_payload()
        cases = [
            {"ms": []},
            {"ms": [1, 0]},
            {"ms": [16, 16]},
            {"ms": ["16"]},
            {"experts": 0},
            {"top_k": "8"},
            {"hidden": -1},
            {"inter": None},
        ]
        for override in cases:
            payload = copy.deepcopy(base)
            payload.update(override)
            name = _write_spec(payload)
            try:
                with self.assertRaises(
                    ValueError, msg=json.dumps(override)
                ):
                    tab.parse_structure_spec(name)
            finally:
                os.unlink(name)


class RosterParity(unittest.TestCase):
    def test_default_roster_matches_hardcoded_cells(self):
        expected = [
            "routed:gate_up:M1",
            "routed:gate_up:M16",
            "routed:gate_up:M2048",
            "routed:gate_up:M4096",
            "routed:down:M1",
            "routed:down:M16",
            "routed:down:M2048",
            "routed:down:M4096",
            "dense:o_proj:M1",
            "dense:o_proj:M16",
            "dense:o_proj:M2048",
            "dense:o_proj:M4096",
            "dense:q_b:M1",
            "dense:q_b:M16",
            "dense:q_b:M2048",
            "dense:q_b:M4096",
            "routed:gate_up:M2048:recorded",
            "routed:gate_up:M4096:recorded",
            "routed:down:M2048:recorded",
            "routed:down:M4096:recorded",
        ]
        self.assertEqual(
            [cell["cell_id"] for cell in tab.roster()], expected
        )

    def test_spec_roster_matches_default_roster(self):
        parsed = tab.parse_structure_spec(FIXTURE)
        self.assertEqual(
            tab.roster(parsed["shapes"], parsed["ms"]), tab.roster()
        )

    def test_resolve_without_flag_returns_defaults(self):
        shapes, ms, owner, record = tab.resolve_sweep_geometry(None)
        self.assertEqual(shapes, tab.SHAPES)
        self.assertEqual(ms, tab.MS)
        self.assertEqual(owner, tab.DEFAULT_SHAPE_OWNER)
        self.assertIsNone(record)

    def test_custom_spec_changes_roster(self):
        payload = _glm_payload()
        payload["spec_id"] = "wide-tp2"
        payload["hidden"] = 5120
        payload["inter"] = 1536
        payload["experts"] = 128
        payload["shapes"] = [
            {
                "kernel_kind": "routed",
                "shape_id": "gate_up",
                "rows": 768,
                "columns": 5120,
                "mode": 0,
            },
            {
                "kernel_kind": "routed",
                "shape_id": "down",
                "rows": 5120,
                "columns": 768,
                "mode": 2,
            },
            {
                "kernel_kind": "dense",
                "shape_id": "o_proj",
                "rows": 5120,
                "columns": 5120,
                "mode": 2,
            },
        ]
        name = _write_spec(payload)
        try:
            shapes, ms, owner, record = tab.resolve_sweep_geometry(name)
            cells = tab.roster(shapes, ms)
            self.assertEqual(len(cells), 3 * 4 + 2 * 2)
            self.assertIn("routed:gate_up:M2048:recorded", {
                cell["cell_id"] for cell in cells
            })
            self.assertNotIn("dense:o_proj:M2048:recorded", {
                cell["cell_id"] for cell in cells
            })
            self.assertIn("wide-tp2", owner)
            self.assertEqual(record["experts"], 128)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertNotEqual(cells, tab.roster())
        finally:
            os.unlink(name)

    def test_recorded_cells_follow_min_m(self):
        payload = _glm_payload()
        payload["ms"] = [1, 512]
        name = _write_spec(payload)
        try:
            parsed = tab.parse_structure_spec(name)
            ids = {
                cell["cell_id"]
                for cell in tab.roster(parsed["shapes"], parsed["ms"])
            }
            self.assertFalse(
                any("recorded" in cell_id for cell_id in ids)
            )
        finally:
            os.unlink(name)


class CliFlag(unittest.TestCase):
    def test_help_lists_structure_spec(self):
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(SPEED, "rung_allowability_table.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--structure-spec", proc.stdout)

    def test_module_imports_without_torch(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, %r); "
                    "import rung_allowability_table; "
                    "assert 'torch' not in sys.modules; "
                    "assert not any(k.startswith('tessera.') "
                    "for k in sys.modules)"
                    % os.path.normpath(SPEED)
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
