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


class StrictSpec(unittest.TestCase):
    """A spec value that is not what it looks like must refuse by name (#1038 review)."""

    def _refuses(self, mutate):
        payload = _glm_payload()
        mutate(payload)
        name = _write_spec(payload)
        try:
            with self.assertRaises(ValueError):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_a_boolean_is_not_a_dimension(self):
        for field in ("rows", "columns"):
            with self.subTest(field=field):
                self._refuses(lambda p, f=field: p["shapes"][0].__setitem__(f, True))
        for field in ("experts", "top_k", "hidden", "inter"):
            with self.subTest(field=field):
                self._refuses(lambda p, f=field: p.__setitem__(f, True))
        with self.subTest(field="ms"):
            self._refuses(lambda p: p["ms"].__setitem__(0, True))

    def test_a_float_is_not_an_integer_field(self):
        with self.subTest(field="rows"):
            self._refuses(lambda p: p["shapes"][0].__setitem__("rows", 1024.0))
        with self.subTest(field="mode"):
            self._refuses(lambda p: p["shapes"][0].__setitem__("mode", 0.0))
        with self.subTest(field="ms"):
            self._refuses(lambda p: p["ms"].__setitem__(0, 1.0))
        with self.subTest(field="experts"):
            self._refuses(lambda p: p.__setitem__("experts", 288.0))

    def test_false_is_not_mode_zero(self):
        self._refuses(lambda p: p["shapes"][0].__setitem__("mode", False))

    def test_top_k_above_experts_refuses(self):
        self._refuses(lambda p: p.__setitem__("top_k", p["experts"] + 1))

    def test_top_k_equal_to_experts_is_valid(self):
        payload = _glm_payload()
        payload["top_k"] = payload["experts"]
        name = _write_spec(payload)
        try:
            self.assertEqual(tab.parse_structure_spec(name)["meta"]["top_k"], payload["experts"])
        finally:
            os.unlink(name)


class MeasurementMatchesSpec(unittest.TestCase):
    """An old measurement joins a spec's table only when it was taken at that spec's geometry."""

    def setUp(self):
        parsed = tab.parse_structure_spec(FIXTURE)
        self.spec = parsed["shapes"][0]
        self.record = dict(parsed["meta"])
        self.meta = dict(parsed["meta"])
        self.group = {"rows": self.spec[2], "cols": self.spec[3], "mode": self.spec[4]}

    def _kept(self, meta=None, group=None):
        return tab.measured_at_spec(meta or self.meta, group or self.group, self.spec, self.record)

    def test_a_measurement_at_the_spec_geometry_is_kept(self):
        self.assertTrue(self._kept())

    def test_a_different_dimension_or_model_field_is_skipped(self):
        for field, value in (("rows", 2048), ("cols", 2048)):
            with self.subTest(field=field):
                self.assertFalse(self._kept(group=dict(self.group, **{field: value})))
        for field in ("experts", "top_k", "hidden", "inter"):
            with self.subTest(field=field):
                self.assertFalse(self._kept(meta=dict(self.meta, **{field: self.meta[field] + 1})))

    def test_a_measurement_at_another_mode_is_skipped(self):
        # A routed group takes its shape name from its mode: mode 0 is gate_up, anything else is down.
        # Mode 0 evidence must not enter a table whose spec declares mode 2 for that name.
        other = 2 if self.spec[4] == 0 else 0
        self.assertFalse(self._kept(group=dict(self.group, mode=other)))

    def test_a_group_that_records_no_mode_counts_as_mode_two(self):
        # measurement() reads a missing mode as 2, so the comparison reads it the same way.
        no_mode = {k: v for k, v in self.group.items() if k != "mode"}
        spec = (self.spec[0], self.spec[1], self.spec[2], self.spec[3], 2)
        self.assertTrue(tab.measured_at_spec(self.meta, no_mode, spec, self.record))
        spec = (self.spec[0], self.spec[1], self.spec[2], self.spec[3], 0)
        self.assertFalse(tab.measured_at_spec(self.meta, no_mode, spec, self.record))

    def test_a_non_integer_mode_is_skipped(self):
        for value in (True, False, 0.0, 2.0, "2"):
            with self.subTest(mode=value):
                spec = (self.spec[0], self.spec[1], self.spec[2], self.spec[3], 2 if value in (2.0, "2", True) else 0)
                self.assertFalse(tab.measured_at_spec(self.meta, dict(self.group, mode=value), spec, self.record))

    def test_a_measurement_that_does_not_record_a_field_is_skipped(self):
        for field in ("rows", "cols"):
            with self.subTest(field=field):
                self.assertFalse(self._kept(group={k: v for k, v in self.group.items() if k != field}))
        for field in ("experts", "top_k", "hidden", "inter"):
            with self.subTest(field=field):
                self.assertFalse(self._kept(meta={k: v for k, v in self.meta.items() if k != field}))


if __name__ == "__main__":
    unittest.main()
