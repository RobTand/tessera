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


class StrictTypes(unittest.TestCase):
    def test_bool_rows_refused(self):
        payload = _glm_payload()
        payload["shapes"][0]["rows"] = True
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_bool_meta_refused(self):
        payload = _glm_payload()
        payload["experts"] = True
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_bool_ms_refused(self):
        payload = _glm_payload()
        payload["ms"] = [1, True]
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "positive integers"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_float_mode_refused(self):
        payload = _glm_payload()
        payload["shapes"][0]["mode"] = 0.0
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "mode must be 0"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_string_mode_refused(self):
        payload = _glm_payload()
        payload["shapes"][1]["mode"] = "2"
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "mode must be 0"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)

    def test_top_k_above_experts_refused(self):
        payload = _glm_payload()
        payload["experts"] = 8
        payload["top_k"] = 16
        name = _write_spec(payload)
        try:
            with self.assertRaisesRegex(ValueError, "top_k"):
                tab.parse_structure_spec(name)
        finally:
            os.unlink(name)


class FullDimensionMatch(unittest.TestCase):
    def test_exact_dims_match(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "routed", "gate_up", 1024, 4096, 0
        )
        self.assertEqual(entry, ("routed", "gate_up", 1024, 4096, 0))
        self.assertTrue(verified)

    def test_dense_none_mode_matches(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "dense", "o_proj", 4096, 4096, None
        )
        self.assertEqual(entry, ("dense", "o_proj", 4096, 4096, 2))
        self.assertTrue(verified)

    def test_same_name_new_dims_mismatch(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "routed", "gate_up", 768, 5120, 0
        )
        self.assertIsNone(entry)
        self.assertTrue(verified)

    def test_wrong_mode_mismatch(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "routed", "gate_up", 1024, 4096, 2
        )
        self.assertIsNone(entry)
        self.assertTrue(verified)

    def test_unknown_name_mismatch(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "dense", "o_new", 4096, 4096, 2
        )
        self.assertIsNone(entry)
        self.assertFalse(verified)

    def test_missing_dims_unverified(self):
        entry, verified = tab.match_shape(
            tab.SHAPES, "routed", "down", None, None, 2
        )
        self.assertEqual(entry, ("routed", "down", 4096, 1024, 2))
        self.assertFalse(verified)


class SerializedScopeParity(unittest.TestCase):
    def test_flag_and_default_scopes_match(self):
        parsed = tab.parse_structure_spec(FIXTURE)
        default = json.dumps(
            tab.scope_geometry(tab.SHAPES, tab.MS), sort_keys=True
        )
        flagged = json.dumps(
            tab.scope_geometry(parsed["shapes"], parsed["ms"]),
            sort_keys=True,
        )
        self.assertEqual(flagged, default)

    def test_spec_scope_carries_model_fields(self):
        shapes, ms, owner, record = tab.resolve_sweep_geometry(FIXTURE)
        self.assertEqual(
            (record["experts"], record["top_k"], record["hidden"],
             record["inter"]),
            (288, 8, 4096, 1024),
        )
        self.assertEqual(len(record["sha256"]), 64)
        self.assertIn("glm53-tp2", owner)


def _needs_harvest_deps():
    try:
        import torch  # noqa: F401
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


NEEDS_HARVEST = _needs_harvest_deps()
HARVEST_FIXTURE = os.path.join(HERE, "fixtures", "d41_structure_spec_harvest")
HARVEST_SPEC = os.path.join(HARVEST_FIXTURE, "glm53_tp2_spec.json")
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))


class HarvestWithFlag(unittest.TestCase):
    """A full harvest with --structure-spec on a committed fixture.

    The fixture carries two rungs over the four GLM TP2 shapes with stamped
    rows, columns and model fields, plus a quality file. Both tests run the
    real CLI in a subprocess and need torch and jsonschema; without them
    they skip instead of failing.
    """

    def _run_harvest(self, out, *extra):
        import shutil
        import tempfile

        if not NEEDS_HARVEST:
            self.skipTest("harvest needs torch and jsonschema")
        target = tempfile.mkdtemp(prefix="d41-harvest-")
        self.addCleanup(shutil.rmtree, target, True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src") + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        cmd = [
            sys.executable,
            os.path.join(SPEED, "rung_allowability_table.py"),
            "--root", HARVEST_FIXTURE,
            "--schema", os.path.join(REPO_ROOT, "docs", "schema",
                                     "allowable-rung-table.v2.schema.json"),
            "--index-schema", os.path.join(REPO_ROOT, "docs", "schema",
                                           "index.v2.schema.json"),
            "--out", target,
            "--version", "9001",
            "--format", "TESSERA_E4M3_K1",
        ] + list(extra)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=600)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        table = json.load(open(os.path.join(target, "table.json")))
        validation = json.load(open(os.path.join(target, "validation.json")))
        return table, validation

    def test_flag_and_default_tables_match_serialized(self):
        default, _ = self._run_harvest(None)
        flagged, _ = self._run_harvest(None, "--structure-spec", HARVEST_SPEC)
        self.assertEqual(len(default["rungs"]), len(flagged["rungs"]))
        self.assertEqual(default["rungs"], flagged["rungs"])
        self.assertEqual(default["scope"]["required_cells"],
                         flagged["scope"]["required_cells"])
        self.assertEqual(default["scope"]["shapes"], flagged["scope"]["shapes"])
        self.assertNotIn("structure_spec", default["scope"])
        record = flagged["scope"]["structure_spec"]
        self.assertEqual(record["spec_id"], "glm53-tp2")
        self.assertEqual((record["experts"], record["top_k"],
                          record["hidden"], record["inter"]),
                         (288, 8, 4096, 1024))
        self.assertNotEqual(default["scope"]["shape_owner"],
                            flagged["scope"]["shape_owner"])

    def test_full_harvest_with_flag(self):
        table, validation = self._run_harvest(
            None, "--structure-spec", HARVEST_SPEC)
        self.assertEqual(validation["status"],
                         "schema_and_semantic_validation_passed")
        self.assertEqual(len(table["scope"]["required_cells"]), 20)
        self.assertEqual(len(table["rungs"]), 385)

    def test_altered_dims_spec_skips_cells(self):
        import shutil
        import tempfile

        payload = json.load(open(HARVEST_SPEC))
        payload["shapes"][0]["rows"] = 768
        tmp = tempfile.mkdtemp(prefix="d41-spec-")
        self.addCleanup(shutil.rmtree, tmp, True)
        altered = os.path.join(tmp, "altered.json")
        json.dump(payload, open(altered, "w"))
        table, validation = self._run_harvest(
            None, "--structure-spec", altered)
        self.assertEqual(validation["status"],
                         "schema_and_semantic_validation_passed")
        seen = {m["cell_id"] for row in table["rungs"]
                for m in row["measurements"]}
        self.assertTrue(seen)
        self.assertFalse(any(cell.startswith("routed:gate_up")
                             for cell in seen))


if __name__ == "__main__":
    unittest.main()
