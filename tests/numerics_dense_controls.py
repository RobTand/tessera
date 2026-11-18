"""CPU fault controls on sealed dense artifacts. Run outside ordinary rows."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.numerics_rows import dense_phases as dp, row, scale_launch, scope as source_scope
from experiments.numerics_rows.common import Invalid, error_text, sha256, write_json
from experiments.numerics_rows.gpu_time import PhaseTimes
from experiments.numerics_rows.join import join, rule_unit_id
from experiments.numerics_rows.launch_state import json_digest
from experiments.numerics_rows.population import load_signed_population, select_modules
from experiments.numerics_rows.scale_index import evaluate


REPLAY_GROUPS = {
    "R0": ("incomplete-C1", "wrong-C1-domain"),
    "R1": ("sweep-outside", "chunk_pairs-outside", "diagnostic-outside", "scale_index-outside", "composition-outside"),
    "R2": ("sweep-inside", "chunk_pairs-inside", "scale_index-inside", "wrong-scale-index"),
    "R3": ("diagnostic-inside",),
    "R4": ("composition-staged",),
}


def artifact_hashes(*roots):
    return {str(path): sha256(path) for root in roots for path in sorted(Path(root).rglob("*")) if path.is_file()}


def experiments_tree(repo, commit, scratch):
    def git(*args, **kwargs):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, **kwargs).strip()
    if git("rev-parse", "HEAD") != commit:
        raise Invalid("control checkout differs from the canary tool commit")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        env = os.environ | {"GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git("read-tree", commit, env=env)
        git("add", "-A", "-f", "--", "experiments", env=env)
        actual = git("rev-parse", git("write-tree", env=env) + ":experiments")
    if actual != git("rev-parse", commit + ":experiments"):
        raise Invalid("control experiments tree differs from the canary tool commit")
    return actual


def selected_unit(receipt, scope):
    unit, = [u for u in receipt["units"] if row.unit_identity(u) == scope["unit_key"] and u["module"] == scope["module"]]
    return unit


def eligible_cases(pairs, receipt):
    return {c["id"] for c, _ in pairs if any(
        row.unit_identity(u) == c["scope"]["unit_key"] and u["module"] == c["scope"]["module"]
        and u["role"] == "unit" and u["admitted_by_v68"] is True for u in receipt["units"])}


def changed(value):
    return 1.0 if float(value) == 0 else 0.0


def source_roots(args, *outputs):
    roots = {root.resolve().parent.parent.parent for root in (*args.prepare, *args.capture)}
    if not roots or any(root.parent.name != "units" for root in roots):
        raise Invalid("controls require unit attempt artifacts")
    if any(output.resolve().is_relative_to(root.parent.parent) or root.parent.parent.is_relative_to(output.resolve())
           for root in roots for output in outputs):
        raise Invalid("control results must be outside original artifacts")
    return sorted(roots)


def check_paths(args):
    roots = source_roots(args, args.scratch, args.out)
    outputs = args.scratch.resolve(), args.out.resolve()
    if outputs[0] == outputs[1] or outputs[0].is_relative_to(outputs[1]):
        raise Invalid("control report must not contain scratch")
    if outputs[0].is_relative_to(Path("/mnt/shared")):
        raise Invalid("control scratch must stay off the shared pool")
    return roots


def module_inputs(args, receipt, groups):
    """Read the newest sealed pair for every TP unit and case set."""
    if len(args.prepare) != len(args.capture):
        raise Invalid("control phase artifact counts differ")
    inputs, selected = [], []
    for prepare_root, capture_root in zip(args.prepare, args.capture):
        document = json.loads((prepare_root / "phase.json").read_text())
        matched = [unit for unit in dp.phase_units(groups) if dp.scopes(unit) == document["scopes"]]
        if len(matched) != 1:
            raise Invalid("control phase artifact has a foreign module or cut")
        unit_groups, = matched
        part = document["case_set"]
        selected.append((unit_groups[0][2]["size"], part))
        common = dp.identities({**receipt, "inputs": {**receipt["inputs"], "selected_cuts": [g[2] for g in unit_groups]}})
        newest = dp.read_attempts(args, unit_groups, prepare_root, capture_root, common, part)[0]
        if newest is None:
            raise Invalid("controls require a finished artifact pair")
        pr, cr, prepared, captured, pd, _ = newest
        inputs.append((pr, cr, prepared, captured, pd, unit_groups))
    expected = sorted((unit[0][2]["size"], part) for unit in dp.phase_units(groups) for part in ("gate", "diagnostics"))
    if sorted(selected) != expected:
        raise Invalid("controls require the full gate and diagnostics union exactly once")
    return sorted(inputs, key=lambda item: item[2]["ordinal_start"])


def replay(prepare, capture, out, faults=(), refusal=None, *, groups, receipt):
    """Copy all attempts, then overlay selected tensors during one module compare."""
    originals = SimpleNamespace(prepare=prepare, capture=capture)
    roots = source_roots(originals, out)
    out.mkdir(parents=True)
    for root in roots:
        shutil.copytree(root, out / "artifacts/units" / root.name)
    def copied(root):
        owner = root.resolve().parent.parent.parent
        return out / "artifacts/units" / owner.name / root.resolve().relative_to(owner)
    result = copy.deepcopy(receipt)
    result["modules"] = []
    result["units"] = [row.new_unit(unit) for unit in result["units"]]
    result["row"].update(finalization_finished=False, outcome="STOPPED")
    original_compare, original_payload = dp.compare_case, dp.payload
    applied = []

    def compare(case, observed, *arguments):
        if refusal is not None and case["scope"] == refusal[0]:
            for field in ("invalid", "unattested"):
                arguments[-1][field].extend(v for v in refusal[1][field] if v not in arguments[-1][field])
        def payload(root, reference):
            value = original_payload(root, reference)
            for case_id, field, index, replacement in faults:
                if case["id"] == case_id and reference is observed["tensors"].get(field):
                    value[index] = replacement
                    applied.append((case_id, field, index, replacement))
            return value
        with patch.object(dp, "payload", payload):
            original_compare(case, observed, *arguments)

    arguments = SimpleNamespace(out=out, prepare_artifact=[copied(p) for p in prepare],
                                capture_artifact=[copied(c) for c in capture])
    with patch.object(dp, "compare_case", compare):
        dp.host_compare(arguments, result, PhaseTimes(result["row"]), groups)
    if sorted(applied) != sorted(faults):
        raise Invalid("a control did not change exactly its selected tensors")
    if not result["row"].get("finalization_finished"):
        raise Invalid("control compare did not finish")
    return result


class BeforeNative(Exception):
    pass


def plan_probe(adapter, unit, module, role, plan):
    """Use the real memo recompute and validation, and stop before a launch."""
    unit = row.new_unit(unit)
    module = {**module, "roles": [{**role, "scale_index": [plan]}]}
    verified, calls = scale_launch.joint_coverage, []

    def record(*args, **kwargs):
        result = verified(*args, **kwargs)
        calls.append(result)
        return result

    def stop_launch(*args, **kwargs):
        raise BeforeNative

    adapter.splits = unit["m_config"]["k_split"]
    with patch.object(scale_launch, "joint_coverage", record), patch.object(
            adapter, "case_writer", SimpleNamespace(scale=stop_launch), create=True):
        try:
            scale_launch.check(adapter, unit, module, None)
        except BeforeNative:
            pass
        except Invalid as error:
            unit["invalid"].append(str(error))
        else:
            if not unit["invalid"] and not unit["unattested"]:
                raise Invalid("control did not reach the prelaunch boundary")
    if len(calls) != 1:
        raise Invalid("control did not run the real C1 memo recompute once")
    return unit, calls[0]


def matching_plan(manifest, case):
    scope = case["scope"]
    module, = [m for m in manifest["modules"] if m["name"] == scope["module"]]
    cut, = [c for c in module["cuts"] if all(c["tp"][k] == scope["cut"][k] for k in ("size", "rank"))]
    module = {**module, **cut}
    role, = [r for r in module["roles"] if r["name"] == case["role"]]
    plan, = [p for p in role["scale_index"] if p["scale"] == case["scale"] and
             p["array"].get("M", case["configuration"]["M"]) == case["configuration"]["M"]]
    read_case, = [c for c in plan["read_cases"] if c["read"] == case["read"]]
    if case["coding"] not in plan["coding"]:
        raise Invalid("captured scale coding is absent from C1")
    return module, role, plan, read_case


class Controls:
    def __init__(self, args):
        check_paths(args)
        self.args, self.records = args, []
        self.population, self.population_digest = load_signed_population(args.population)
        self.manifest = json.loads(args.manifest.read_text())
        first = json.loads((args.capture[0] / "phase.json").read_text())
        self.receipt = copy.deepcopy(first["receipt"])
        receipt = self.receipt
        if len(receipt["inputs"]["selected_modules"]) != 1:
            raise Invalid("controls require one module receipt")
        self.groups = row.planned_units(source_scope.contract_document(args.contract)[0], receipt["runtime"]["image_digest"],
            select_modules(self.population, receipt["inputs"]["selected_modules"]), self.manifest)
        for _, module, _, units in self.groups:
            for unit in units:
                unit["launch_state_sha256"] = receipt["runtime"]["launch_state_sha256"]
                unit["tp"].update(cut_axis=[r["cut_axis"] for r in module["roles"]], cut_range=[r["cut_range"] for r in module["roles"]])
        receipt["inputs"]["selected_cuts"] = [g[2] for g in self.groups]
        receipt["units"] = [row.new_unit(u) for *_, units in self.groups for u in units]
        self.inputs = module_inputs(args, receipt, self.groups)
        captured_units = {}
        for _, _, _, captured, _, _ in self.inputs:
            for key, unit in dp.unit_map(captured["receipt"]["units"]).items():
                if key in captured_units and any(unit[field] != captured_units[key][field] for field in ("role", "admitted_by_v68")):
                    raise Invalid("control phase eligibility differs")
                captured_units[key] = unit
        receipt["units"] = [row.new_unit(captured_units[key]) for key in dp.unit_map(receipt["units"])]
        binding = receipt["inputs"]
        if (sha256(args.manifest) != binding["case_manifest_sha256"] or sha256(args.q1) != binding["q1_table_sha256"]
                or self.population_digest != binding["population_sha256"]):
            raise Invalid("control C1, Q1 or population differs from captured inputs")

    @property
    def pairs(self):
        for pr, cr, prepared, captured, pd, groups in self.inputs:
            yield from dp.check_pair(self.args, groups, prepared, captured, pd, pr, cr)

    def eligible(self, case):
        unit = selected_unit(self.receipt, case["scope"])
        return unit["role"] == "unit" and unit["admitted_by_v68"] is True

    def tensor(self, case, reference, *, captured=False):
        for pr, cr, prepared, _, _, _ in self.inputs:
            if case["scope"] in prepared["scopes"] and (prepared["case_set"] == "diagnostics") == (case["category"] == "diagnostic"):
                return dp.payload(cr if captured else pr, reference)
        raise Invalid("control case has no phase artifact")

    def joined(self, receipt, target):
        try:
            result = join(self.population, self.population_digest, [(sha256(self.args.manifest), self.manifest)],
                          [(json_digest(receipt), receipt)], self.args.contract)
        except Invalid as error:
            if receipt.get("canary") and str(error) == "a canary cannot supply attestation":
                return "CANARY_REFUSED"
            raise
        key = rule_unit_id(target)
        joined, = [u for u in result["units"] if u["unit_id"] == key]
        module, = [m for m in joined["modules"] if m["module"] == target["module"]]
        return module["outcome"]

    def run(self, name, case, expected, *, faults=(), refusal=None, checked=1, details=None, check=None):
        if not hasattr(self, "plants") and getattr(self.args, "only", None) not in (None, name):
            return
        plant = dict(name=name, case=case, expected=expected, faults=faults, refusal=refusal,
                     checked=checked, details=details, check=check)
        if hasattr(self, "plants"):
            self.plants.append(plant)
        else:
            self.run_plants(name, [plant], expected)

    def run_plants(self, name, plants, expected):
        if len({p["case"]["id"] for p in plants}) != len(plants):
            raise Invalid("combined plants must use distinct cases")
        start = time.monotonic()
        faults = [f for p in plants for f in p["faults"]]
        refusals = [p["refusal"] for p in plants if p["refusal"]]
        if len(refusals) > 1:
            raise Invalid("mapping controls require separate replays")
        result = replay(self.args.prepare, self.args.capture, self.args.scratch / name, faults,
                        refusals[0] if refusals else None, groups=self.groups, receipt=self.receipt)
        valid = result["row"]["outcome"] == expected and result["row"]["exit_status"] == (
            3 if expected in ("FAIL", "INVALID", "STOPPED") else 0)
        targets, joined_keys = [], {}
        plants = sorted(plants, key=lambda plant: plant["case"]["ordinal"])
        stream_root = Path(result["phase_artifacts"]["compare"]["path"]).parent
        document, _ = dp.read(stream_root, "compare", dp.identities(result), dp.scopes(self.groups))
        stream_reasons = {json_digest([u["module"], row.unit_identity(u)]): {k: [] for k in ("findings", "invalid", "unattested")}
                          for u in result["units"]}
        for record in dp.cases(stream_root, document):
            local = [p for p in plants if p["case"]["id"] == record["id"] or
                     p["refusal"] and p["refusal"][0] == record["scope"]]
            valid &= all(record[field] == [p["check"][1] for p in local if p["check"] and p["check"][0] == field]
                         for field in ("findings", "invalid", "unattested"))
            reasons = stream_reasons[json_digest([record["scope"]["module"], record["scope"]["unit_key"]])]
            for field in reasons:
                reasons[field].extend(value for value in record[field] if value not in reasons[field])
        for target in result["units"]:
            selected = [p for p in plants if selected_unit(result, p["case"]["scope"]) is target]
            reasons = {k: target[k] for k in ("findings", "invalid", "unattested")}
            expected_reasons = {k: [p["check"][1] for p in selected if p["check"] and p["check"][0] == k] for k in reasons}
            target_expected = expected if selected else "PASS"
            key = json_digest(rule_unit_id(target))
            if selected and key not in joined_keys:
                joined_keys[key] = self.joined(result, target)
            joined = joined_keys.get(key) if selected else None
            full_reasons = stream_reasons[json_digest([target["module"], row.unit_identity(target)])]
            valid &= (reasons == {k: v[:1] for k, v in expected_reasons.items()} and full_reasons == expected_reasons
                      and target["outcome"] == target_expected and
                      (not selected or joined == ("CANARY_REFUSED" if result["canary"] else target_expected)))
            targets.append(dict(unit_id=rule_unit_id(target), reasons=reasons, stream_reasons=full_reasons, expected_reasons=expected_reasons,
                                outcome=target["outcome"], join_outcome=joined))
        for plant in plants:
            self.records.append({"name": plant["name"], "case_id": plant["case"]["id"], "group": name,
                "scope": "in" if plant["checked"] else "out", "checked_elements": plant["checked"],
                "not_checked_elements": 0 if plant["checked"] else 1, "expected": plant["expected"],
                "control_result": "MISSED" if not valid else "MAPPING_ONLY" if plant["refusal"] else
                                  "DETECTED" if plant["checked"] else "NOT_CHECKED",
                "row_outcome": result["row"]["outcome"], "exit_status": result["row"]["exit_status"],
                "join_outcome": joined_keys[json_digest(rule_unit_id(selected_unit(result, plant["case"]["scope"])))], "valid": bool(valid),
                "expected_check": plant["check"], "targets": targets, "details": plant["details"],
                "fault_indices": [list(f[2]) for f in plant["faults"]], "elapsed_seconds": time.monotonic() - start})
        if not valid:
            raise Invalid("control was not flagged: " + name + "; use --only for each control in this group")
        return result

    def output_controls(self, category):
        candidates = ((c, o) for c, o in self.pairs if c["category"] == category and self.eligible(c))
        first = next(candidates, None)
        if first is None:
            raise Invalid("required control category is absent: " + category)
        case, observed = first
        output = self.tensor(case, observed["tensors"]["output"], captured=True)
        if case["kind"] == "scale":
            index = tuple(case["output_index"])
            outside = torch.ones_like(output, dtype=torch.bool)
            outside[index] = False
        elif category == "diagnostic":
            index, outside = tuple(0 for _ in output.shape), torch.ones_like(output, dtype=torch.bool)
            outside[index] = False
        else:
            mask = self.tensor(case, case["tensors"]["x_mask"])
            index, outside = tuple(mask.nonzero()[0].tolist()), ~mask
        replacement = changed(output[index])
        fields = ("output",) if case["kind"] == "scale" or category == "diagnostic" else ("output", "staged")
        faults = [(case["id"], field, index, replacement) for field in fields]
        self.run(category + "-inside", case, "FAIL", faults=faults, check=("findings",
                 "apply differs from the same staged launches" if category == "diagnostic" else case["id"] + ": exact output differs"))
        if category == "chunk_pairs" and not bool(outside.any()):
            for other, other_observed in candidates:
                mask = self.tensor(other, other["tensors"]["x_mask"])
                if bool((~mask).any()):
                    case, observed, outside = other, other_observed, ~mask
                    output = self.tensor(case, observed["tensors"]["output"], captured=True)
                    break
        if bool(outside.any()):
            index = tuple(outside.nonzero()[0].tolist())
            fields = ("output",) if case["kind"] == "scale" else ("output", "staged")
            self.run(category + "-outside", case, "PASS", checked=0,
                     faults=[(case["id"], f, index, changed(output[index])) for f in fields])
        else:
            self.not_applicable(category + "-outside", case)
        return case

    def not_applicable(self, name, case):
        if not hasattr(self, "plants") and getattr(self.args, "only", None) not in (None, name):
            return
        record = dict(name=name, case_id=case["id"], scope="out", not_checked_elements=0,
                      control_result="NOT_APPLICABLE", expected="NOT_APPLICABLE", valid=True)
        if hasattr(self, "plants"):
            self.plants.append(dict(name=name, case=case, expected="NOT_APPLICABLE", check=None, record=record))
        else:
            self.records.append(record)

    def scale_controls(self):
        inside = next(c["id"] for c, _ in self.pairs if c["category"] == "scale_index" and self.eligible(c))
        for case, _ in self.pairs:
            if case["kind"] != "scale" or not self.eligible(case) or case["id"] == inside:
                continue
            module, role, plan, read_case = matching_plan(self.manifest, case)
            values = self.tensor(case, case["tensors"]["coded_values"]).tolist()
            expected = float(self.tensor(case, case["tensors"]["reference"])[0])
            if evaluate(read_case, case["read"], values) != expected:
                raise Invalid("C1 scale evaluator differs from the captured case reference")
            wrong = next((i for i in range(len(values)) if [i] != case["read"] and evaluate(read_case, [i], values) != expected), None)
            if wrong is not None:
                break
        else:
            raise Invalid("no recorded scale pass can expose a wrong index")
        self.run("wrong-scale-index", case, "FAIL", faults=[(case["id"], "output", tuple(case["output_index"]), evaluate(read_case, [wrong], values))],
                 check=("findings", case["id"] + ": exact output differs"), details={"read": case["read"], "wrong_read": [wrong]})
        self.run("incomplete-C1", case, "UNATTESTED", check=("unattested", "incomplete scale-index coverage before the freeze"))
        self.run("wrong-C1-domain", case, "INVALID", check=("invalid", "C1 scale case has the wrong read tuple or array size"))

    def mapping(self, plant):
        case = plant["case"]
        module, role, plan, _ = matching_plan(self.manifest, case)
        target = selected_unit(self.receipt, case["scope"])
        entry, = [e for e in self.population["modules"] if e["name"] == target["module"]]
        adapter = dp.HostDense(entry, {k: target["tp"][k] for k in ("size", "rank")}, module, json.loads(self.args.q1.read_text()))
        positive, verified = plan_probe(adapter, target, module, role, plan)
        if positive["invalid"] or positive["unattested"] or verified["reads"] != verified["reads_covered"]:
            raise Invalid("real C1 positive control is not complete and valid")
        plan = copy.deepcopy(plan)
        if plant["name"] == "incomplete-C1":
            plan["reads_covered"] -= 1
        else:
            plan["array"]["N"] += 1
        finding, verified = plan_probe(adapter, target, module, role, plan)
        return {**plant, "refusal": (case["scope"], finding), "details": {"recomputed": verified}}

    def plan(self):
        self.plants = []
        for category in ("sweep", "chunk_pairs", "diagnostic", "scale_index"):
            case = self.output_controls(category)
            if category == "sweep":
                staged = self.tensor(case, next(o for c, o in self.pairs if c["id"] == case["id"])["tensors"]["staged"], captured=True)
                index = tuple(0 for _ in staged.shape)
                self.run("composition-staged", case, "FAIL", faults=[(case["id"], "staged", index, changed(staged[index]))],
                         check=("findings", "apply differs from the same staged launches"))
                self.not_applicable("composition-outside", case)
        self.scale_controls()
        choices = {"wrong-scale-index": "first other scale case with a wrong read that changes the reference",
                   "incomplete-C1": "same case as wrong-scale-index; counted row only",
                   "wrong-C1-domain": "same case as wrong-scale-index; counted row only",
                   "composition-staged": "first eligible sweep case; zero index",
                   "composition-outside": "first eligible sweep case; no plant",
                   "chunk_pairs-outside": "first eligible chunk_pairs case with an outside element"}
        self.table = [{"selection": choices.get(p["name"], "first eligible " + p["case"]["category"] + " case"), "control": p["name"], "group": next(g for g, names in REPLAY_GROUPS.items() if p["name"] in names),
                       "case_id": p["case"]["id"], "expected_check": list(p["check"]) if p["check"] else None, "expected": p["expected"]} for p in self.plants]
        # Write real case ids before the first replay.
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.args.out, {"verdict": "IN_PROGRESS", "control_table": self.table})

    def source_tree(self):
        return experiments_tree(Path(__file__).resolve().parents[1], self.receipt["tool"]["commit"], self.args.scratch)

    def group(self, name):
        tree_before = self.source_tree()
        before = artifact_hashes(*source_roots(self.args))
        self.records = []
        plants = [p for p in self.plants if p["name"] in REPLAY_GROUPS[name]]
        active = [p for p in plants if "record" not in p]
        self.records.extend(p["record"] for p in plants if "record" in p)
        baseline = None
        if name == "R0":
            baseline = self.run_plants("R0", [], "PASS")
            for plant in active:
                if self.receipt["canary"]:
                    self.records.append(dict(name=plant["name"], case_id=plant["case"]["id"], valid=True,
                                             control_result="DEFERRED_TO_COUNTED_ROW"))
                else:
                    self.run_plants(plant["name"], [self.mapping(plant)], plant["expected"])
        else:
            self.run_plants(name, active, "PASS" if name == "R1" else "FAIL")
        after = artifact_hashes(*source_roots(self.args))
        tree_after = self.source_tree()
        return dict(group=name, verdict="PASS" if before == after and tree_before == tree_after else "INVALID", controls=self.records[:],
                    experiments_tree_before=tree_before, experiments_tree_after=tree_after,
                    baseline=baseline, artifact_hashes_before=before, artifact_hashes_after=after,
                    artifacts_unchanged=before == after, control_table=self.table)

    def assemble(self, reports):
        by_group = {r["group"]: r for r in reports}
        if len(reports) != 5 or set(by_group) != set(REPLAY_GROUPS):
            raise Invalid("assembly requires one report for each of R0..R4")
        hashes = artifact_hashes(*source_roots(self.args))
        tree = self.source_tree()
        for name, report in by_group.items():
            if (report["verdict"] != "PASS" or report["control_table"] != self.table or
                report["artifact_hashes_before"] != hashes or report["artifact_hashes_after"] != hashes or
                report["experiments_tree_before"] != tree or report["experiments_tree_after"] != tree or
                not report["artifacts_unchanged"] or not all(r["valid"] for r in report["controls"]) or
                sorted(r["name"] for r in report["controls"]) != sorted(REPLAY_GROUPS[name])):
                raise Invalid("invalid group report: " + name)
        baseline = by_group["R0"]["baseline"]
        if baseline["row"]["outcome"] != "PASS" or baseline["row"]["exit_status"] != 0:
            raise Invalid("unmodified artifacts do not produce a PASS row with rc 0")
        targets = {json_digest(rule_unit_id(selected_unit(baseline, c["scope"]))): selected_unit(baseline, c["scope"])
                   for c, _ in self.pairs if self.eligible(c)}
        if any(self.joined(baseline, target) != ("CANARY_REFUSED" if baseline["canary"] else "PASS")
               for target in targets.values()):
            raise Invalid("unmodified target cannot supply joined attestation")
        if baseline["canary"] is not self.receipt["canary"]:
            raise Invalid("baseline row kind differs from the capture")
        mappings = by_group["R0"]["controls"]
        status = "DEFERRED_TO_COUNTED_ROW" if baseline["canary"] else "MAPPING_ONLY"
        if any(r["control_result"] != status for r in mappings):
            raise Invalid("C1 mapping status differs from the baseline row kind")
        self.records = [r for report in reports for r in report["controls"]]
        return not baseline["canary"]

    def check(self):
        self.source_before = self.source_tree()
        self.plan()
        if self.args.assemble:
            reports = [json.loads(path.read_text()) for path in self.args.assemble]
            counted = self.assemble(reports)
        elif self.args.only:
            plant, = [p for p in self.plants if p["name"] == self.args.only]
            if "record" in plant:
                self.records = [plant["record"]]
            elif plant["name"] in REPLAY_GROUPS["R0"]:
                if self.receipt["canary"]:
                    self.records = [dict(name=plant["name"], valid=True, control_result="DEFERRED_TO_COUNTED_ROW")]
                else:
                    self.run_plants(plant["name"], [self.mapping(plant)], plant["expected"])
            else:
                self.run_plants(plant["name"], [plant], plant["expected"])
            counted = False
        else:
            reports = [self.group(name) for name in ([self.args.group] if self.args.group else REPLAY_GROUPS)]
            self.group_report = reports[0] if self.args.group else None
            counted = False if self.args.group else self.assemble(reports)
        self.source_after = self.source_tree()
        return counted


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepare", "capture"):
        parser.add_argument("--" + name, type=Path, nargs="+", required=True)
    for name in ("manifest", "population", "q1", "contract", "scratch", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only", help="Run one named control")
    mode.add_argument("--group", choices=REPLAY_GROUPS, help="Run one independent group")
    mode.add_argument("--assemble", type=Path, nargs=5, help="Read reports for R0..R4")
    args = parser.parse_args()
    roots = check_paths(args)
    before, controls = artifact_hashes(*roots), None
    report = {"verdict": "INVALID", "counting_allowed": False, "group": args.group}
    try:
        controls = Controls(args)
        counted = controls.check()
        report.update(getattr(controls, "group_report", None) or {})
        report.update(verdict=report.get("verdict") if args.group else "PASS", counting_allowed=counted)
    except Exception as error:
        report.update(exception_type=type(error).__name__, reason=error_text(error))
    after = artifact_hashes(*roots)
    report.update(control_table=getattr(controls, "table", []), controls=[] if controls is None else controls.records, artifact_hashes_before=before,
                  artifact_hashes_after=after, artifacts_unchanged=before == after,
                  experiments_tree_before=getattr(controls, "source_before", None), experiments_tree_after=getattr(controls, "source_after", None),
                  elapsed_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start)
    if before != after:
        report.update(verdict="INVALID", counting_allowed=False, reason="original artifact files changed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    raise SystemExit(0 if report["verdict"] == "PASS" else 3)


if __name__ == "__main__":
    main()
