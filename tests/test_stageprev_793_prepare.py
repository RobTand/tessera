"""New coordinator-proposal controls only; no old73/ELF/readset/CUDA replay."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/t8r_speed"))
import stageprev_793_prepare as preparation


@pytest.fixture
def contracts():
    return (json.loads((ROOT / preparation.PACKET_PATH).read_text()),
            json.loads((ROOT / preparation.EXPECTED_PATH).read_text()))


def test_actual_frozen_sl_coordinator_preparation(contracts):
    # SL-local frozen checkout is the reason this CPU control is placed on SL.
    record = preparation.prepare(preparation.SL_COORDINATOR)
    assert record["frozen_commit"] == preparation.FROZEN_COMMIT
    assert record["gpu_submitted"] is False
    assert record["status"] == "GPU_HOLD"
    assert record["coordinator_temp_directory"] == tempfile.gettempdir()
    assert record["requires_before_publication"] == contracts[0]["requires_before_publication"]


def test_proposal_preserves_one_worker_four_processes_and_all_frozen_inputs(contracts):
    packet, expected = contracts
    record = preparation.proposal(packet, expected, preparation.SL_COORDINATOR)
    argv = record["argv_proposal_only"]
    assert record["process_order"] == ["master", "fix", "fix", "master"]
    assert record["action_count"] == record["worker_count"] == 1
    assert argv[argv.index("--") + 1:] == packet["entrypoint"]
    assert "--tag" not in argv and "--here" not in argv and "--anywhere" not in argv
    for name in ("cases", "readset", "native_files", "source", "resources", "entrypoint"):
        assert record["unchanged"][name] == packet[name]
    exported = dict(value.split("=", 1) for flag, value in zip(argv, argv[1:]) if flag == "--env")
    assert {key: exported[key] for key in packet["environment"]} == packet["environment"]
    assert all(exported[key] == "1" for key in preparation.NATIVE_THREADS)
    assert exported["BENCH_STRICT_STAGED"] == "1"
    assert record["placement"]["hostname_pin"] is None


@pytest.mark.parametrize("defect", ["missing_case", "wrong_native", "changed_budget",
                                   "swapped_arms", "changed_output", "timing", "origin_routing",
                                   "wrong_sdk", "wrong_reader", "missing_mount", "floating_image"])
def test_changed_numeric_contract_refuses_before_any_invocation(contracts, defect):
    packet, expected = deepcopy(contracts)
    if defect == "missing_case":
        packet["cases"].pop()
    elif defect == "wrong_native":
        packet["native_files"]["fix"]["sha256"] = "0" * 64
    elif defect == "changed_budget":
        packet["resources"]["aggregate_memory_gib"] = 8
    elif defect == "swapped_arms":
        packet["entrypoint"][3:] = ["fix", "master"]
    elif defect == "changed_output":
        packet["entrypoint"][2] += "-new"
    elif defect == "timing":
        packet["environment"]["AB_BENCH_ARGS"] = "--iters 30"
    elif defect == "origin_routing":
        packet["environment"]["AB_ROUTED"] = "experts.R1024.L10"
    elif defect == "wrong_sdk":
        packet["environment"]["PB_CLIENT_ROOT"] = "/home/rob/old-pb"
    elif defect == "wrong_reader":
        packet["environment"]["AB_INPUT_MANIFEST"] = "/mnt/shared/other-inputs.json"
    elif defect == "missing_mount":
        packet["environment"]["BENCH_RO_MOUNTS"] = ""
    elif defect == "floating_image":
        packet["image"] = packet["environment"]["ORACLE_IMAGE"] = "image:latest"
    with pytest.raises(ValueError):
        preparation.proposal(packet, expected, preparation.SL_COORDINATOR)


def test_sp_or_alternate_nas_coordinator_is_not_this_proposal(contracts):
    for checkout in (preparation.OLD_COORDINATOR, "/mnt/shared/new-coordinator"):
        with pytest.raises(ValueError, match="owned frozen SL checkout"):
            preparation.proposal(*contracts, checkout)


def test_preparation_never_promotes_literal_two_box_floor_or_root_go(contracts):
    packet, expected = contracts
    record = preparation.proposal(packet, expected, preparation.SL_COORDINATOR)
    assert "Root exact #793 GPU GO" in record["requires_before_publication"]
    assert "fresh floors >=5%, capacity and quiet/residency checks, timestamped both-Spark CPU/power/clocks/residency" in record["requires_before_publication"]
    assert record["status"] == "GPU_HOLD"
    assert "not a measured peak or hard cap" in record["allowance_scope"]


def test_actual_published_pbrun_class_scope_is_not_submitter_pin():
    # Exercise the real published owner, not a second placement implementation.
    path = Path(preparation.PB_ROOT) / "tools/pbrun.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("stageprev_live_pbrun", path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    from prismabuild.client import SDK_VERSION
    assert SDK_VERSION == 4
    owner.require_host_class_scope(measurement=True, host_class="gb10", transport="pool")
    for hostname in ("sparky", "sparklina"):
        assert owner.placement_tags(Path(preparation.SL_COORDINATOR), explicit=["gb10"],
                                    here=False, hostname=hostname, portable_checkout=True,
                                    command=["bash", "experiments/t8r_speed/ab_arms.sh"]) == ["gb10"]
        # Preserve a genuine --here restriction; do not reinterpret it as class.
        assert owner.placement_tags(Path(preparation.SL_COORDINATOR), explicit=["gb10"],
                                    here=True, hostname=hostname) == ["gb10", hostname]
    with pytest.raises(SystemExit, match="drop --anywhere"):
        owner.require_host_class_scope(measurement=True, host_class="gb10", transport="pool", anywhere=True)
