"""Archived with the ts113 sparklina campaign scripts (T-SLICE5): the four ts113 tests moved out of test_serve_build_identity.py.
Not collected; imports and helpers are those of the original module."""
_TS113_CAMPAIGN = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ts113_sparklina_campaign.sh"
)
_TS113_PREFLIGHT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ts113_sparklina_preflight.sh"
)


def test_ts113_launch_gate_counts_source_roles_not_module_containers(tmp_path):
    """One GEMV launches per source role, while fallback refuses per module."""
    body = _TS113_CAMPAIGN.read_text()
    anchor = body.index("read -r DECODE_POSITIONS")
    start = body.index("<<'PY'\n", anchor) + len("<<'PY'\n")
    end = body.index("\nPY\n", start)
    program = body[start:end]

    closure = tmp_path / ".pbrun-closure.test.json"
    manifest = tmp_path / "manifest.json"
    contract = tmp_path / "runtime_contract.json"
    corpus = tmp_path / "corpus.json"
    identity = tmp_path / "CAMPAIGN_IDENTITY.json"
    closure.write_text(json.dumps({
        "cwd": ".",
        "dirty_sha256": hashlib.sha256(b"").hexdigest(),
        "head": "a" * 40,
    }))
    manifest.write_text(json.dumps({
        "modules": {
            "model.layers.0.qkv_proj": {
                "grid": "E4M3", "q256": 1024, "roles": ["q", "k", "v"],
            },
            "model.layers.0.gate_up_proj": {
                "grid": "E4M3", "q256": 1024, "roles": ["gate", "up"],
            },
        },
        "totals": {"modules": 2, "units": 5},
    }))
    contract.write_text(json.dumps({
        "formats": [{"grid": "E4M3", "family": "TESSERA_FP8"}],
        "lane_eligibility": {"cells": [{
            "platform": "sm_121",
            "structure": "dense",
            "regime": "decode",
            "family": "TESSERA_FP8",
            "rungs_q256": [1024],
            "requires_serve_flags": ["TESSERA_SERVE_MODE=streamed"],
            "executes": [{
                "symbol": "tessera_window_gemv::gemv",
                "decoder": "window_gemv",
            }],
        }]},
    }))
    corpus.write_text(json.dumps({
        "n_chunks": 1,
        "seqlen": 5,
        "chunks": [[1, 2, 3, 4, 5]],
        "contract_sha256": "b" * 64,
    }))

    result = subprocess.run(
        [sys.executable, "-", str(closure), "c" * 40, PIN,
         str(manifest), str(contract), str(corpus), str(identity), "2",
         "device:inode"],
        input=program,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert [int(value) for value in result.stdout.split()] == [2, 2, 5, 10]
    derived = json.loads(identity.read_text())["derived_gate"]
    assert derived["eligible_modules"] == 2
    assert derived["eligible_units"] == 5
    assert derived["expected_window_gemv_launches"] == 10
    assert derived["fallback_refusals"] == 2


def test_ts113_preflight_defaults_to_the_campaign_population():
    """The standalone preflight must guard the namespace the campaign writes."""
    campaign = _TS113_CAMPAIGN.read_text().splitlines()
    preflight = _TS113_PREFLIGHT.read_text().splitlines()

    def literal(lines, name):
        return next(line.split("=", 1)[1] for line in lines
                    if line.startswith(f"{name}="))

    def default(lines, name):
        value = literal(lines, name)
        return value.split(":-", 1)[1][:-1]

    assert default(preflight, "POP_ROOT") == literal(
        campaign, "PROMOTIONAL_POP_ROOT")
    assert default(preflight, "LOCAL_ROOT") == literal(
        campaign, "PROMOTIONAL_LOCAL_ROOT")


def test_ts113_extension_hashes_ignore_unrelated_private_tempdirs(tmp_path):
    from tessera.serving.ext import WINDOW_GEMV_MODULE_NAME

    body = _TS113_CAMPAIGN.read_text()
    start = body.index('  (\n    cd "$EXT_LANE"')
    end = body.index('  [ -s "$dir/extension-files.sha256" ]', start)
    root, out = tmp_path / "ext", tmp_path / "out"
    module = root / WINDOW_GEMV_MODULE_NAME
    module.mkdir(parents=True)
    (module / (WINDOW_GEMV_MODULE_NAME + ".so")).write_bytes(b"served extension")
    private = root / "private-vllm-temp"
    private.mkdir()
    private.chmod(0)
    out.mkdir()
    env = dict(os.environ, EXT_LANE=str(root), WT=str(ROOT), PY=sys.executable,
               dir=str(out))
    try:
        result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + body[start:end]],
                                env=env, capture_output=True, text=True)
    finally:
        private.chmod(0o700)
    assert result.returncode == 0, result.stderr
    manifest = (out / "extension-files.sha256").read_text()
    assert WINDOW_GEMV_MODULE_NAME + ".so" in manifest
    assert "private-vllm-temp" not in manifest


@pytest.mark.parametrize("seal", ["STAGE_SHA256", "STAGE_COMPLETE"])
def test_ts113_recovery_never_reseals_a_failed_existing_seal(tmp_path, seal):
    body = (ROOT / "experiments/ts113_resume_r6.sh").read_text()
    start = body.index('if ! verify_stage "$POP_ROOT/stages/A1"; then')
    end = body.index('\nrun_arm A2 ', start)
    stage = tmp_path / "stages" / "A1"
    stage.mkdir(parents=True)
    (stage / seal).write_text("an existing seal that no longer verifies")
    program = ("set -euo pipefail\nverify_stage() { return 1; }\n"
               "run_arm() { echo RECOVERED; }\n" + body[start:end])
    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                            env=dict(os.environ, POP_ROOT=str(tmp_path), ARMA="unused"))
    assert result.returncode != 0
    assert "RECOVERED" not in result.stdout
    assert (stage / seal).read_text() == "an existing seal that no longer verifies"


