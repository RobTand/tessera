"""CPU checks for ``tessera.serving.glm53_graphs`` (tessera#702).

Without vLLM: config stand-ins for the operator pin and the branch plan, and a
toy runner that keeps the three facts of vLLM's V2 runner the frozen branch
comes from -- a FULL capture builds its metadata at ``max_seq_len =
max_model_len``, the indexer decides its branch on the host from that value,
and a replay re-runs no Python.  The toy's indexer is GLM's rule
(``max_seq_len <= index_topk`` takes the causal fill), so a graph is held to
the branch the eager step takes at every ``max_seq_len`` a serve can run.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from tessera.serving import glm53_graphs as gg
from tessera.serving import graph_equivalence as ge

TOPK = 2048          # GLM-5.3's index_topk
RELEASE_LEN = 8448   # the release serve's max_model_len


# ------------------------------------------------------------------ pure class rule


@pytest.mark.parametrize("thresholds, max_model_len, bounds", [
    ({TOPK}, RELEASE_LEN, (TOPK, RELEASE_LEN)),
    ({TOPK}, TOPK, (TOPK,)),               # at the threshold: one class, vLLM's own capture
    ({TOPK}, 1024, (1024,)),
    ({TOPK, 512}, RELEASE_LEN, (512, TOPK, RELEASE_LEN)),
])
def test_capture_bounds_split_at_every_threshold_below_max_model_len(thresholds, max_model_len,
                                                                   bounds):
    assert ge.branch_capture_bounds(thresholds, max_model_len) == bounds


@pytest.mark.parametrize("max_seq_len", [1, 5, TOPK - 1, TOPK, TOPK + 1, 4000, RELEASE_LEN])
def test_a_step_replays_the_class_whose_capture_took_its_branch(max_seq_len):
    bounds = ge.branch_capture_bounds({TOPK}, RELEASE_LEN)
    bound = ge.branch_bound(bounds, max_seq_len)
    # The indexer's own rule, at the step and at the capture of the class it replays.
    assert (max_seq_len <= TOPK) == (bound <= TOPK)
    assert max_seq_len <= bound


def test_a_step_beyond_every_bound_is_refused():
    with pytest.raises(ValueError, match="exceeds every captured bound"):
        ge.branch_bound((TOPK, RELEASE_LEN), RELEASE_LEN + 1)


# ------------------------------------------------------------------ config stand-ins


def _config(*, quantization="tessera", arch="Glm5NextForConditionalGeneration", eager=False,
            mode_name="FULL_DECODE_ONLY", max_model_len=RELEASE_LEN, topk=TOPK, spec=None,
            max_num_seqs=8, sizes=(1, 2, 3, 4, 5, 6, 7, 8), custom_ops=None, priority=None,
            parallel=None):
    mode = NS(name=mode_name)
    par = dict(pipeline_parallel_size=1, data_parallel_size=1,
               prefill_context_parallel_size=1, decode_context_parallel_size=1)
    par.update(parallel or {})
    prio = NS(rms_norm=[], fused_add_rms_norm=[])
    for op, order in (priority or {}).items():
        setattr(prio, op, list(order))
    return NS(
        model_config=NS(quantization=quantization, enforce_eager=eager,
                        max_model_len=max_model_len,
                        hf_config=NS(architectures=[arch]),
                        hf_text_config=NS(index_topk=topk)),
        compilation_config=NS(cudagraph_mode=mode, custom_ops=list(custom_ops or []),
                              cudagraph_capture_sizes=list(sizes),
                              splitting_ops_contain_attention=lambda: False, backend="inductor"),
        kernel_config=NS(ir_op_priority=prio),
        scheduler_config=NS(max_num_seqs=max_num_seqs),
        parallel_config=NS(**par),
        speculative_config=spec,
        lora_config=None)


EAGER_PRIORITY = {"rms_norm": ["vllm_c", "native"], "fused_add_rms_norm": ["vllm_c", "native"]}


def _eager_priority(config):
    return EAGER_PRIORITY


# ------------------------------------------------------------------ cause 1: operators


def test_operator_pin_fills_what_a_graph_serve_left_unset():
    cfg = _config()
    pinned = gg.pin_eager_operators(cfg, eager_priority=_eager_priority)
    assert cfg.compilation_config.custom_ops == ["all"]
    assert cfg.kernel_config.ir_op_priority.rms_norm == ["vllm_c", "native"]
    assert cfg.kernel_config.ir_op_priority.fused_add_rms_norm == ["vllm_c", "native"]
    assert len(pinned) == 3
    # Once resolved this way, the one rule that judges operators finds no gap.
    assert ge.op_implementation_gap(cfg) is None


def test_operator_pin_leaves_a_serves_own_choice_for_the_worker_to_judge():
    cfg = _config(custom_ops=["none"], priority={"rms_norm": ["native"]})
    gg.pin_eager_operators(cfg, eager_priority=_eager_priority)
    assert cfg.compilation_config.custom_ops == ["none"]
    assert cfg.kernel_config.ir_op_priority.rms_norm == ["native"]
    assert ge.op_implementation_gap(cfg) is not None


@pytest.mark.parametrize("over", [dict(eager=True), dict(quantization="fp8"),
                                  dict(arch="Qwen3ForCausalLM")])
def test_operator_pin_touches_only_tessera_glm5next_graph_serves(over):
    cfg = _config(**over)
    assert gg.pin_eager_operators(cfg, eager_priority=_eager_priority) == []
    assert cfg.compilation_config.custom_ops == []
    assert cfg.kernel_config.ir_op_priority.rms_norm == []


# ------------------------------------------------------------------ cause 2: the plan


def test_release_serve_needs_one_class_per_side_of_index_topk():
    assert gg.branch_plan(_config()) == ((TOPK, RELEASE_LEN), [])


def test_release_serve_with_mtp_k1_reads_the_drafts_index_topk_too():
    draft = NS(hf_text_config=NS(index_topk=TOPK))
    spec = NS(method="mtp", num_speculative_tokens=1, draft_model_config=draft)
    assert gg.branch_plan(_config(spec=spec, sizes=range(1, 17))) == ((TOPK, RELEASE_LEN), [])


@pytest.mark.parametrize("over", [dict(eager=True), dict(max_model_len=TOPK),
                                  dict(mode_name="NONE"), dict(quantization="fp8")])
def test_no_plan_where_stock_capture_is_already_eagers(over):
    assert gg.branch_plan(_config(**over)) == (None, [])


@pytest.mark.parametrize("over, needle", [
    (dict(spec=NS(method="mtp", num_speculative_tokens=2,
                  draft_model_config=NS(hf_text_config=NS(index_topk=TOPK)))),
     "2 speculative tokens"),
    (dict(max_num_seqs=128, sizes=range(1, 129)), "sampled path"),
    (dict(max_model_len=gg.PERSISTENT_TOPK_RADIX_THRESHOLD + 1), "radix path"),
    (dict(parallel=dict(pipeline_parallel_size=2)), "pipeline_parallel_size 2"),
    (dict(parallel=dict(data_parallel_size=2)), "data_parallel_size 2"),
])
def test_a_serve_outside_the_inspected_structure_names_why(over, needle):
    bounds, reasons = gg.branch_plan(_config(**over))
    assert bounds is not None
    assert any(needle in r for r in reasons), reasons


def test_install_refuses_by_name_when_the_operators_differ(monkeypatch):
    monkeypatch.setattr(gg, "_STATE", NS(installed=False, bounds=None, capture_bound=None,
                                         bound_applied=False, step_max_seq_len=None))
    cfg = _config(custom_ops=["none"], priority={"rms_norm": ["native"],
                                                 "fused_add_rms_norm": ["native"]})
    with pytest.raises(RuntimeError, match=r"custom_ops resolves to \['none'\]"):
        gg.install_branch_capture(cfg, breakable=lambda: False)


def test_install_refuses_an_uninspected_interface(monkeypatch):
    monkeypatch.setattr(gg, "_STATE", NS(installed=False, bounds=None, capture_bound=None,
                                         bound_applied=False, step_max_seq_len=None))
    monkeypatch.setattr(gg, "import_modules", lambda names: ((NS(),) * len(names), ""))
    monkeypatch.setattr(gg, "match_modules", lambda *a: (None, "no inspected interface matches"))
    cfg = _config(custom_ops=["all"], priority=EAGER_PRIORITY)
    with pytest.raises(RuntimeError, match="no inspected interface matches.*--enforce-eager"):
        gg.install_branch_capture(cfg, breakable=lambda: False)


# ------------------------------------------------------------------ cause 2: the toy runner


class _Indexer:
    """GLM's host branch: the causal fill at ``max_seq_len <= index_topk``, else logits + top-k."""

    @staticmethod
    def branch(max_seq_len):
        return "fill" if max_seq_len <= TOPK else "topk"


def _toy_vllm():
    """A toy ``cudagraph_utils`` / ``mamba_hybrid`` / speculator trio with vLLM's capture semantics."""
    model_states = NS()
    model_states.build_attn_metadata = lambda **kw: NS(
        max_seq_len=kw["max_seq_len"], branch=_Indexer.branch(kw["max_seq_len"]))

    class State:
        max_model_len = RELEASE_LEN

        def prepare_attn(self, *, for_capture, step_max_seq_len=None):
            # MambaHybridModelState.prepare_attn: the worst case at capture, the step's otherwise.
            seq = self.max_model_len if for_capture else step_max_seq_len
            return model_states.build_attn_metadata(max_seq_len=seq,
                                                    for_cudagraph_capture=for_capture)

    class CudaGraphManager:
        def __init__(self, descs):
            self.graphs, self._graphs_captured, self.descs = {}, False, descs

        def capture(self, create_forward_fn):
            for desc in self.descs:
                assert desc not in self.graphs, "Graph already captured"
                # The "graph" is the branch the capture-time forward took: a replay re-runs no Python.
                self.graphs[desc] = create_forward_fn(desc).branch
            self._graphs_captured = True

        def run_fullgraph(self, desc):
            return self.graphs[desc]

        def release_graphs(self):
            self.graphs.clear()
            self._graphs_captured = False

    class ModelCudaGraphManager(CudaGraphManager):
        def capture(self, model, model_state):
            super().capture(lambda desc: model_state.prepare_attn(for_capture=True))

    class SpeculatorCudaGraphManager(CudaGraphManager):
        def capture(self, forward_fn, model_state):
            super().capture(lambda desc: model_state.prepare_attn(for_capture=True))

    utils = NS(CudaGraphManager=CudaGraphManager, ModelCudaGraphManager=ModelCudaGraphManager,
               _extrapolate_full_graph_memory=lambda samples, total: total)
    return utils, model_states, SpeculatorCudaGraphManager, State


class _Model:
    def __init__(self, *topks):
        self._mods = [type("SparseAttnIndexerKpool", (), {"topk_tokens": t})() for t in topks]

    def modules(self):
        return iter(self._mods)


def _install(monkeypatch):
    utils, states, spec_cls, state_cls = _toy_vllm()
    monkeypatch.setattr(gg, "_STATE", NS(installed=True, bounds=(TOPK, RELEASE_LEN),
                                         capture_bound=None, bound_applied=False,
                                         step_max_seq_len=None))
    monkeypatch.setattr(gg, "REPLAYS", type(gg.REPLAYS)())
    monkeypatch.setattr(gg, "CAPTURED", {})
    states.build_attn_metadata = gg._record_build(states.build_attn_metadata)
    gg._patch_managers(utils, (utils.ModelCudaGraphManager, spec_cls))
    return utils, spec_cls, state_cls


def _step(state, manager, max_seq_len):
    state.prepare_attn(for_capture=False, step_max_seq_len=max_seq_len)  # execute_model
    return manager.run_fullgraph("decode-b1")                            # then the replay


STEPS = [1, 5, 100, TOPK - 1, TOPK, TOPK + 1, 4000, RELEASE_LEN]


def test_stock_capture_freezes_the_long_branch():
    """The defect the toy must reproduce, or it proves nothing: stock graphs take top-k at every step."""
    utils, _, _, state_cls = _toy_vllm()
    state, manager = state_cls(), utils.ModelCudaGraphManager(["decode-b1"])
    manager.capture(_Model(TOPK), state)
    taken = {seq: _step(state, manager, seq) for seq in STEPS}
    eager = {seq: _Indexer.branch(seq) for seq in STEPS}
    assert taken != eager
    assert set(taken.values()) == {"topk"}


@pytest.mark.parametrize("seq", STEPS)
def test_per_class_capture_replays_eagers_branch_at_every_context(monkeypatch, seq):
    utils, spec_cls, state_cls = _install(monkeypatch)
    state = state_cls()
    target = utils.ModelCudaGraphManager(["decode-b1"])
    draft = spec_cls(["decode-b1"])
    target.capture(_Model(TOPK), state)
    draft.capture(None, state)
    assert _step(state, target, seq) == _Indexer.branch(seq)
    # The draft prefill replays on the target step's metadata, so the same record selects it.
    assert draft.run_fullgraph("decode-b1") == _Indexer.branch(seq)
    bound = ge.branch_bound((TOPK, RELEASE_LEN), seq)
    assert gg.REPLAYS[("ModelCudaGraphManager", bound)] == 1
    assert gg.REPLAYS[("SpeculatorCudaGraphManager", bound)] == 1
    assert gg.CAPTURED == {"ModelCudaGraphManager": {TOPK: 1, RELEASE_LEN: 1},
                           "SpeculatorCudaGraphManager": {TOPK: 1, RELEASE_LEN: 1}}


def test_a_replay_with_no_recorded_step_is_refused(monkeypatch):
    utils, _, state_cls = _install(monkeypatch)
    manager = utils.ModelCudaGraphManager(["decode-b1"])
    manager.capture(_Model(TOPK), state_cls())
    with pytest.raises(RuntimeError, match="no step max_seq_len"):
        manager.run_fullgraph("decode-b1")


def test_a_capture_that_bypasses_the_inspected_builder_is_refused(monkeypatch):
    utils, _, state_cls = _install(monkeypatch)

    class Bypass(state_cls):
        def prepare_attn(self, *, for_capture, step_max_seq_len=None):
            return NS(branch=_Indexer.branch(self.max_model_len))  # never calls build_attn_metadata

    with pytest.raises(RuntimeError, match="would freeze the long-context branch"):
        utils.ModelCudaGraphManager(["decode-b1"]).capture(_Model(TOPK), Bypass())


def test_an_indexer_branching_at_an_unsplit_threshold_is_refused(monkeypatch):
    utils, _, state_cls = _install(monkeypatch)
    with pytest.raises(RuntimeError, match=r"indexers branch at \[1024\]"):
        utils.ModelCudaGraphManager(["decode-b1"]).capture(_Model(TOPK, 1024), state_cls())


def test_release_drops_every_class(monkeypatch):
    utils, _, state_cls = _install(monkeypatch)
    manager = utils.ModelCudaGraphManager(["decode-b1"])
    manager.capture(_Model(TOPK), state_cls())
    manager.release_graphs()
    assert manager._tessera_graphs_by_bound is None and manager.graphs == {}


def test_memory_profile_counts_every_class(monkeypatch):
    utils, _, _ = _install(monkeypatch)
    assert utils._extrapolate_full_graph_memory([1, 1], 8) == 16


def test_operators_are_judged_even_where_one_class_suffices(monkeypatch):
    """Cause 1 does not depend on cause 2: a graph serve at max_model_len <= index_topk that runs
    other operators than eager is refused too (sbG1 departed at step 0 at any context)."""
    monkeypatch.setattr(gg, "_STATE", NS(installed=False, bounds=None, capture_bound=None,
                                         bound_applied=False, step_max_seq_len=None))
    cfg = _config(max_model_len=TOPK, custom_ops=["none"],
                  priority={"rms_norm": ["native"], "fused_add_rms_norm": ["native"]})
    with pytest.raises(RuntimeError, match=r"custom_ops resolves to \['none'\]"):
        gg.install_branch_capture(cfg, breakable=lambda: False)
    # And a one-class serve that runs eager's operators installs nothing.
    assert gg.install_branch_capture(_config(max_model_len=TOPK, custom_ops=["all"],
                                             priority=EAGER_PRIORITY),
                                     breakable=lambda: False) is False


# ------------------------------------------------------------------ cause 3: padded replays


def test_capture_sizes_left_unset_are_pinned_to_every_decode_count():
    cfg = _config(sizes=())
    cfg.compilation_config.cudagraph_capture_sizes = None
    assert gg.pin_unpadded_capture_sizes(cfg) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ge.padded_token_counts(cfg, cfg.compilation_config.cudagraph_mode) == []


def test_with_a_drafter_the_pinned_sizes_are_whole_requests():
    spec = NS(method="mtp", num_speculative_tokens=1,
              draft_model_config=NS(hf_text_config=NS(index_topk=TOPK)))
    cfg = _config(spec=spec, max_num_seqs=4, sizes=())
    cfg.compilation_config.cudagraph_capture_sizes = None
    assert gg.pin_unpadded_capture_sizes(cfg) == [2, 4, 6, 8]


def test_capture_sizes_a_serve_set_are_left_for_the_worker_to_judge():
    cfg = _config(sizes=(1, 2, 4, 8))
    assert gg.pin_unpadded_capture_sizes(cfg) == []
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]


def test_a_padded_decode_replay_is_an_eager_gap():
    """vLLM's default sizes [1, 2, 4, 8] replay batches of 3, 5, 6 and 7 in a larger graph; at
    max_model_len 8448 that moved b5 and b7 (rG1, rGR, rP1 of 2026-10-04: 46/48)."""
    gaps = gg.eager_gaps(_config(sizes=(1, 2, 4, 8), custom_ops=["all"], priority=EAGER_PRIORITY),
                         breakable=False)
    assert any("[3, 5, 6, 7]" in g for g in gaps), gaps


def test_breakable_piecewise_graphs_are_an_eager_gap():
    cfg = _config(mode_name="FULL_AND_PIECEWISE", custom_ops=["all"], priority=EAGER_PRIORITY)
    assert any("breakable" in g for g in gg.eager_gaps(cfg, breakable=True))
    assert gg.eager_gaps(cfg, breakable=False) == []


def test_an_unpadded_eager_operator_serve_has_no_gap():
    assert gg.eager_gaps(_config(custom_ops=["all"], priority=EAGER_PRIORITY), breakable=False) == []
