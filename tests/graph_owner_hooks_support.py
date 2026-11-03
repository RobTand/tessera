"""Shared vLLM stand-ins for the real graph-owner observer hooks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


HOOK = ROOT / "experiments/glm53_508_graph_qual/digest/usercustomize.py"


MANAGER = '''
from dataclasses import dataclass
from types import SimpleNamespace as NS
@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int = 1
    @property
    def cg_mode(self):
        return NS(name="FULL")
class CudaGraphManager:
    def __init__(self):
        self.graphs = {}
        self._graphs_captured = False
        self.fail = False
        self.incomplete = False
        self.replay_fail = False
        self.result = object()
        self.fail_bound = self.incomplete_bound = self.bypass_bound = None
    def capture(self, *args):
        import sys
        gg = sys.modules.get("tessera.serving.glm53_graphs")
        bound = gg._STATE.capture_bound if gg is not None else None
        if bound is not None and bound != self.bypass_bound:
            gg._record_build(lambda **kw: None)(max_seq_len=8448, for_cudagraph_capture=True)
        self.graphs = {Descriptor(1): object(), Descriptor(2): object()}
        if self.fail or (bound is not None and bound == self.fail_bound):
            raise ValueError("capture-sentinel")
        self._graphs_captured = not (self.incomplete or
            (bound is not None and bound == self.incomplete_bound))
        return self.result
    def captured_token_counts(self):
        return [d.num_tokens for d in self.graphs]
    def dispatch(self, desc):
        return desc
    def run_fullgraph(self, desc):
        if self.replay_fail:
            raise LookupError("replay-sentinel")
        self.graphs[desc]  # a real unknown descriptor cannot be replayed
        return self.result
    def release_graphs(self):
        self.graphs = {}
        self._graphs_captured = False
class ModelCudaGraphManager(CudaGraphManager):
    def capture(self, *args):
        return super().capture(*args)
def _extrapolate_full_graph_memory(samples, total):
    return total
'''


RUNNER = '''
class GPUModelRunner:
    def execute_model(self, manager, desc, *, replay=True, dummy_run=False):
        selected = manager.dispatch(desc)
        return manager.run_fullgraph(selected) if replay else selected
    def sample_tokens(self, manager, desc):
        return manager.run_fullgraph(manager.dispatch(desc))
'''
