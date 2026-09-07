"""Stock worker extension for whole-census timing observation, without admission."""
import hashlib
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from experiments.full_engine_kv import inspect_worker_kv
from experiments.full_engine_timing_boundaries import resolve_apply_boundaries
from experiments.full_engine_timings import FullEngineTimingRecorder
from experiments.full_engine_worker import full_engine_runtime_observation


class TimingCaptureWorker(Worker):
    def __init__(self, *args, **kwargs):
        self._timing_plan = json.loads(Path(os.environ["TESSERA_ENGINE_TIMING_PLAN"]).read_text())
        self._timing_recorder = None
        self._timing_boundaries = None
        self._timing_kv = None
        super().__init__(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._timing_boundaries = resolve_apply_boundaries(self.model_runner.model, self._timing_plan["canonical_roster"])
        return result

    def initialize_from_config(self, kv_cache_config):
        result = super().initialize_from_config(kv_cache_config)
        self._timing_kv = inspect_worker_kv(self, self._timing_plan["selected_configuration"]["capacity_assertions"],
                                          received=kv_cache_config)
        if not self._timing_kv["capacity_assertions"]["passed"]:
            raise RuntimeError("timing worker KV capacity differs from selected assertions")
        return result

    def timing_capture_arm(self, arm, sample):
        if self._timing_recorder is not None:
            raise RuntimeError("timing worker is already armed")
        if type(sample) is not int or not 0 <= sample < self._timing_plan["timing_samples"]:
            raise ValueError("timing sample is outside the explicit plan")
        directory = Path(self._timing_plan["output_directory"]) / f"worker-{os.getpid()}" / f"sample-{sample}-{arm}"
        self._timing_recorder = FullEngineTimingRecorder(self._timing_boundaries, arm=arm, output=directory)
        return {"pid": os.getpid(), "arm": arm, "sample": sample, "native_units": len(self._timing_boundaries)}

    def execute_model(self, scheduler_output):
        if self._timing_recorder is not None:
            if scheduler_output.total_num_scheduled_tokens == 0:
                with self._timing_recorder.housekeeping(scheduler_output):
                    return super().execute_model(scheduler_output)
            runner = self.model_runner
            self._timing_recorder.begin_step(scheduler_output, runner.main_stream, runner.output_copy_stream)
        return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        result = super().sample_tokens(grammar_output)
        if self._timing_recorder is not None:
            self._timing_recorder.end_step(result)
        return result

    def timing_capture_finish(self):
        if self._timing_recorder is None:
            raise RuntimeError("timing worker was not armed")
        def runtime():
            result = full_engine_runtime_observation(self._timing_plan)
            result["source"]["full_engine_timing_worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            result["kv_configuration"] = self._timing_kv
            return result
        result = self._timing_recorder.finish(identity=self._timing_plan["identity"], runtime=runtime)
        self._timing_recorder = None
        return result
