"""Optional bounded before/after intake evidence for the native TP2 control.

Only observation: the stock loader, parser, sharder and collective run unchanged.
The trace covers construction plus the first three projection callbacks; peak
allocator counters cover the complete owner's intake, before its stock oracle.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import time


class IntakeObservation(AbstractContextManager):
    def __init__(self, root, options):
        self.root, self.options = root, options
        self.profile = None
        self.count = 0
        self.record = {}
        if options is not None:
            if (set(options) != {'expected_mode', 'profile_initial_callbacks'}
                    or options['expected_mode'] not in ('padded', 'rank_local')
                    or options['profile_initial_callbacks'] != 3):
                raise ValueError('native intake observation requires padded/rank_local and exactly 3 callbacks')

    def __enter__(self):
        if self.options is not None:
            import torch
            self.record.update(options=self.options, started_epoch=time.time())
            self.profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=False, with_stack=False)
            self.profile.__enter__()
        return self

    def __exit__(self, *exc):
        if self.options is not None:
            self.stop_profile(*exc)
            self.record.update(finished_epoch=time.time(), completed_without_exception=exc[0] is None)
            (self.root/'intake-observation.json').write_text(json.dumps(self.record, indent=2)+'\n')
        return False

    def stop_profile(self, *exc):
        if self.profile is None:
            return
        profile, self.profile = self.profile, None
        profile.__exit__(*(exc or (None, None, None)))
        trace = self.root/'intake-initial-callbacks-trace.json'
        profile.export_chrome_trace(str(trace))
        self.record['profile'] = {
            'path':str(trace), 'sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
            'scope':'construction plus initial projection callbacks, not full intake timing',
            'completed_callbacks':self.count,
            'events':[{'key':x.key, 'calls':x.count, 'cpu_time_us':x.cpu_time_total,
                'device_time_us':x.device_time_total,
                'self_device_memory_bytes':x.self_device_memory_usage} for x in profile.key_averages()]}

    def after_create(self, layer, method):
        if self.options is None:
            return
        sizes = {name:getattr(layer, name).numel() for name in ('w13_wire', 'w2_wire')}
        local = self.options['expected_mode'] == 'rank_local'
        assert (sum(sizes.values()) == 0) == local, sizes
        assert (getattr(method, '_rank_local_intake', None) is not None) == local
        self.record['construction_wire_parameter_bytes'] = sizes

    def after_callback(self, method):
        self.count += 1
        if self.options is None:
            return
        if self.count == self.options['profile_initial_callbacks']:
            self.stop_profile()
            self.record['after_initial_callbacks'] = self.ownership(method)

    def ownership(self, method):
        intake = getattr(method, '_rank_local_intake', None)
        modules = ([] if intake is None else
            [m for group in intake.prepared.values() for expert in group for m in expert if m is not None])
        sizes = [m.wire_bytes_resident() + m.rows * 4 for m in modules]
        return {'completed_loader_callbacks':self.count, 'prepared_local_projections':len(modules),
                'prepared_local_bytes':sum(sizes)}

    def after_load(self, method, expected_count):
        if self.options is None:
            return
        assert self.count == expected_count, (self.count, expected_count)
        row = self.ownership(method)
        expected_prepared = expected_count if self.options['expected_mode'] == 'rank_local' else 0
        assert row['prepared_local_projections'] == expected_prepared, row
        self.record['after_complete_load'] = row

    def after_finalize(self, method):
        if self.options is None:
            return
        assert getattr(method, '_rank_local_intake', None) is None
        self.record['final_owner_bytes'] = method.research_resident_bytes()
