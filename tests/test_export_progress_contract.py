"""Export tools publish durable progress without scheduler-specific channels."""
from pathlib import Path


def test_the_export_tools_name_no_prismabuild_progress_channel():
    root=Path(__file__).resolve().parents[1]
    for name in ('tools/run_glm_cached_cpu_export.py','tools/pq2459_export_progress.py','experiments/glm_routed_owner_inputs.py'):
        text=(root/name).read_text()
        assert 'PRISMABUILD_ACTION_PROGRESS' not in text and 'runpy.run_path' not in text,name
