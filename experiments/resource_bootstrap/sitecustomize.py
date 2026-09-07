"""Opt-in early bootstrap; failure must stop Python instead of being ignored."""
import os
import sys

if os.environ.get("TESSERA_ENGINE_RESOURCE_PLAN"):
    try:
        from experiments.full_engine_bootstrap import start
        start()
    except BaseException as exc:
        print(f"Fatal full-engine resource bootstrap: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        os._exit(78)
