"""Frozen import boundary before 8cf85dbc; collection only, never CUDA work.

The imports below are the original runtime test's import block, retained so
regression checks do not require historical Git objects on fleet snapshots.
"""
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import pytest
import torch
pytest.importorskip('vllm.v1.attention.backends.mla.flashinfer_mla_sparse')
from tessera.serving import mla_sparse_sm120 as module
from tessera.serving import mla_prefill
