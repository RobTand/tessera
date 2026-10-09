"""A small loader consumer for CPU byte-observer tests, not a vLLM qualification."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace


def loader(monkeypatch, root: Path, *, device="cpu", reject=False, copied=False):
    import torch
    from safetensors.torch import load_file, save_file
    from tessera.serving import endpoint_runtime as er

    root.mkdir(exist_ok=True)
    save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(root / "model.safetensors"))

    def default_weight_loader(parameter, loaded):
        parameter.data.copy_(loaded)

    module = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    module.default_weight_loader = default_weight_loader
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, device=device), requires_grad=False)

            def load(param, raw):
                if reject:
                    raise ValueError("original numerical gate refuses this tensor")
                param.data.copy_(raw)

            self.weight.weight_loader = load

    class Base:
        def load_model(self, config, model_config):
            model = Model()
            self.load_weights(model, model_config)
            return model

    class Default(Base):
        def _prepare_weights(self, path):
            return path, [str(Path(path) / "model.safetensors")], True, "model.safetensors.index.json"

        def _get_weights_iterator(self, source):
            _, paths, _, _ = self._prepare_weights(source.model_or_path)
            for name, tensor in load_file(paths[0]).items():
                yield source.prefix + name, tensor

        def load_weights(self, model, model_config):
            source = SimpleNamespace(model_or_path=model_config.model, prefix="")
            for name, raw in self._get_weights_iterator(source):
                if copied:
                    raw = raw.clone()
                parameter = dict(model.named_parameters())[name]
                parameter.weight_loader(parameter, raw)

    er.install_loader_hooks(Base, Default)
    config = SimpleNamespace(model=str(root))
    return Default(), config


class Tokenizer:
    def __init__(self, root):
        import json
        from tokenizers import Tokenizer as Backend

        self.name_or_path = str(root)
        self.backend_tokenizer = Backend.from_file(str(root / "tokenizer.json"))
        config = json.loads((root / "tokenizer_config.json").read_text())
        self.bos_token_id = self.backend_tokenizer.token_to_id(config["bos_token"])

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()
