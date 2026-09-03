"""Which vision-tower prefixes is a quantization config actually offered?

Constructs the pinned build's GLM-5.3-Flash vision tower on meta device with a
RECORDING quant config, so the prefixes are observed rather than argued from
the source.  Then does the same through the model class's own call site.
"""
import json, torch
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm.config import VllmConfig, DeviceConfig, set_current_vllm_config
_ctx = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu")))
_ctx.__enter__()
init_distributed_environment(world_size=1, rank=0,
                             distributed_init_method="tcp://127.0.0.1:29591",
                             local_rank=0, backend="gloo")
initialize_model_parallel(1, 1)

asked = []

class Recorder(QuantizationConfig):
    def get_name(self): return "recorder"
    def get_supported_act_dtypes(self): return [torch.bfloat16]
    @classmethod
    def get_min_capability(cls): return 0
    @staticmethod
    def get_config_filenames(): return []
    @classmethod
    def from_config(cls, c): return cls()
    def get_quant_method(self, layer, prefix):
        asked.append((prefix, type(layer).__name__, isinstance(layer, LinearBase)))
        return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

from vllm.transformers_utils.config import get_config
cfg = get_config("/models/GLM-5.3-Flash-4layer", trust_remote_code=False)
text_cfg = cfg.text_config
vis_cfg = cfg.vision_config
from vllm.models.glm5next.nvidia.multimodal import Glm5NextVisionTransformer

with torch.device("meta"):
    tower = Glm5NextVisionTransformer(text_cfg, vis_cfg, quant_config=Recorder(),
                                      prefix="visual")
lin = sorted(n for n, m in tower.named_modules() if isinstance(m, LinearBase))
print("LINEARBASE MODULES:", json.dumps(lin, indent=0))
print("OFFERED PREFIXES:", json.dumps(sorted(set(p for p, _, _ in asked)), indent=0))
print("NON LINEAR OFFERS:", [a for a in asked if not a[2]])

# And the call site the model class actually uses: quant_config=None.
asked.clear()
with torch.device("meta"):
    tower_none = Glm5NextVisionTransformer(text_cfg, vis_cfg, quant_config=None,
                                           prefix="visual")
print("OFFERS WITH quant_config=None:", asked)
print("LINEARBASE COUNT (none case):",
      sum(1 for _n, m in tower_none.named_modules() if isinstance(m, LinearBase)))
print("QUANT METHODS (none case):", sorted({type(m.quant_method).__name__
      for _n, m in tower_none.named_modules() if isinstance(m, LinearBase)}))
