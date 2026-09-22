"""Per-tensor byte identity of the v3 re-export against the v2 artifact.

The manifest had to move (tessera#557 prices resident rows the exporter did not
price before).  Nothing else may have.  This hashes every tensor of both
checkpoints independently, so a difference names the tensor rather than the
file.
"""
import hashlib
import json
import sys

from safetensors import safe_open

A, B = sys.argv[1], sys.argv[2]


def digests(path):
    out = {}
    with safe_open(path + "/model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            out[key] = (str(tensor.dtype), tuple(tensor.shape),
                        hashlib.sha256(tensor.contiguous().view(-1).view(
                            __import__("torch").uint8).numpy().tobytes()).hexdigest())
    return out


a, b = digests(A), digests(B)
only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
differ = sorted(k for k in set(a) & set(b) if a[k] != b[k])
report = {"schema": "tessera.export_byte_identity.v1", "a": A, "b": B,
          "tensors_a": len(a), "tensors_b": len(b),
          "only_in_a": only_a, "only_in_b": only_b,
          "differing": [{"tensor": k, "a": a[k], "b": b[k]} for k in differ],
          "identical": len(set(a) & set(b)) - len(differ)}
print(json.dumps(report, indent=1))
sys.exit(0 if not (only_a or only_b or differ) else 1)
