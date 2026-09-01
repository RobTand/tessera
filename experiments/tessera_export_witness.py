"""Re-encode units of a written checkpoint from source, through its own config.

The exporter's defaults moved (span 2, LUT plane, refit 4, scale-weighted
trellis; 2026-09-01) after the 151 GiB GLM export was written.  "The export is
reproducible from source" is only true if a replay reads its settings off the
checkpoint's config rather than the exporter's defaults --
``encode_settings_from_config`` is that path, and this witness proves it on
real units: encode from the BF16 source with exactly those settings and
compare the artifact bytes byte for byte with the shard on disk.

    python experiments/tessera_export_witness.py /mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901 \
        --units 'layers\\.5\\.mlp\\.experts\\.0\\.(gate|up)_proj'
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.export import (_shard_holding, encode_linear, encode_settings_from_config,
                            grid_from_config, read_checkpoint_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--units", default=r"layers\.5\.mlp\.experts\.0\.(gate|up)_proj",
                    help="regex over plan names")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--source", default=None, help="override config['source_model']")
    a = ap.parse_args()
    out = Path(a.checkpoint)
    config = read_checkpoint_config(out)
    settings = encode_settings_from_config(config)
    grid = grid_from_config(config)
    src = Path(a.source or config["source_model"])
    src_index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    suffix = config.get("blob_suffix", ".tessera")
    names = [n for n in config["plan"] if re.search(a.units, n)][: a.limit]
    print(f"settings from config: { {k: (v.generators if k == 'code' else v) for k, v in settings.items()} }")
    print(f"grid {grid.name}  units {names}")
    ok = True
    for name in names:
        with safe_open(str(src / src_index[name]), framework="pt") as f:
            w = f.get_tensor(name).contiguous().cuda()
        t0 = time.time()
        unit = encode_linear(w, grid=grid, q256=int(config["plan"][name]), name=name, **settings)
        enc = time.time() - t0
        key = name + suffix
        with safe_open(str(_shard_holding(out, key)), framework="pt") as f:
            disk = bytes(f.get_tensor(key).numpy().tobytes())
        same = disk == unit.blob
        ok &= same
        if same:
            print(f"  {name}: byte-identical ({len(disk)} bytes, encode {enc:.1f}s)")
        else:
            first = next((i for i, (x, y) in enumerate(zip(disk, unit.blob)) if x != y), min(len(disk), len(unit.blob)))
            print(f"  {name}: DIFFERS at byte {first} (disk {len(disk)} bytes, replay {len(unit.blob)} bytes)")
    print("WITNESS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
