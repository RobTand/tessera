#!/usr/bin/env python3
"""tessera#695: a four-layer GLM-5.3-Flash stub that carries the MTP drafter layer.

The #508 qualification stub has no MTP layer, so no drafter graph path can be
measured on it. This builds one from two existing artifacts, with no GPU:

  STUB  the four-layer stub (``num_hidden_layers`` 4, ``num_nextn_predict_layers`` 0)
  FULL  a 45-layer export that carries the MTP layer as ``layers.45``

and writes OUT with:

- every STUB file hardlinked into OUT except the two JSON files rewritten below
  (a hardlink across file systems is refused, never replaced by a copy);
- one new shard holding every ``.layers.45.`` tensor of FULL renamed to
  ``.layers.4.``, bytes copied verbatim. vLLM builds the MTP layer at index
  ``num_hidden_layers`` (``models/glm5next/nvidia/mtp.py``: ``mtp_start_layer_idx``)
  and as MLA through ``is_mtp_layer``, whatever the index;
- ``config.json``: ``num_nextn_predict_layers`` 1, plus FULL's layer-45
  ``config_groups`` and ``ignore`` entries renamed to layer 4;
- ``model.safetensors.index.json`` extended by the new shard.

The stub's ``tessera_serving_manifest.json`` is hardlinked unchanged: the
serving path does not read it, and this artifact is a research stub, not a
shippable export. The drafter follows a four-layer body, so its acceptance is
expected to be low; the stub measures the drafter's graph mechanics, not its
quality.

Stdlib only. Prints one JSON receipt on stdout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

SRC_LAYER = 45
NEW_SHARD = "model-mtp-layer4.safetensors"
REWRITTEN = ("config.json", "model.safetensors.index.json")
CHUNK = 64 << 20


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def _rename(name: str, dst_layer: int) -> str:
    out, n = re.subn(rf"\.layers\.{SRC_LAYER}\.", f".layers.{dst_layer}.", name)
    if n != 1:
        raise SystemExit(f"expected one .layers.{SRC_LAYER}. in {name!r}")
    return out


def _text_config(config: dict) -> dict:
    return config.get("text_config", config)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stub", type=Path)
    ap.add_argument("full", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    stub_cfg = json.loads((args.stub / "config.json").read_text())
    full_cfg = json.loads((args.full / "config.json").read_text())
    stub_text, full_text = _text_config(stub_cfg), _text_config(full_cfg)
    dst_layer = int(stub_text["num_hidden_layers"])
    if int(stub_text.get("num_nextn_predict_layers") or 0) != 0:
        raise SystemExit("the stub already carries an MTP layer")
    if int(full_text["num_hidden_layers"]) != SRC_LAYER or int(full_text["num_nextn_predict_layers"]) != 1:
        raise SystemExit(f"FULL must have {SRC_LAYER} layers and one MTP layer")
    # Everything but the layer count and its per-layer lists must agree, or the
    # MTP layer was trained for a different geometry.
    per_layer = {"num_hidden_layers", "num_nextn_predict_layers", "layer_types", "indexer_types",
                 "mlp_layer_types", "linear_attn_config", "first_k_dense_replace", "quantization_config"}
    mismatched = sorted(k for k in set(stub_text) | set(full_text)
                        if k not in per_layer and stub_text.get(k) != full_text.get(k))
    if mismatched:
        raise SystemExit(f"stub and FULL disagree outside the per-layer fields: {mismatched}")

    full_index = json.loads((args.full / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in full_index.items() if f".layers.{SRC_LAYER}." in k}
    if not wanted:
        raise SystemExit(f"FULL has no .layers.{SRC_LAYER}. tensors")

    args.out.mkdir(parents=True, exist_ok=False)
    # Hardlink the stub, never copy it.
    for entry in sorted(args.stub.iterdir()):
        if entry.name in REWRITTEN or not entry.is_file():
            continue
        os.link(entry, args.out / entry.name)

    # Gather the MTP tensors' headers, then write the new shard in one pass.
    plan = []  # (new_name, src_path, src_begin, src_end, meta)
    for shard in sorted(set(wanted.values())):
        header, base = _header(args.full / shard)
        for name, meta in header.items():
            if name in wanted:
                begin, end = meta["data_offsets"]
                plan.append((_rename(name, dst_layer), args.full / shard, base + begin, base + end, meta))
    if len(plan) != len(wanted):
        raise SystemExit(f"index names {len(wanted)} tensors, shard headers {len(plan)}")
    plan.sort(key=lambda p: p[0])
    new_header, offset = {}, 0
    for new_name, _, begin, end, meta in plan:
        size = end - begin
        new_header[new_name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                                "data_offsets": [offset, offset + size]}
        offset += size
    new_header["__metadata__"] = {"format": "pt", "tessera_695": f"layers.{SRC_LAYER} of {args.full} as layers.{dst_layer}"}
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    digest = hashlib.sha256()
    with (args.out / NEW_SHARD).open("wb") as out:
        head = struct.pack("<Q", len(blob)) + blob
        out.write(head)
        digest.update(head)
        for _, src, begin, end, _ in plan:
            with src.open("rb") as f:
                f.seek(begin)
                left = end - begin
                while left:
                    buf = f.read(min(CHUNK, left))
                    if not buf:
                        raise SystemExit(f"short read in {src}")
                    out.write(buf)
                    digest.update(buf)
                    left -= len(buf)

    # Index: the stub's map plus the new shard.
    stub_index = json.loads((args.stub / "model.safetensors.index.json").read_text())
    index = copy.deepcopy(stub_index)
    for new_name, *_ in plan:
        if new_name in index["weight_map"]:
            raise SystemExit(f"{new_name} already in the stub")
        index["weight_map"][new_name] = NEW_SHARD
    index.setdefault("metadata", {})["total_size"] = int(index.get("metadata", {}).get("total_size", 0)) + offset
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    # Config: one MTP layer, plus FULL's layer-45 quantization entries at layer 4.
    cfg = copy.deepcopy(stub_cfg)
    _text_config(cfg)["num_nextn_predict_layers"] = 1
    q_full = full_cfg.get("quantization_config") or full_text.get("quantization_config")
    q_new = cfg.get("quantization_config") or _text_config(cfg).get("quantization_config")
    groups, ignores = [], []
    for name, group in q_full.get("config_groups", {}).items():
        targets = group.get("targets", [])
        hits = [t for t in targets if f".layers.{SRC_LAYER}." in t]
        if not hits:
            continue
        if len(hits) != len(targets):
            raise SystemExit(f"config group {name} mixes layer {SRC_LAYER} with other targets")
        new_group = copy.deepcopy(group)
        new_group["targets"] = [_rename(t, dst_layer) for t in targets]
        new_name = name.replace(f"layers_{SRC_LAYER}_", f"layers_{dst_layer}_")
        if new_name in q_new["config_groups"]:
            raise SystemExit(f"config group {new_name} already in the stub")
        q_new["config_groups"][new_name] = new_group
        groups.append(new_name)
    for entry in q_full.get("ignore", []):
        if f".layers.{SRC_LAYER}." in entry:
            renamed = _rename(entry, dst_layer)
            if renamed not in q_new["ignore"]:
                q_new["ignore"].append(renamed)
            ignores.append(renamed)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    json.dump({"out": str(args.out), "mtp_layer_index": dst_layer, "tensors": len(plan),
               "bytes": offset, "shard": NEW_SHARD, "shard_sha256": digest.hexdigest(),
               "config_groups": groups, "ignore": ignores,
               "hardlinked": sorted(p.name for p in args.out.iterdir()
                                    if p.name not in REWRITTEN and p.name != NEW_SHARD)},
              sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
