# Endpoint witness on a live GPU serve, 2026-10-09 (tessera#1134)

This note records one live probe of the endpoint witness producer.
It proves the byte join on a tiny model. It grants no D50 qualification.

## What ran

- PrismaBuild action `25ea6519e540a7f8b30bead59a3895b45c519941e5b5310252cb01b8347c5240`, priority 0, on sparklina.
- Image `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` (vLLM 0.28.0, tokenizers 0.22.2, transformers 5.15.1).
- Model: a tiny untied Qwen3 at `/mnt/shared/tessera-witness-val/tiny-qwen3-untied`, one rank, `--network host`.
- `tools/probe_endpoint_witness.py` posted one fresh nonce. `tools/verify_endpoint_witness.py` checked the receipt against the served and tokenizer directories.

## Result

- The verifier printed `runtime witness valid` for attempt `sparklina/6be8b8a9-d932-46f1-a563-d5c0b2ec645a/1/2722855`, ranks `[0]`.
- Receipt fingerprint: `9df7e024c5be2253b73b07dfa0ab822687baa3b0cc81b01d571fbe07b8148c5c`. The receipt is 29 MB and stays on `/mnt/shared/tessera-witness-val/receipt-9df7e024.json`.
- The lifetime nonce `24eeef2628a2adeb5a7d81fa5e834ddb` equals the rank 0 nonce and the tokenizer nonce.
- Served alias: `tiny-untied-live` at `http://127.0.0.1:8158`.
- Coverage: 25 loader inputs and 19 resident tensors on rank 0 cover 315889664 payload bytes of `model.safetensors`.
- `model.safetensors` sha256 `56f13b39…a8140c`. `tokenizer.json` sha256 `aeb13307…92dae4`. Both equal the receipt.

## Defects this run found

1. The tokenizer gate refused the real backend, because transformers rewrites four ByteLevel flags after it loads `tokenizer.json`. Action `d6b138586061ea66302167547df25c1db81d5f9ce1948d880c91312381721adc` shows the `tokenizers` library alone reproduces the file exactly. The gate now ignores the flags that no token ID depends on.
2. A container port mapping makes the observed listener port differ from the contacted endpoint, so the probe refuses it. The run used `--network host`.
3. The receipt file that the container writes is readable by root only. The run copied it out with `docker exec`.

## Not measured

- A tied-weight model. vLLM skips `lm_head` for those models, so the coverage rule cannot pass.
- More than one rank. No pinned-image tensor-parallel serve ran.
- Served KL or any quality metric.
