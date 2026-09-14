# Folded BF16 selected MoE, one-box stock-vLLM numerical control

On 2026-09-13 local time, Sparklina ran
`experiments/bf16_selected_moe_native_control.py` (source SHA-256
`3afec9b8c2bccb6bbb5543231c900e1887072d7857c89b2bdd03fd330fed8e83`)
against Tessera route source `bb203936bc505fa175241f7dd40cb3aa341c9d93`.
The container used unmodified pinned stock image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
(local image ID `sha256:44426342348a2fb1ae1191e09e5d4e19ab1231ab6dbdf0e17e8192bfdbeff025`).
The vLLM test was exempt from PrismaBuild. The owned container
`tessera-bf16-native-control` exited 0 during
`2026-09-14T01:24:23.330269182Z`–`01:24:39.172024641Z`; it was removed and
Sparklina had no remaining compute process. The exact run command used
`--gpus all --ipc host --network host`, `PYTHONPATH=/work/src:/work`,
`TESSERA_SERVE_MODE=resident`, and `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`,
mounting the local source read-only. The retained raw log and result are at
`/home/rob/dq-runs/glm-campaign-takeover-20260913/serving-format-closure/native/`.

The test encoded four distinct locally generated expert stacks at BF16-grid
`q256=1792` with hidden size 256, intermediate size 128 and top-K 2. It loaded
all twelve containers through stock `RoutedExperts.load_weights`, then ran
the **actual** stock `UnquantizedMoeBackend.TRITON` modular kernel, not a fake
kernel or constructor-only oracle. Five input tokens routed to global experts
`{0,1,3}` in mixed order, leaving expert 2 unselected. Selected compressed
execution was bit-exact to a full stock unquantized MoE loaded with the same
folded BF16 weights (`max_abs=0`). The stock `MoERunner` shared-expert result
was bit-exact to the routed result plus the same shared module's output.
Swapping two global-to-compact IDs in an independent stock-kernel call changed
the answer (`max_abs=100.25`), so the parity test was sensitive to the map.
The configured SwiGLU clamp was active: an FP32 unclamped reference differed
from the clamped one by relative L2 `18.2229`; the native output differed from
the clamped FP32 reference by relative L2 `0.00255`.

The [result JSON](bf16_selected_moe_native_2026-09-13.json) is copied byte for
byte from the run (SHA-256
`4d116381add06d2e36adf9d3ec2812e23d5ba1290ed10b45f2020a3842f19b37`).
This control covers one-box TP1 numerical loading and execution on a tiny
random fixture. It does not cover TP2 stock-kernel execution, full GLM,
long-context generation, latency, memory fit, power or a production runtime
cell. No runtime contract or consumer pin was promoted.
