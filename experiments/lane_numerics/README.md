# Lane numerics: where two served forwards part

Scripts behind `docs/measurements/tessera-gridbook-lane-served-2026-09-02.md`.
They run inside the vanilla vLLM 0.28 image via
`/home/rob/tessera-runs/gbfam/gbrun.sh` (mount `/home/rob/tessera-runs` and
`/mnt/shared`; `pip install --no-deps -e /gb` before the Gridbook arm).

| script | what |
|---|---|
| `layer_dump.py <ckpt> <out.npz>` | one in-process eager prefill of a corpus chunk; hooks every decoder layer and every quantized Linear (in + out) |
| `layer_compare.py a.npz b.npz` | relative Frobenius / per-row differences, tensor by tensor |
| `gemm_real.py a.npz b.npz L...` | each arm's per-Linear GEMM error on ITS OWN captured input vs an fp64 reference (stock tiles + vLLM's `scaled_fp4_quant`); splits the arm difference into local vs propagated |
| `noise_control.py <ckpt> <out.npz> eps...` | multiplicative noise of relative std `eps` at every quantized Linear output of the stock model; final hidden states per chunk |
| `hidden_kl.py noise.npz [a.npz b.npz]` | exact full-vocab KL from final hidden states (tied lm_head): the noise curve, and the stock-vs-lane KL on the dumped chunk |
