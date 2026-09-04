# Serve wrapper support-size request

During the final LFM driver review, both `serve_and_dump_kl.sh` and
`tessera_plugin_served.sh` passed `TESSERA_KL_TOPK` to the server's
`--max-logprobs` but omitted the dump client's `--top-k`. Thus a nondefault
request only changed the server limit; the client still requested its own
default. The current LFM campaign uses 1,024 in both places and is unaffected.

The fix passes the same value explicitly to each real dump invocation.
`tests/test_serve_wrapper_topk.py` executes those shell argument expressions
with captured argv at the default and a nondefault support size. It does not
start a server or fake a quality measurement.

Before the fix, PB action `2b3f2666287d` reported all four new cases failing
at line 32: `AssertionError: server max-logprobs is not the dump request's
top-K`. Population: dl380g10 CPU, torch 2.11.0+cpu, no CUDA, 0 skipped and
0 uncollected modules. The deployed pool retried the failing action; preserved
surface files retain those repeated red observations.

After the fix, action
`6a4e9a925b11b81f75789a488e3b5e175e214df1eb396642a0bda57fab7d658f`
returned zero: 8 passed in 1.39 seconds across the new argv tests, existing
teacher-cleanup tests and MoE campaign serving gate. Population: dl380g10 CPU,
torch 2.11.0+cpu, no CUDA, 0 skipped and 0 uncollected modules. Receipt CAS:
`262c10ec0bf259c3f1002f41ab4fadbcc5ec3bc841695ed2d2e3974ed75bac4e`.
No GPU claim or new top-K policy follows from this regression test.

The impacted selector action
`15c34c0eca15e047c734e05629f162027ae20ab0288495192719058c34b731d0`
used the exact fetched base and narrowed to the new argv tests, teacher cleanup,
MoE serving gates and build identity. The remaining build-identity file ran in
PB action `c402fb0365cd8c39a19264d73c94f23ba42b51af569c390565de839abc347d5a`:
44 passed, 9 skipped, zero uncollected modules, serial CPU/no CUDA. Receipt CAS:
`e4cbb62aefa3b5e193b5b920e7c8f41826c4d94b3bf5dd7aa9618aaffeb68410`.
Verbatim skip reasons and counts:

- 6: `/home/rob/tessera-runs/compile-dispatch is not on this box`
- 1: `the two surviving compile caches from 2026-09-02 are not on this box`
- 1: `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box`
- 1: `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box`
