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
