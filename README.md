# marshmallow-rust

A standalone Rust port of [marshmallow](https://github.com/marshmallow-code/marshmallow)
object serialization/validation: a compiled schema dump/load kernel with a
pure-Python fallback. Upstream attribution in NOTICE; MIT licensed.

`marshmallow_rust.Schema` subclasses `marshmallow.Schema`. On first
`dump`/`load` the schema tree is compiled to a Rust-core plan; every batch
then crosses the FFI boundary once (`many=True` batches cross once per
batch). Anything the compiler cannot prove — decorated schema hooks,
overridden entry points — takes the pure-Python upstream path with
identical behavior. Custom `Field` subclasses and per-value user callables
do not force fallback: they compile to opaque nodes that invoke the same
callables upstream would call (same cost as upstream for that field, full
speed elsewhere).

## Layout

- `marshmallow-core/` — reusable Rust dump/load kernel (`marshmallow_core`),
  no Python dependency beyond the C API.
- `src/lib.rs` + root `Cargo.toml` — the PyO3 binding crate
  (`marshmallow-py`), compiled as the `marshmallow_rust._core` extension module.
- `marshmallow_rust/` — Python face: `Schema`, the plan compiler
  (`compile.py`), and the error materializer (`errors.py`).
- `tests/` — port tests, equivalence corpus, and the upstream-suite gate.
- `benches/two_track.py` — two-track wall/RSS bench (measured only).

## Correctness bar

- Full upstream `marshmallow` suite green against the port (import alias —
  every upstream test exercises the Rust core wherever the schema
  compiles, the pure-Python path elsewhere).

## Benches

`testvenv/bin/python benches/two_track.py` (same inputs both sides, fresh
process per side for honest peak RSS).

<!-- BENCH:TABLE -->
| workload | upstream wall | port wall | upstream peak RSS | port peak RSS | speedup |
| --- | --- | --- | --- | --- | --- |
| nested-batch-500x4 (dump+load `many=True` x4, 500 nested objects) | 147.82 ms | 33.35 ms | 23176 KiB | 25204 KiB | 4.43x |

Frozen workload (`benches/two_track.py --reps 5`, medians of 5 fresh-process
runs): port side asserts extension engagement; kill line <1.5x (lane 3).
Measured 2026-10-28 on AMD Ryzen 9 5950X, CPython 3.12 + marshmallow 4.3.1
vs release `_core` extension. Port peak RSS includes the extension pages.

