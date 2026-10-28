# marshmallow-core

Reusable Rust dump/load kernel for the
[marshmallow-rust](https://github.com/John-Cusack/marshmallow-rust) port of
[marshmallow](https://github.com/marshmallow-code/marshmallow)
object serialization/validation.

The kernel replays upstream `Schema` dump/load control flow over a plan
compiled from the bound schema tree (`marshmallow_rust.compile` builds the
plan; see `Plan::from_dict`). Every leaf that depends on CPython semantics
(conversions, comparisons, regex matches, user callables) is a C-API call
with identical observable behavior; exact-type fast paths (int, float, str,
bool, exact-dict extraction/storage) eliminate the per-value calls on the
common path. Errors are collected as a structured tree and formatted
Python-side with the upstream templates, so messages are identical by
construction.

No Python dependency beyond the C API; no network, no I/O.
