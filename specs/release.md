# Release spec — `marshmallow-rust` 0.1.0

Release lane spec for the Rust port of marshmallow. It records what ships,
how it is verified, and what it does not claim. The lane-equivalent config
is `release.toml`; attribution and the accepted divergences are in `NOTICE`;
this port's own license is `LICENSE` (MIT).

## Scope

- Python distribution `marshmallow-rust` (import name `marshmallow_rust`),
  version `0.1.0`.
- Reusable Rust core crate `marshmallow-core` 0.1.0 on crates.io.
- Covered: the `Schema` dump/load/validate kernel — every field codec in
  `marshmallow.fields`, every std validator in `marshmallow.validate`,
  nested schemas, unknown/partial/many/index_errors handling — with exact
  `ValidationError` message/valid_data parity against marshmallow 4.3.1.

This is **not** a drop-in replacement for `marshmallow`: the upstream
package name and import stay with upstream. `marshmallow_rust.Schema`
subclasses `marshmallow.Schema`; schemas with decorated hooks or
overridden entry points take the pure-Python upstream path by construction.
Custom `Field` subclasses compile to opaque nodes that invoke the field's
own methods per value (same cost as upstream for that field).

## Artifacts

| Artifact | Content |
|---|---|
| wheel | `marshmallow_rust` package with the compiled `marshmallow_rust._core` extension (PyO3) |
| sdist | source tree incl. `marshmallow-core/` (path dependency), `NOTICE`, `LICENSE`, `release.toml` |
| crate | `marshmallow-core` 0.1.0 (rlib + cdylib bindings, no Python dependency in the core logic) |

## Verification battery (all green before any upload)

1. `cargo clippy -- -D warnings` clean on both crates.
2. Full upstream marshmallow suite green against the port (import alias;
   1188 passed; `tests/mypy_test_cases` excluded — static-typing fixtures
   that fail identically on upstream).
3. `tests/test_fuzz.py` (200 randomized schemas + hostile data, deterministic
   seed) and `tests/test_port.py` (seams, entry points, no-extension path).
4. `benches/two_track.py` measured: ≥1.5x on the frozen 500-object nested
   dump/load x4 workload (kill line; lane measured 4.4x), wall + peak RSS
   both sides, per-workload labels. Workers run with a neutral cwd and
   assert extension engagement.

## Publish route

Release prep -> TestPyPI -> prod PyPI + crates.io once proven (human-gated;
standing rule). Each upload is verified by installing from the index into a
fresh venv and rerunning the battery.

## Follow-ups (not blocking 0.1.0)

- Hand-rolled ASCII fast paths already cover exact int/float/str/bool plus
  exact-dict extraction/storage; remaining per-value C calls (regex-backed
  Email/URL validators, UUID/IP/temporal/decimal conversions, custom
  callables) are exact by construction and candidates for later
  fuzz-proven native ports.
- `Decimal`/`TimeDelta`/IP-family are fully covered via C calls; no
  fallback involved.
- Schema-level hooks and overridden entry points stay on the upstream path
  (documented seam); per-node opaque fallback exists but is currently
  unreachable — whole-tree fallback triggers first.
