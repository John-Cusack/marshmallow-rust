"""Port-specific tests: seams, entry points, and fallback paths.

Behavioral parity is carried by the upstream suite (1188 tests, run via the
gate harness) and the differential fuzzer; these tests pin the port's own
seams: compiled vs fallback routing, the opaque-field slow path, and the
no-extension pure-Python path.
"""

import marshmallow as mm
import pytest

import marshmallow_rust as mr
from marshmallow_rust import Schema, fields


class UserSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(load_default=0)


def test_compiled_schema_uses_core():
    s = UserSchema()
    assert s.dump({"name": "a"}) == {"name": "a"}
    assert s._rust_handle is not None


def test_hooks_force_fallback_with_identical_behavior():
    class Hooked(Schema):
        x = fields.Int()

        @mm.pre_load
        def _pre(self, data, **kwargs):
            return data

    s = Hooked()
    assert s.dump({"x": 1}) == {"x": 1}
    assert s.load({"x": "2"}) == {"x": 2}
    assert s._rust_handle is None


def test_custom_field_takes_opaque_path_with_identical_behavior():
    class Upper(mm.fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            return value.upper() if isinstance(value, str) else value

        def _deserialize(self, value, attr, data, **kwargs):
            return value.lower() if isinstance(value, str) else value

    class S(Schema):
        name = Upper()

    s = S()
    assert s.dump({"name": "AbC"}) == {"name": "ABC"}
    assert s.load({"name": "AbC"}) == {"name": "abc"}
    assert s._rust_handle is not None  # compiles; field runs per-value calls


def test_no_extension_fallback(monkeypatch):
    monkeypatch.setattr(mr, "_core", None)

    class S(Schema):
        a = fields.Int(required=True)

    s = S()
    assert s._rust_handle is None
    assert s.dump({"a": 1}) == {"a": 1}
    assert s.load({"a": "2"}) == {"a": 2}
    with pytest.raises(mm.ValidationError):
        s.load({})


def test_entry_points():
    s = UserSchema()
    assert s.loads('{"name": "a", "age": 3}') == {"name": "a", "age": 3}
    assert s.load([{"name": "a"}], many=True) == [{"name": "a", "age": 0}]
    assert s.validate({"name": "a"}) == {}
    assert UserSchema(many=True).dump([{"name": "a"}]) == [{"name": "a"}]


def test_unknown_and_partial_variants():
    class S(Schema):
        a = fields.Int()

    s = S()
    assert s.load({"a": 1, "zzz": 2}, unknown="include") == {"a": 1, "zzz": 2}
    assert s.load({"a": 1, "zzz": 2}, unknown="exclude") == {"a": 1}
    with pytest.raises(mm.ValidationError):
        s.load({"a": 1, "zzz": 2}, unknown="raise")
    # garbage unknown behaves like exclude (matches upstream comparisons)
    assert s.load({"a": 1, "zzz": 2}, unknown="bogus") == {"a": 1}
    assert s.load({}, partial=True) == {}
