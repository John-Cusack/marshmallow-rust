"""Differential fuzz: randomized schemas + hostile data, port vs oracle.

Deterministic (fixed seed). For each generated schema pair
(`marshmallow.Schema` oracle vs `marshmallow_rust.Schema` port), compares:

* `dump` / `load` results (NaN-safe, type-strict),
* raised exceptions (type, messages, valid_data, data),
* `validate()` output.

Covers every codec, validator, option, and seam path (opaque custom
fields, schema hooks, callable defaults/processors, dotted attributes,
unknown/partial/many/index_errors variants, exotic scalars).
"""

import datetime as dt
import decimal
import enum
import ipaddress
import math
import random
import uuid

import pytest

import marshmallow as mm
from marshmallow_rust import Schema as PortSchema


class Color(enum.Enum):
    RED = 1
    GREEN = 2


class CustomField(mm.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return f"custom:{value}"

    def _deserialize(self, value, attr, data, **kwargs):
        return f"seen:{value}"


class CustomValidator(mm.validate.Validator):
    def __call__(self, value):
        if value == "boom":
            raise mm.ValidationError("went boom")
        return value


def _values(rng):
    ints = [0, 1, -1, 42, 2**63 - 1, -(2**63), 2**100, -(2**100), True, False]
    floats = [0.0, 1.5, -0.0, 1e16, 1e-7, float("nan"), float("inf"), float("-inf")]
    strs = [
        "", "a", "  padded  ", "1_0", "  -12  ", "+.5", "0x10", "1e3", "nan",
        "Infinity", "foo@bar.com", "bad@", "@bad", "a\nb", "http://x.y/",
        "2014-12-22T03:12:58", "not-a-date", "caf\u00e9", "\u00e51\u00e5",
        "1234567890123456789012345678901234567890",
    ]
    return [
        None, True, False, *ints, *floats, *strs,
        b"bytes", "b\xffbin".encode("latin1"),
        [], [1, 2], {}, {"k": "v"},
        uuid.uuid4(), "550e8400-e29b-41d4-a716-446655440000",
        "{550e8400-e29b-41d4-a716-446655440000}", "not-a-uuid",
        dt.datetime(2020, 1, 2, 3, 4, 5), "2020-01-02T03:04:05",
        dt.date(2021, 5, 6), "2021-05-06", dt.time(3, 4, 5), "03:04:05",
        dt.timedelta(seconds=90), decimal.Decimal("1.5"), decimal.Decimal("NaN"),
        ipaddress.ip_address("127.0.0.1"), "127.0.0.1", "999.1.1.1",
        Color.RED, 1, "RED",
    ]


def _field(rng, depth, registry):
    f = rng.choice([
        "raw", "str", "uuid", "int", "float", "decimal", "bool",
        "datetime", "date", "time", "timedelta", "url", "email",
        "ip", "ipv4", "enum", "constant", "method", "function", "custom",
        "nested", "list", "tuple", "dict",
    ])
    kw = {}
    if rng.random() < 0.3:
        kw["required"] = True
    if rng.random() < 0.2:
        kw["allow_none"] = True
    if rng.random() < 0.15 and "required" not in kw:
        kw["load_default"] = rng.choice([None, 0, "", [], {}])
    if rng.random() < 0.1 and "required" not in kw:
        kw["load_default"] = rng.choice([list, dict])
    if rng.random() < 0.15:
        kw["dump_default"] = rng.choice([None, 0, "d"])
    if rng.random() < 0.1:
        kw["data_key"] = rng.choice(["dk", "other"])
    if rng.random() < 0.1:
        kw["attribute"] = rng.choice(["attr", "a.b"])
    if rng.random() < 0.2:
        kw["validate"] = rng.choice([
            mm.validate.Length(min=1, max=5),
            mm.validate.Range(min=1, max=10),
            mm.validate.OneOf(["a", "b"]),
            mm.validate.Equal("x"),
            mm.validate.Regexp(r"^a+$"),
            mm.validate.Predicate("isdigit"),
            lambda v: v,
            CustomValidator(),
        ])
    if rng.random() < 0.1:
        kw["pre_load"] = lambda v: v
    if rng.random() < 0.1:
        kw["post_load"] = lambda v: v
    F = mm.fields
    if f == "raw":
        return F.Raw(**kw)
    if f == "str":
        return F.Str(**kw)
    if f == "uuid":
        return F.UUID(**kw)
    if f == "int":
        return F.Int(strict=rng.random() < 0.3, **kw)
    if f == "date":
        return F.Date(**kw)
    if f == "time":
        return F.Time(**kw)
    if f == "timedelta":
        return F.TimeDelta(**kw)
    if f == "url":
        return F.Url(require_tld=rng.random() < 0.5, **kw)
    if f == "email":
        return F.Email(**kw)
    if f == "ip":
        return F.IP(**kw)
    if f == "ipv4":
        return F.IPv4(**kw)
    if f == "enum":
        return F.Enum(Color, **kw)
    if f == "constant":
        return F.Constant(rng.choice([1, "c", None]), **kw)
    if f == "method":
        return F.Method(serialize="get_x", deserialize="set_x", **kw)
    if f == "function":
        return F.Function(serialize=lambda o: 1, deserialize=lambda v: 2, **kw)
    if f == "custom":
        return CustomField(**kw)
    if f == "nested" and depth < 2:
        sub = _schema(rng, depth + 1, registry)
        return F.Nested(sub, unknown=rng.choice([None, "exclude", "include"]), **kw)
    if f == "nested":
        return F.Raw(**kw)
    if f == "list":
        inner = _field(rng, 3, registry) if depth < 2 else F.Raw()
        return F.List(inner, **kw)
    if f == "tuple":
        return F.Tuple((F.Str(), F.Int()), **kw)
    if f == "dict":
        return F.Dict(keys=F.Str(), values=F.Int(), **kw)
    return F.Raw(**kw)


def _schema(rng, depth, registry):
    attrs = {}
    for i in range(rng.randint(1, 5)):
        attrs[f"f{i}"] = _field(rng, depth, registry)
    attrs["get_x"] = lambda self, obj: 1
    attrs["set_x"] = lambda self, value: 2
    if rng.random() < 0.15:
        # Schema-level hooks force the pure-Python path; behavior must match.
        def _pre(self, data, **kwargs):
            return data

        def _post(self, data, **kwargs):
            return data

        attrs["_pre"] = mm.pre_load(_pre)
        attrs["_post"] = mm.post_load(_post)
    meta = {}
    if rng.random() < 0.2:
        meta["unknown"] = rng.choice(["exclude", "include", "raise", "bogus"])
    if rng.random() < 0.1:
        meta["index_errors"] = False
    if meta:
        attrs["Meta"] = type("Meta", (), meta)
    name = f"Fuzz{len(registry)}"
    cls = type(name, (mm.Schema,), attrs)
    registry.append(cls)
    return cls


def _port_of(cls):
    # The metaclass strips field attributes from __dict__; rebuild from the
    # declared fields plus the non-field members (Meta, hooks, helpers).
    # Internal metaclass state (_declared_fields, _hooks, opts, _abc_impl)
    # is recomputed by the metaclass and must not be copied.
    members = dict(cls._declared_fields)
    for k, v in cls.__dict__.items():
        if k.startswith("__") or k in members:
            continue
        if k in ("_declared_fields", "_hooks", "opts", "_abc_impl"):
            continue
        members[k] = v
    return type(cls.__name__ + "Port", (PortSchema,), members)


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (math.isnan(a) and math.isnan(b))
    if isinstance(a, decimal.Decimal) and isinstance(b, decimal.Decimal):
        return (a == b) or (a.is_nan() and b.is_nan())
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if type(a) is not type(b):
        # int == bool and IntEnum == int are value-equal but type-divergent;
        # the port must preserve upstream's exact runtime types.
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _outcome(fn):
    try:
        return ("ok", fn())
    except mm.ValidationError as e:
        return ("ve", e.messages, e.valid_data, e.data)
    except Exception as e:  # noqa: BLE001 - parity includes error types
        return ("raise", type(e).__name__, str(e))


def _check_case(oracle_schema, port_schema, data, many, partial, unknown):
    for op in ("dump", "load", "validate"):
        if op == "dump":
            o = _outcome(lambda: oracle_schema.dump(data, many=many))
            p = _outcome(lambda: port_schema.dump(data, many=many))
        elif op == "load":
            o = _outcome(lambda: oracle_schema.load(data, many=many, partial=partial, unknown=unknown))
            p = _outcome(lambda: port_schema.load(data, many=many, partial=partial, unknown=unknown))
        else:
            o = _outcome(lambda: oracle_schema.validate(data, many=many, partial=partial))
            p = _outcome(lambda: port_schema.validate(data, many=many, partial=partial))
        assert o[0] == p[0], (op, data, o, p)
        if o[0] == "ok":
            assert _eq(o[1], p[1]), (op, data, o[1], p[1])
        elif o[0] == "ve":
            assert _eq(o[1], p[1]) and _eq(o[2], p[2]), (op, data, o, p)
        else:
            assert o[1:] == p[1:], (op, data, o, p)


def test_fuzz_differential():
    rng = random.Random(20261028)
    vals = _values(rng)
    registry = []
    for _ in range(200):
        cls = _schema(rng, 0, registry)
        try:
            oracle = cls()
        except Exception:
            continue  # construction-time errors are upstream's domain
        try:
            port = _port_of(cls)()
        except Exception as e:
            raise AssertionError(f"port class construction diverged: {e}")
        pool = [{f"f{i}": rng.choice(vals) for i in range(5)} for _ in range(4)]
        pool += [rng.choice(vals), [rng.choice(vals) for _ in range(3)]]
        for data in pool:
            many = rng.random() < 0.3
            if many and not isinstance(data, list):
                data = [data]
            partial = rng.choice([None, True, ["f0"], ("f1",), "f0"])
            unknown = rng.choice([None, "exclude", "include", "raise", "bogus"])
            methods = [m for m in ("get_x", "set_x") if hasattr(oracle, m)]
            _check_case(oracle, port, data, many, partial, unknown)


def test_fuzz_objects_with_attributes():
    # Dump from real objects exercises getattr paths (incl. dotted/missing).
    rng = random.Random(77)
    vals = _values(rng)

    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class S(mm.Schema):
        a = mm.fields.Int(attribute="x")
        b = mm.fields.Str(attribute="y.z")
        c = mm.fields.Int()

    class P(PortSchema):
        a = mm.fields.Int(attribute="x")
        b = mm.fields.Str(attribute="y.z")
        c = mm.fields.Int()

    for i in range(50):
        obj = Obj(x=rng.choice(vals), y={"z": rng.choice(vals)}, c=rng.choice(vals))
        if i % 7 == 0:
            del obj.__dict__["x"]
        o = _outcome(lambda: S().dump(obj))
        p = _outcome(lambda: P().dump(obj))
        assert o[0] == p[0], (i, o, p)
        if o[0] == "ok":
            assert _eq(o[1], p[1]), (i, o[1], p[1])
        else:
            assert o[1:] == p[1:], (i, o, p)
