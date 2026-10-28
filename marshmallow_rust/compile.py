"""Compile a marshmallow Schema into a Rust-core plan dict.

The plan mirrors the bound schema tree (fields, codecs, validators,
error-message templates) with all host objects held by reference. The Rust
core replays upstream control flow exactly; every leaf operation it cannot
prove (conversions, user callables, regex matches, comparisons) is a
CPython C-API call with identical observable behavior.

Whole-tree fallback (return None -> pure-Python upstream path) triggers on:
  * a schema node with non-empty ``_hooks`` (decorated pre/post/validates),
  * a Schema subclass overriding dump/load/_do_load/_deserialize/_serialize,
  * any defensive bail (unexpected shape, unresolvable nested schema).

Custom Field subclasses do NOT force fallback: they compile to an opaque
node that calls the field's own serialize/deserialize per value (same cost
as upstream for that field, full speed elsewhere).
"""

import datetime as dt
import uuid as uuid_mod

import marshmallow
from marshmallow import fields as mm_fields
from marshmallow import validate as mm_validate
from marshmallow.constants import EXCLUDE, INCLUDE, RAISE, missing

#: Field classes with inline Rust codecs, keyed by EXACT type.
_CODECS = {
    mm_fields.Raw: "raw",
    mm_fields.String: "string",
    mm_fields.UUID: "uuid",
    mm_fields.Integer: "integer",
    mm_fields.Float: "float",
    mm_fields.Decimal: "decimal",
    mm_fields.Boolean: "boolean",
    mm_fields.DateTime: "datetime",
    mm_fields.NaiveDateTime: "naive_datetime",
    mm_fields.AwareDateTime: "aware_datetime",
    mm_fields.Time: "time",
    mm_fields.Date: "date",
    mm_fields.TimeDelta: "timedelta",
    mm_fields.Nested: "nested",
    mm_fields.Pluck: "pluck",
    mm_fields.List: "list",
    mm_fields.Tuple: "tuple",
    mm_fields.Mapping: "mapping",
    mm_fields.Dict: "dict",
    mm_fields.Url: "url",
    mm_fields.Email: "email",
    mm_fields.IP: "ip",
    mm_fields.IPv4: "ipv4",
    mm_fields.IPv6: "ipv6",
    mm_fields.IPInterface: "ipinterface",
    mm_fields.IPv4Interface: "ipv4interface",
    mm_fields.IPv6Interface: "ipv6interface",
    mm_fields.Enum: "enum",
    mm_fields.Method: "method",
    mm_fields.Function: "function",
    mm_fields.Constant: "constant",
}

#: Validator classes with inline Rust handling, keyed by EXACT type.
_VALIDATORS = {
    mm_validate.Length: "length",
    mm_validate.Range: "range",
    mm_validate.Equal: "equal",
    mm_validate.Regexp: "regexp",
    mm_validate.Predicate: "predicate",
    mm_validate.NoneOf: "noneof",
    mm_validate.OneOf: "oneof",
    mm_validate.ContainsOnly: "containsonly",
    mm_validate.ContainsNoneOf: "containsnoneof",
    mm_validate.Email: "email",
    mm_validate.URL: "url",
}

_BYPASSED_METHODS = (
    "dump",
    "load",
    "loads",
    "dumps",
    "validate",
    "_do_load",
    "_deserialize",
    "_serialize",
)


class _Bail(Exception):
    """Raised for any unexpected shape: caller falls back to pure Python."""
def _get_bypass(schema):
    # The port's own dump/_do_load overrides are the mechanism, not a
    # bypass; nested upstream schemas are inlined, never called. Bail only
    # when the method matches neither base (a real user override).
    from . import Schema as _PortSchema

    for name in _BYPASSED_METHODS:
        meth = getattr(type(schema), name, None)
        if meth is not getattr(_PortSchema, name, None) and meth is not getattr(
            marshmallow.Schema, name, None
        ):
            return name
    return None


def compile_schema(schema):
    """Build the Rust-core plan for `schema`, or None for fallback."""
    try:
        compiler = _Compiler()
        root = compiler.compile_node(schema)
        return {"root": root, "nodes": compiler.nodes}
    except _Bail:
        return None
    except Exception:
        # Defensive: any surprise in the walker means fallback, never a crash.
        return None


class _Compiler:
    def __init__(self):
        self.nodes = []
        self.memo = {}  # id(resolved nested instance) -> node idx
        self.active = {}  # structural key -> node idx (nodes being compiled)

    # -- schemas ------------------------------------------------------

    def _struct_key(self, schema):
        only = schema.only
        return (
            type(schema),
            frozenset(only) if only is not None else None,
            frozenset(schema.exclude),
            bool(schema.many),
            schema.unknown,
            frozenset(schema.load_only),
            frozenset(schema.dump_only),
        )

    def compile_node(self, schema):
        key = self._struct_key(schema)
        if key in self.active:
            return {"backref": self.active[key]}
        ident = id(schema)
        if ident in self.memo:
            return self.memo[ident]
        node_idx = len(self.nodes)
        self.nodes.append(None)  # placeholder for cycles
        self.active[key] = node_idx
        try:
            if self._get_bypass(schema) is not None:
                raise _Bail()
            try:
                hooks_empty = not any(schema._hooks.values())
            except Exception:
                raise _Bail()
            if not hooks_empty:
                raise _Bail()
            try:
                error_messages = dict(schema.error_messages)
                unknown = schema.unknown
                index_errors = bool(schema.opts.index_errors)
                get_attr = schema.get_attribute
            except Exception:
                raise _Bail()
            try:
                default_get = type(schema).get_attribute is marshmallow.Schema.get_attribute
            except Exception:
                raise _Bail()
            # (node placeholder registered above; fields compiled below)
            try:
                dump_fields = [
                    self.compile_field(name, f, "dump")
                    for name, f in schema.dump_fields.items()
                ]
                load_fields = [
                    self.compile_field(name, f, "load")
                    for name, f in schema.load_fields.items()
                ]
            except _Bail:
                raise
            node = {
                "dump_fields": dump_fields,
                "load_fields": load_fields,
                "dict_class": schema.dict_class,
                "partial": schema.partial,
                "unknown": unknown,
                "unknown_msg": error_messages.get("unknown", "Unknown field."),
                "type_msg": error_messages.get("type", "Invalid input type."),
                "index_errors": index_errors,
                "get_attribute": get_attr,
                "get_attribute_is_default": bool(default_get),
                "schema": schema,
            }
            self.nodes[node_idx] = node
            self.memo[ident] = node_idx
            return node_idx
        finally:
            del self.active[key]

    def _get_bypass(self, schema):
        return _get_bypass(schema)

    # -- fields -------------------------------------------------------

    def compile_field(self, name, field, direction):
        try:
            kind = _CODECS.get(type(field))
            codec = self.compile_codec(field) if kind else {"kind": "opaque", "field": field}
            try:
                validators = [self.compile_validator(v) for v in field.validators]
                pre_load = list(field.pre_load)
                post_load = list(field.post_load)
                error_messages = dict(field.error_messages)
                for message in error_messages.values():
                    if not isinstance(message, str):
                        raise _Bail()
            except _Bail:
                raise
            except Exception:
                raise _Bail()
            load_default = field.load_default
            dump_default = field.dump_default
            spec = {
                "name": name,
                "data_key": field.data_key if field.data_key is not None else name,
                "attribute": field.attribute,
                "required": bool(field.required),
                "allow_none": bool(field.allow_none),
                "load_default": (load_default is not missing, load_default),
                "dump_default": (dump_default is not missing, dump_default),
                "required_msg": error_messages.get(
                    "required", "Missing data for required field."
                ),
                "null_msg": error_messages.get("null", "Field may not be null."),
                "error_messages": error_messages,
                "validators": validators,
                "pre_load": pre_load,
                "post_load": post_load,
                "check_attribute": bool(field._CHECK_ATTRIBUTE),
                "codec": codec,
            }
            return spec
        except _Bail:
            raise
        except Exception:
            raise _Bail()

    def compile_inner(self, field):
        """Compile a nested/inner field (List inner, Tuple/Dict/Key members)."""
        return self.compile_field(None, field, "both")

    # -- codecs -------------------------------------------------------

    def compile_codec(self, field):
        kind = _CODECS[type(field)]
        if kind == "integer":
            return {
                "kind": kind,
                "strict": bool(field.strict),
                "as_string": bool(field.as_string),
            }
        if kind == "float":
            return {
                "kind": kind,
                "allow_nan": bool(field.allow_nan),
                "as_string": bool(field.as_string),
            }
        if kind == "decimal":
            return {
                "kind": kind,
                "places": field.places,
                "rounding": field.rounding,
                "allow_nan": bool(field.allow_nan),
                "as_string": bool(field.as_string),
            }
        if kind == "boolean":
            return {"kind": kind, "truthy": field.truthy, "falsy": field.falsy}
        if kind in ("datetime", "naive_datetime", "aware_datetime", "time", "date"):
            return self.compile_temporal(field, kind)
        if kind == "timedelta":
            return {"kind": kind, "precision": field.precision}
        if kind in ("nested", "pluck"):
            return self.compile_nested(field, kind)
        if kind == "list":
            return {"kind": kind, "inner": self.compile_inner(field.inner)}
        if kind == "tuple":
            return {
                "kind": kind,
                "inners": [self.compile_inner(f) for f in field.tuple_fields],
                "length_msg": mm_validate.Length.message_equal,
            }
        if kind in ("mapping", "dict"):
            key_spec = (
                self.compile_inner(field.key_field) if field.key_field is not None else None
            )
            value_spec = (
                self.compile_inner(field.value_field)
                if field.value_field is not None
                else None
            )
            return {
                "kind": kind,
                "mapping_type": field.mapping_type,
                "key": key_spec,
                "value": value_spec,
            }
        if kind in ("ip", "ipv4", "ipv6"):
            return {
                "kind": kind,
                "exploded": bool(field.exploded),
                "deser_class": field.DESERIALIZATION_CLASS,
            }
        if kind in ("ipinterface", "ipv4interface", "ipv6interface"):
            return {
                "kind": kind,
                "exploded": bool(field.exploded),
                "deser_class": field.DESERIALIZATION_CLASS,
            }
        if kind == "enum":
            return self.compile_enum(field)
        if kind == "method":
            return {
                "kind": kind,
                "serialize_method": field._serialize_method,
                "deserialize_method": field._deserialize_method,
            }
        if kind == "function":
            return {
                "kind": kind,
                "serialize_func": field.serialize_func,
                "deserialize_func": field.deserialize_func,
            }
        if kind == "constant":
            return {"kind": kind, "constant": field.constant}
        return {"kind": kind}

    def compile_temporal(self, field, kind):
        try:
            root_opts = field.root.opts if field.root is not None else None
        except Exception:
            raise _Bail()
        opt_name = {
            "datetime": "datetimeformat",
            "naive_datetime": "datetimeformat",
            "aware_datetime": "datetimeformat",
            "time": "timeformat",
            "date": "dateformat",
        }[kind]
        default = {
            "datetime": "iso",
            "naive_datetime": "iso",
            "aware_datetime": "iso",
            "time": "iso",
            "date": "iso",
        }[kind]
        try:
            opt_value = getattr(root_opts, opt_name, None) if root_opts is not None else None
            fmt = field.format or opt_value or default
        except Exception:
            raise _Bail()
        spec = {"kind": kind, "format": fmt}
        if kind == "naive_datetime":
            spec["timezone"] = field.timezone
        if kind == "aware_datetime":
            spec["default_timezone"] = field.default_timezone
        return spec

    def compile_nested(self, field, kind):
        try:
            nested_schema = field.schema
        except Exception:
            # RegistryError / ValueError surface at call time upstream;
            # bail so the pure-Python path raises identically.
            raise _Bail()
        if not isinstance(nested_schema, marshmallow.Schema):
            raise _Bail()
        target = self.compile_node(nested_schema)
        try:
            many = bool(nested_schema.many or field.many)
            unknown = field.unknown
        except Exception:
            raise _Bail()
        spec = {"kind": kind, "target": target, "many": many, "unknown": unknown}
        if kind == "pluck":
            try:
                only_field = nested_schema.fields[field.field_name]
                data_key = only_field.data_key or field.field_name
            except Exception:
                raise _Bail()
            spec["pluck_key"] = data_key
        return spec

    def compile_enum(self, field):
        by_value = field.by_value
        if by_value is False:
            inner = None
            inner_kind = "name"
        elif by_value is True:
            inner = None
            inner_kind = "raw"
        else:
            try:
                from marshmallow.fields import _resolve_field_instance

                inner_field = _resolve_field_instance(by_value)
            except Exception:
                raise _Bail()
            inner = self.compile_inner(inner_field)
            inner_kind = "custom"
        try:
            choices_text = field.choices_text
        except Exception:
            raise _Bail()
        return {
            "kind": "enum",
            "enum_class": field.enum,
            "inner_kind": inner_kind,
            "inner": inner,
            "choices_text": choices_text,
        }

    # -- validators ---------------------------------------------------

    def compile_validator(self, validator):
        kind = _VALIDATORS.get(type(validator))
        if kind is None:
            if isinstance(validator, mm_validate.Validator):
                return {"kind": "custom", "validator": validator}
            if callable(validator):
                return {"kind": "custom", "validator": validator}
            raise _Bail()
        if kind == "length":
            return {
                "kind": kind,
                "min": validator.min,
                "max": validator.max,
                "equal": validator.equal,
                "messages": {
                    "min": validator.message_min,
                    "max": validator.message_max,
                    "all": validator.message_all,
                    "equal": validator.message_equal,
                },
                "error": validator.error,
            }
        if kind == "range":
            return {
                "kind": kind,
                "min_present": validator.min is not None,
                "max_present": validator.max is not None,
                "min": validator.min,
                "max": validator.max,
                "min_inclusive": bool(validator.min_inclusive),
                "max_inclusive": bool(validator.max_inclusive),
                "messages": {
                    "min": validator.message_min,
                    "max": validator.message_max,
                    "all": validator.message_all,
                },
                "error": validator.error,
            }
        if kind == "equal":
            return {
                "kind": kind,
                "comparable": validator.comparable,
                "message": validator.error,
            }
        if kind == "regexp":
            try:
                pattern_text = validator.regex.pattern
            except Exception:
                raise _Bail()
            return {
                "kind": kind,
                "pattern": validator.regex,
                "pattern_text": pattern_text,
                "message": validator.error,
            }
        if kind == "predicate":
            try:
                kwargs = dict(validator.kwargs)
            except Exception:
                raise _Bail()
            return {
                "kind": kind,
                "method": validator.method,
                "kwargs": kwargs,
                "message": validator.error,
            }
        if kind == "noneof":
            return {
                "kind": kind,
                "iterable": validator.iterable,
                "values_text": validator.values_text,
                "message": validator.error,
            }
        if kind == "oneof":
            return {
                "kind": kind,
                "choices": validator.choices,
                "choices_text": validator.choices_text,
                "labels_text": validator.labels_text,
                "message": validator.error,
            }
        if kind in ("containsonly", "containsnoneof"):
            key = "choices" if kind == "containsonly" else "iterable"
            return {
                "kind": kind,
                "items": getattr(validator, key),
                "choices_text": validator.choices_text
                if kind == "containsonly"
                else None,
                "labels_text": validator.labels_text
                if kind == "containsonly"
                else None,
                "values_text": validator.values_text
                if kind == "containsnoneof"
                else None,
                "message": validator.error,
            }
        if kind == "email":
            return {"kind": kind, "message": validator.error}
        if kind == "url":
            try:
                pattern = validator._regex(
                    relative=validator.relative,
                    absolute=validator.absolute,
                    require_tld=validator.require_tld,
                )
            except Exception:
                raise _Bail()
            return {
                "kind": kind,
                "relative": bool(validator.relative),
                "absolute": bool(validator.absolute),
                "schemes": set(validator.schemes),
                "pattern": pattern,
                "message": validator.error,
            }
        raise _Bail()
