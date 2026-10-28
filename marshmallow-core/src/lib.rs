//! `marshmallow-core`: Rust dump/load kernel for the marshmallow-rust port.
//!
//! The plan is built once from a Python dict (`compile`) into structs that
//! hold every host object the replay needs. Execution mirrors upstream
//! control flow exactly; every leaf operation that depends on CPython
//! semantics (conversions, comparisons, regex matches, user callables) is
//! a C-API call with identical observable behavior, including exceptions.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyTuple, PyTupleMethods,
};
use std::collections::HashMap;
mod exec;
pub use exec::Exec;

// ---------------------------------------------------------------------------
// Small C-API helpers with upstream-identical semantics
// ---------------------------------------------------------------------------


fn has_attr(obj: &Bound<PyAny>, name: &str) -> bool {
    obj.hasattr(name).unwrap_or(false)
}

/// `getattr(obj, key, default)`: only AttributeError falls back.
/// Non-string keys raise the same TypeError CPython raises.
fn getattr_default<'py>(
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let name = match key.downcast::<pyo3::types::PyString>() {
        Ok(s) => s.clone(),
        Err(_) => {
            let tp = key.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "<unknown>".to_string());
            return Err(PyTypeError::new_err(format!(
                "attribute name must be string, not '{tp}'"
            )));
        }
    };
    match obj.getattr(&name) {
        Ok(v) => Ok(v),
        Err(e) => {
            if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                Ok(default.clone())
            } else {
                Err(e)
            }
        }
    }
}

fn matches_gotten<'py>(
    py: Python<'py>,
    result: PyResult<Bound<'py, PyAny>>,
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            let is_lookup = e.is_instance_of::<pyo3::exceptions::PyKeyError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py)
                || e.is_instance_of::<PyTypeError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py);
            if is_lookup {
                getattr_default(obj, key, default)
            } else {
                Err(e)
            }
        }
    }
}

/// `marshmallow.utils._get_value_for_key`
fn get_value_for_key<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !has_attr(obj, "__getitem__") {
        return getattr_default(obj, key, default);
    }
    matches_gotten(py, obj.get_item(key), obj, key, default)
}

/// `marshmallow.utils.get_value` (dotted paths walk like `_get_value_for_keys`)
/// `marshmallow.utils.get_value` (dotted paths walk like `_get_value_for_keys`).
/// The `"." in key` check and the split both go through the key object so
/// exotic keys behave exactly like upstream.
pub(crate) fn get_value<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dot = pyo3::types::PyString::new(py, ".");
    if key.contains(&dot)? {
        let parts: Bound<PyAny> = key.call_method("split", (".",), None)?;
        let mut cur = obj.clone();
        for part in parts.try_iter()? {
            // No short-circuit: upstream recurses into the default value.
            cur = get_value_for_key(py, &cur, &part?, default)?;
        }
        return Ok(cur);
    }
    get_value_for_key(py, obj, key, default)
}

/// Call `func()` if callable else return it (`default() if callable`).
pub(crate) fn maybe_call<'py>(
    _py: Python,
    func: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if func.is_callable() {
        func.call0()
    } else {
        Ok(func.clone())
    }
}


/// `marshmallow.utils.set_value` with the identical conflict error.
/// Operates on bound values so exotic keys (`__contains__` overrides,
/// non-str keys) behave exactly like upstream.
pub(crate) fn set_value_any(
    py: Python,
    dct: &Bound<PyAny>,
    key: &Bound<PyAny>,
    value: Bound<PyAny>,
) -> PyResult<()> {
    let dot = pyo3::types::PyString::new(py, ".");
    if key.contains(&dot)? {
        let parts: Bound<PyAny> = key.call_method("split", (".", 1), None)?;
        let head = parts.get_item(0)?;
        let rest = parts.get_item(1)?;
        let target =
            dct.call_method("setdefault", (&head, PyDict::new(py)), None)?;
        if target.is_instance_of::<PyDict>() {
            set_value_any(py, &target, &rest, value)
        } else {
            let msg = format!(
                "Cannot set {} in {} due to existing value: {}",
                key.str()?,
                head.str()?,
                target.str()?
            );
            Err(PyValueError::new_err(msg))
        }
    } else {
        dct.set_item(key, value)?;
        Ok(())
    }
}

pub(crate) fn is_collection(py: Python, obj: &Bound<PyAny>) -> PyResult<bool> {
    let utils = py.import("marshmallow.utils")?;
    utils
        .getattr("is_collection")?
        .call1((obj,))?
        .extract::<bool>()
}

fn is_sequence_but_not_string(py: Python, obj: &Bound<PyAny>) -> PyResult<bool> {
    let utils = py.import("marshmallow.utils")?;
    utils
        .getattr("is_sequence_but_not_string")?
        .call1((obj,))?
        .extract::<bool>()
}

// ---------------------------------------------------------------------------
// Error tree (mirrors ErrorStore.store_error / merge_errors)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ErrKey {
    S(String),
    I(usize),
    K(Py<PyAny>),
}

const SCHEMA_KEY: &str = "_schema";

#[derive(Debug)]
pub enum Msg {
    ReadyStr(String),
    Ready(Py<PyAny>),
    Deferred {
        template: String,
        args: Vec<(String, Py<PyAny>)>,
        input: Py<PyAny>,
        join: bool,
    },
}

#[derive(Debug)]
pub enum ErrNode {
    Msgs(Vec<Msg>),
    Map(Vec<(ErrKey, ErrNode)>),
    One(Msg),
}

impl ErrNode {
    fn is_empty(&self) -> bool {
        match self {
            ErrNode::Msgs(m) => m.is_empty(),
            ErrNode::Map(m) => m.is_empty(),
            ErrNode::One(_) => false,
        }
    }
    fn is_map(&self) -> bool {
        matches!(self, ErrNode::Map(_))
    }
    fn as_map_mut(&mut self) -> Option<&mut Vec<(ErrKey, ErrNode)>> {
        match self {
            ErrNode::Map(m) => Some(m),
            _ => None,
        }
    }
}

fn key_eq(py: Python, a: &ErrKey, b: &ErrKey) -> bool {
    match (a, b) {
        (ErrKey::S(x), ErrKey::S(y)) => x == y,
        (ErrKey::I(x), ErrKey::I(y)) => x == y,
        (ErrKey::K(x), ErrKey::K(y)) => x.bind(py).eq(y.bind(py)).unwrap_or(false),
        _ => false,
    }
}

/// Direct port of `merge_errors`.
fn merge_errors(py: Python, mut e1: ErrNode, mut e2: ErrNode) -> ErrNode {
    if e1.is_empty() {
        return e2;
    }
    if e2.is_empty() {
        return e1;
    }
    let e1_is_list = matches!(e1, ErrNode::Msgs(_));
    let e2_is_list = matches!(e2, ErrNode::Msgs(_));
    let e1_is_map = e1.is_map();
    let e2_is_map = e2.is_map();
    if e1_is_list {
        if e2_is_list {
            let (ErrNode::Msgs(a), ErrNode::Msgs(b)) = (e1, e2) else {
                unreachable!()
            };
            let mut a = a;
            a.extend(b);
            return ErrNode::Msgs(a);
        }
        if e2_is_map {
            let sub = match e2.as_map_mut().unwrap().iter().position(|(k, _)| {
                matches!(k, ErrKey::S(s) if s == SCHEMA_KEY)
            }) {
                Some(pos) => {
                    let (_, old) = e2.as_map_mut().unwrap().remove(pos);
                    merge_errors(py, e1, old)
                }
                None => e1,
            };
            e2.as_map_mut().unwrap().push((
                ErrKey::S(SCHEMA_KEY.to_string()),
                sub,
            ));
            return e2;
        }
        // e2 is a single message: append it.
        let ErrNode::Msgs(mut a) = e1 else {
            unreachable!()
        };
        let one = match e2 {
            ErrNode::One(m) => m,
            _ => unreachable!(),
        };
        a.push(one);
        return ErrNode::Msgs(a);
    }
    if e1_is_map {
        if e2_is_map {
            let mut e2map = match e2 {
                ErrNode::Map(m) => m,
                _ => unreachable!(),
            };
            let e1map = match &mut e1 {
                ErrNode::Map(m) => m,
                _ => unreachable!(),
            };
            for (k2, v2) in e2map.drain(..) {
                if let Some(slot) = e1map.iter_mut().find(|(k1, _)| key_eq(py, k1, &k2)) {
                    let old = std::mem::replace(&mut slot.1, ErrNode::Msgs(vec![]));
                    slot.1 = merge_errors(py, old, v2);
                } else {
                    e1map.push((k2, v2));
                }
            }
            return e1;
        }
        let map = e1.as_map_mut().unwrap();
        let pos = map
            .iter()
            .position(|(k, _)| matches!(k, ErrKey::S(s) if s == SCHEMA_KEY));
        let old = pos.map(|p| map.remove(p).1).unwrap_or(ErrNode::Msgs(vec![]));
        let merged = merge_errors(py, old, e2);
        map.push((ErrKey::S(SCHEMA_KEY.to_string()), merged));
        return e1;
    }
    // e1 is a single message.
    if e2_is_list {
        let mut out = match e2 {
            ErrNode::Msgs(m) => m,
            _ => unreachable!(),
        };
        let one = match e1 {
            ErrNode::One(m) => m,
            _ => unreachable!(),
        };
        out.insert(0, one);
        return ErrNode::Msgs(out);
    }
    if e2_is_map {
        let sub = match e2.as_map_mut().unwrap().iter().position(|(k, _)| {
            matches!(k, ErrKey::S(s) if s == SCHEMA_KEY)
        }) {
            Some(pos) => {
                let (_, old) = e2.as_map_mut().unwrap().remove(pos);
                merge_errors(py, e1, old)
            }
            None => e1,
        };
        e2.as_map_mut().unwrap().push((
            ErrKey::S(SCHEMA_KEY.to_string()),
            sub,
        ));
        return e2;
    }
    let m1 = match e1 {
        ErrNode::One(m) => m,
        _ => unreachable!(),
    };
    let m2 = match e2 {
        ErrNode::One(m) => m,
        _ => unreachable!(),
    };
    ErrNode::Msgs(vec![m1, m2])
}

struct ErrStore {
    root: ErrNode,
}

impl ErrStore {
    fn new() -> Self {
        ErrStore {
            root: ErrNode::Map(vec![]),
        }
    }
    fn is_empty(&self) -> bool {
        self.root.is_empty()
    }
    /// Mirror of `ErrorStore.store_error`.
    fn store(&mut self, py: Python, messages: ErrNode, field: ErrKey, index: Option<usize>) {
        let mut node = if !(matches!(field, ErrKey::S(ref s) if s == SCHEMA_KEY) && messages.is_map()) {
            ErrNode::Map(vec![(field, messages)])
        } else {
            messages
        };
        if let Some(idx) = index {
            node = ErrNode::Map(vec![(ErrKey::I(idx), node)]);
        }
        let old = std::mem::replace(&mut self.root, ErrNode::Msgs(vec![]));
        self.root = merge_errors(py, old, node);
    }
}

// Convert the tree to the wire protocol (see marshmallow_rust.errors).
fn errkey_to_py(py: Python, key: &ErrKey) -> PyResult<Py<PyAny>> {
    match key {
        ErrKey::S(s) => {
            let inner = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$s").into_any(),
                    pyo3::types::PyString::new(py, s).into_any(),
                ],
            )?;
            Ok(inner.into_any().unbind())
        }
        ErrKey::I(i) => {
            let inner = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$i").into_any(),
                    pyo3::types::PyInt::new(py, *i).into_any(),
                ],
            )?;
            Ok(inner.into_any().unbind())
        }
        ErrKey::K(o) => {
            let inner = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$k").into_any(),
                    o.bind(py).clone().into_any(),
                ],
            )?;
            Ok(inner.into_any().unbind())
        }
    }
}


fn msg_to_py(py: Python, msg: &Msg) -> PyResult<Py<PyAny>> {
    match msg {
        Msg::ReadyStr(s) => Ok(pyo3::types::PyString::new(py, s).into_any().unbind()),
        Msg::Ready(o) => {
            let inner = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$ready").into_any(),
                    o.bind(py).clone().into_any(),
                ],
            )?;
            Ok(inner.into_any().unbind())
        }
        Msg::Deferred {
            template,
            args,
            input,
            join,
        } => {
            let d = PyDict::new(py);
            for (k, v) in args {
                d.set_item(k, v.bind(py))?;
            }
            let inner = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$deferred").into_any(),
                    pyo3::types::PyString::new(py, template).into_any(),
                    d.into_any(),
                    input.bind(py).clone().into_any(),
                    pyo3::types::PyBool::new(py, *join).to_owned().into_any(),
                ],
            )?;
            Ok(inner.into_any().unbind())
        }
    }
}

pub fn node_to_py(py: Python, node: &ErrNode) -> PyResult<Py<PyAny>> {
    match node {
        ErrNode::Msgs(ms) => {
            let list = PyList::empty(py);
            for m in ms {
                list.append(msg_to_py(py, m)?)?;
            }
            Ok(list.into_any().unbind())
        }
        ErrNode::Map(pairs) => {
            let list = PyList::empty(py);
            for (k, v) in pairs {
                let pair = PyTuple::new(py, [errkey_to_py(py, k)?, node_to_py(py, v)?])?;
                list.append(pair)?;
            }
            let outer = PyTuple::new(
                py,
                [
                    pyo3::types::PyString::new(py, "$map").into_any(),
                    list.into_any(),
                ],
            )?;
            Ok(outer.into_any().unbind())
        }
        ErrNode::One(m) => msg_to_py(py, m),
    }
}

// ---------------------------------------------------------------------------
// Plan structs
// ---------------------------------------------------------------------------

struct Lib {
    int_cls: Py<PyAny>,
    float_cls: Py<PyAny>,
    str_cls: Py<PyAny>,
    bool_cls: Py<PyAny>,
    decimal_cls: Py<PyAny>,
    datetime_cls: Py<PyAny>,
    date_cls: Py<PyAny>,
    time_cls: Py<PyAny>,
    timedelta_cls: Py<PyAny>,
    uuid_cls: Py<PyAny>,
    ip_address_fn: Py<PyAny>,
    ip_interface_fn: Py<PyAny>,
    fromiso_datetime: Py<PyAny>,
    fromiso_date: Py<PyAny>,
    fromiso_time: Py<PyAny>,
    strptime_fn: Py<PyAny>,
    format_datetime_fn: Py<PyAny>,
    parsedate_to_datetime_fn: Py<PyAny>,
    from_timestamp_fn: Py<PyAny>,
    from_timestamp_ms_fn: Py<PyAny>,
    timestamp_fn: Py<PyAny>,
    timestamp_ms_fn: Py<PyAny>,
    timedelta_to_microseconds_fn: Py<PyAny>,
    ensure_text_type_fn: Py<PyAny>,
    is_aware_fn: Py<PyAny>,
    numbers_integral_abc: Py<PyAny>,
    mapping_abc: Py<PyAny>,
    decimal_invalid_operation: Py<PyAny>,
    validation_error_cls: Py<PyAny>,
    missing: Py<PyAny>,
    email_user_regex: Py<PyAny>,
    email_domain_regex: Py<PyAny>,
    email_whitelist: Py<PyAny>,
}

impl Lib {
    fn resolve(py: Python) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let dt = py.import("datetime")?;
        let mm_utils = py.import("marshmallow.utils")?;
        let mm_exc = py.import("marshmallow.exceptions")?;
        let mm_const = py.import("marshmallow.constants")?;
        let ipaddress = py.import("ipaddress")?;
        let email_utils = py.import("email.utils")?;
        let mm_validate = py.import("marshmallow.validate")?;
        let numbers = py.import("numbers")?;
        let abc = py.import("collections.abc")?;
        let decimal_mod = py.import("decimal")?;
        Ok(Lib {
            int_cls: builtins.getattr("int")?.unbind(),
            float_cls: builtins.getattr("float")?.unbind(),
            str_cls: builtins.getattr("str")?.unbind(),
            bool_cls: builtins.getattr("bool")?.unbind(),
            decimal_cls: decimal_mod.getattr("Decimal")?.unbind(),
            datetime_cls: dt.getattr("datetime")?.unbind(),
            date_cls: dt.getattr("date")?.unbind(),
            time_cls: dt.getattr("time")?.unbind(),
            timedelta_cls: dt.getattr("timedelta")?.unbind(),
            uuid_cls: py.import("uuid")?.getattr("UUID")?.unbind(),
            ip_address_fn: ipaddress.getattr("ip_address")?.unbind(),
            ip_interface_fn: ipaddress.getattr("ip_interface")?.unbind(),
            strptime_fn: dt.getattr("datetime")?.getattr("strptime")?.unbind(),
            fromiso_datetime: dt.getattr("datetime")?.getattr("fromisoformat")?.unbind(),
            fromiso_date: dt.getattr("date")?.getattr("fromisoformat")?.unbind(),
            fromiso_time: dt.getattr("time")?.getattr("fromisoformat")?.unbind(),
            format_datetime_fn: email_utils.getattr("format_datetime")?.unbind(),
            parsedate_to_datetime_fn: email_utils.getattr("parsedate_to_datetime")?.unbind(),
            from_timestamp_fn: mm_utils.getattr("from_timestamp")?.unbind(),
            from_timestamp_ms_fn: mm_utils.getattr("from_timestamp_ms")?.unbind(),
            timestamp_fn: mm_utils.getattr("timestamp")?.unbind(),
            timestamp_ms_fn: mm_utils.getattr("timestamp_ms")?.unbind(),
            timedelta_to_microseconds_fn: mm_utils.getattr("timedelta_to_microseconds")?.unbind(),
            ensure_text_type_fn: mm_utils.getattr("ensure_text_type")?.unbind(),
            is_aware_fn: mm_utils.getattr("is_aware")?.unbind(),
            numbers_integral_abc: numbers.getattr("Integral")?.unbind(),
            mapping_abc: abc.getattr("Mapping")?.unbind(),
            decimal_invalid_operation: decimal_mod.getattr("InvalidOperation")?.unbind(),
            validation_error_cls: mm_exc.getattr("ValidationError")?.unbind(),
            missing: mm_const.getattr("missing")?.unbind(),
            email_user_regex: mm_validate.getattr("Email")?.getattr("USER_REGEX")?.unbind(),
            email_domain_regex: mm_validate.getattr("Email")?.getattr("DOMAIN_REGEX")?.unbind(),
            email_whitelist: mm_validate.getattr("Email")?.getattr("DOMAIN_WHITELIST")?.unbind(),
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum UnknownOpt {
    Raise,
    Exclude,
    Include,
    /// Garbage values behave like exclude (matches upstream comparisons).
    Other,
}

#[derive(Clone, Copy, PartialEq)]
enum Precision {
    Weeks,
    Days,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
    Microseconds,
}

impl Precision {
    fn micros_per_unit(&self) -> i64 {
        match self {
            Precision::Weeks => 1000000 * 60 * 60 * 24 * 7,
            Precision::Days => 1000000 * 60 * 60 * 24,
            Precision::Hours => 1000000 * 60 * 60,
            Precision::Minutes => 1000000 * 60,
            Precision::Seconds => 1000000,
            Precision::Milliseconds => 1000,
            Precision::Microseconds => 1,
        }
    }
}

#[derive(Clone)]
enum DateFmt {
    Iso,
    Rfc,
    Timestamp,
    TimestampMs,
    Custom(String),
}

struct TemporalSpec {
    fmt: DateFmt,
}

struct FieldPlan {
    name: Option<String>,
    data_key: Option<Py<PyAny>>,
    attribute: Option<Py<PyAny>>,
    required: bool,
    allow_none: bool,
    load_default_present: bool,
    load_default: Py<PyAny>,
    dump_default_present: bool,
    dump_default: Py<PyAny>,
    required_msg: String,
    null_msg: String,
    error_messages: HashMap<String, String>,
    validators: Vec<ValidatorPlan>,
    pre_load: Vec<Py<PyAny>>,
    post_load: Vec<Py<PyAny>>,
    check_attribute: bool,
    constant_missing_check_only_null: bool,
    codec: Codec,
}

enum Codec {
    Raw,
    Str,
    Uuid,
    Integer { strict: bool, as_string: bool },
    Float { allow_nan: bool, as_string: bool },
    Decimal { places: Option<Py<PyAny>>, rounding: Option<Py<PyAny>>, allow_nan: bool, as_string: bool },
    Boolean { truthy: Py<PyAny>, falsy: Py<PyAny> },
    DateTime(TemporalSpec),
    NaiveDateTime { spec: TemporalSpec, timezone: Option<Py<PyAny>> },
    AwareDateTime { spec: TemporalSpec, default_timezone: Option<Py<PyAny>> },
    Time(TemporalSpec),
    Date(TemporalSpec),
    TimeDelta { precision: Precision },
    Nested { target: usize, many: bool, unknown: Option<UnknownOpt> },
    Pluck { target: usize, many: bool, unknown: Option<UnknownOpt>, pluck_key: String },
    List { inner: Box<FieldPlan> },
    Tuple { inners: Vec<FieldPlan>, length_msg: String },
    Mapping { mapping_type: Py<PyAny>, key: Option<Box<FieldPlan>>, value: Option<Box<FieldPlan>> },
    Ip { exploded: bool, deser_class: Option<Py<PyAny>> },
    IpInterface { exploded: bool, deser_class: Option<Py<PyAny>> },
    Enum { enum_class: Py<PyAny>, inner_kind: EnumInner, choices_text: String },
    Method { serialize_method: Option<Py<PyAny>>, deserialize_method: Option<Py<PyAny>> },
    Function { serialize_func: Option<Py<PyAny>>, deserialize_func: Option<Py<PyAny>> },
    Constant { constant: Py<PyAny> },
    Opaque { field: Py<PyAny> },
}

#[derive(Clone)]
enum EnumInner {
    Name,
    Raw,
    Custom(usize), // index into shared custom-inner pool on Plan
}

struct ValidatorPlan {
    kind: ValidatorKind,
}

enum ValidatorKind {
    Length { min: Option<Py<PyAny>>, max: Option<Py<PyAny>>, equal: Option<Py<PyAny>>, msgs: LenMsgs, error: Option<String> },
    Range { min_present: bool, max_present: bool, min: Option<Py<PyAny>>, max: Option<Py<PyAny>>, min_inclusive: bool, max_inclusive: bool, msgs: RangeMsgs, error: Option<String> },
    Equal { comparable: Py<PyAny>, message: String },
    Regexp { pattern: Py<PyAny>, pattern_text: String, message: String },
    Predicate { method: String, kwargs: Py<PyAny>, message: String },
    NoneOf { iterable: Py<PyAny>, values_text: String, message: String },
    OneOf { choices: Py<PyAny>, choices_text: String, labels_text: String, message: String },
    ContainsOnly { choices: Py<PyAny>, choices_text: String, labels_text: String, message: String },
    ContainsNoneOf { iterable: Py<PyAny>, values_text: String, message: String },
    EmailV { message: String },
    UrlV { relative: bool, absolute: bool, schemes: Py<PyAny>, pattern: Py<PyAny>, message: String },
    Custom { validator: Py<PyAny> },
}

struct LenMsgs { min: String, max: String, all: String, equal: String }
struct RangeMsgs { min: String, max: String, all: String }

struct NodePlan {
    dump_fields: Vec<FieldPlan>,
    load_fields: Vec<FieldPlan>,
    dict_class: Py<PyAny>,
    partial: Py<PyAny>,
    unknown: UnknownOpt,
    unknown_msg: String,
    type_msg: String,
    index_errors: bool,
    get_attribute: Py<PyAny>,
    get_attribute_is_default: bool,
}

pub struct Plan {
    nodes: Vec<NodePlan>,
    custom_enum_inners: Vec<FieldPlan>,
    lib: Lib,
}

// ---------------------------------------------------------------------------
// Plan building from the Python dict
// ---------------------------------------------------------------------------

fn get_str(d: &Bound<PyDict>, key: &str) -> PyResult<String> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("plan missing {key}")))?
        .extract::<String>()
}

fn get_bool(d: &Bound<PyDict>, key: &str) -> PyResult<bool> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("plan missing {key}")))?
        .extract::<bool>()
}

fn get_opt_py(d: &Bound<PyDict>, key: &str) -> PyResult<Option<Py<PyAny>>> {
    match d.get_item(key)? {
        Some(v) => {
            if v.is_none() {
                Ok(None)
            } else {
                Ok(Some(v.unbind()))
            }
        }
        None => Ok(None),
    }
}

fn get_py<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("plan missing {key}")))
}

fn get_opt_string(d: &Bound<PyDict>, key: &str) -> PyResult<Option<String>> {
    match d.get_item(key)? {
        Some(v) => {
            if v.is_none() {
                Ok(None)
            } else {
                Ok(Some(v.extract::<String>()?))
            }
        }
        None => Ok(None),
    }
}

impl UnknownOpt {
    pub fn parse(s: &str) -> UnknownOpt {
        match s {
            "raise" => UnknownOpt::Raise,
            "exclude" => UnknownOpt::Exclude,
            "include" => UnknownOpt::Include,
            _ => UnknownOpt::Other,
        }
    }
}

fn parse_precision(s: &str) -> PyResult<Precision> {
    match s {
        "weeks" => Ok(Precision::Weeks),
        "days" => Ok(Precision::Days),
        "hours" => Ok(Precision::Hours),
        "minutes" => Ok(Precision::Minutes),
        "seconds" => Ok(Precision::Seconds),
        "milliseconds" => Ok(Precision::Milliseconds),
        "microseconds" => Ok(Precision::Microseconds),
        _ => Err(PyValueError::new_err("bad precision")),
    }
}

impl Plan {
    pub fn from_dict(py: Python, plan: &Bound<PyAny>) -> PyResult<Self> {
        let lib = Lib::resolve(py)?;
        let nodes_obj: Bound<PyAny> = plan.get_item("nodes")?;
        let nodes_list = nodes_obj.downcast::<PyList>()?;
        let mut this = Plan {
            nodes: vec![],
            custom_enum_inners: vec![],
            lib,
        };
        for node in nodes_list.iter() {
            if node.is_none() {
                return Err(PyValueError::new_err("unresolved backref placeholder"));
            }
            let built = this.build_node(py, &node.downcast::<PyDict>()?.clone())?;
            this.nodes.push(built);
        }
        Ok(this)
    }

    fn build_node(&mut self, py: Python, d: &Bound<PyDict>) -> PyResult<NodePlan> {
        let dump_raw: Bound<PyAny> = get_py(d, "dump_fields")?;
        let load_raw: Bound<PyAny> = get_py(d, "load_fields")?;
        let mut dump_fields = vec![];
        for f in dump_raw.downcast::<PyList>()?.iter() {
            dump_fields.push(self.build_field(py, &f.downcast::<PyDict>()?.clone())?);
        }
        let mut load_fields = vec![];
        for f in load_raw.downcast::<PyList>()?.iter() {
            load_fields.push(self.build_field(py, &f.downcast::<PyDict>()?.clone())?);
        }
        Ok(NodePlan {
            dump_fields,
            load_fields,
            dict_class: get_py(d, "dict_class")?.unbind(),
            partial: get_py(d, "partial")?.unbind(),
            unknown: UnknownOpt::parse(&get_str(d, "unknown")?),
            unknown_msg: get_str(d, "unknown_msg")?,
            type_msg: get_str(d, "type_msg")?,
            index_errors: get_bool(d, "index_errors")?,
            get_attribute: get_py(d, "get_attribute")?.unbind(),
            get_attribute_is_default: get_bool(d, "get_attribute_is_default")?,
        })
    }

    fn build_field(&mut self, py: Python, d: &Bound<PyDict>) -> PyResult<FieldPlan> {
        let name = get_opt_string(d, "name")?;
        let data_key = match d.get_item("data_key")? {
            Some(v) => {
                if v.is_none() {
                    None
                } else {
                    Some(v.unbind())
                }
            }
            None => None,
        };
        let attribute = get_opt_py(d, "attribute")?;
        let (load_present, load_default) = {
            let t: Bound<PyAny> = get_py(d, "load_default")?;
            let tup = t.downcast::<PyTuple>()?;
            (
                tup.get_item(0)?.extract::<bool>()?,
                tup.get_item(1)?.unbind(),
            )
        };
        let (dump_present, dump_default) = {
            let t: Bound<PyAny> = get_py(d, "dump_default")?;
            let tup = t.downcast::<PyTuple>()?;
            (
                tup.get_item(0)?.extract::<bool>()?,
                tup.get_item(1)?.unbind(),
            )
        };
        let errmsgs: Bound<PyAny> = get_py(d, "error_messages")?;
        let mut error_messages = HashMap::new();
        for (k, v) in errmsgs.downcast::<PyDict>()?.iter() {
            error_messages.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        let validators_raw: Bound<PyAny> = get_py(d, "validators")?;
        let mut validators = vec![];
        for v in validators_raw.downcast::<PyList>()?.iter() {
            validators.push(self.build_validator(py, &v.downcast::<PyDict>()?.clone())?);
        }
        let pre_raw: Bound<PyAny> = get_py(d, "pre_load")?;
        let post_raw: Bound<PyAny> = get_py(d, "post_load")?;
        let codec_raw: Bound<PyAny> = get_py(d, "codec")?;
        let codec_d = codec_raw.downcast::<PyDict>()?.clone();
        let kind = get_str(&codec_d, "kind")?;
        let codec = self.build_codec(py, &kind, &codec_d)?;
        let constant_missing_check_only_null = kind == "constant";
        Ok(FieldPlan {
            name,
            data_key,
            attribute,
            required: get_bool(d, "required")?,
            allow_none: get_bool(d, "allow_none")?,
            load_default_present: load_present,
            load_default,
            dump_default_present: dump_present,
            dump_default,
            required_msg: get_str(d, "required_msg")?,
            null_msg: get_str(d, "null_msg")?,
            error_messages,
            validators,
            pre_load: pre_raw
                .downcast::<PyList>()?
                .iter()
                .map(|v| v.unbind())
                .collect(),
            post_load: post_raw
                .downcast::<PyList>()?
                .iter()
                .map(|v| v.unbind())
                .collect(),
            check_attribute: get_bool(d, "check_attribute")?,
            constant_missing_check_only_null,
            codec,
        })
    }

    fn build_validator(&mut self, _py: Python, d: &Bound<PyDict>) -> PyResult<ValidatorPlan> {
        let kind = get_str(d, "kind")?;
        let k = match kind.as_str() {
            "length" => {
                let msgs_raw: Bound<PyAny> = get_py(d, "messages")?;
                let msgs = msgs_raw.downcast::<PyDict>()?;
                ValidatorKind::Length {
                    min: get_opt_py(d, "min")?,
                    max: get_opt_py(d, "max")?,
                    equal: get_opt_py(d, "equal")?,
                    msgs: LenMsgs {
                        min: get_str(msgs, "min")?,
                        max: get_str(msgs, "max")?,
                        all: get_str(msgs, "all")?,
                        equal: get_str(msgs, "equal")?,
                    },
                    error: get_opt_string(d, "error")?,
                }
            }
            "range" => {
                let msgs_raw: Bound<PyAny> = get_py(d, "messages")?;
                let msgs = msgs_raw.downcast::<PyDict>()?;
                ValidatorKind::Range {
                    min_present: get_bool(d, "min_present")?,
                    max_present: get_bool(d, "max_present")?,
                    min: get_opt_py(d, "min")?,
                    max: get_opt_py(d, "max")?,
                    min_inclusive: get_bool(d, "min_inclusive")?,
                    max_inclusive: get_bool(d, "max_inclusive")?,
                    msgs: RangeMsgs {
                        min: get_str(msgs, "min")?,
                        max: get_str(msgs, "max")?,
                        all: get_str(msgs, "all")?,
                    },
                    error: get_opt_string(d, "error")?,
                }
            }
            "equal" => ValidatorKind::Equal {
                comparable: get_py(d, "comparable")?.unbind(),
                message: get_str(d, "message")?,
            },
            "regexp" => ValidatorKind::Regexp {
                pattern: get_py(d, "pattern")?.unbind(),
                pattern_text: get_str(d, "pattern_text")?,
                message: get_str(d, "message")?,
            },
            "predicate" => ValidatorKind::Predicate {
                method: get_str(d, "method")?,
                kwargs: get_py(d, "kwargs")?.unbind(),
                message: get_str(d, "message")?,
            },
            "noneof" => ValidatorKind::NoneOf {
                iterable: get_py(d, "iterable")?.unbind(),
                values_text: get_str(d, "values_text")?,
                message: get_str(d, "message")?,
            },
            "oneof" => ValidatorKind::OneOf {
                choices: get_py(d, "choices")?.unbind(),
                choices_text: get_str(d, "choices_text")?,
                labels_text: get_str(d, "labels_text")?,
                message: get_str(d, "message")?,
            },
            "containsonly" => ValidatorKind::ContainsOnly {
                choices: get_py(d, "items")?.unbind(),
                choices_text: get_str(d, "choices_text")?,
                labels_text: get_str(d, "labels_text")?,
                message: get_str(d, "message")?,
            },
            "containsnoneof" => ValidatorKind::ContainsNoneOf {
                iterable: get_py(d, "items")?.unbind(),
                values_text: get_str(d, "values_text")?,
                message: get_str(d, "message")?,
            },
            "email" => ValidatorKind::EmailV {
                message: get_str(d, "message")?,
            },
            "url" => ValidatorKind::UrlV {
                relative: get_bool(d, "relative")?,
                absolute: get_bool(d, "absolute")?,
                schemes: get_py(d, "schemes")?.unbind(),
                pattern: get_py(d, "pattern")?.unbind(),
                message: get_str(d, "message")?,
            },
            "custom" => ValidatorKind::Custom {
                validator: get_py(d, "validator")?.unbind(),
            },
            _ => return Err(PyValueError::new_err("bad validator kind")),
        };
        Ok(ValidatorPlan { kind: k })
    }

    fn build_codec(
        &mut self,
        py: Python,
        kind: &str,
        d: &Bound<PyDict>,
    ) -> PyResult<Codec> {
        Ok(match kind {
            "raw" => Codec::Raw,
            "string" | "url" | "email" => Codec::Str,
            "uuid" => Codec::Uuid,
            "integer" => Codec::Integer {
                strict: get_bool(d, "strict")?,
                as_string: get_bool(d, "as_string")?,
            },
            "float" => Codec::Float {
                allow_nan: get_bool(d, "allow_nan")?,
                as_string: get_bool(d, "as_string")?,
            },
            "decimal" => Codec::Decimal {
                places: get_opt_py(d, "places")?,
                rounding: get_opt_py(d, "rounding")?,
                allow_nan: get_bool(d, "allow_nan")?,
                as_string: get_bool(d, "as_string")?,
            },
            "boolean" => Codec::Boolean {
                truthy: get_py(d, "truthy")?.unbind(),
                falsy: get_py(d, "falsy")?.unbind(),
            },
            "datetime" => Codec::DateTime(self.build_temporal(d)?),
            "naive_datetime" => Codec::NaiveDateTime {
                spec: self.build_temporal(d)?,
                timezone: get_opt_py(d, "timezone")?,
            },
            "aware_datetime" => Codec::AwareDateTime {
                spec: self.build_temporal(d)?,
                default_timezone: get_opt_py(d, "default_timezone")?,
            },
            "time" => Codec::Time(self.build_temporal(d)?),
            "date" => Codec::Date(self.build_temporal(d)?),
            "timedelta" => Codec::TimeDelta {
                precision: parse_precision(&get_str(d, "precision")?)?,
            },
            "nested" => Codec::Nested {
                target: self.build_target(d)?,
                many: get_bool(d, "many")?,
                unknown: match d.get_item("unknown")? {
                    Some(v) => {
                        if v.is_none() {
                            None
                        } else {
                            Some(UnknownOpt::parse(&v.extract::<String>()?))
                        }
                    }
                    None => None,
                },
            },
            "pluck" => Codec::Pluck {
                target: self.build_target(d)?,
                many: get_bool(d, "many")?,
                unknown: match d.get_item("unknown")? {
                    Some(v) => {
                        if v.is_none() {
                            None
                        } else {
                            Some(UnknownOpt::parse(&v.extract::<String>()?))
                        }
                    }
                    None => None,
                },
                pluck_key: get_str(d, "pluck_key")?,
            },
            "list" => {
                let inner_raw: Bound<PyAny> = get_py(d, "inner")?;
                Codec::List {
                    inner: Box::new(self.build_field(py, &inner_raw.downcast::<PyDict>()?.clone())?),
                }
            }
            "tuple" => {
                let inners_raw: Bound<PyAny> = get_py(d, "inners")?;
                let mut inners = vec![];
                for f in inners_raw.downcast::<PyList>()?.iter() {
                    inners.push(self.build_field(py, &f.downcast::<PyDict>()?.clone())?);
                }
                Codec::Tuple {
                    inners,
                    length_msg: get_str(d, "length_msg")?,
                }
            }
            "mapping" | "dict" => {
                let key = match get_opt_py(d, "key")? {
                    Some(k) => {
                        let kd = k.bind(py).downcast::<PyDict>()?.clone();
                        Some(Box::new(self.build_field(py, &kd)?))
                    }
                    None => None,
                };
                let value = match get_opt_py(d, "value")? {
                    Some(v) => {
                        let vd = v.bind(py).downcast::<PyDict>()?.clone();
                        Some(Box::new(self.build_field(py, &vd)?))
                    }
                    None => None,
                };
                Codec::Mapping {
                    mapping_type: get_py(d, "mapping_type")?.unbind(),
                    key,
                    value,
                }
            }
            "ip" | "ipv4" | "ipv6" => Codec::Ip {
                exploded: get_bool(d, "exploded")?,
                deser_class: get_opt_py(d, "deser_class")?,
            },
            "ipinterface" | "ipv4interface" | "ipv6interface" => Codec::IpInterface {
                exploded: get_bool(d, "exploded")?,
                deser_class: get_opt_py(d, "deser_class")?,
            },
            "enum" => {
                let inner_kind = get_str(d, "inner_kind")?;
                let inner = match inner_kind.as_str() {
                    "name" => EnumInner::Name,
                    "raw" => EnumInner::Raw,
                    "custom" => {
                        let inner_raw: Bound<PyAny> = get_py(d, "inner")?;
                        let spec =
                            self.build_field(py, &inner_raw.downcast::<PyDict>()?.clone())?;
                        let idx = self.custom_enum_inners.len();
                        self.custom_enum_inners.push(spec);
                        EnumInner::Custom(idx)
                    }
                    _ => return Err(PyValueError::new_err("bad enum inner")),
                };
                Codec::Enum {
                    enum_class: get_py(d, "enum_class")?.unbind(),
                    inner_kind: inner,
                    choices_text: get_str(d, "choices_text")?,
                }
            }
            "method" => Codec::Method {
                serialize_method: get_opt_py(d, "serialize_method")?,
                deserialize_method: get_opt_py(d, "deserialize_method")?,
            },
            "function" => Codec::Function {
                serialize_func: get_opt_py(d, "serialize_func")?,
                deserialize_func: get_opt_py(d, "deserialize_func")?,
            },
            "constant" => Codec::Constant {
                constant: get_py(d, "constant")?.unbind(),
            },
            "opaque" => Codec::Opaque {
                field: get_py(d, "field")?.unbind(),
            },
            _ => return Err(PyValueError::new_err("bad codec kind")),
        })
    }

    fn build_temporal(&mut self, d: &Bound<PyDict>) -> PyResult<TemporalSpec> {
        let fmt = get_str(d, "format")?;
        let parsed = match fmt.as_str() {
            "iso" | "iso8601" => DateFmt::Iso,
            "rfc" | "rfc822" => DateFmt::Rfc,
            "timestamp" => DateFmt::Timestamp,
            "timestamp_ms" => DateFmt::TimestampMs,
            other => DateFmt::Custom(other.to_string()),
        };
        Ok(TemporalSpec { fmt: parsed })
    }

    fn build_target(&mut self, d: &Bound<PyDict>) -> PyResult<usize> {
        let t: Bound<PyAny> = get_py(d, "target")?;
        if let Ok(i) = t.extract::<usize>() {
            return Ok(i);
        }
        let td = t.downcast::<PyDict>()?;
        let b: Bound<PyAny> = td
            .get_item("backref")?
            .ok_or_else(|| PyValueError::new_err("bad target"))?;
        b.extract::<usize>()
    }
}
