//! Execution engine: replays upstream dump/load control flow over a plan.
//!
//! Every leaf that depends on CPython semantics (conversions, comparisons,
//! regex matches, user callables) is a C-API call with identical observable
//! behavior, including which exception type propagates.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyListMethods,
    PyString, PyStringMethods, PyTuple, PyTupleMethods, PyType, PyTypeMethods,
};

use crate::{
    is_collection, is_sequence_but_not_string, Codec, DateFmt, EnumInner, ErrKey, ErrNode,
    ErrStore, FieldPlan, Msg, NodePlan, Plan, Precision, UnknownOpt, ValidatorKind, ValidatorPlan,
};

pub enum DumpOut<'py> {
    Value(Bound<'py, PyAny>),
    Skip,
}

pub enum LoadOut {
    Value,
    Missing,
}

enum FieldErr {
    Ve { node: ErrNode, valid_data: Option<Py<PyAny>> },
    Other(PyErr),
}

impl From<PyErr> for FieldErr {
    fn from(e: PyErr) -> Self {
        FieldErr::Other(e)
    }
}

struct Partial<'py> {
    raw: Bound<'py, PyAny>,
    is_none: bool,
    is_true: bool,
    is_coll: bool,
}

pub struct Exec<'a, 'py> {
    py: Python<'py>,
    plan: &'a Plan,
    builtins: Bound<'py, PyAny>,
}

impl<'a, 'py> Exec<'a, 'py> {
    pub fn new(py: Python<'py>, plan: &'a Plan) -> PyResult<Self> {
        Ok(Exec {
            py,
            plan,
            builtins: py.import("builtins")?.into_any(),
        })
    }

    fn lib(&self) -> &crate::Lib {
        &self.plan.lib
    }

    fn is_instance(&self, obj: &Bound<'py, PyAny>, cls: &Py<PyAny>) -> PyResult<bool> {
        obj.is_instance(&cls.bind(self.py).downcast::<PyType>()?.clone())
    }

    fn missing(&self) -> Bound<'py, PyAny> {
        self.lib().missing.bind(self.py).clone()
    }

    fn is_missing(&self, v: &Bound<'py, PyAny>) -> bool {
        v.is(self.lib().missing.bind(self.py))
    }

    fn is_ve(&self, e: &PyErr) -> bool {
        e.is_instance(self.py, self.lib().validation_error_cls.bind(self.py))
    }

    fn none(&self) -> Bound<'py, PyAny> {
        self.py.None().bind(self.py).clone()
    }
    /// Exact-dict result store (C setitem, no override dispatch); dotted
    /// keys go through set_value_any (nested dicts); otherwise the generic
    /// path that respects __setitem__ overrides.
    fn dict_set(
        &self,
        ret: &Bound<'py, PyAny>,
        ret_dict: Option<&Bound<PyDict>>,
        key: &Bound<'py, PyAny>,
        val: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let py = self.py;
        if fast_key_is_dotted(key)? {
            return crate::set_value_any(py, ret, key, val);
        }
        if let Some(rd) = ret_dict {
            rd.set_item(key, val)
        } else {
            ret.set_item(key, val)
        }
    }

    // -- messages ---------------------------------------------------

    /// Deferred message: formats exactly like the upstream
    /// make_error/_format_error site with the same template and objects.
    fn msg(&self, template: &str, kwargs: Vec<(String, Bound<'py, PyAny>)>) -> Msg {
        Msg::Deferred {
            template: template.to_string(),
            args: kwargs.into_iter().map(|(k, v)| (k, v.unbind())).collect(),
            input: self.py.None(),
            join: false,
        }
    }

    fn msg_in(
        &self,
        template: &str,
        input: &Bound<'py, PyAny>,
        extra: Vec<(String, Bound<'py, PyAny>)>,
    ) -> Msg {
        let mut args = vec![("input".to_string(), input.clone().unbind())];
        args.extend(extra.into_iter().map(|(k, v)| (k, v.unbind())));
        Msg::Deferred {
            template: template.to_string(),
            args,
            input: input.clone().unbind(),
            join: false,
        }
    }

    fn msg_join(
        &self,
        template: &str,
        input: &Bound<'py, PyAny>,
        extra: Vec<(String, Bound<'py, PyAny>)>,
    ) -> Msg {
        let mut args = vec![("input".to_string(), input.clone().unbind())];
        args.extend(extra.into_iter().map(|(k, v)| (k, v.unbind())));
        Msg::Deferred {
            template: template.to_string(),
            args,
            input: input.clone().unbind(),
            join: true,
        }
    }

    fn tmpl(&self, fp: &FieldPlan, key: &str, default: &str) -> String {
        fp.error_messages.get(key).cloned().unwrap_or_else(|| default.to_string())
    }
    fn ve_msg(&self, template: String, kwargs: Vec<(String, Bound<'py, PyAny>)>) -> FieldErr {
        FieldErr::Ve {
            node: ErrNode::Msgs(vec![self.msg(&template, kwargs)]),
            valid_data: None,
        }
    }

    // -- dump -------------------------------------------------------

    pub fn dump_root(
        &self,
        node_idx: usize,
        obj: &Bound<'py, PyAny>,
        many: bool,
    ) -> PyResult<Py<PyAny>> {
        let node = &self.plan.nodes[node_idx];
        // `if many and obj is not None`
        if many && !obj.is_none() {
            let list = PyList::empty(self.py);
            for item in obj.try_iter()? {
                list.append(self.dump_one(node, &item?)?)?;
            }
            return Ok(list.into_any().unbind());
        }
        Ok(self.dump_one(node, obj)?.unbind())
    }

    fn dump_one(&self, node: &NodePlan, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let ret: Bound<PyAny> = node.dict_class.bind(self.py).call0()?;
        let obj_dict: Option<Bound<PyDict>> = if is_exact_dict(obj) {
            obj.downcast::<PyDict>().ok().cloned()
        } else {
            None
        };
        let ret_dict: Option<Bound<PyDict>> = if is_exact_dict(&ret) {
            ret.downcast::<PyDict>().ok().cloned()
        } else {
            None
        };
        for fp in &node.dump_fields {
            match self.field_dump(node, fp, obj, obj_dict.as_ref())? {
                DumpOut::Skip => {}
                DumpOut::Value(v) => {
                    let key = match &fp.data_key {
                        Some(k) => k.bind(self.py).clone(),
                        None => return Err(pyo3::exceptions::PyTypeError::new_err("no key")),
                    };
                    self.dict_set(&ret, ret_dict.as_ref(), &key, v)?;
                }
            }
        }
        Ok(ret.into_any())
    }

    /// Dump extraction key: attribute if not None else name (None-check).
    fn extract_key(&self, fp: &FieldPlan) -> PyResult<Bound<'py, PyAny>> {
        match &fp.attribute {
            Some(a) => Ok(a.bind(self.py).clone()),
            None => match &fp.name {
                Some(n) => Ok(pyo3::types::PyString::new(self.py, n).into_any()),
                None => Err(pyo3::exceptions::PyTypeError::new_err("no key")),
            },
        }
    }

    fn field_dump(
        &self,
        node: &NodePlan,
        fp: &FieldPlan,
        obj: &Bound<'py, PyAny>,
        obj_dict: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<DumpOut<'py>> {
        let py = self.py;
        if let Codec::Opaque { field } = &fp.codec {
            // Full serialize() with the node's bound get_attribute.
            let name = self.extract_key(fp)?;
            let got = if node.get_attribute_is_default {
                field.bind(py).call_method("serialize", (name, obj), None)?
            } else {
                let kwargs = PyDict::new(py);
                kwargs.set_item("accessor", &node.get_attribute)?;
                field.bind(py).call_method("serialize", (name, obj), Some(&kwargs))?
            };
            if self.is_missing(&got) {
                return Ok(DumpOut::Skip);
            }
            return Ok(DumpOut::Value(got));
        }
        let value = if fp.check_attribute {
            let key = self.extract_key(fp)?;
            // Exact-dict hit avoids the accessor call; a miss still falls
            // back to getattr, exactly like _get_value_for_key.
            let got = match (node.get_attribute_is_default, obj_dict, fast_key(&key)) {
                (true, Some(od), Some(k)) => match od.get_item(&k).ok().flatten() {
                    Some(v) => v,
                    None => crate::getattr_default(obj, &key, &self.missing())?,
                },
                _ => {
                    if node.get_attribute_is_default {
                        crate::get_value(py, obj, &key, &self.missing())?
                    } else {
                        node.get_attribute.bind(py).call1((obj, &key, self.missing()))?
                    }
                }
            };
            if self.is_missing(&got) {
                let v = if fp.dump_default_present {
                    crate::maybe_call(py, fp.dump_default.bind(py))?
                } else {
                    self.missing()
                };
                if self.is_missing(&v) {
                    return Ok(DumpOut::Skip);
                }
                v
            } else {
                got
            }
        } else {
            self.none()
        };
        let attr = match &fp.name {
            Some(n) => pyo3::types::PyString::new(py, n).into_any(),
            None => self.none(),
        };
        match self.codec_dump(fp, &value, obj, &attr)? {
            None => Ok(DumpOut::Skip),
            Some(v) => Ok(DumpOut::Value(v)),
        }
    }

    /// Codec serialize. Ok(None) means missing (skip the field).
    /// `attr`/`obj` thread exactly like upstream `_serialize` args.
    fn codec_dump(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
        obj: &Bound<'py, PyAny>,
        attr: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let lib = self.lib();
        let py = self.py;
        match &fp.codec {
            Codec::Raw => Ok(Some(value.clone())),
            Codec::Str => {
                if value.is_none() {
                    return Ok(Some(value.clone()));
                }
                Ok(Some(lib.ensure_text_type_fn.bind(py).call1((value,))?))
            }
            Codec::Uuid => {
                if value.is_none() {
                    return Ok(Some(value.clone()));
                }
                Ok(Some(lib.str_cls.bind(py).call1((value,))?))
            }
            Codec::Integer { .. } | Codec::Float { .. } | Codec::Decimal { .. } => {
                if value.is_none() {
                    return Ok(Some(value.clone()));
                }
                self.number_dump(fp, value)
            }
            Codec::Boolean { .. } => Ok(Some(value.clone())),
            Codec::DateTime(spec) => self.temporal_dump(spec, "datetime", value),
            Codec::NaiveDateTime { spec, .. } => self.temporal_dump(spec, "datetime", value),
            Codec::AwareDateTime { spec, .. } => self.temporal_dump(spec, "datetime", value),
            Codec::Time(spec) => self.temporal_dump(spec, "time", value),
            Codec::Date(spec) => self.temporal_dump(spec, "date", value),
            Codec::TimeDelta { precision } => self.timedelta_dump(*precision, value),
            Codec::Nested { target, many, .. } => self.nested_dump(*target, *many, None, value),
            Codec::Pluck { target, many, pluck_key, .. } => {
                self.nested_dump(*target, *many, Some(pluck_key), value)
            }
            Codec::List { inner } => {
                if value.is_none() {
                    return Ok(Some(value.clone()));
                }
                let out = PyList::empty(py);
                for each in value.try_iter()? {
                    let each = each?;
                    let v = self
                        .inner_dump(inner, &each, obj, attr)?
                        .unwrap_or_else(|| self.none());
                    out.append(v)?;
                }
                Ok(Some(out.into_any()))
            }
            Codec::Tuple { inners, .. } => {
                if value.is_none() {
                    return Ok(Some(value.clone()));
                }
                let mut iter = value.try_iter()?;
                let mut items = vec![];
                for f in inners.iter() {
                    match iter.next() {
                        Some(Ok(each)) => {
                            let v = self
                                .inner_dump(f, &each, obj, attr)?
                                .unwrap_or_else(|| self.none());
                            items.push(v);
                        }
                        Some(Err(e)) => return Err(e),
                        None => {
                            return Err(PyValueError::new_err(
                                "zip() argument 2 is shorter than argument 1",
                            ));
                        }
                    }
                }
                if iter.next().is_some() {
                    return Err(PyValueError::new_err(
                        "zip() argument 2 is longer than argument 1",
                    ));
                }
                Ok(Some(PyTuple::new(py, items)?.into_any()))
            }
            Codec::Mapping { mapping_type, key, value: vf } => {
                self.mapping_dump(mapping_type, key.as_deref(), vf.as_deref(), value)
            }
            Codec::Ip { exploded, .. } => self.ip_dump(*exploded, value),
            Codec::IpInterface { exploded, .. } => self.ip_dump(*exploded, value),
            Codec::Enum { inner_kind, .. } => self.enum_dump(inner_kind, value, obj, attr),
            Codec::Method { serialize_method, .. } => match serialize_method {
                None => Ok(None),
                Some(m) => Ok(Some(m.bind(py).call1((obj,))?)),
            },
            Codec::Function { serialize_func, .. } => match serialize_func {
                None => Err(pyo3::exceptions::PyTypeError::new_err(
                    "'NoneType' object is not callable",
                )),
                Some(f) => Ok(Some(f.bind(py).call1((obj,))?)),
            },
            Codec::Constant { constant } => Ok(Some(constant.bind(py).clone())),
            Codec::Opaque { .. } => unreachable!("opaque handled by callers"),
        }
    }

    /// Inner dump: List/Tuple pass (attr, obj) through; Mapping uses None;
    /// opaque inner calls `_serialize` directly.
    fn inner_dump(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
        obj: &Bound<'py, PyAny>,
        attr: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Codec::Opaque { field } = &fp.codec {
            return Ok(Some(field.bind(self.py).call_method(
                "_serialize",
                (value, attr, obj),
                None,
            )?));
        }
        if matches!(fp.codec, Codec::Mapping { .. }) {
            return self.codec_dump(fp, value, &self.none(), &self.none());
        }
        self.codec_dump(fp, value, obj, attr)
    }

    fn number_dump(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let lib = self.lib();
        let py = self.py;
        let num = match &fp.codec {
            Codec::Integer { .. } => lib.int_cls.bind(py).call1((value,))?,
            Codec::Float { .. } => lib.float_cls.bind(py).call1((value,))?,
            Codec::Decimal { places, rounding, allow_nan, .. } => {
                let s = lib.str_cls.bind(py).call1((value,))?;
                let mut num = lib.decimal_cls.bind(py).call1((s,))?;
                if *allow_nan {
                    let is_nan: bool = num.call_method0("is_nan")?.extract()?;
                    if is_nan {
                        num = lib.decimal_cls.bind(py).call1(("NaN",))?;
                    }
                }
                if let Some(places) = places {
                    let is_finite: bool = num.call_method0("is_finite")?.extract()?;
                    if is_finite {
                        let kwargs = PyDict::new(py);
                        if let Some(r) = rounding {
                            kwargs.set_item("rounding", r)?;
                        }
                        num = num.call_method("quantize", (places,), Some(&kwargs))?;
                    }
                }
                num
            }
            _ => unreachable!(),
        };
        let (as_string, is_decimal) = match &fp.codec {
            Codec::Float { as_string, .. } => (*as_string, false),
            Codec::Decimal { as_string, .. } => (*as_string, true),
            Codec::Integer { as_string, .. } => (*as_string, false),
            _ => unreachable!(),
        };
        if as_string {
            if is_decimal {
                return Ok(Some(self.builtins.getattr("format")?.call1((num, "f"))?));
            }
            return Ok(Some(lib.str_cls.bind(py).call1((num,))?));
        }
        Ok(Some(num))
    }

    fn temporal_dump(
        &self,
        spec: &crate::TemporalSpec,
        obj_type: &str,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        let lib = self.lib();
        let py = self.py;
        // Upstream calls the UNBOUND method (dt.datetime.isoformat(value)),
        // so non-datetime inputs raise TypeError, not AttributeError.
        let iso = match obj_type {
            "datetime" => &lib.isoformat_datetime_fn,
            "time" => &lib.isoformat_time_fn,
            _ => &lib.isoformat_date_fn,
        };
        Ok(Some(match &spec.fmt {
            DateFmt::Iso => iso.bind(py).call1((value,))?,
            DateFmt::Rfc => lib.format_datetime_fn.bind(py).call1((value,))?,
            DateFmt::Timestamp => lib.timestamp_fn.bind(py).call1((value,))?,
            DateFmt::TimestampMs => lib.timestamp_ms_fn.bind(py).call1((value,))?,
            DateFmt::Custom(fmt) => {
                value.call_method("strftime", (fmt.clone(),), None)?
            }
        }))
    }

    fn timedelta_dump(
        &self,
        precision: Precision,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        let lib = self.lib();
        let py = self.py;
        let micros: Bound<PyAny> = lib.timedelta_to_microseconds_fn.bind(py).call1((value,))?;
        let unit = lib.int_cls.bind(py).call1((precision.micros_per_unit(),))?;
        Ok(Some(micros.call_method("__truediv__", (unit,), None)?))
    }

    fn nested_dump(
        &self,
        target: usize,
        many: bool,
        pluck_key: Option<&str>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        let node = &self.plan.nodes[target];
        let dumped = if many {
            let out = PyList::empty(self.py);
            for item in value.try_iter()? {
                out.append(self.dump_one(node, &item?)?)?;
            }
            out.into_any()
        } else {
            self.dump_one(node, value)?
        };
        if let Some(key) = pluck_key {
            if many {
                let out = PyList::empty(self.py);
                for item in dumped.try_iter()? {
                    out.append(item?.get_item(key)?)?;
                }
                return Ok(Some(out.into_any()));
            }
            return Ok(Some(dumped.get_item(key)?));
        }
        Ok(Some(dumped))
    }

    fn mapping_dump(
        &self,
        mapping_type: &Py<PyAny>,
        key: Option<&FieldPlan>,
        vf: Option<&FieldPlan>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        let py = self.py;
        if key.is_none() && vf.is_none() {
            return Ok(Some(mapping_type.bind(py).call1((value,))?));
        }
        // Serialize keys (identity when no key field).
        let mut keys: Vec<(Bound<PyAny>, Bound<PyAny>)> = vec![];
        for k in value.try_iter()? {
            let k = k?;
            let sk = match key {
                Some(kf) => self
                    .codec_dump(kf, &k, &self.none(), &self.none())?
                    .unwrap_or_else(|| self.none()),
                None => k.clone(),
            };
            keys.push((k, sk));
        }
        let result: Bound<PyAny> = mapping_type.bind(py).call0()?;
        // Serialize values (`if k in keys` is always true here, kept for shape).
        if let Some(vf) = vf {
            for pair in value.call_method0("items")?.try_iter()? {
                let pair = pair?;
                let t = pair.downcast::<PyTuple>()?;
                let (k, v) = (t.get_item(0)?, t.get_item(1)?);
                if let Some((_, sk)) = keys.iter().find(|(ok, _)| k.is(ok) || k.eq(ok).unwrap_or(false)) {
                    let sv = self
                        .codec_dump(vf, &v, &self.none(), &self.none())?
                        .unwrap_or_else(|| self.none());
                    result.set_item(sk, &sv)?;
                }
            }
        } else {
            for pair in value.call_method0("items")?.try_iter()? {
                let pair = pair?;
                let t = pair.downcast::<PyTuple>()?;
                let (k, v) = (t.get_item(0)?, t.get_item(1)?);
                if let Some((_, sk)) = keys.iter().find(|(ok, _)| k.is(ok) || k.eq(ok).unwrap_or(false)) {
                    result.set_item(sk, &v)?;
                }
            }
        }
        Ok(Some(result.into_any()))
    }

    fn ip_dump(
        &self,
        exploded: bool,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        Ok(Some(if exploded {
            value.getattr("exploded")?
        } else {
            value.getattr("compressed")?
        }))
    }

    fn enum_dump(
        &self,
        inner_kind: &EnumInner,
        value: &Bound<'py, PyAny>,
        obj: &Bound<'py, PyAny>,
        attr: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if value.is_none() {
            return Ok(Some(value.clone()));
        }
        let lib = self.lib();
        let py = self.py;
        match inner_kind {
            EnumInner::Name => {
                let name = value.getattr("name")?;
                Ok(Some(lib.ensure_text_type_fn.bind(py).call1((name,))?))
            }
            EnumInner::Raw => Ok(Some(value.getattr("value")?)),
            EnumInner::Custom(idx) => {
                let inner = &self.plan.custom_enum_inners[*idx];
                let val = value.getattr("value")?;
                self.inner_dump(inner, &val, obj, attr)
            }
        }
    }
    // -- load -------------------------------------------------------

    pub fn load_root(
        &self,
        node_idx: usize,
        data: &Bound<'py, PyAny>,
        many: bool,
        partial: &Bound<'py, PyAny>,
        unknown: UnknownOpt,
    ) -> PyResult<(Py<PyAny>, Option<crate::ErrNode>)> {
        let node = &self.plan.nodes[node_idx];
        let p = self.classify_partial(partial)?;
        if many {
            let mut store = ErrStore::new();
            let out = PyList::empty(self.py);
            for (idx, item) in data.try_iter()?.enumerate() {
                let item = item?;
                let index = if node.index_errors { Some(idx) } else { None };
                let v = self.load_single(node, &item, &mut store, &p, unknown, index)?;
                out.append(v)?;
            }
            let errors = if store.is_empty() { None } else { Some(store.root) };
            return Ok((out.into_any().unbind(), errors));
        }
        let mut store = ErrStore::new();
        let v = self.load_single(node, data, &mut store, &p, unknown, None)?;
        let errors = if store.is_empty() { None } else { Some(store.root) };
        Ok((v.unbind(), errors))
    }

    fn classify_partial(&self, partial: &Bound<'py, PyAny>) -> PyResult<Partial<'py>> {
        if partial.is_none() {
            return Ok(Partial { raw: partial.clone(), is_none: true, is_true: false, is_coll: false });
        }
        if partial.is(PyBool::new(self.py, true).as_any()) {
            return Ok(Partial { raw: partial.clone(), is_none: false, is_true: true, is_coll: false });
        }
        if is_collection(self.py, partial)? {
            return Ok(Partial { raw: partial.clone(), is_none: false, is_true: false, is_coll: true });
        }
        Ok(Partial { raw: partial.clone(), is_none: false, is_true: false, is_coll: false })
    }

    fn partial_skip(&self, name: &str, p: &Partial) -> PyResult<bool> {
        if p.is_true {
            return Ok(true);
        }
        if p.is_coll {
            let s = pyo3::types::PyString::new(self.py, name);
            return p.raw.contains(&s);
        }
        Ok(false)
    }

    /// d_kwargs partial for one field: None (no kwarg), True/scalar passthrough,
    /// or the stripped sub-list for collections.
    fn sub_partial(&self, name: &str, p: &Partial<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if p.is_none {
            return Ok(None);
        }
        if p.is_true || !p.is_coll {
            return Ok(Some(p.raw.clone()));
        }
        let prefix = format!("{name}.");
        let n = prefix.chars().count();
        let out = PyList::empty(self.py);
        for f in p.raw.try_iter()? {
            let f = f?;
            let starts: bool = f.call_method("startswith", (&prefix,), None)?.extract()?;
            if starts {
                let s: String = f.extract()?;
                let rest: String = s.chars().skip(n).collect();
                out.append(pyo3::types::PyString::new(self.py, &rest))?;
            }
        }
        Ok(Some(out.into_any()))
    }

    fn out_key(&self, fp: &FieldPlan) -> PyResult<Bound<'py, PyAny>> {
        if let Some(a) = &fp.attribute {
            let ab = a.bind(self.py);
            if ab.is_truthy()? {
                return Ok(ab.clone());
            }
        }
        match &fp.name {
            Some(n) => Ok(pyo3::types::PyString::new(self.py, n).into_any()),
            None => Err(pyo3::exceptions::PyTypeError::new_err("no key")),
        }
    }

    fn load_single(
        &self,
        node: &NodePlan,
        data: &Bound<'py, PyAny>,
        store: &mut ErrStore,
        partial: &Partial,
        unknown: UnknownOpt,
        index: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let ret: Bound<PyAny> = node.dict_class.bind(py).call0()?;
        if !self.is_instance(data, &self.lib().mapping_abc)? {
            store.store(
                py,
                ErrNode::Msgs(vec![Msg::ReadyStr(node.type_msg.clone())]),
                ErrKey::S(crate::SCHEMA_KEY.to_string()),
                index,
            );
            return Ok(ret.into_any());
        }
        // Fast paths need an exact dict on both sides.
        let data_dict: Option<Bound<PyDict>> = if is_exact_dict(data) {
            data.downcast::<PyDict>().ok().cloned()
        } else {
            None
        };
        let ret_dict: Option<Bound<PyDict>> = if is_exact_dict(&ret) {
            ret.downcast::<PyDict>().ok().cloned()
        } else {
            None
        };
        for fp in &node.load_fields {
            match self.field_load(node, fp, data, data_dict.as_ref(), ret_dict.as_ref(), store, partial, index, &ret)? {
                LoadOut::Missing => {}
                LoadOut::Value => {}
            }
        }
        // Garbage `unknown` values behave like exclude (upstream compares
        // `!= EXCLUDE` to enter but only INCLUDE/RAISE act inside).
        if unknown == UnknownOpt::Include || unknown == UnknownOpt::Raise {
            // fields = {data_key-or-name}; for key in set(data) - fields
            let mut fkeys: Vec<Bound<PyAny>> = vec![];
            for fp in &node.load_fields {
                fkeys.push(match &fp.data_key {
                    Some(k) => k.bind(py).clone(),
                    None => self.out_key(fp)?,
                });
            }
            let fset = self.builtins.call_method("frozenset", (fkeys,), None)?;
            let dset: Bound<PyAny> = self.builtins.call_method("set", (data,), None)?;
            let diff = dset.call_method("__sub__", (fset,), None)?;
            for key in diff.try_iter()? {
                let key = key?;
                if unknown == UnknownOpt::Include {
                    ret.set_item(&key, &data.get_item(&key)?)?;
                } else {
                    let ek = match key.extract::<String>() {
                        Ok(s) => ErrKey::S(s),
                        Err(_) => ErrKey::K(key.unbind()),
                    };
                    store.store(
                        py,
                        ErrNode::Msgs(vec![Msg::ReadyStr(node.unknown_msg.clone())]),
                        ek,
                        index,
                    );
                }
            }
        }
        Ok(ret.into_any())
    }

    // Ten args is the honest shape of the per-field context.
    #[allow(clippy::too_many_arguments)]
    fn field_load(
        &self,
        _node: &NodePlan,
        fp: &FieldPlan,
        data: &Bound<'py, PyAny>,
        data_dict: Option<&Bound<'py, PyDict>>,
        ret_dict: Option<&Bound<'py, PyDict>>,
        store: &mut ErrStore,
        partial: &Partial,
        index: Option<usize>,
        ret: &Bound<PyAny>,
    ) -> PyResult<LoadOut> {
        let py = self.py;
        let data_key = match &fp.data_key {
            Some(k) => k.bind(py).clone(),
            None => return Err(pyo3::exceptions::PyTypeError::new_err("no key")),
        };
        let raw = match (data_dict, fast_key(&data_key)) {
            (Some(dd), Some(k)) => dd.get_item(&k).ok().flatten().unwrap_or_else(|| self.missing()),
            _ => data.call_method("get", (&data_key, self.missing()), None)?,
        };
        let name = fp.name.clone().unwrap_or_default();
        if self.is_missing(&raw) && self.partial_skip(&name, partial)? {
            return Ok(LoadOut::Missing);
        }
        let sub = self.sub_partial(&name, partial)?;
        let out_key = self.out_key(fp)?;
        match self.field_value(fp, &raw, data, sub.as_ref()) {
            Ok(v) => {
                if self.is_missing(&v) {
                    return Ok(LoadOut::Missing);
                }
                self.dict_set(ret, ret_dict, &out_key, v)?;
                Ok(LoadOut::Value)
            }
            Err(FieldErr::Ve { node, valid_data }) => {
                let ek = match data_key.extract::<String>() {
                    Ok(s) => ErrKey::S(s),
                    Err(_) => ErrKey::K(data_key.unbind()),
                };
                store.store(py, node, ek, index);
                match valid_data {
                    Some(vd) => {
                        let b = vd.bind(py);
                        if b.is_truthy()? {
                            self.dict_set(ret, ret_dict, &out_key, b.clone())?;
                            return Ok(LoadOut::Value);
                        }
                        Ok(LoadOut::Missing)
                    }
                    None => Ok(LoadOut::Missing),
                }
            }
            Err(FieldErr::Other(e)) => Err(e),
        }
    }

    /// Full `Field.deserialize` replay for one value.
    fn field_value(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let py = self.py;
        // 1. _validate_missing
        if self.is_missing(value) {
            if !fp.constant_missing_check_only_null && fp.required {
                return Err(FieldErr::Ve {
                    node: ErrNode::Msgs(vec![self.msg(&fp.required_msg, vec![])]),
                    valid_data: None,
                });
            }
        } else if value.is_none() && !fp.allow_none {
            return Err(FieldErr::Ve {
                node: ErrNode::Msgs(vec![self.msg(&fp.null_msg, vec![])]),
                valid_data: None,
            });
        }
        // 2. missing -> default (validators skipped)
        if self.is_missing(value) {
            if fp.load_default_present {
                let d = fp.load_default.bind(py);
                let v = crate::maybe_call(py, d)?;
                return Ok(v);
            }
            return Ok(self.missing());
        }
        // 3. pre_load
        // 3. pre_load (a ValidationError here is stored under the field,
        //    like upstream where it escapes deserialize into _call_and_store)
        let mut v = value.clone();
        for f in &fp.pre_load {
            v = match f.bind(py).call1((v,)) {
                Ok(nv) => nv,
                Err(e) => return Err(self.field_ve(e)?),
            };
        }
        // 4. allow_none
        if v.is_none() && fp.allow_none {
            return Ok(v);
        }
        // 5. codec
        let mut out = match self.codec_load(fp, &v, data, sub) {
            Ok(o) => o,
            Err(FieldErr::Other(e)) => return Err(self.field_ve(e)?),
            Err(other) => return Err(other),
        };
        // 6. validators (And loop). A failed _validate carries no valid_data
        //    (upstream And drops it), so the field is excluded from results.
        if let Some(msgs) = self.run_validators(fp, &out)? {
            return Err(FieldErr::Ve {
                node: ErrNode::Msgs(msgs),
                valid_data: None,
            });
        }
        // 7. post_load
        for f in &fp.post_load {
            out = match f.bind(py).call1((out,)) {
                Ok(no) => no,
                Err(e) => return Err(self.field_ve(e)?),
            };
        }
        Ok(out)
    }

    /// Convert a ValidationError escaping a field-level call into the stored
    /// shape (messages + valid_data-or-missing); other errors propagate.
    fn field_ve(&self, e: PyErr) -> Result<FieldErr, FieldErr> {
        if !self.is_ve(&e) {
            return Err(FieldErr::Other(e));
        }
        let py = self.py;
        let val = e.value(py);
        let messages = val
            .getattr("messages")
            .map_err(FieldErr::Other)?
            .unbind();
        let valid = val.getattr("valid_data").map_err(FieldErr::Other)?;
        let valid_opt = if valid.is_truthy().map_err(FieldErr::Other)? {
            Some(valid.unbind())
        } else {
            None
        };
        Ok(FieldErr::Ve {
            node: ve_messages_node(messages.bind(py)),
            valid_data: valid_opt,
        })
    }

    /// Codec deserialize. Errors become FieldErr::Ve; other exceptions propagate.
    fn codec_load(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let _lib = self.lib();
        let py = self.py;
        match &fp.codec {
            Codec::Raw => Ok(value.clone()),
            Codec::Str => self.string_load(fp, value),
            Codec::Uuid => self.uuid_load(fp, value),
            Codec::Integer { strict, .. } => self.integer_load(fp, *strict, value),
            Codec::Float { allow_nan, .. } => self.float_load(fp, *allow_nan, value),
            Codec::Decimal { places, rounding, allow_nan, .. } => {
                self.decimal_load(fp, places.as_ref(), rounding.as_ref(), *allow_nan, value)
            }
            Codec::Boolean { truthy, falsy } => self.boolean_load(fp, truthy, falsy, value),
            Codec::DateTime(spec) => self.temporal_load(fp, spec, "datetime", 0, None, value),
            Codec::NaiveDateTime { spec, timezone } => {
                self.temporal_load(fp, spec, "datetime", 1, timezone.as_ref(), value)
            }
            Codec::AwareDateTime { spec, default_timezone } => {
                self.temporal_load(fp, spec, "datetime", 2, default_timezone.as_ref(), value)
            }
            Codec::Time(spec) => self.temporal_load(fp, spec, "time", 0, None, value),
            Codec::Date(spec) => self.temporal_load(fp, spec, "date", 0, None, value),
            Codec::TimeDelta { precision } => self.timedelta_load(fp, *precision, value),
            Codec::Nested { target, many, unknown } => {
                self.nested_load(fp, *target, *many, *unknown, None, value, data, sub)
            }
            Codec::Pluck { target, many, unknown, pluck_key } => {
                self.nested_load(fp, *target, *many, *unknown, Some(pluck_key), value, data, sub)
            }
            Codec::List { inner } => self.list_load(fp, inner, value, data, sub),
            Codec::Tuple { inners, length_msg } => {
                self.tuple_load(fp, inners, length_msg, value, data, sub)
            }
            Codec::Mapping { mapping_type, key, value: vf } => {
                self.mapping_load(fp, mapping_type, key.as_deref(), vf.as_deref(), value, data, sub)
            }
            Codec::Ip { exploded: _, deser_class } => self.ip_load(
                fp, deser_class.as_ref(), &self.lib().ip_address_fn, "invalid_ip", "Not a valid IP address.", value,
            ),
            Codec::IpInterface { exploded: _, deser_class } => self.ip_load(
                fp, deser_class.as_ref(), &self.lib().ip_interface_fn, "invalid_ip_interface",
                "Not a valid IP interface.", value,
            ),
            Codec::Enum { enum_class, inner_kind, choices_text } => {
                self.enum_load(fp, enum_class, inner_kind, choices_text, value, data, sub)
            }
            Codec::Method { deserialize_method, .. } => match deserialize_method {
                None => Ok(value.clone()),
                Some(m) => Ok(m.bind(py).call1((value,))?),
            },
            Codec::Function { deserialize_func, .. } => match deserialize_func {
                None => Ok(value.clone()),
                Some(f) => Ok(f.bind(py).call1((value,))?),
            },
            Codec::Constant { constant } => Ok(constant.bind(py).clone()),
            Codec::Opaque { field } => self.opaque_load(fp, field, value, data, sub),
        }
    }

    fn opaque_load(
        &self,
        fp: &FieldPlan,
        field: &Py<PyAny>,
        value: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let py = self.py;
        let data_key = match &fp.data_key {
            Some(k) => k.bind(py).clone(),
            None => return Err(pyo3::exceptions::PyTypeError::new_err("no key").into()),
        };
        let res = if let Some(p) = sub {
            let kwargs = PyDict::new(py);
            kwargs.set_item("partial", p)?;
            field.bind(py).call_method("deserialize", (&value, &data_key, data), Some(&kwargs))
        } else {
            field.bind(py).call_method("deserialize", (&value, &data_key, data), None)
        };
        match res {
            Ok(v) => Ok(v),
            Err(e) if self.is_ve(&e) => {
                let val = e.value(py);
                let messages = val.getattr("messages")?.unbind();
                let valid = val.getattr("valid_data")?;
                let valid_opt = if valid.is_truthy()? {
                    Some(valid.unbind())
                } else {
                    None
                };
                Err(FieldErr::Ve {
                    node: ve_messages_node(messages.bind(py)),
                    valid_data: valid_opt,
                })
            }
            Err(e) => Err(FieldErr::Other(e)),
        }
    }

    /// Inner (List/Tuple/Dict/Enum-custom) load: full field_value replay.
    fn inner_load(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        if let Codec::Opaque { field } = &fp.codec {
            // Direct `deserialize(each, **kwargs)`: name/data are None here.
            let res = if let Some(p) = sub {
                let kwargs = PyDict::new(self.py);
                kwargs.set_item("partial", p)?;
                field.bind(self.py).call_method(
                    "deserialize",
                    (value, self.none(), self.none()),
                    Some(&kwargs),
                )
            } else {
                field.bind(self.py).call_method(
                    "deserialize",
                    (value, self.none(), self.none()),
                    None,
                )
            };
            return match res {
                Ok(v) => Ok(v),
                Err(e) if self.is_ve(&e) => {
                    let val = e.value(self.py);
                    let messages = val.getattr("messages")?.unbind();
                    let valid = val.getattr("valid_data")?;
                    let valid_opt = if valid.is_truthy()? {
                        Some(valid.unbind())
                    } else {
                        None
                    };
                    Err(FieldErr::Ve {
                        node: ve_messages_node(messages.bind(self.py)),
                        valid_data: valid_opt,
                    })
                }
                Err(e) => Err(FieldErr::Other(e)),
            };
        }
        let none = self.none();
        self.field_value(fp, value, &none, sub)
    }
    fn string_load(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        // Exact-unicode fast path: str(str) is the identical object.
        if is_exact_unicode(value) {
            return Ok(value.clone());
        }
        use pyo3::types::{PyBytes, PyString};
        if !(value.is_instance_of::<PyString>() || value.is_instance_of::<PyBytes>()) {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid string."), vec![]));
        }
        if let Ok(b) = value.downcast::<PyBytes>() {
            let lib = self.lib();
            match lib.ensure_text_type_fn.bind(self.py).call1((b,)) {
                Ok(s) => return Ok(s),
                Err(e) => {
                    if e.is_instance_of::<pyo3::exceptions::PyUnicodeDecodeError>(self.py) {
                        return Err(self.ve_msg(
                            self.tmpl(fp, "invalid_utf8", "Not a valid utf-8 string."),
                            vec![],
                        ));
                    }
                    return Err(e.into());
                }
            }
        }
        Ok(value.clone())
    }

    fn uuid_load(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        use pyo3::types::PyBytes;
        let lib = self.lib();
        let py = self.py;
        if self.is_instance(value, &lib.uuid_cls).unwrap_or(false) {
            return Ok(value.clone());
        }
        let res = if let Ok(b) = value.downcast::<PyBytes>() {
            if b.len().unwrap_or(0) == 16 {
                let kwargs = PyDict::new(py);
                kwargs.set_item("bytes", b)?;
                lib.uuid_cls.bind(py).call((), Some(&kwargs))
            } else {
                lib.uuid_cls.bind(py).call1((value,))
            }
        } else {
            lib.uuid_cls.bind(py).call1((value,))
        };
        match res {
            Ok(v) => Ok(v),
            Err(e) => {
                if e.is_instance_of::<PyValueError>(py)
                    || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                    || e.is_instance_of::<PyTypeError>(py)
                {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid_uuid", "Not a valid UUID."), vec![]));
                }
                Err(e.into())
            }
        }
    }

    fn integer_load(
        &self,
        fp: &FieldPlan,
        strict: bool,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let lib = self.lib();
        // Exact-int fast path: int() of an exact int is the identical object
        // (and exact ints satisfy Integral, so strict passes without a call).
        if is_exact_long(value) {
            return Ok(value.clone());
        }
        // `if self.strict and not isinstance(...)` (flag pre-resolved to bool).
        if strict && !self.is_instance(value, &lib.numbers_integral_abc).unwrap_or(false) {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid integer."), vec![("input".to_string(), value.clone())]));
        }
        if value.is(PyBool::new(self.py, true).as_any()) || value.is(PyBool::new(self.py, false).as_any()) {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid integer."), vec![("input".to_string(), value.clone())]));
        }
        match lib.int_cls.bind(self.py).call1((value,)) {
            Ok(v) => Ok(v),
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(self.py) || e.is_instance_of::<PyValueError>(self.py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid integer."), vec![("input".to_string(), value.clone())]));
                }
                if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(self.py) {
                    let t = self.tmpl(fp, "too_large", "Number too large.");
                    return Err(self.ve_msg(t, vec![("input".to_string(), value.clone())]));
                }
                Err(e.into())
            }
        }
    }
    fn float_load(
        &self,
        fp: &FieldPlan,
        allow_nan: bool,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        // Exact-float fast path: float() is identity; nan/inf checked inline.
        if is_exact_float(value) {
            if !allow_nan {
                if let Ok(f) = value.downcast::<PyFloat>() {
                    let d = f.value();
                    if d.is_nan() || d.is_infinite() {
                        return Err(self.ve_msg(self.tmpl(fp, "special", "Special numeric values (nan or infinity) are not permitted."), vec![]));
                    }
                }
            }
            return Ok(value.clone());
        }
        // Exact-int fast path: float(int) via correctly-rounded C conversion.
        if is_exact_long(value) {
            if let Ok(d) = value.extract::<f64>() {
                return Ok(PyFloat::new(self.py, d).into_any());
            }
        }
        let num = match self.lib().float_cls.bind(self.py).call1((value,)) {
            Ok(v) => v,
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(self.py) || e.is_instance_of::<PyValueError>(self.py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![("input".to_string(), value.clone())]));
                }
                if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(self.py) {
                    let t = self.tmpl(fp, "too_large", "Number too large.");
                    return Err(self.ve_msg(t, vec![("input".to_string(), value.clone())]));
                }
                return Err(e.into());
            }
        };
        if !allow_nan {
            use pyo3::types::PyFloat;
            if let Ok(f) = num.downcast::<PyFloat>() {
                let d = f.value();
                if d.is_nan() || d.is_infinite() {
                    return Err(self.ve_msg(self.tmpl(fp, "special", "Special numeric values (nan or infinity) are not permitted."), vec![]));
                }
            }
        }
        Ok(num)
    }

    fn decimal_load(
        &self,
        fp: &FieldPlan,
        places: Option<&Py<PyAny>>,
        rounding: Option<&Py<PyAny>>,
        allow_nan: bool,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let lib = self.lib();
        let py = self.py;
        if value.is(PyBool::new(py, true).as_any()) || value.is(PyBool::new(py, false).as_any()) {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![("input".to_string(), value.clone())]));
        }
        let s = match lib.str_cls.bind(py).call1((value,)) {
            Ok(s) => s,
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![("input".to_string(), value.clone())]));
                }
                return Err(e.into());
            }
        };
        let mut num = match lib.decimal_cls.bind(py).call1((s,)) {
            Ok(v) => v,
            Err(e) => {
                if e.is_instance(py, lib.decimal_invalid_operation.bind(py)) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![]));
                }
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![("input".to_string(), value.clone())]));
                }
                if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(py) {
                    let t = self.tmpl(fp, "too_large", "Number too large.");
                    return Err(self.ve_msg(t, vec![("input".to_string(), value.clone())]));
                }
                return Err(e.into());
            }
        };
        if allow_nan {
            let is_nan: bool = num.call_method0("is_nan")?.extract()?;
            if is_nan {
                num = lib.decimal_cls.bind(py).call1(("NaN",))?;
            }
        }
        if let Some(places) = places {
            let is_finite: bool = num.call_method0("is_finite")?.extract()?;
            if is_finite {
                let kwargs = PyDict::new(py);
                if let Some(r) = rounding {
                    kwargs.set_item("rounding", r)?;
                }
                match num.call_method("quantize", (places,), Some(&kwargs)) {
                    Ok(q) => num = q,
                    Err(e) => {
                        if e.is_instance(py, lib.decimal_invalid_operation.bind(py)) {
                            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid number."), vec![]));
                        }
                        return Err(e.into());
                    }
                }
            }
        }
        if !allow_nan {
            let is_nan: bool = num.call_method0("is_nan")?.extract()?;
            let is_inf: bool = num.call_method0("is_infinite")?.extract()?;
            if is_nan || is_inf {
                return Err(self.ve_msg(self.tmpl(fp, "special", "Special numeric values (nan or infinity) are not permitted."), vec![]));
            }
        }
        Ok(num)
    }

    fn boolean_load(
        &self,
        fp: &FieldPlan,
        truthy: &Py<PyAny>,
        falsy: &Py<PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        // Exact-bool fast path: the value is already the canonical result.
        if value.is(PyBool::new(self.py, true).as_any()) || value.is(PyBool::new(self.py, false).as_any()) {
            return Ok(value.clone());
        }
        let t = truthy.bind(self.py);
        if !t.is_truthy()? {
            return Ok(self.lib().bool_cls.bind(self.py).call1((value,))?);
        }
        let bad = || self.ve_msg(self.tmpl(fp, "invalid", "Not a valid boolean."), vec![("input".to_string(), value.clone())]);
        match t.contains(value) {
            Ok(true) => return Ok(PyBool::new(self.py, true).to_owned().into_any()),
            Ok(false) => {}
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(self.py) {
                    return Err(bad());
                }
                return Err(e.into());
            }
        }
        match falsy.bind(self.py).contains(value) {
            Ok(true) => return Ok(PyBool::new(self.py, false).to_owned().into_any()),
            Ok(false) => {}
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(self.py) {
                    return Err(bad());
                }
                return Err(e.into());
            }
        }
        Err(bad())
    }
    /// `kind`: 0 = plain, 1 = naive (reject aware unless `tz`), 2 = aware.
    fn temporal_load(
        &self,
        fp: &FieldPlan,
        spec: &crate::TemporalSpec,
        obj_type: &str,
        kind: u8,
        tz: Option<&Py<PyAny>>,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let lib = self.lib();
        let py = self.py;
        let internal = match obj_type {
            "datetime" => &lib.datetime_cls,
            "time" => &lib.time_cls,
            _ => &lib.date_cls,
        };
        let mut ret = if self.is_instance(value, internal).unwrap_or(false) {
            value.clone()
        } else {
            let conv: PyResult<Bound<PyAny>> = match &spec.fmt {
                DateFmt::Iso => match obj_type {
                    "datetime" => lib.fromiso_datetime.bind(py).call1((value,)),
                    "time" => lib.fromiso_time.bind(py).call1((value,)),
                    _ => lib.fromiso_date.bind(py).call1((value,)),
                },
                DateFmt::Rfc => lib.parsedate_to_datetime_fn.bind(py).call1((value,)),
                DateFmt::Timestamp => lib.from_timestamp_fn.bind(py).call1((value,)),
                DateFmt::TimestampMs => lib.from_timestamp_ms_fn.bind(py).call1((value,)),
                DateFmt::Custom(fmt) => {
                    lib.strptime_fn.bind(py).call1((value, fmt.clone()))
                }
            };
            match conv {
                Ok(v) => v,
                Err(e) => {
                    if e.is_instance_of::<PyTypeError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                        || e.is_instance_of::<PyValueError>(py)
                    {
                        let t = self.tmpl(fp, "invalid", "Not a valid datetime.");
                        return Err(self.ve_msg(
                            t,
                            vec![
                                ("input".to_string(), value.clone()),
                                ("obj_type".to_string(), pyo3::types::PyString::new(py, obj_type).into_any()),
                            ],
                        ));
                    }
                    return Err(e.into());
                }
            }
        };
        if kind == 1 {
            let aware: bool = lib.is_aware_fn.bind(py).call1((ret.clone(),))?.extract()?;
            if aware {
                match tz {
                    None => {
                        let t = self.tmpl(fp, "invalid_awareness", "Not a valid naive datetime.");
                        return Err(self.ve_msg(
                            t,
                            vec![
                                ("awareness".to_string(), pyo3::types::PyString::new(py, "naive").into_any()),
                                ("obj_type".to_string(), pyo3::types::PyString::new(py, obj_type).into_any()),
                            ],
                        ));
                    }
                    Some(timezone) => {
                        ret = ret.call_method("astimezone", (timezone,), None)?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("tzinfo", self.none())?;
                        ret = ret.call_method("replace", (), Some(&kwargs))?;
                    }
                }
            }
        } else if kind == 2 {
            let aware: bool = lib.is_aware_fn.bind(py).call1((ret.clone(),))?.extract()?;
            if !aware {
                match tz {
                    None => {
                        let t = self.tmpl(fp, "invalid_awareness", "Not a valid aware datetime.");
                        return Err(self.ve_msg(
                            t,
                            vec![
                                ("awareness".to_string(), pyo3::types::PyString::new(py, "aware").into_any()),
                                ("obj_type".to_string(), pyo3::types::PyString::new(py, obj_type).into_any()),
                            ],
                        ));
                    }
                    Some(default_tz) => {
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("tzinfo", default_tz)?;
                        ret = ret.call_method("replace", (), Some(&kwargs))?;
                    }
                }
            }
        }
        Ok(ret)
    }
    fn timedelta_load(
        &self,
        fp: &FieldPlan,
        precision: Precision,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let lib = self.lib();
        let py = self.py;
        if self.is_instance(value, &lib.timedelta_cls).unwrap_or(false) {
            return Ok(value.clone());
        }
        let f = match lib.float_cls.bind(py).call1((value,)) {
            Ok(v) => v,
            Err(e) => {
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid period of time."), vec![]));
                }
                return Err(e.into());
            }
        };
        let prec_name = match precision {
            Precision::Weeks => "weeks",
            Precision::Days => "days",
            Precision::Hours => "hours",
            Precision::Minutes => "minutes",
            Precision::Seconds => "seconds",
            Precision::Milliseconds => "milliseconds",
            Precision::Microseconds => "microseconds",
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item(prec_name, f)?;
        match lib.timedelta_cls.bind(py).call((), Some(&kwargs)) {
            Ok(v) => Ok(v),
            Err(e) => {
                if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid period of time."), vec![]));
                }
                Err(e.into())
            }
        }
    }

    fn ip_load(
        &self,
        fp: &FieldPlan,
        deser_class: Option<&Py<PyAny>>,
        fallback: &Py<PyAny>,
        err_key: &str,
        err_default: &str,
        value: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let lib = self.lib();
        let py = self.py;
        let text = lib.ensure_text_type_fn.bind(py).call1((value,));
        let text = match text {
            Ok(t) => t,
            Err(e) => {
                if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, err_key, err_default), vec![]));
                }
                return Err(e.into());
            }
        };
        let ctor = match deser_class {
            Some(c) => c.bind(py).clone(),
            None => fallback.bind(py).clone(),
        };
        match ctor.call1((text,)) {
            Ok(v) => Ok(v),
            Err(e) => {
                if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) {
                    return Err(self.ve_msg(self.tmpl(fp, err_key, err_default), vec![]));
                }
                Err(e.into())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enum_load(
        &self,
        fp: &FieldPlan,
        enum_class: &Py<PyAny>,
        inner_kind: &EnumInner,
        choices_text: &str,
        value: &Bound<'py, PyAny>,
        _data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let py = self.py;
        if self.is_instance(value, enum_class).unwrap_or(false) {
            return Ok(value.clone());
        }
        let val = match inner_kind {
            // String() inner (no validators/processors on the bare instance).
            EnumInner::Name => self.string_load(fp, value)?,
            // Raw() inner: no-op passthrough.
            EnumInner::Raw => value.clone(),
            EnumInner::Custom(idx) => {
                let inner = &self.plan.custom_enum_inners[*idx];
                let none = self.none();
                self.field_value(inner, value, &none, sub)?
            }
        };
        let unknown = || {
            self.ve_msg(
                self.tmpl(fp, "unknown", "Must be one of: {choices}."),
                vec![("choices".to_string(), pyo3::types::PyString::new(py, choices_text).into_any())],
            )
        };
        match inner_kind {
            EnumInner::Name => match enum_class.bind(py).get_item(&val) {
                Ok(m) => Ok(m),
                Err(e) => {
                    if e.is_instance_of::<pyo3::exceptions::PyKeyError>(py) {
                        return Err(unknown());
                    }
                    Err(e.into())
                }
            },
            _ => match enum_class.bind(py).call1((val,)) {
                Ok(m) => Ok(m),
                Err(e) => {
                    if e.is_instance_of::<PyValueError>(py) {
                        return Err(unknown());
                    }
                    Err(e.into())
                }
            },
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn nested_load(
        &self,
        _fp: &FieldPlan,
        target: usize,
        many: bool,
        unknown_override: Option<UnknownOpt>,
        pluck_key: Option<&str>,
        value: &Bound<'py, PyAny>,
        _data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let node = &self.plan.nodes[target];
        if many && !is_collection(self.py, value)? {
            let type_name = value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            return Err(self.ve_msg(
                self.tmpl(_fp, "type", "Invalid type."),
                vec![
                    ("input".to_string(), value.clone()),
                    ("type".to_string(), pyo3::types::PyString::new(self.py, &type_name).into_any()),
                ],
            ));
        }
        let wrapped;
        let eff_value = if let Some(key) = pluck_key {
            if many {
                let list = PyList::empty(self.py);
                for v in value.try_iter()? {
                    let d = PyDict::new(self.py);
                    d.set_item(key, v?)?;
                    list.append(d)?;
                }
                wrapped = list.into_any();
                &wrapped
            } else {
                let d = PyDict::new(self.py);
                d.set_item(key, value)?;
                wrapped = d.into_any();
                &wrapped
            }
        } else {
            value
        };
        let unknown = unknown_override.unwrap_or(node.unknown);
        // `partial` falls back to the nested schema's own partial, exactly
        // like `_do_load` (`if partial is None: partial = self.partial`).
        let node_partial = node.partial.bind(self.py).clone();
        let child_partial = match sub {
            Some(p) => self.classify_partial(p)?,
            None => self.classify_partial(&node_partial)?,
        };
        let py = self.py;
        if many {
            let mut store = ErrStore::new();
            let out = PyList::empty(py);
            for (idx, item) in eff_value.try_iter()?.enumerate() {
                let item = item?;
                let index = if node.index_errors { Some(idx) } else { None };
                let v = self.load_single(node, &item, &mut store, &child_partial, unknown, index)?;
                out.append(v)?;
            }
            if store.is_empty() {
                return Ok(out.into_any());
            }
            return Err(FieldErr::Ve { node: store.root, valid_data: Some(out.into_any().unbind()) });
        }
        let mut store = ErrStore::new();
        let v = self.load_single(node, eff_value, &mut store, &child_partial, unknown, None)?;
        if store.is_empty() {
            return Ok(v);
        }
        Err(FieldErr::Ve { node: store.root, valid_data: Some(v.unbind()) })
    }

    fn list_load(
        &self,
        fp: &FieldPlan,
        inner: &FieldPlan,
        value: &Bound<'py, PyAny>,
        _data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        if !is_collection(self.py, value)? {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid list."), vec![]));
        }
        let out = PyList::empty(self.py);
        let mut errors: Vec<(usize, ErrNode)> = vec![];
        for (idx, each) in value.try_iter()?.enumerate() {
            let each = each?;
            match self.inner_load(inner, &each, sub) {
                Ok(v) => {
                    out.append(v)?;
                }
                Err(FieldErr::Ve { node, valid_data }) => {
                    if let Some(vd) = valid_data {
                        if !vd.bind(self.py).is_none() {
                            out.append(vd)?;
                        }
                    }
                    errors.push((idx, node));
                }
                Err(FieldErr::Other(e)) => return Err(FieldErr::Other(e)),
            }
        }
        if errors.is_empty() {
            return Ok(out.into_any());
        }
        let mut map = vec![];
        for (idx, node) in errors {
            map.push((ErrKey::I(idx), node));
        }
        Err(FieldErr::Ve { node: ErrNode::Map(map), valid_data: Some(out.into_any().unbind()) })
    }

    #[allow(clippy::too_many_arguments)]
    fn tuple_load(
        &self,
        fp: &FieldPlan,
        inners: &[FieldPlan],
        length_msg: &str,
        value: &Bound<'py, PyAny>,
        _data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        if !is_sequence_but_not_string(self.py, value)? {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid tuple."), vec![]));
        }
        let len: usize = value.len()?;
        if len != inners.len() {
            return Err(self.ve_msg(
                length_msg.to_string(),
                vec![
                    ("input".to_string(), value.clone()),
                    ("min".to_string(), self.none()),
                    ("max".to_string(), self.none()),
                    ("equal".to_string(), pyo3::types::PyInt::new(self.py, inners.len()).into_any()),
                ],
            ));
        }
        let mut iter = value.try_iter()?;
        let mut items: Vec<Bound<PyAny>> = vec![];
        let mut result: Vec<Bound<PyAny>> = vec![];
        let mut errors: Vec<(usize, ErrNode)> = vec![];
        for (idx, f) in inners.iter().enumerate() {
            let each = match iter.next() {
                Some(Ok(v)) => v,
                Some(Err(e)) => return Err(e.into()),
                None => {
                    return Err(PyValueError::new_err(
                        "zip() argument 2 is shorter than argument 1",
                    )
                    .into());
                }
            };
            match self.inner_load(f, &each, sub) {
                Ok(v) => {
                    result.push(v.clone());
                    items.push(v);
                }
                Err(FieldErr::Ve { node, valid_data }) => {
                    if let Some(vd) = valid_data {
                        if !vd.bind(self.py).is_none() {
                            let b = vd.bind(self.py).clone();
                            result.push(b.clone());
                            items.push(b);
                        }
                    }
                    errors.push((idx, node));
                }
                Err(FieldErr::Other(e)) => return Err(FieldErr::Other(e)),
            }
        }
        if iter.next().is_some() {
            return Err(PyValueError::new_err(
                "zip() argument 2 is longer than argument 1",
            )
            .into());
        }
        if errors.is_empty() {
            return Ok(PyTuple::new(self.py, items)?.into_any());
        }
        let mut map = vec![];
        for (idx, node) in errors {
            map.push((ErrKey::I(idx), node));
        }
        let valid_list = PyList::empty(self.py);
        for r in result {
            valid_list.append(r)?;
        }
        Err(FieldErr::Ve { node: ErrNode::Map(map), valid_data: Some(valid_list.into_any().unbind()) })
    }
    #[allow(clippy::too_many_arguments)]
    fn mapping_load(
        &self,
        fp: &FieldPlan,
        mapping_type: &Py<PyAny>,
        key_fp: Option<&FieldPlan>,
        vf: Option<&FieldPlan>,
        value: &Bound<'py, PyAny>,
        _data: &Bound<'py, PyAny>,
        sub: Option<&Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, FieldErr> {
        let py = self.py;
        if !self.is_instance(value, &self.lib().mapping_abc).unwrap_or(false) {
            return Err(self.ve_msg(self.tmpl(fp, "invalid", "Not a valid mapping type."), vec![]));
        }
        if key_fp.is_none() && vf.is_none() {
            return Ok(mapping_type.bind(py).call1((value,))?);
        }
        let result: Bound<PyAny> = mapping_type.bind(py).call0()?;
        // Deserialize keys.
        let mut keys: Vec<(Bound<PyAny>, Option<Bound<PyAny>>)> = vec![];
        let mut store = ErrStore::new();
        if let Some(kf) = key_fp {
            for k in value.try_iter()? {
                let k = k?;
                match self.inner_load(kf, &k, sub) {
                    Ok(sk) => keys.push((k, Some(sk))),
                    Err(FieldErr::Ve { node, .. }) => {
                        store.store(py, ErrNode::Map(vec![(ErrKey::S("key".to_string()), node)]), key_to_errkey(py, &k)?, None);
                        keys.push((k, None));
                    }
                    Err(FieldErr::Other(e)) => return Err(FieldErr::Other(e)),
                }
            }
        } else {
            for k in value.try_iter()? {
                let k = k?;
                keys.push((k.clone(), Some(k)));
            }
        }
        // Deserialize values.
        for pair in value.call_method0("items")?.try_iter()? {
            let pair = pair?;
            let t = pair.downcast::<PyTuple>().map_err(|e| FieldErr::Other(e.into()))?;
            let (k, v) = (t.get_item(0)?, t.get_item(1)?);
            let slot = keys.iter().find(|(ok, _)| k.is(ok) || k.eq(ok).unwrap_or(false));
            if let Some(vf) = vf {
                match self.inner_load(vf, &v, sub) {
                    Ok(dv) => {
                        if let Some((_, Some(sk))) = slot {
                            result.set_item(sk, &dv)?;
                        }
                    }
                    Err(FieldErr::Ve { node, valid_data }) => {
                        store.store(py, ErrNode::Map(vec![(ErrKey::S("value".to_string()), node)]), key_to_errkey(py, &k)?, None);
                        if let Some(vd) = valid_data {
                            if !vd.bind(py).is_none() {
                                if let Some((_, Some(sk))) = slot {
                                    result.set_item(sk, &vd)?;
                                }
                            }
                        }
                    }
                    Err(FieldErr::Other(e)) => return Err(FieldErr::Other(e)),
                }
            } else if let Some((_, Some(sk))) = slot {
                result.set_item(sk, &v)?;
            }
        }
        if store.is_empty() {
            return Ok(result.into_any());
        }
        Err(FieldErr::Ve { node: store.root, valid_data: Some(result.into_any().unbind()) })
    }
    // -- validators -------------------------------------------------

    /// And-loop over a field's validators. Returns the collected messages.
    fn run_validators(
        &self,
        fp: &FieldPlan,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<Msg>>> {
        let mut out: Vec<Msg> = vec![];
        for v in &fp.validators {
            match self.run_validator(v, value)? {
                VRes::Pass => {}
                VRes::Fail(m) => out.push(m),
                VRes::Expand(list) => {
                    for item in list.bind(self.py).try_iter()? {
                        out.push(Msg::Ready(item?.unbind()));
                    }
                }
            }
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    fn run_validator(
        &self,
        v: &ValidatorPlan,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<VRes> {
        let py = self.py;
        match &v.kind {
            ValidatorKind::Length { min, max, equal, msgs, error } => {
                let len: usize = value.len()?;
                let err = error.clone();
                if let Some(eq) = equal {
                    let eq_n: usize = eq.bind(py).extract()?;
                    if len != eq_n {
                        let t = err.unwrap_or_else(|| msgs.equal.clone());
                        return Ok(VRes::Fail(self.msg_in(&t, value, vec![
                            ("min".to_string(), none_or(py, min)?),
                            ("max".to_string(), none_or(py, max)?),
                            ("equal".to_string(), eq.bind(py).clone()),
                        ])));
                    }
                    return Ok(VRes::Pass);
                }
                let ge = |b: &Option<Py<PyAny>>| -> PyResult<Option<usize>> {
                    match b {
                        Some(x) => Ok(Some(x.bind(py).extract()?)),
                        None => Ok(None),
                    }
                };
                if let Some(mn) = ge(min)? {
                    if len < mn {
                        let t = err.clone().unwrap_or_else(|| {
                            if max.is_some() { msgs.all.clone() } else { msgs.min.clone() }
                        });
                        return Ok(VRes::Fail(self.msg_in(&t, value, vec![
                            ("min".to_string(), none_or(py, min)?),
                            ("max".to_string(), none_or(py, max)?),
                            ("equal".to_string(), self.none()),
                        ])));
                    }
                }
                if let Some(mx) = ge(max)? {
                    if len > mx {
                        let t = err.unwrap_or_else(|| {
                            if min.is_some() { msgs.all.clone() } else { msgs.max.clone() }
                        });
                        return Ok(VRes::Fail(self.msg_in(&t, value, vec![
                            ("min".to_string(), none_or(py, min)?),
                            ("max".to_string(), none_or(py, max)?),
                            ("equal".to_string(), self.none()),
                        ])));
                    }
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::Range { min_present, max_present, min, max, min_inclusive, max_inclusive, msgs, error } => {
                if *min_present {
                    let bound = min.as_ref().unwrap().bind(py).clone();
                    let fail = if *min_inclusive { value.lt(&bound)? } else { value.le(&bound)? };
                    if fail {
                        let t = error.clone().unwrap_or_else(|| {
                            if *max_present { msgs.all.clone() } else { msgs.min.clone() }
                        });
                        return Ok(VRes::Fail(self.msg_in(&t, value, vec![
                            ("min".to_string(), bound.clone()),
                            ("max".to_string(), none_or_obj(py, max.as_ref())?),
                        ])));
                    }
                }
                if *max_present {
                    let bound = max.as_ref().unwrap().bind(py).clone();
                    let fail = if *max_inclusive { value.gt(&bound)? } else { value.ge(&bound)? };
                    if fail {
                        let t = error.clone().unwrap_or_else(|| {
                            if *min_present { msgs.all.clone() } else { msgs.max.clone() }
                        });
                        return Ok(VRes::Fail(self.msg_in(&t, value, vec![
                            ("min".to_string(), none_or_obj(py, min.as_ref())?),
                            ("max".to_string(), bound.clone()),
                        ])));
                    }
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::Equal { comparable, message } => {
                if !value.eq(comparable.bind(py))? {
                    return Ok(VRes::Fail(self.msg_in(message, value, vec![
                        ("other".to_string(), comparable.bind(py).clone()),
                    ])));
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::Regexp { pattern, pattern_text, message } => {
                let m = pattern.bind(py).call_method("match", (value,), None)?;
                if m.is_none() {
                    return Ok(VRes::Fail(self.msg_in(message, value, vec![
                        ("regex".to_string(), pyo3::types::PyString::new(py, pattern_text).into_any()),
                    ])));
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::Predicate { method, kwargs, message } => {
                let m = value.getattr(method.as_str())?;
                let r = m.call((), Some(&kwargs.bind(py).downcast::<PyDict>()?.clone()))?;
                if !r.is_truthy()? {
                    return Ok(VRes::Fail(self.msg_in(message, value, vec![
                        ("method".to_string(), pyo3::types::PyString::new(py, method).into_any()),
                    ])));
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::NoneOf { iterable, values_text, message } => {
                match iterable.bind(py).contains(value) {
                    Ok(true) => Ok(VRes::Fail(self.msg_in(message, value, vec![
                        ("values".to_string(), pyo3::types::PyString::new(py, values_text).into_any()),
                    ]))),
                    Ok(false) => Ok(VRes::Pass),
                    Err(e) => {
                        // `in` raising TypeError means "not a member" -> pass.
                        if e.is_instance_of::<PyTypeError>(py) {
                            return Ok(VRes::Pass);
                        }
                        Err(e)
                    }
                }
            }
            ValidatorKind::OneOf { choices, choices_text, labels_text, message } => {
                match choices.bind(py).contains(value) {
                    Ok(true) => Ok(VRes::Pass),
                    Ok(false) => Ok(VRes::Fail(self.msg_in(message, value, vec![
                        ("choices".to_string(), pyo3::types::PyString::new(py, choices_text).into_any()),
                        ("labels".to_string(), pyo3::types::PyString::new(py, labels_text).into_any()),
                    ]))),
                    Err(e) => {
                        if e.is_instance_of::<PyTypeError>(py) {
                            return Ok(VRes::Fail(self.msg_in(message, value, vec![
                                ("choices".to_string(), pyo3::types::PyString::new(py, choices_text).into_any()),
                                ("labels".to_string(), pyo3::types::PyString::new(py, labels_text).into_any()),
                            ])));
                        }
                        Err(e)
                    }
                }
            }
            ValidatorKind::ContainsOnly { choices, choices_text, labels_text, message } => {
                for val in value.try_iter()? {
                    let val = val?;
                    if !choices.bind(py).contains(&val)? {
                        return Ok(VRes::Fail(self.msg_join(message, value, vec![
                            ("choices".to_string(), pyo3::types::PyString::new(py, choices_text).into_any()),
                            ("labels".to_string(), pyo3::types::PyString::new(py, labels_text).into_any()),
                        ])));
                    }
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::ContainsNoneOf { iterable, values_text, message } => {
                for val in value.try_iter()? {
                    let val = val?;
                    if iterable.bind(py).contains(&val)? {
                        return Ok(VRes::Fail(self.msg_join(message, value, vec![
                            ("values".to_string(), pyo3::types::PyString::new(py, values_text).into_any()),
                        ])));
                    }
                }
                Ok(VRes::Pass)
            }
            ValidatorKind::EmailV { message } => self.email_check(value, message).map(|ok| if ok { VRes::Pass } else { VRes::Fail(self.msg_in(message, value, vec![])) }),
            ValidatorKind::UrlV { relative, absolute, schemes, pattern, message } => {
                self.url_check(value, *relative, *absolute, schemes, pattern)
                    .map(|ok| if ok { VRes::Pass } else { VRes::Fail(self.msg_in(message, value, vec![])) })
            }
            ValidatorKind::Custom { validator } => {
                match validator.bind(py).call1((value,)) {
                    Ok(_) => Ok(VRes::Pass),
                    Err(e) if self.is_ve(&e) => {
                        let val = e.value(py);
                        let messages = val.getattr("messages")?;
                        if messages.downcast::<PyDict>().is_ok() {
                            Ok(VRes::Fail(Msg::Ready(messages.unbind())))
                        } else {
                            Ok(VRes::Expand(messages.unbind()))
                        }
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }
    fn email_check(&self, value: &Bound<'py, PyAny>, _message: &str) -> PyResult<bool> {
        let py = self.py;
        let lib = self.lib();
        if !value.is_truthy()? {
            return Ok(false);
        }
        if !value.contains(pyo3::types::PyString::new(py, "@"))? {
            return Ok(false);
        }
        // rsplit returns a list; unpack positionally like upstream.
        let parts: Bound<PyAny> = value.call_method("rsplit", ("@", 1), None)?;
        let (user, domain) = (parts.get_item(0)?, parts.get_item(1)?);
        if lib.email_user_regex.bind(py).call_method("match", (user,), None)?.is_none() {
            return Ok(false);
        }
        if !lib.email_whitelist.bind(py).contains(&domain)?
            && lib.email_domain_regex.bind(py).call_method("match", (domain,), None)?.is_none()
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn url_check(
        &self,
        value: &Bound<'py, PyAny>,
        _relative: bool,
        _absolute: bool,
        schemes: &Py<PyAny>,
        pattern: &Py<PyAny>,
    ) -> PyResult<bool> {
        let py = self.py;
        if !value.is_truthy()? {
            return Ok(false);
        }
        let has_scheme_sep = value.contains(pyo3::types::PyString::new(py, "://"))?;
        let mut scheme: Option<Bound<PyAny>> = None;
        if has_scheme_sep {
            // split returns a list; unpack positionally like upstream.
            let parts: Bound<PyAny> = value.call_method("split", ("://", 1), None)?;
            let s = parts.get_item(0)?.call_method0("lower")?;
            if !schemes.bind(py).contains(&s)? {
                return Ok(false);
            }
            scheme = Some(s);
        }
        let matched = if scheme.as_ref().map(|s| s.extract::<String>().unwrap_or_default() == "file").unwrap_or(false)
            && value.call_method0("lower")?.call_method("startswith", ("file:///",), None)?.is_truthy()?
        {
            let lowered: String = value.call_method0("lower")?.extract()?;
            let rest: String = lowered.chars().skip(8).collect();
            pattern.bind(py).call_method("search", (format!("file://localhost/{rest}"),), None)?
        } else {
            pattern.bind(py).call_method("search", (value,), None)?
        };
        Ok(!matched.is_none())
    }
}

enum VRes {
    Pass,
    Fail(Msg),
    Expand(Py<PyAny>),
}

fn none_or<'py>(py: Python<'py>, v: &Option<Py<PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    match v {
        Some(x) => Ok(x.bind(py).clone()),
        None => Ok(py.None().bind(py).clone()),
    }
}

fn none_or_obj<'py>(py: Python<'py>, v: Option<&Py<PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    match v {
        Some(x) => Ok(x.bind(py).clone()),
        None => Ok(py.None().bind(py).clone()),
    }
}

fn key_to_errkey(_py: Python, key: &Bound<PyAny>) -> PyResult<ErrKey> {
    match key.extract::<String>() {
        Ok(s) => Ok(ErrKey::S(s)),
        Err(_) => Ok(ErrKey::K(key.clone().unbind())),
    }
}
/// An already-raised ValidationError's messages as an error node: a list
/// contributes its elements (upstream stores the list as the value), any
/// other shape passes through verbatim.
fn ve_messages_node(messages: &Bound<PyAny>) -> ErrNode {
    use pyo3::types::PyList;
    if let Ok(list) = messages.downcast::<PyList>() {
        ErrNode::Msgs(list.iter().map(|m| Msg::Ready(m.unbind())).collect())
    } else {
        ErrNode::Msgs(vec![Msg::Ready(messages.clone().unbind())])
    }
}
/// Fast-path predicates. Exact-type checks only: subclasses keep the slow
/// path (identical behavior, e.g. int() canonicalization, overridden hooks).
fn is_exact_dict(obj: &Bound<PyAny>) -> bool {
    obj.is_exact_instance_of::<PyDict>()
}

fn is_exact_long(obj: &Bound<PyAny>) -> bool {
    obj.is_exact_instance_of::<PyInt>()
}

fn is_exact_float(obj: &Bound<PyAny>) -> bool {
    obj.is_exact_instance_of::<PyFloat>()
}

fn is_exact_unicode(obj: &Bound<PyAny>) -> bool {
    obj.is_exact_instance_of::<PyString>()
}

/// Exact-unicode, dot-free key for the dict fast path.
fn fast_key<'py>(key: &Bound<'py, PyAny>) -> Option<Bound<'py, PyString>> {
    if is_exact_unicode(key) {
        if let Ok(s) = key.downcast::<PyString>() {
            if let Ok(text) = s.to_str() {
                if !text.contains('.') {
                    return Some(s.clone());
                }
            }
        }
    }
    None
}
/// True when the key needs dotted-path store. Exact-unicode keys decide by
/// content; anything else takes the slow path (which performs the same
/// `"." in key` check upstream does), so returning true is always safe.
fn fast_key_is_dotted(key: &Bound<PyAny>) -> PyResult<bool> {
    if is_exact_unicode(key) {
        if let Ok(s) = key.downcast::<PyString>() {
            if let Ok(text) = s.to_str() {
                return Ok(text.contains('.'));
            }
        }
    }
    Ok(true)
}
