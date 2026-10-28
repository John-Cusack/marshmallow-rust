//! PyO3 bindings: `marshmallow_rust._core`.

use marshmallow_core::{node_to_py, Exec, Plan, UnknownOpt};
use pyo3::prelude::*;

#[pyclass]
struct PlanHandle {
    plan: Plan,
}

#[pyfunction]
fn compile(plan_dict: &Bound<PyAny>) -> PyResult<PlanHandle> {
    let py = plan_dict.py();
    Ok(PlanHandle {
        plan: Plan::from_dict(py, plan_dict)?,
    })
}

#[pyfunction]
fn dump(handle: &PlanHandle, obj: &Bound<PyAny>, many: bool) -> PyResult<Py<PyAny>> {
    let py = obj.py();
    let exec = Exec::new(py, &handle.plan)?;
    exec.dump_root(0, obj, many)
}

#[pyfunction]
fn load(
    handle: &PlanHandle,
    data: &Bound<PyAny>,
    many: bool,
    partial: &Bound<PyAny>,
    unknown: &Bound<PyAny>,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>)> {
    let py = data.py();
    let exec = Exec::new(py, &handle.plan)?;
    let unk = match unknown.extract::<String>() {
        Ok(s) => UnknownOpt::parse(&s),
        Err(_) => UnknownOpt::Other,
    };
    let (result, errors) = exec.load_root(0, data, many, partial, unk)?;
    let errors_py = errors.map(|n| node_to_py(py, &n)).transpose()?;
    Ok((result, errors_py))
}

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(dump, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add("version", "0.1.0")?;
    Ok(())
}
