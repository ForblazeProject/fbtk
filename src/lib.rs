pub mod core;
pub mod parsers;
pub mod python;

use pyo3::prelude::*;
use crate::python::{PyBuilder, PySystem, PyMolecule};
use crate::python::functions::*;

#[pymodule]
fn fbtk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Thread pool initialization
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global();
    }

    m.add_class::<PyBuilder>()?;
    m.add_class::<PySystem>()?;
    m.add_class::<PyMolecule>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(compute_rdf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_msd, m)?)?;
    m.add_function(wrap_pyfunction!(build_system, m)?)?;
    m.add_function(wrap_pyfunction!(from_ase, m)?)?;
    m.add_function(wrap_pyfunction!(from_rdkit, m)?)?;

    Ok(())
}