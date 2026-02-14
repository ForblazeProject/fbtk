use pyo3::prelude::*;
use crate::python::{PyBuilder, PySystem};
use crate::python::converter::extract_traj_data;
use crate::core::selection::SelectionEngine;
use crate::core::rdf::{RdfParams, compute_rdf_core};
use crate::core::msd::compute_msd_core;
use numpy::ToPyArray;
use ndarray::Array3;

/// Compute RDF with smart query.
#[pyfunction]
#[pyo3(signature = (input, query, r_max=10.0, n_bins=200))]
pub fn compute_rdf(
    py: Python,
    input: PyObject,
    query: String,
    r_max: f64,
    n_bins: usize,
) -> PyResult<(PyObject, PyObject)> {
    let (all_pos, all_cells, info) = extract_traj_data(py, input)?;
    let engine = SelectionEngine::new(info);
    let (idx_a, idx_b) = engine.select_pair(&query).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    let n_frames = all_pos.len();
    let n_atoms = all_pos[0].len();

    // Bounds check
    for &i in idx_a.iter().chain(idx_b.iter()) {
        if i >= n_atoms {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Index {} is out of bounds for system with {} atoms", i, n_atoms)
            ));
        }
    }

    let mut pos_array = Array3::zeros((n_frames, n_atoms, 3));
    let mut cell_array = Array3::zeros((n_frames, 3, 3));

    for f in 0..n_frames {
        for a in 0..n_atoms {
            for i in 0..3 { pos_array[[f, a, i]] = all_pos[f][a][i]; }
        }
        if all_cells[f].len() == 9 {
            for i in 0..3 {
                for j in 0..3 {
                    cell_array[[f, i, j]] = all_cells[f][i * 3 + j];
                }
            }
        } else if all_cells[f].len() == 3 {
            cell_array[[f, 0, 0]] = all_cells[f][0];
            cell_array[[f, 1, 1]] = all_cells[f][1];
            cell_array[[f, 2, 2]] = all_cells[f][2];
        }
    }

    let params = RdfParams { r_max, n_bins };
    let result = compute_rdf_core(pos_array.view(), cell_array.view(), &idx_a, &idx_b, params);
    Ok((result.r_axis.to_pyarray(py).into_any().unbind(), result.g_r.to_pyarray(py).into_any().unbind()))
}

/// Compute MSD with smart query.
#[pyfunction]
#[pyo3(signature = (input, query, max_lag=0, dt=1.0))]
pub fn compute_msd(
    py: Python,
    input: PyObject,
    query: String,
    max_lag: usize,
    dt: f64,
) -> PyResult<PyObject> {
    let (all_pos, all_cells, info) = extract_traj_data(py, input)?;
    let engine = SelectionEngine::new(info);
    let idx = engine.select(&query).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    let n_frames = all_pos.len();
    let n_atoms = all_pos[0].len();
    let mut pos_array = Array3::zeros((n_frames, n_atoms, 3));
    let mut cell_array = Array3::zeros((n_frames, 3, 3));

    for f in 0..n_frames {
        for a in 0..n_atoms {
            for i in 0..3 { pos_array[[f, a, i]] = all_pos[f][a][i]; }
        }
        if all_cells[f].len() == 9 {
            for i in 0..3 {
                for j in 0..3 {
                    cell_array[[f, i, j]] = all_cells[f][i * 3 + j];
                }
            }
        } else if all_cells[f].len() == 3 {
            cell_array[[f, 0, 0]] = all_cells[f][0];
            cell_array[[f, 1, 1]] = all_cells[f][1];
            cell_array[[f, 2, 2]] = all_cells[f][2];
        }
    }

    let res = compute_msd_core(pos_array.view(), cell_array.view(), &idx, max_lag, dt);
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("time", res.time.to_pyarray(py))?;
    dict.set_item("msd", res.msd_total.to_pyarray(py))?;
    Ok(dict.into_any().unbind())
}

/// Build a system from a recipe YAML file.
#[pyfunction]
pub fn build_system(py: Python, recipe_path: String) -> PyResult<PySystem> {
    let mut builder = PyBuilder::new(None, None)?;
    builder.load_recipe(recipe_path)?;
    builder.build(py)
}

/// Create a System object from an ASE Atoms object.
#[pyfunction]
pub fn from_ase(py: Python, atoms: PyObject) -> PyResult<PySystem> {
    let (all_pos, all_cells, info) = extract_traj_data(py, atoms)?;
    if all_pos.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty ASE object"));
    }
    
    let pos = &all_pos[0];
    let cell_raw = &all_cells[0]; 
    
    let mut cell = [[0.0; 3]; 3];
    if cell_raw.len() == 3 {
        cell[0][0] = cell_raw[0];
        cell[1][1] = cell_raw[1];
        cell[2][2] = cell_raw[2];
    } else if cell_raw.len() == 9 {
        for i in 0..3 { for j in 0..3 { cell[i][j] = cell_raw[i * 3 + j]; } }
    }

    let mut atoms_list = Vec::new();
    for (i, p) in pos.iter().enumerate() {
        atoms_list.push(crate::core::builder::types::Atom {
            id: i,
            residue_name: info[i].resname.clone(),
            residue_index: 0,
            element: info[i].element.clone(),
            atom_type: info[i].element.clone(), // Fix XX issue here too
            position: [p[0], p[1], p[2]].into(),
            charge: 0.0,
            chain_index: 0,
        });
    }

    Ok(PySystem {
        n_atoms: atoms_list.len(),
        cell,
        atoms: atoms_list,
        bonds: Vec::new(),
    })
}

/// Create a System object from an RDKit Mol object.
#[pyfunction]
pub fn from_rdkit(py: Python, mol: PyObject) -> PyResult<PySystem> {
    let template = crate::python::converter::extract_rdkit_template(py, mol, "MOL".to_string())?;
    let cell = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
    
    Ok(PySystem {
        n_atoms: template.atoms.len(),
        cell,
        atoms: template.atoms,
        bonds: template.bonds,
    })
}

/// Helper function to run the analyze CLI logic from Python
#[pyfunction]
pub fn run_analyze_cli(_py: Python, args: Vec<String>) -> PyResult<()> {
    crate::core::cli::analyze::run_analyze_cli(args).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })
}

#[pyfunction]
pub fn run_build_cli(_py: Python, args: Vec<String>) -> PyResult<()> {
    crate::core::cli::build::run_build_cli(args).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })
}
