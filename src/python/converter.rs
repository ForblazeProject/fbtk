use pyo3::prelude::*;
use crate::core::builder::types::{Atom, Bond, MoleculeTemplate};
use crate::python::PySystem;
use numpy::PyReadonlyArray2;
use crate::core::selection::AtomInfo;

pub fn extract_traj_data(py: Python, obj: PyObject) -> PyResult<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<AtomInfo>)> {
    // 1. PySystem
    if let Ok(system) = obj.extract::<PySystem>(py) {
        let pos = vec![system.atoms.iter().map(|a| a.position.to_array().to_vec()).collect()];
        let cell_mat = glam::DMat3::from_cols_array_2d(&system.cell).transpose();
        let cell_flattened: Vec<f64> = cell_mat.transpose().to_cols_array().to_vec();
        let cells = vec![cell_flattened];
        let info = system.atoms.iter().map(|a| AtomInfo {
            index: a.id,
            element: a.element.clone(),
            resname: a.residue_name.clone(),
            atom_type: 0, // Not strictly used for PySystem
        }).collect();
        return Ok((pos, cells, info));
    }

    // 2. ASE Atoms
    let ase = py.import("ase")?;
    let atoms_cls = ase.getattr("Atoms")?;
    if obj.bind(py).is_instance(&atoms_cls)? {
        let pos: Vec<Vec<f64>> = obj.call_method0(py, "get_positions")?.extract(py)?;
        // Get full 3x3 cell matrix and flatten it to length 9
        let cell_obj = obj.call_method0(py, "get_cell")?;
        let cell_flat: Vec<f64> = cell_obj.call_method0(py, "flatten")?.extract(py)?;
        
        let symbols: Vec<String> = obj.call_method0(py, "get_chemical_symbols")?.extract(py)?;
        let info = symbols.into_iter().enumerate().map(|(i, s)| AtomInfo {
            index: i,
            element: s,
            resname: "UNK".to_string(),
            atom_type: 0,
        }).collect();
        return Ok((vec![pos], vec![cell_flat], info));
    }

    // 3. Trajectory (List)
    if let Ok(list) = obj.bind(py).downcast::<pyo3::types::PyList>() {
        let mut all_pos = Vec::new();
        let mut all_cells = Vec::new();
        let mut info = Vec::new();
        
        for (i, item) in list.iter().enumerate() {
            let pos: Vec<Vec<f64>> = item.call_method0("get_positions")?.extract()?;
            let cell_obj = item.call_method0("get_cell")?;
            let cell_flat: Vec<f64> = cell_obj.call_method0("flatten")?.extract()?;
            if i == 0 {
                let symbols: Vec<String> = item.call_method0("get_chemical_symbols")?.extract()?;
                info = symbols.into_iter().enumerate().map(|(idx, s)| AtomInfo {
                    index: idx,
                    element: s,
                    resname: "UNK".to_string(),
                    atom_type: 0,
                }).collect();
            }
            all_pos.push(pos);
            all_cells.push(cell_flat);
        }
        return Ok((all_pos, all_cells, info));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Input must be a System, ASE Atoms, or List[Atoms]"))
}

pub fn extract_rdkit_template(py: Python, mol: PyObject, name: String) -> PyResult<MoleculeTemplate> {
    let conf = mol.call_method0(py, "GetConformer")?;
    let pos_obj = conf.call_method0(py, "GetPositions")?;
    let positions: PyReadonlyArray2<f64> = pos_obj.extract(py)?;
    let pos = positions.as_array();

    let atoms_list = mol.call_method0(py, "GetAtoms")?;
    let elements: Vec<String> = atoms_list.bind(py).try_iter()?
        .map(|a| a.unwrap().call_method0("GetSymbol").unwrap().extract().unwrap())
        .collect();

    let bonds_list = mol.call_method0(py, "GetBonds")?;
    let mut my_bonds = Vec::new();
    for bond in bonds_list.bind(py).try_iter()? {
        let b = bond.unwrap();
        let i: usize = b.call_method0("GetBeginAtomIdx")?.extract()?;
        let j: usize = b.call_method0("GetEndAtomIdx")?.extract()?;
        let order: f64 = b.call_method0("GetBondTypeAsDouble")?.extract()?;
        my_bonds.push(Bond { atom_i: i, atom_j: j, order });
    }

    let mut atoms = Vec::new();
    for i in 0..pos.shape()[0] {
        atoms.push(Atom {
            id: i,
            residue_name: name.clone(),
            residue_index: 0,
            element: elements[i].clone(),
            atom_type: "XX".to_string(),
            position: [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]].into(),
            charge: 0.0,
            chain_index: 0,
        });
    }

    Ok(MoleculeTemplate { atoms, bonds: my_bonds })
}