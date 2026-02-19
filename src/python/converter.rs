use pyo3::prelude::*;
use crate::core::builder::types::{Atom, Bond, MoleculeTemplate};
use crate::python::PySystem;
use numpy::PyReadonlyArray2;
use crate::core::selection::AtomInfo;

/// Extracts coordinate and topology data from various Python objects (System, ASE Atoms, etc.)
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

/// Converts RDKit Molecule to an internal FBTK MoleculeTemplate.
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
            formal_charge: 0.0,
            chain_index: 0,
        });
    }

    Ok(MoleculeTemplate { atoms, bonds: my_bonds })
}

/// Converts internal atom and bond data to an RDKit Mol object using a robust Python-based construction sequence.
pub fn to_rdkit_impl(py: Python, atoms: &[Atom], bonds: &[Bond], pos: &[[f64; 3]]) -> PyResult<PyObject> {
    let _ = py.import("rdkit.Chem").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'rdkit' library is required for RDKit conversion. Please install it via 'pip install rdkit'.")
    })?;

    // Prepare raw data for Python construction
    let atoms_data: Vec<(String, f32, String, usize)> = atoms.iter().map(|a| {
        let res_name = if a.residue_name.is_empty() { "RES".to_string() } else { a.residue_name.to_uppercase() };
        (a.element.clone(), a.formal_charge, res_name, a.residue_index)
    }).collect();

    let bonds_data: Vec<(usize, usize, f64)> = bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect();
    
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("atoms_data", atoms_data)?;
    locals.set_item("bonds_data", bonds_data)?;
    locals.set_item("pos", pos)?;

    // Build RDKit Mol in Python context with strict control over sequence (to handle anions/cations correctly)
    py.run(pyo3::ffi::c_str!(r#"
from rdkit import Chem
from rdkit.Geometry import Point3D

mol = Chem.RWMol()
for element, charge, res_name, res_id in atoms_data:
    a = Chem.Atom(element)
    a.SetFormalCharge(int(charge))
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(0)
    a.SetNumRadicalElectrons(0)
    
    idx = mol.AddAtom(a)
    
    # Residue info
    res_info = Chem.AtomPDBResidueInfo(element)
    res_info.SetResidueName(res_name)
    res_info.SetResidueNumber(int(res_id) if res_id > 0 else 1)
    mol.GetAtomWithIdx(idx).SetMonomerInfo(res_info)

for i, j, order in bonds_data:
    if abs(order - 1.5) < 0.1:
        btype = Chem.BondType.AROMATIC
    elif abs(order - 2.0) < 0.1:
        btype = Chem.BondType.DOUBLE
    elif abs(order - 3.0) < 0.1:
        btype = Chem.BondType.TRIPLE
    else:
        btype = Chem.BondType.SINGLE
    mol.AddBond(i, j, btype)

# FINAL GUARD: Ensure RDKit doesn't add implicit hydrogens after bond creation
for atom in mol.GetAtoms():
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)

# Setup geometry
conf = Chem.Conformer(len(atoms_data))
for i, p in enumerate(pos):
    conf.SetAtomPosition(i, Point3D(p[0], p[1], p[2]))
mol.AddConformer(conf, assignId=True)

# THE CRITICAL SEQUENCE for Ion support
mol.UpdatePropertyCache(strict=False)
# Perform sanitization but skip property validation (which includes strict valence checks)
# that often fails for custom ions even with correct charges.
flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
Chem.SanitizeMol(mol, sanitizeOps=flags)

result_mol = mol
"#), None, Some(&locals))?;

    let result_mol: PyObject = locals.get_item("result_mol")?.unwrap().into();
    Ok(result_mol)
}

/// Converts atom and bond data to an OpenFF Molecule or Topology via RDKit bridge.
pub fn to_openff_impl(py: Python, atoms: &[Atom], bonds: &[Bond], pos: &[[f64; 3]], cell_nm: Option<Vec<f64>>) -> PyResult<PyObject> {
    let rd_mol = to_rdkit_impl(py, atoms, bonds, pos)?;

    let _ = py.import("openff.toolkit").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-toolkit' library is required for OpenFF conversion.")
    })?;
    let _ = py.import("openff.units").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-units' library is required for OpenFF conversion.")
    })?;

    // Prepare coordinates in Nanometers (0.1x)
    let pos_nm: Vec<f64> = pos.iter().flat_map(|p| [p[0] * 0.1, p[1] * 0.1, p[2] * 0.1]).collect();
    
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("rd_mol", rd_mol)?;
    locals.set_item("pos_nm", pos_nm)?;
    locals.set_item("is_system", cell_nm.is_some())?;
    if let Some(c) = cell_nm {
        locals.set_item("cell_nm", c)?;
    }

    py.run(pyo3::ffi::c_str!(r#"
import openff.toolkit
import openff.units
from rdkit import Chem
import numpy as np

all_pos = np.array(pos_nm).reshape((-1, 3))
unit = openff.units.unit

if is_system:
    # Handle as System (Topology with multiple fragments)
    frags = Chem.GetMolFrags(rd_mol, asMols=True)
    off_mols = []
    ptr = 0
    for f in frags:
        m = openff.toolkit.Molecule.from_rdkit(f, allow_undefined_stereo=True)
        m_pos = all_pos[ptr : ptr + m.n_atoms]
        m._conformers = [unit.Quantity(m_pos, unit.nanometer)]
        off_mols.append(m)
        ptr += m.n_atoms
    
    result = openff.toolkit.Topology.from_molecules(off_mols)
    box_matrix = np.array(cell_nm).reshape((3, 3))
    result.box_vectors = unit.Quantity(box_matrix, unit.nanometer)
else:
    # Handle as single Molecule
    result = openff.toolkit.Molecule.from_rdkit(rd_mol, allow_undefined_stereo=True)
    result._conformers = [unit.Quantity(all_pos, unit.nanometer)]

"#), None, Some(&locals))?;

    let result: PyObject = locals.get_item("result")?.unwrap().into();
    Ok(result)
}
