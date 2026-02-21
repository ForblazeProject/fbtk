use pyo3::prelude::*;
use crate::core::builder::types::{Atom, Bond, MoleculeTemplate};
use crate::python::PySystem;
use numpy::PyReadonlyArray2;
use crate::core::selection::AtomInfo;

/// Extracts coordinate and topology data from various Python objects (System, ASE Atoms, etc.)
pub fn extract_traj_data(py: Python, obj: PyObject) -> PyResult<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<AtomInfo>)> {
    if let Ok(system) = obj.extract::<PySystem>(py) {
        let pos = vec![system.atoms.iter().map(|a| a.position.to_array().to_vec()).collect()];
        let cell_mat = glam::DMat3::from_cols_array_2d(&system.cell).transpose();
        let cell_flattened: Vec<f64> = cell_mat.transpose().to_cols_array().to_vec();
        let cells = vec![cell_flattened];
        let info = system.atoms.iter().map(|a| AtomInfo {
            index: a.id, element: a.element.clone(), resname: a.residue_name.clone(), atom_type: 0,
        }).collect();
        return Ok((pos, cells, info));
    }
    let ase = py.import("ase")?;
    let atoms_cls = ase.getattr("Atoms")?;
    if obj.bind(py).is_instance(&atoms_cls)? {
        let pos: Vec<Vec<f64>> = obj.call_method0(py, "get_positions")?.extract(py)?;
        let cell_obj = obj.call_method0(py, "get_cell")?;
        let cell_flat: Vec<f64> = cell_obj.call_method0(py, "flatten")?.extract(py)?;
        let symbols: Vec<String> = obj.call_method0(py, "get_chemical_symbols")?.extract(py)?;
        let info = symbols.into_iter().enumerate().map(|(i, s)| AtomInfo {
            index: i, element: s, resname: "UNK".to_string(), atom_type: 0,
        }).collect();
        return Ok((vec![pos], vec![cell_flat], info));
    }
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
                    index: idx, element: s, resname: "UNK".to_string(), atom_type: 0,
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
            id: i, residue_name: name.clone(), residue_index: 0,
            element: elements[i].clone(), atom_type: "XX".to_string(),
            position: [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]].into(),
            charge: 0.0, formal_charge: 0.0, chain_index: 0,
        });
    }
    Ok(MoleculeTemplate { atoms, bonds: my_bonds })
}

pub fn to_rdkit_impl(py: Python, atoms: &[Atom], bonds: &[Bond], pos: &[[f64; 3]]) -> PyResult<PyObject> {
    let _ = py.import("rdkit.Chem")?;
    let atoms_data: Vec<(String, f32, String, usize)> = atoms.iter().map(|a| {
        let res_name = if a.residue_name.is_empty() { "RES".to_string() } else { a.residue_name.to_uppercase() };
        (a.element.clone(), a.formal_charge, res_name, a.residue_index)
    }).collect();
    let bonds_data: Vec<(usize, usize, f64)> = bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect();
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("atoms_data", atoms_data)?;
    locals.set_item("bonds_data", bonds_data)?;
    locals.set_item("pos", pos)?;
    py.run(pyo3::ffi::c_str!(r#"
from rdkit import Chem
from rdkit.Geometry import Point3D
mol = Chem.RWMol()
for element, charge, res_name, res_id in atoms_data:
    a = Chem.Atom(element); a.SetFormalCharge(int(charge)); a.SetNoImplicit(True); a.SetNumExplicitHs(0)
    idx = mol.AddAtom(a)
    res_info = Chem.AtomPDBResidueInfo(element); res_info.SetResidueName(res_name); res_info.SetResidueNumber(int(res_id) if res_id > 0 else 1)
    mol.GetAtomWithIdx(idx).SetMonomerInfo(res_info)
for i, j, order in bonds_data:
    btype = Chem.BondType.SINGLE
    if abs(order - 1.5) < 0.1: btype = Chem.BondType.AROMATIC
    elif abs(order - 2.0) < 0.1: btype = Chem.BondType.DOUBLE
    elif abs(order - 3.0) < 0.1: btype = Chem.BondType.TRIPLE
    mol.AddBond(i, j, btype)
conf = Chem.Conformer(len(atoms_data))
for i, p in enumerate(pos): conf.SetAtomPosition(i, Point3D(p[0], p[1], p[2]))
mol.AddConformer(conf, assignId=True)
mol.UpdatePropertyCache(strict=False)
Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
result_mol = mol
"#), None, Some(&locals))?;
    let result_mol: PyObject = locals.get_item("result_mol")?.unwrap().into();
    Ok(result_mol)
}

pub fn to_openff_impl(py: Python, atoms: &[Atom], bonds: &[Bond], pos: &[[f64; 3]], cell_nm: Option<Vec<f64>>, ff: Option<PyObject>) -> PyResult<PyObject> {
    let rd_mol = to_rdkit_impl(py, atoms, bonds, pos)?;
    let _ = py.import("openff.toolkit")?;
    let _ = py.import("openff.units")?;
    let pos_nm: Vec<f64> = pos.iter().flat_map(|p| [p[0] * 0.1, p[1] * 0.1, p[2] * 0.1]).collect();
    let partial_charges: Vec<f64> = atoms.iter().map(|a| a.charge as f64).collect();
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("rd_mol", rd_mol)?;
    locals.set_item("pos_nm", pos_nm)?;
    locals.set_item("partial_charges", partial_charges)?;
    locals.set_item("is_system", cell_nm.is_some())?;
    locals.set_item("ff", ff)?;
    if let Some(c) = cell_nm { locals.set_item("cell_nm", c)?; }

    py.run(pyo3::ffi::c_str!(r#"
import openff.toolkit, openff.units, numpy as np
from rdkit import Chem
all_pos = np.array(pos_nm).reshape((-1, 3)); all_charges = np.array(partial_charges); unit = openff.units.unit
frag_indices = Chem.GetMolFrags(rd_mol, asMols=False); frags = Chem.GetMolFrags(rd_mol, asMols=True, sanitizeFrags=False)
off_mols = []
for i, idx_list in enumerate(frag_indices):
    m = openff.toolkit.Molecule.from_rdkit(frags[i], allow_undefined_stereo=True)
    m.partial_charges = unit.Quantity(all_charges[list(idx_list)], unit.elementary_charge)
    m._conformers = [unit.Quantity(all_pos[list(idx_list)], unit.nanometer)]; off_mols.append(m)
topology = openff.toolkit.Topology.from_molecules(off_mols)
if is_system: topology.box_vectors = unit.Quantity(np.array(cell_nm).reshape((3, 3)), unit.nanometer)
result = topology
if ff is not None:
    from openff.interchange import Interchange
    ff_obj = openff.toolkit.ForceField(ff, allow_cosmetic_attributes=True) if isinstance(ff, str) else ff
    seen_isomorphs = []; unique_charge_mols = []
    for m in off_mols:
        is_new = True
        for seen in seen_isomorphs:
            if openff.toolkit.Molecule.are_isomorphic(m, seen): is_new = False; break
        if is_new: seen_isomorphs.append(m); unique_charge_mols.append(m)
    interchange = Interchange.from_smirnoff(topology=topology, force_field=ff_obj, charge_from_molecules=unique_charge_mols)
    if is_system: interchange.box = topology.box_vectors
    result = interchange
"#), None, Some(&locals))?;
    let result: PyObject = locals.get_item("result")?.unwrap().into();
    Ok(result)
}

pub fn from_openff_impl(py: Python, obj: PyObject) -> PyResult<(Vec<Atom>, Vec<Bond>, [[f64; 3]; 3])> {
    let _ = py.import("openff.toolkit")?;
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("obj", obj)?;
    py.run(pyo3::ffi::c_str!(r#"
import openff.toolkit, numpy as np, openff.units
input_obj = obj; all_charges = None
if hasattr(input_obj, "topology"): 
    topology = input_obj.topology
    if hasattr(input_obj, "collections") and "Electrostatics" in input_obj.collections:
        coll = input_obj.collections["Electrostatics"]; temp_q = {}
        for k, v in coll.key_map.items(): temp_q[k.atom_indices[0]] = float(coll.potentials[v].parameters["charge"].m)
        all_charges = []
        for i in range(len(temp_q)): all_charges.append(temp_q[i])
elif isinstance(input_obj, openff.toolkit.Molecule):
    topology = input_obj.to_topology()
elif isinstance(input_obj, openff.toolkit.Topology):
    topology = input_obj
else:
    raise TypeError(f"Unsupported type: {type(input_obj)}")
mols = list(topology.molecules); atoms_data = []; ptr = 0
for m_idx, mol in enumerate(mols):
    pos = mol.conformers[0].to(openff.units.unit.angstrom).m if mol.conformers else np.zeros((mol.n_atoms, 3))
    charges = all_charges[ptr : ptr + mol.n_atoms] if all_charges is not None else (mol.partial_charges.to(openff.units.unit.elementary_charge).m if mol.partial_charges is not None else np.zeros(mol.n_atoms))
    for i, atom in enumerate(mol.atoms):
        atoms_data.append({"id": ptr + i, "element": atom.symbol, "charge": float(charges[i]), "formal_charge": float(atom.formal_charge.m), "pos": pos[i].tolist(), "res_name": mol.name if mol.name else "UNL", "res_id": m_idx + 1})
    ptr += mol.n_atoms
bonds_data = [{"i": b.atom1_index, "j": b.atom2_index, "order": float(b.bond_order) if b.bond_order else 1.0} for b in topology.bonds]
if hasattr(input_obj, "box") and input_obj.box is not None:
    cell_ang = input_obj.box.to(openff.units.unit.angstrom).m.reshape((3, 3)).tolist()
elif topology.box_vectors is not None:
    cell_ang = topology.box_vectors.to(openff.units.unit.angstrom).m.reshape((3, 3)).tolist()
else:
    cell_ang = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
"#), None, Some(&locals))?;
    let atoms_raw: Vec<PyObject> = locals.get_item("atoms_data")?.unwrap().extract()?;
    let bonds_raw: Vec<PyObject> = locals.get_item("bonds_data")?.unwrap().extract()?;
    let cell_py: [[f64; 3]; 3] = locals.get_item("cell_ang")?.unwrap().extract()?;
    let mut atoms = Vec::new();
    for a_obj in atoms_raw {
        let a = a_obj.bind(py).downcast::<pyo3::types::PyDict>()?;
        let pos: Vec<f64> = a.get_item("pos")?.unwrap().extract()?;
        atoms.push(Atom {
            id: a.get_item("id")?.unwrap().extract()?, element: a.get_item("element")?.unwrap().extract()?,
            charge: a.get_item("charge")?.unwrap().extract::<f64>()?,
            formal_charge: a.get_item("formal_charge")?.unwrap().extract::<f64>()? as f32,
            position: [pos[0], pos[1], pos[2]].into(), residue_name: a.get_item("res_name")?.unwrap().extract()?,
            residue_index: a.get_item("res_id")?.unwrap().extract()?, atom_type: "XX".to_string(), chain_index: 0,
        });
    }
    let mut bonds = Vec::new();
    for b_obj in bonds_raw {
        let b = b_obj.bind(py).downcast::<pyo3::types::PyDict>()?;
        bonds.push(Bond { atom_i: b.get_item("i")?.unwrap().extract()?, atom_j: b.get_item("j")?.unwrap().extract()?, order: b.get_item("order")?.unwrap().extract()?, });
    }
    Ok((atoms, bonds, cell_py))
}