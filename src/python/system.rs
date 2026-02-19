use pyo3::prelude::*;
use crate::core::builder::types::{Atom, Bond};
use numpy::{PyArray2, ToPyArray, PyArrayMethods};

#[derive(Clone)]
#[pyclass(name = "System")]
pub struct PySystem {
    #[pyo3(get)]
    pub n_atoms: usize,
    #[pyo3(get)]
    pub cell: [[f64; 3]; 3],
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

#[pymethods]
impl PySystem {
    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    pub fn from_file(path: &str, format: Option<&str>) -> PyResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
        })?;
        
        let ext = format.unwrap_or_else(|| {
            std::path::Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
        }).to_lowercase();

        if ext != "mol2" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Only .mol2 is supported for System.from_file currently"));
        }

        let system = crate::parsers::mol2::parse_mol2_as_system(&content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse MOL2 as system: {}", e))
        })?;

        Ok(PySystem {
            n_atoms: system.atoms.len(),
            cell: system.cell.transpose().to_cols_array_2d(),
            atoms: system.atoms,
            bonds: system.bonds,
        })
    }

    #[getter]
    pub fn n_atoms(&self) -> usize { self.n_atoms }

    pub fn get_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        use numpy::PyArray1;
        let mut pos = Vec::with_capacity(self.n_atoms * 3);
        for atom in &self.atoms {
            pos.push(atom.position[0]);
            pos.push(atom.position[1]);
            pos.push(atom.position[2]);
        }
        PyArray1::from_vec(py, pos).reshape([self.n_atoms, 3]).unwrap()
    }

    pub fn get_bonds(&self) -> Vec<(usize, usize, f64)> {
        self.bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect()
    }

    #[pyo3(signature = (mic=false))]
    pub fn get_all_distances<'py>(&self, py: Python<'py>, mic: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        
        let dists = sys.get_all_distances(mic);
        let n = self.n_atoms;
        let array = dists.to_pyarray(py).reshape([n, n])?;
        Ok(array)
    }

    pub fn get_center_of_mass(&self) -> [f64; 3] {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_center_of_mass().into()
    }

    #[pyo3(signature = (indices, mic=false))]
    pub fn get_angles(&self, indices: Vec<[usize; 3]>, mic: bool) -> Vec<f64> {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_angles(&indices, mic)
    }

    #[pyo3(signature = (indices, mic=false))]
    pub fn get_dihedrals(&self, indices: Vec<[usize; 4]>, mic: bool) -> Vec<f64> {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_dihedrals(&indices, mic)
    }

    pub fn wrap(&mut self) {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.wrap();
        self.atoms = sys.atoms;
    }

    pub fn unwrap(&mut self) {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        sys.unwrap();
        self.atoms = sys.atoms;
    }

    pub fn get_neighbor_list(&self, cutoff: f64) -> Vec<(usize, usize, f64)> {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_neighbor_list(cutoff)
    }

    pub fn get_volume(&self) -> f64 {
        crate::core::builder::system::System::new(self.cell).get_volume()
    }

    pub fn get_total_mass(&self) -> f64 {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_total_mass()
    }

    pub fn get_density(&self) -> f64 {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.get_density()
    }

    #[pyo3(signature = (steps=None, threshold=None, verbose=true, num_threads=0, cutoff=6.0, history_size=10))]
    pub fn relax(&mut self, steps: Option<usize>, threshold: Option<f64>, verbose: bool, num_threads: usize, cutoff: f64, history_size: usize) -> PyResult<()> {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        
        let mut p = crate::core::builder::relax::RelaxParams::default();
        if let Some(s) = steps { p.steps = s; }
        if let Some(t) = threshold { p.threshold = t; }
        p.verbose = verbose;
        p.num_threads = num_threads;
        p.cutoff = cutoff;
        p.history_size = history_size;
        
        crate::core::builder::relax::minimize(&mut sys, p);
        
        sys.unwrap(); 
        self.atoms = sys.atoms;
        Ok(())
    }

    pub fn stack(&mut self, other: &PySystem, axis: usize) -> PyResult<()> {
        let mut core_sys = crate::core::builder::system::System::new(self.cell);
        core_sys.atoms = self.atoms.clone();
        core_sys.bonds = self.bonds.clone();

        let mut other_core = crate::core::builder::system::System::new(other.cell);
        other_core.atoms = other.atoms.clone();
        other_core.bonds = other.bonds.clone();

        core_sys.stack(&other_core, axis).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        self.atoms = core_sys.atoms;
        self.bonds = core_sys.bonds;
        self.cell = core_sys.cell.transpose().to_cols_array_2d();
        self.n_atoms = self.atoms.len();
        Ok(())
    }

    pub fn to_ase(&self, py: Python) -> PyResult<PyObject> {
        let ase = py.import("ase").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'ase' library is required for to_ase(). Please install it via 'pip install ase'.")
        })?;
        let atoms_cls = ase.getattr("Atoms")?;
        
        let mut symbols = Vec::new();
        let mut positions = Vec::new();
        for atom in &self.atoms {
            let mut s = atom.element.to_lowercase();
            if let Some(first) = s.get_mut(0..1) { first.make_ascii_uppercase(); }
            symbols.push(s);
            positions.push(atom.position.to_array().to_vec());
        }

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("symbols", symbols)?;
        kwargs.set_item("positions", positions)?;
        
        let cell_vec: Vec<Vec<f64>> = self.cell.iter().map(|row| row.to_vec()).collect();
        kwargs.set_item("cell", cell_vec)?;
        kwargs.set_item("pbc", true)?;

        let atoms_obj = atoms_cls.call((), Some(&kwargs))?;
        Ok(atoms_obj.into_any().unbind())
    }

    pub fn to_rdkit(&self, py: Python) -> PyResult<PyObject> {
        let _ = py.import("rdkit.Chem").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'rdkit' library is required for to_rdkit(). Please install it via 'pip install rdkit'.")
        })?;

        // 1. Prepare raw data for Python construction
        let atoms_data: Vec<(String, f32, String, usize)> = self.atoms.iter().map(|a| {
            let res_name = if a.residue_name.is_empty() { "RES".to_string() } else { a.residue_name.to_uppercase() };
            (a.element.clone(), a.formal_charge, res_name, a.residue_index)
        }).collect();

        let bonds_data: Vec<(usize, usize, f64)> = self.bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect();
        
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        sys.unwrap();
        let pos: Vec<[f64; 3]> = sys.atoms.iter().map(|a| a.position.to_array()).collect();

        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("atoms_data", atoms_data)?;
        locals.set_item("bonds_data", bonds_data)?;
        locals.set_item("pos", pos)?;

        // 2. Build RDKit Mol in Python context with strict control over sequence
        py.run(pyo3::ffi::c_str!(r#"
from rdkit import Chem
from rdkit.Geometry import Point3D

mol = Chem.RWMol()
# IMPORTANT: Construct without any automatic sanitization or valence checks
for element, charge, res_name, res_id in atoms_data:
    a = Chem.Atom(element)
    a.SetFormalCharge(int(charge))
    # Disable implicit hydrogens to prevent RDKit from adding them before we set bonds
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(0)
    
    idx = mol.AddAtom(a)
    
    # Set residue info
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

# THE CRITICAL SEQUENCE:
# 1. Update property cache with strict=False to acknowledge formal charges and bonds
mol.UpdatePropertyCache(strict=False)
# 2. Sanitize only after properties are cached correctly. [O-] is now valid.
Chem.SanitizeMol(mol)

result_mol = mol
"#), None, Some(&locals))?;

        let result_mol: PyObject = locals.get_item("result_mol")?.unwrap().into();
        Ok(result_mol)
    }

    pub fn to_openff_topology(&self, py: Python) -> PyResult<PyObject> {
        let rd_mol = self.to_rdkit(py)?;
        
        // 1. Prepare coordinates in Nanometers (0.1x) directly in Rust
        let pos_nm: Vec<f64> = self.atoms.iter().flat_map(|a| [a.position.x * 0.1, a.position.y * 0.1, a.position.z * 0.1]).collect();
        let cell_nm: Vec<f64> = self.cell.iter().flatten().map(|v| v * 0.1).collect();

        let _ = py.import("openff.toolkit").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-toolkit' library is required for to_openff_topology(). Please install it via 'conda install openff-toolkit'.")
        })?;
        let _ = py.import("openff.units").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-units' library is required for to_openff_topology().")
        })?;
        
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rd_mol", rd_mol)?;
        locals.set_item("pos_nm", pos_nm)?;
        locals.set_item("cell_nm", cell_nm)?;

        // 2. Execute conversion in pure Python environment with explicit NM units
        py.run(pyo3::ffi::c_str!(r#"
import openff.toolkit
import openff.units
from rdkit import Chem
import numpy as np

# 1. Fragment the system
frags = Chem.GetMolFrags(rd_mol, asMols=True)

# 2. Convert each fragment to an OpenFF Molecule
off_mols = []
atom_ptr = 0
all_pos = np.array(pos_nm).reshape((-1, 3))

for f in frags:
    m = openff.toolkit.Molecule.from_rdkit(f, allow_undefined_stereo=True)
    n = m.n_atoms
    # Assign the pre-converted nanometer coordinates
    m_pos = all_pos[atom_ptr : atom_ptr + n]
    m._conformers = [openff.units.unit.Quantity(m_pos, openff.units.unit.nanometer)]
    off_mols.append(m)
    atom_ptr += n

# 3. Create Topology
topology = openff.toolkit.Topology.from_molecules(off_mols)

# 4. Set box vectors via property assignment (already in NM)
box_matrix = np.array(cell_nm).reshape((3, 3))
topology.box_vectors = openff.units.unit.Quantity(box_matrix, openff.units.unit.nanometer)
"#), None, Some(&locals))?;

        let topology: PyObject = locals.get_item("topology")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to generate OpenFF Topology")
        })?.into();

        Ok(topology)
    }

    pub fn to_file(&self, path: &str) -> PyResult<()> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext != "mol2" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Only .mol2 is supported for to_file currently"));
        }

        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        
        let content = crate::parsers::mol2::write_mol2(&sys);
        std::fs::write(path, content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write file: {}", e))
        })?;
        Ok(())
    }

    pub fn assign_partial_charges(&mut self) {
        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        sys.assign_partial_charges();
        self.atoms = sys.atoms;
    }
}