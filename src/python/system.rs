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
        let ase = py.import("ase")?;
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
        let rdkit = py.import("rdkit.Chem")?;
        let mol_cls = rdkit.getattr("RWMol")?;
        let mol = mol_cls.call0()?;

        let mut sys = crate::core::builder::system::System::new(self.cell);
        sys.atoms = self.atoms.clone();
        sys.bonds = self.bonds.clone();
        sys.unwrap();

        let mut atom_map = std::collections::HashMap::new();
        for atom in &sys.atoms {
            let rd_atom = rdkit.getattr("Atom")?.call1((atom.element.clone(),))?;
            let idx = mol.call_method1("AddAtom", (rd_atom,))?.extract::<usize>()?;
            atom_map.insert(atom.id, idx);
        }

        for bond in &sys.bonds {
            let i = *atom_map.get(&bond.atom_i).unwrap();
            let j = *atom_map.get(&bond.atom_j).unwrap();
            let btype = if bond.order == 1.5 {
                rdkit.getattr("BondType")?.getattr("AROMATIC")?
            } else if bond.order == 2.0 {
                rdkit.getattr("BondType")?.getattr("DOUBLE")?
            } else {
                rdkit.getattr("BondType")?.getattr("SINGLE")?
            };
            mol.call_method1("AddBond", (i, j, btype))?;
        }

        let conf_cls = py.import("rdkit.Chem")?.getattr("Conformer")?;
        let conf = conf_cls.call1((sys.atoms.len(),))?;
        for (i, atom) in sys.atoms.iter().enumerate() {
            let point_cls = py.import("rdkit.Geometry")?.getattr("Point3D")?;
            let point = point_cls.call1((atom.position[0], atom.position[1], atom.position[2]))?;
            conf.call_method1("SetAtomPosition", (i, point))?;
        }
        mol.call_method1("AddConformer", (conf, true))?;

        Ok(mol.into_any().unbind())
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