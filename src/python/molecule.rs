use pyo3::prelude::*;
use crate::core::builder::types::MoleculeTemplate;
use crate::core::builder::smiles::parse_smiles;

#[pyclass(name = "Molecule")]
#[derive(Clone)]
pub struct PyMolecule {
    pub inner: MoleculeTemplate,
    pub name: String,
}

#[pymethods]
impl PyMolecule {
    #[staticmethod]
    #[pyo3(signature = (smiles, name=None))]
    pub fn from_smiles(smiles: &str, name: Option<String>) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| "MOL".to_string());
        let mut template = parse_smiles(smiles).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse SMILES: {}", e))
        })?;
        template.assign_partial_charges();
        Ok(Self { inner: template, name })
    }

    #[staticmethod]
    #[pyo3(signature = (mol, name=None))]
    pub fn from_rdkit(mol: PyObject, name: Option<String>, py: Python) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| "MOL".to_string());
        let template = crate::python::converter::extract_rdkit_template(py, mol, name.clone())?;
        Ok(Self { inner: template, name })
    }

    #[staticmethod]
    #[pyo3(signature = (path, name=None, format=None))]
    pub fn from_file(path: &str, name: Option<String>, format: Option<&str>) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| "MOL".to_string());
        let content = std::fs::read_to_string(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
        })?;
        
        let ext = format.unwrap_or_else(|| {
            std::path::Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
        }).to_lowercase();

        let template = match ext.as_str() {
            "mol" => {
                crate::parsers::mol::parse_mol(&content).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse MOL: {}", e))
                })?
            }
            "mol2" => {
                crate::parsers::mol2::parse_mol2(&content).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse MOL2: {}", e))
                })?
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported file format: {}", ext)));
            }
        };

        Ok(Self { inner: template, name })
    }

    #[staticmethod]
    #[pyo3(signature = (monomer, degree, head=None, tail=None, h_leaving=None, t_leaving=None, tacticity=None, name=None))]
    pub fn from_polymer(
        monomer: &PyMolecule,
        degree: usize,
        head: Option<usize>,
        tail: Option<usize>,
        h_leaving: Option<usize>,
        t_leaving: Option<usize>,
        tacticity: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        use crate::core::builder::config::{PolymerParams, Tacticity};
        use crate::core::builder::model::Builder;
        use crate::core::builder::smiles::resolve_polymer_indices;

        let tact = match tacticity.as_deref() {
            Some("syndiotactic") => Some(Tacticity::Syndiotactic),
            Some("atactic") => Some(Tacticity::Atactic),
            Some("isotactic") => Some(Tacticity::Isotactic),
            _ => None,
        };

        let mut params = PolymerParams {
            degree,
            n_chains: 1,
            head_index: head,
            tail_index: tail,
            head_leaving_index: h_leaving,
            tail_leaving_index: t_leaving,
            tacticity: tact,
        };

        // Resolve indices if missing
        if params.head_index.is_none() || params.tail_index.is_none() {
            let (h, t, hl, tl) = resolve_polymer_indices(&monomer.inner, head, tail).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to resolve polymer indices: {}", e))
            })?;
            if params.head_index.is_none() { params.head_index = Some(h); }
            if params.tail_index.is_none() { params.tail_index = Some(t); }
            if params.head_leaving_index.is_none() { params.head_leaving_index = hl; }
            if params.tail_leaving_index.is_none() { params.tail_leaving_index = tl; }
        }

        let mut chain = Builder::generate_chain(&monomer.inner, &params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to generate polymer chain: {}", e))
        })?;
        
        // Automatically assign charges after polymerization
        chain.assign_partial_charges();

        let final_name = name.unwrap_or_else(|| format!("{}_{}", monomer.name, degree));
        Ok(Self { inner: chain, name: final_name })
    }

    pub fn to_file(&self, path: &str) -> PyResult<()> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "mol2" => {
                let mut sys = crate::core::builder::model::System::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
                sys.atoms = self.inner.atoms.clone();
                sys.bonds = self.inner.bonds.clone();
                let content = crate::parsers::mol2::write_mol2_full(&sys, false);
                std::fs::write(path, content).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write MOL2: {}", e))
                })?;
            }
            "mol" => {
                let content = crate::parsers::mol::write_mol(&self.inner, &self.name);
                std::fs::write(path, content).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write MOL: {}", e))
                })?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Supported formats for Molecule.to_file: .mol, .mol2"));
            }
        }
        Ok(())
    }

            #[pyo3(signature = (steps=None, threshold=None, verbose=true, num_threads=0, cutoff=6.0, history_size=10))]
            pub fn relax(&mut self, steps: Option<usize>, threshold: Option<f64>, verbose: bool, num_threads: usize, cutoff: f64, history_size: usize) -> PyResult<()> {
                // Create a temporary large box
                let mut sys = crate::core::builder::model::System::new([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]);
                sys.atoms = self.inner.atoms.clone();
                sys.bonds = self.inner.bonds.clone();
                
                // Ensure atom IDs match indices (critical for uff-relax)
                for (i, atom) in sys.atoms.iter_mut().enumerate() {
                    atom.id = i;
                }
        
                let mut p = crate::core::builder::relax::RelaxParams::default();
                if let Some(s) = steps { p.steps = s; }
                if let Some(t) = threshold { p.threshold = t; }
                p.verbose = verbose;
                p.num_threads = num_threads;
                p.cutoff = cutoff;
                p.history_size = history_size;
        
                crate::core::builder::relax::minimize(&mut sys, p);
        
                // Write back relaxed coordinates
                self.inner.atoms = sys.atoms;
                Ok(())
            }

        

    

        #[getter]

        pub fn n_atoms(&self) -> usize {

            self.inner.atoms.len()

        }

    

        #[getter]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn get_positions<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        use numpy::{PyArray1, PyArrayMethods};
        let n = self.inner.atoms.len();
        let mut pos = Vec::with_capacity(n * 3);
        for atom in &self.inner.atoms {
            pos.push(atom.position[0]);
            pos.push(atom.position[1]);
            pos.push(atom.position[2]);
        }
        PyArray1::from_vec(py, pos).reshape([n, 3]).unwrap()
    }

    pub fn get_bonds(&self) -> Vec<(usize, usize, f64)> {
        self.inner.bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect()
    }

    pub fn assign_partial_charges(&mut self) {
        self.inner.assign_partial_charges();
    }

    pub fn to_rdkit(&self, py: Python) -> PyResult<PyObject> {
        let _ = py.import("rdkit.Chem").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'rdkit' library is required for to_rdkit(). Please install it via 'pip install rdkit'.")
        })?;

        let atoms_data: Vec<(String, f32, String, usize)> = self.inner.atoms.iter().map(|a| {
            let res_name = if a.residue_name.is_empty() { "RES".to_string() } else { a.residue_name.to_uppercase() };
            (a.element.clone(), a.formal_charge, res_name, a.residue_index)
        }).collect();

        let bonds_data: Vec<(usize, usize, f64)> = self.inner.bonds.iter().map(|b| (b.atom_i, b.atom_j, b.order)).collect();
        let pos: Vec<[f64; 3]> = self.inner.atoms.iter().map(|a| a.position.to_array()).collect();

        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("atoms_data", atoms_data)?;
        locals.set_item("bonds_data", bonds_data)?;
        locals.set_item("pos", pos)?;

        py.run(pyo3::ffi::c_str!(r#"
from rdkit import Chem
from rdkit.Geometry import Point3D

mol = Chem.RWMol()
for element, charge, res_name, res_id in atoms_data:
    a = Chem.Atom(element)
    a.SetFormalCharge(int(charge))
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(0)
    
    idx = mol.AddAtom(a)
    
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

for atom in mol.GetAtoms():
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)

conf = Chem.Conformer(len(atoms_data))
for i, p in enumerate(pos):
    conf.SetAtomPosition(i, Point3D(p[0], p[1], p[2]))
mol.AddConformer(conf, assignId=True)

# THE CRITICAL SEQUENCE:
# 1. Update property cache with strict=False
mol.UpdatePropertyCache(strict=False)
# 2. Sanitize only after properties are cached correctly
Chem.SanitizeMol(mol)

result_mol = mol
"#), None, Some(&locals))?;

        let result_mol: PyObject = locals.get_item("result_mol")?.unwrap().into();
        Ok(result_mol)
    }

    pub fn to_openff(&self, py: Python) -> PyResult<PyObject> {
        let rd_mol = self.to_rdkit(py)?;
        let _ = py.import("openff.toolkit").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-toolkit' library is required for to_openff(). Please install it via 'conda install openff-toolkit'.")
        })?;
        let _ = py.import("openff.units").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>("The 'openff-units' library is required for to_openff().")
        })?;

        let pos_nm: Vec<f64> = self.inner.atoms.iter().flat_map(|a| [a.position.x * 0.1, a.position.y * 0.1, a.position.z * 0.1]).collect();
        
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rd_mol", rd_mol)?;
        locals.set_item("pos_nm", pos_nm)?;

        py.run(pyo3::ffi::c_str!(r#"
import openff.toolkit
import openff.units
import numpy as np

# Use allow_undefined_stereo=True to improve robustness for built structures
molecule = openff.toolkit.Molecule.from_rdkit(rd_mol, allow_undefined_stereo=True)
m_pos = np.array(pos_nm).reshape((-1, 3))
molecule._conformers = [openff.units.unit.Quantity(m_pos, openff.units.unit.nanometer)]
"#), None, Some(&locals))?;

        let molecule: PyObject = locals.get_item("molecule")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to generate OpenFF Molecule")
        })?.into();

        Ok(molecule)
    }
}

    