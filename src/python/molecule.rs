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
        let template = parse_smiles(smiles).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse SMILES: {}", e))
        })?;
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

    pub fn to_file(&self, path: &str) -> PyResult<()> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext != "mol2" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Only .mol2 is supported for to_file currently"));
        }

        let mut sys = crate::core::builder::model::System::new([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]);
        sys.atoms = self.inner.atoms.clone();
        sys.bonds = self.inner.bonds.clone();
        
        let content = crate::parsers::mol2::write_mol2(&sys);
        std::fs::write(path, content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write file: {}", e))
        })?;
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
}

    