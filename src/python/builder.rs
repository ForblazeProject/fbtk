use pyo3::prelude::*;
use crate::core::builder::model::{Builder as CoreBuilder};
use crate::core::builder::types::MoleculeTemplate;
use crate::core::builder::config::Recipe;
use crate::python::system::PySystem;
use crate::python::converter::extract_rdkit_template;
use numpy::PyReadonlyArray2;

use crate::python::molecule::PyMolecule;

#[pyclass(name = "Builder")]
pub struct PyBuilder {
    pub inner: CoreBuilder,
}

#[pymethods]
impl PyBuilder {
    #[new]
    #[pyo3(signature = (box_size=None, density=None))]
    pub fn new(box_size: Option<Vec<f64>>, density: Option<f64>) -> PyResult<Self> {
        let mut builder = Self { inner: CoreBuilder::new() };
        if let Some(bs) = box_size {
            builder.set_cell(bs)?;
        }
        if let Some(d) = density {
            builder.set_density(d);
        }
        Ok(builder)
    }

    #[pyo3(signature = (molecule, count))]
    pub fn add_molecule(&mut self, molecule: &PyMolecule, count: usize) {
        let name = molecule.name.clone();
        self.inner.add_template(name.clone(), molecule.inner.clone());
        
        let recipe = self.ensure_recipe();
        recipe.components.push(crate::core::builder::config::ComponentConfig {
            name,
            role: crate::core::builder::config::ComponentRole::Molecule,
            input: crate::core::builder::config::InputSource { smiles: None, file: None, format: None },
            count: Some(count),
            polymer_params: None,
        });
    }

    pub fn load_recipe(&mut self, path: String) -> PyResult<()> {
        let recipe = Recipe::from_yaml(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load recipe: {}", e))
        })?;
        self.inner.set_recipe(recipe);
        Ok(())
    }

    pub fn set_density(&mut self, density: f64) {
        self.ensure_recipe().system.density = density;
    }

    pub fn set_cell(&mut self, box_size: Vec<f64>) -> PyResult<()> {
        if box_size.len() != 3 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Box size must be a list of 3 floats"));
        }
        self.ensure_recipe().system.cell_shape = Some([box_size[0], box_size[1], box_size[2]]);
        Ok(())
    }

    #[pyo3(signature = (name, count, smiles=None))]
    pub fn add_molecule_smiles(&mut self, name: String, count: usize, smiles: Option<String>) {
        let recipe = self.ensure_recipe();
        recipe.components.push(crate::core::builder::config::ComponentConfig {
            name,
            role: crate::core::builder::config::ComponentRole::Molecule,
            input: crate::core::builder::config::InputSource { smiles, file: None, format: None },
            count: Some(count),
            polymer_params: None,
        });
    }

    #[pyo3(signature = (name, count, degree, smiles=None, head=None, tail=None, h_leaving=None, t_leaving=None, tacticity=None))]
    pub fn add_polymer(
        &mut self, 
        name: String, 
        count: usize, 
        degree: usize, 
        smiles: Option<String>,
        head: Option<usize>,
        tail: Option<usize>,
        h_leaving: Option<usize>,
        t_leaving: Option<usize>,
        tacticity: Option<String>,
    ) {
        let tact = match tacticity.as_deref() {
            Some("syndiotactic") => Some(crate::core::builder::config::Tacticity::Syndiotactic),
            Some("atactic") => Some(crate::core::builder::config::Tacticity::Atactic),
            Some("isotactic") => Some(crate::core::builder::config::Tacticity::Isotactic),
            _ => None,
        };

        let recipe = self.ensure_recipe();
        recipe.components.push(crate::core::builder::config::ComponentConfig {
            name,
            role: crate::core::builder::config::ComponentRole::Polymer,
            input: crate::core::builder::config::InputSource { smiles, file: None, format: None },
            count: None,
            polymer_params: Some(crate::core::builder::config::PolymerParams {
                degree,
                n_chains: count,
                head_index: head,
                tail_index: tail,
                head_leaving_index: h_leaving,
                tail_leaving_index: t_leaving,
                tacticity: tact,
            }),
        });
    }

    pub fn add_rdkit_mol(&mut self, name: String, mol: PyObject, py: Python) -> PyResult<()> {
        let template = extract_rdkit_template(py, mol, name.clone())?;
        self.inner.add_template(name, template);
        Ok(())
    }

    pub fn add_template(
        &mut self,
        name: String,
        positions: PyReadonlyArray2<f64>,
        elements: Vec<String>,
        bonds: Vec<(usize, usize, f64)>,
    ) -> PyResult<()> {
        let pos = positions.as_array();
        let mut atoms = Vec::new();
        for i in 0..pos.shape()[0] {
            atoms.push(crate::core::builder::model::Atom {
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
        let my_bonds = bonds.into_iter().map(|(i, j, o)| crate::core::builder::model::Bond { atom_i: i, atom_j: j, order: o }).collect();
        self.inner.add_template(name, MoleculeTemplate { atoms, bonds: my_bonds });
        Ok(())
    }

    pub fn build(&mut self, _py: Python) -> PyResult<PySystem> {
        // Step 1: Identify missing templates that have SMILES defined
        let mut missing_smiles = Vec::new();
        if let Some(recipe) = &self.inner.recipe {
            for comp in &recipe.components {
                if !self.inner.templates.contains_key(&comp.name) {
                    if let Some(smiles) = &comp.input.smiles {
                        missing_smiles.push((comp.name.clone(), smiles.clone()));
                    }
                }
            }
        }

        // Step 2: Generate and add missing templates
        for (name, smiles) in missing_smiles {
            let mut tmpl = crate::core::builder::smiles::parse_smiles(&smiles)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse SMILES for {}: {}", name, e)))?;
            tmpl.assign_partial_charges();
            self.inner.add_template(name, tmpl);
        }

        // Step 3: Resolve Polymer Indices for components with SMILES
        if let Some(recipe) = &mut self.inner.recipe {
            for comp in &mut recipe.components {
                if comp.role == crate::core::builder::config::ComponentRole::Polymer && comp.input.smiles.is_some() {
                    if let Some(params) = &mut comp.polymer_params {
                        if let Some(tmpl) = self.inner.templates.get(&comp.name) {
                            // Resolve indices: User provided "Heavy Atom Index" -> We map to "Atom ID"
                            let (h_id, t_id, hl_id, tl_id) = crate::core::builder::smiles::resolve_polymer_indices(
                                tmpl, 
                                params.head_index, 
                                params.tail_index
                            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to resolve polymer indices for {}: {}", comp.name, e)))?;
                            
                            // Update params with resolved actual IDs
                            params.head_index = Some(h_id);
                            params.tail_index = Some(t_id);
                            
                            // Only update leaving atoms if not manually specified
                            if params.head_leaving_index.is_none() {
                                params.head_leaving_index = hl_id;
                            }
                            if params.tail_leaving_index.is_none() {
                                params.tail_leaving_index = tl_id;
                            }
                        }
                    }
                }
            }
        }

        // Step 4: Build
        self.inner.build().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Build failed: {}", e))
        })?;
        let system = self.inner.system.as_ref().unwrap();
        Ok(PySystem {
            n_atoms: system.atoms.len(),
            cell: system.cell.transpose().to_cols_array_2d(),
            atoms: system.atoms.clone(),
            bonds: system.bonds.clone(),
        })
    }

    pub fn get_system(&self, _py: Python) -> PyResult<PySystem> {
        let system = self.inner.system.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("System not built yet")
        })?;
        Ok(PySystem {
            n_atoms: system.atoms.len(),
            cell: system.cell.transpose().to_cols_array_2d(),
            atoms: system.atoms.clone(),
            bonds: system.bonds.clone(),
        })
    }

    #[pyo3(signature = (steps=None, threshold=None, verbose=true, num_threads=0, cutoff=6.0, history_size=10))]
    pub fn relax(&mut self, steps: Option<usize>, threshold: Option<f64>, verbose: bool, num_threads: usize, cutoff: f64, history_size: usize) -> PyResult<()> {
        let mut p = crate::core::builder::relax::RelaxParams::default();
        if let Some(s) = steps { p.steps = s; }
        if let Some(t) = threshold { p.threshold = t; }
        p.verbose = verbose;
        p.num_threads = num_threads;
        p.cutoff = cutoff;
        p.history_size = history_size;

        if let Some(sys) = self.inner.system.as_mut() {
            crate::core::builder::relax::minimize(sys, p);
            sys.unwrap();
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("System not built yet"));
        }
        Ok(())
    }
}

impl PyBuilder {
    fn ensure_recipe(&mut self) -> &mut Recipe {
        if self.inner.recipe.is_none() {
            self.inner.recipe = Some(Recipe {
                system: crate::core::builder::config::SystemConfig {
                    density: 0.0,
                    cell_shape: None,
                    temperature: None,
                },
                components: Vec::new(),
            });
        }
        self.inner.recipe.as_mut().unwrap()
    }
}