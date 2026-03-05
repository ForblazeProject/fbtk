use super::config::{Recipe, ComponentRole};
use super::polymer;
use super::placement;
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;

pub use super::types::{Atom, Bond, MoleculeTemplate};
pub use super::system::System;

pub struct Builder {
    pub recipe: Option<Recipe>,
    pub system: Option<System>,
    pub templates: HashMap<String, MoleculeTemplate>,
}

impl Builder {
    pub fn new() -> Self { Self { recipe: None, system: None, templates: HashMap::new() } }
    pub fn set_recipe(&mut self, recipe: Recipe) { self.recipe = Some(recipe); }
    pub fn add_template(&mut self, name: String, template: MoleculeTemplate) { self.templates.insert(name, template); }
    
    pub fn load_template_file(&mut self, name: &str, path: &str) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let ext = std::path::Path::new(path).extension().and_then(|s| s.to_str()).unwrap_or("");
        
        let template = match ext {
            "mol2" => crate::parsers::mol2::parse_mol2(&content)?,
            "mol" | "sdf" => crate::parsers::mol::parse_mol(&content)?,
            _ => return Err(anyhow::anyhow!("Unsupported template format: {}. Only .mol2 and .mol are supported.", ext)),
        };
        
        self.add_template(name.to_string(), template);
        Ok(())
    }

    pub fn build(&mut self) -> Result<()> {
        let recipe = self.recipe.as_mut().ok_or_else(|| anyhow::anyhow!("No recipe"))?;
        
        // Calculate box size if not explicitly set
        if recipe.system.cell_shape.is_none() && recipe.system.density > 0.0 {
            let mut total_mass_amu = 0.0;
            for comp in &recipe.components {
                let base = self.templates.get(&comp.name).ok_or_else(|| anyhow::anyhow!("No template for {}", comp.name))?;
                
                if comp.role == ComponentRole::Polymer {
                    let p = comp.polymer_params.as_ref().ok_or_else(|| anyhow::anyhow!("No polymer params for {}", comp.name))?;
                    
                    // Precise mass of the generated chain
                    let chain = polymer::generate_chain(base, p)?;
                    let chain_mass: f64 = chain.atoms.iter().map(|a| crate::core::elements::get_atomic_mass(&a.element)).sum();
                    total_mass_amu += chain_mass * p.n_chains as f64;
                } else {
                    let n_molecules = comp.count.unwrap_or(1) as f64;
                    let mw: f64 = base.atoms.iter()
                        .map(|a| crate::core::elements::get_atomic_mass(&a.element))
                        .sum();
                    total_mass_amu += mw * n_molecules;
                };
            }
            
            let avogadro = 0.602214076; // (amu/A^3) / (g/cm^3)
            let vol_a3 = total_mass_amu / (recipe.system.density * avogadro);
            let l = vol_a3.powf(1.0/3.0);
            recipe.system.cell_shape = Some([l, l, l]);
        }

        let box_size = recipe.system.cell_shape.unwrap_or([20.0, 20.0, 20.0]);
        let cell_mat = [[box_size[0], 0.0, 0.0], [0.0, box_size[1], 0.0], [0.0, 0.0, box_size[2]]];
        
        let mut instances = Vec::new();
        let vsepr = vsepr_rs::VseprOptimizer::new();

        for (idx, comp) in recipe.components.iter().enumerate() {
            let mut base = self.templates.get(&comp.name).ok_or_else(|| anyhow::anyhow!("No template"))?.clone();
            
            // Step 1: Pre-relax the monomer/molecule template
            println!("Optimizing template for {}...", comp.name);
            let _ = std::io::stdout().flush();
            vsepr.optimize(&mut base.atoms, &base.bonds);
            base.assign_partial_charges();
            let mut base_uff = base.as_uff_system();
            uff_relax::UffOptimizer::new(500, 0.1).optimize(&mut base_uff);
            base.update_from_uff_atoms(&base_uff.atoms);

            // Step 2: Generate instances
            let (tmpl, n) = if comp.role == ComponentRole::Polymer {
                let p = comp.polymer_params.as_ref().ok_or_else(|| anyhow::anyhow!("No params"))?;
                // generate_chain now handles its own relaxation
                let chain = polymer::generate_chain(&base, p)?;
                (chain, p.n_chains)
            } else { 
                (base.clone(), comp.count.unwrap_or(1)) 
            };
            
            for _ in 0..n { instances.push((tmpl.clone(), comp.name.clone(), idx)); }
        }

        self.system = Some(System::new(cell_mat));
        let system = self.system.as_mut().unwrap();
        
        placement::place_molecules(system, instances, box_size);

        Ok(())
    }

    pub fn relax(&mut self, steps: Option<usize>, threshold: Option<f64>) -> Result<()> {
        if let Some(sys) = self.system.as_mut() {
            let mut p = super::relax::RelaxParams::default();
            if let Some(s) = steps { p.steps = s; }
            if let Some(t) = threshold { p.threshold = t; }
            super::relax::minimize(sys, p); Ok(())
        } else { Err(anyhow::anyhow!("Not built")) }
    }
}


