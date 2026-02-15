use super::config::{Recipe, ComponentRole, PolymerParams};
use anyhow::Result;
use std::collections::HashMap;
use glam::{DVec3};
use rand::seq::SliceRandom;

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

    fn generate_chain(monomer: &MoleculeTemplate, params: &PolymerParams) -> Result<MoleculeTemplate> {
        let mut chain_atoms = Vec::new(); 
        let mut chain_bonds = Vec::new();
        let n_mon = monomer.atoms.len();
        let p_head = monomer.atoms[params.head_index.unwrap_or(0)].position;
        let p_tail = monomer.atoms[params.tail_index.unwrap_or(n_mon-1)].position;
        let mut shift = p_tail - p_head;
        let mag = shift.length();
        let buffer = 1.54;
        if mag < 0.1 { shift.x = buffer; } else { shift *= 1.0 + (buffer/mag); }

        let mut atom_cnt = 0;
        let mut prev_tail: Option<usize> = None;
        for m in 0..params.degree {
            let mut l2g = vec![None; n_mon];
            let origin = shift * m as f64;
            for i in 0..n_mon {
                // If it's a connection point marker (R)
                if monomer.atoms[i].element == "R" {
                    let is_head = Some(i) == params.head_index || Some(i) == params.head_leaving_index;
                    let is_tail = Some(i) == params.tail_index || Some(i) == params.tail_leaving_index;
                    
                    if is_head && m == 0 {
                        // At the very start of the chain: Turn R into H
                        let mut atom = monomer.atoms[i].clone();
                        atom.id = atom_cnt; atom.residue_index = m+1;
                        atom.element = "H".to_string(); atom.atom_type = "H".to_string();
                        atom.position += origin;
                        chain_atoms.push(atom); l2g[i] = Some(atom_cnt); atom_cnt += 1;
                        continue;
                    } else if is_tail && m == params.degree - 1 {
                        // At the very end of the chain: Turn R into H
                        let mut atom = monomer.atoms[i].clone();
                        atom.id = atom_cnt; atom.residue_index = m+1;
                        atom.element = "H".to_string(); atom.atom_type = "H".to_string();
                        atom.position += origin;
                        chain_atoms.push(atom); l2g[i] = Some(atom_cnt); atom_cnt += 1;
                        continue;
                    } else {
                        // It's a junction: Remove the marker
                        continue;
                    }
                }

                // Normal removal of leaving atoms (if they were actual atoms, not R)
                if (Some(i) == params.head_leaving_index && m > 0) || 
                   (Some(i) == params.tail_leaving_index && m < params.degree-1) { 
                    continue; 
                }
                
                let mut atom = monomer.atoms[i].clone();
                atom.id = atom_cnt; atom.residue_index = m+1;
                atom.position += origin;
                chain_atoms.push(atom); l2g[i] = Some(atom_cnt); atom_cnt += 1;
            }
            for b in &monomer.bonds {
                if let (Some(gi), Some(gj)) = (l2g[b.atom_i], l2g[b.atom_j]) {
                    chain_bonds.push(Bond { atom_i: gi, atom_j: gj, order: b.order });
                }
            }
            if let Some(pt) = prev_tail {
                if let Some(ch) = l2g[params.head_index.unwrap_or(0)] {
                    chain_bonds.push(Bond { atom_i: pt, atom_j: ch, order: 1.0 });
                }
            }
            prev_tail = l2g[params.tail_index.unwrap_or(n_mon-1)];
        }
        Ok(MoleculeTemplate { atoms: chain_atoms, bonds: chain_bonds })
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
                    let chain = Self::generate_chain(base, p)?;
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
        let uff = uff_relax::UffOptimizer::new(200, 1.0); 

        for (idx, comp) in recipe.components.iter().enumerate() {
            let base = self.templates.get(&comp.name).ok_or_else(|| anyhow::anyhow!("No template"))?;
            let (mut tmpl, n) = if comp.role == ComponentRole::Polymer {
                let p = comp.polymer_params.as_ref().ok_or_else(|| anyhow::anyhow!("No params"))?;
                (Self::generate_chain(base, p)?, p.n_chains)
            } else { (base.clone(), comp.count.unwrap_or(1)) };
            
            vsepr.optimize(&mut tmpl.atoms, &tmpl.bonds);

            let mut uff_sys = tmpl.as_uff_system();
            uff.optimize(&mut uff_sys);
            tmpl.update_from_uff_atoms(&uff_sys.atoms);

            for _ in 0..n { instances.push((tmpl.clone(), comp.name.clone(), idx)); }
        }

        let n_grid = (instances.len() as f64).powf(1.0/3.0).ceil() as usize;
        let mut points = Vec::new();
        for i in 0..n_grid { for j in 0..n_grid { for k in 0..n_grid {
            points.push(DVec3::new(
                (i as f64+0.5)*(box_size[0]/n_grid as f64),
                (j as f64+0.5)*(box_size[1]/n_grid as f64),
                (k as f64+0.5)*(box_size[2]/n_grid as f64)
            ));
        }}}
        points.shuffle(&mut rand::thread_rng());

        self.system = Some(System::new(cell_mat));
        let system = self.system.as_mut().unwrap();
        let mut atom_id = 0;
        for (idx, (tmpl, res, c_idx)) in instances.into_iter().enumerate() {
            let origin = points[idx];
            let q = crate::core::linalg::random_quaternion();
            let quat = glam::DQuat::from_xyzw(q[1], q[2], q[3], q[0]);
            let start = atom_id;
            for atom in &tmpl.atoms {
                let mut new_a = atom.clone();
                new_a.id = atom_id; new_a.residue_name = res.clone(); new_a.chain_index = c_idx;
                new_a.position = (quat * new_a.position) + origin;
                system.wrap_position(&mut new_a.position);
                system.add_atom(new_a); atom_id += 1;
            }
            for b in &tmpl.bonds { system.add_bond(Bond { atom_i: start+b.atom_i, atom_j: start+b.atom_j, order: b.order }); }
        }
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
