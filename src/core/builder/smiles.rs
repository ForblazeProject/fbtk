use purr::graph::Builder;
use purr::feature::{AtomKind, BondKind};
use purr::read::read;
use anyhow::{Result, anyhow};
use super::types::{Atom as MyAtom, Bond as MyBond, MoleculeTemplate};
use rand::Rng;

pub fn parse_smiles(smiles: &str) -> Result<MoleculeTemplate> {
    parse_smiles_with_hydrogens(smiles)
}

pub fn parse_smiles_with_hydrogens(smiles: &str) -> Result<MoleculeTemplate> {
    let mut builder = Builder::new();
    read(smiles, &mut builder, None)
        .map_err(|e| anyhow!("Failed to parse SMILES '{}': {:?}", smiles, e))?;

    let nodes = builder.build()
        .map_err(|e| anyhow!("Failed to build graph: {:?}", e))?;

    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut is_aromatic = Vec::new();
    let mut rng = rand::thread_rng();
    
    // 1. Create Heavy Atoms
    let mut explicit_h_counts = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let mut aromatic = false;
        let mut charge = 0.0;
        let mut h_count = 0;
        let element = match &node.kind {
            AtomKind::Star => "R".to_string(), // Treat * as Dummy "R"
            AtomKind::Aliphatic(e) => e.to_string(),
            AtomKind::Aromatic(e) => {
                aromatic = true;
                let mut s = e.to_string();
                if let Some(first) = s.get_mut(0..1) {
                    let f: &mut str = first;
                    f.make_ascii_uppercase();
                }
                s
            },
            AtomKind::Bracket { symbol, charge: c, hcount: h, .. } => {
                h_count = if let Some(val) = h {
                    let s = format!("{:?}", val);
                    match s.as_str() {
                        "H0" | "Zero" => 0,
                        "H1" | "One" => 1,
                        "H2" | "Two" => 2,
                        "H3" | "Three" => 3,
                        "H4" | "Four" => 4,
                        _ => 0,
                    }
                } else { 0 };
                charge = if let Some(val) = c {
                    let s = format!("{:?}", val);
                    match s.as_str() {
                        "One" => 1.0,
                        "Two" => 2.0,
                        "Three" => 3.0,
                        "Four" => 4.0,
                        "MinusOne" => -1.0,
                        "MinusTwo" => -2.0,
                        "MinusThree" => -3.0,
                        "MinusFour" => -4.0,
                        _ => 0.0,
                    }
                } else { 0.0 };
                let mut s = symbol.to_string();
                if s == "*" {
                    "R".to_string()
                } else {
                    if s.chars().next().unwrap_or(' ').is_lowercase() {
                        aromatic = true;
                        if let Some(first) = s.get_mut(0..1) {
                            let f: &mut str = first;
                            f.make_ascii_uppercase();
                        }
                    }
                    s
                }
            },
        };
        is_aromatic.push(aromatic);
        explicit_h_counts.push(h_count);

        // Simple linear layout with noise to help relaxation
        let noise_y: f64 = rng.gen_range(-0.3..0.3);
        let noise_z: f64 = rng.gen_range(-0.3..0.3);

        atoms.push(MyAtom {
            id: i,
            residue_name: "MOL".to_string(),
            residue_index: 0,
            element: element.clone(),
            atom_type: element.clone(),
            position: [i as f64 * 1.5, noise_y, noise_z].into(),
            charge: 0.0,
            formal_charge: charge as f32,
            chain_index: 0,
        });
    }

    // 2. Add Bonds between Heavy Atoms
    for (i, node) in nodes.iter().enumerate() {
        for bond in &node.bonds {
            if i < bond.tid {
                let order = match bond.kind {
                    BondKind::Single => 1.0,
                    BondKind::Double => 2.0,
                    BondKind::Triple => 3.0,
                    BondKind::Aromatic => 1.5,
                    BondKind::Elided => {
                        if is_aromatic[i] && is_aromatic[bond.tid] {
                            1.5
                        } else {
                            1.0
                        }
                    }
                    _ => 1.0,
                };
                bonds.push(MyBond { atom_i: i, atom_j: bond.tid, order });
            }
        }
    }

    // 3. Add Hydrogens
    let n_heavy = atoms.len();
    let mut atom_counter = n_heavy;

    for i in 0..n_heavy {
        let element = &atoms[i].element;
        if element == "R" { continue; } // Do not add H to dummy atoms
        
        let h_needed = if explicit_h_counts[i] > 0 {
            // Priority 1: Explicitly specified in SMILES (e.g. [NH4+])
            explicit_h_counts[i] as i32
        } else {
            // Priority 2: Inferred for neutral atoms (skip for anions)
            if atoms[i].formal_charge < -0.01 {
                0
            } else {
                let mut current_valence = 0.0;
                for b in &bonds {
                    if b.atom_i == i || b.atom_j == i {
                        current_valence += b.order;
                    }
                }

                let target_valence = match element.to_uppercase().as_str() {
                    "C" => 4.0,
                    "N" | "P" => if atoms[i].formal_charge > 0.5 { 4.0 } else { 3.0 },
                    "O" | "S" => 2.0,
                    "F" | "CL" | "BR" | "I" => 1.0,
                    "H" => 1.0,
                    _ => 0.0,
                };
                (target_valence - current_valence).round() as i32
            }
        };
        
        if h_needed > 0 {
            for k in 0..h_needed {
                let h_idx = atom_counter;
                atom_counter += 1;
                
                // Offset hydrogen slightly with noise
                let p = atoms[i].position;
                let h_noise_x: f64 = rng.gen_range(-0.1..0.1);
                let h_noise_y: f64 = rng.gen_range(-0.1..0.1);
                let h_noise_z: f64 = rng.gen_range(-0.1..0.1);

                let offset = match k % 4 {
                    0 => [0.0, 1.0, 0.0],
                    1 => [0.0, -1.0, 0.0],
                    2 => [0.0, 0.0, 1.0],
                    _ => [0.0, 0.0, -1.0],
                };
                
                atoms.push(MyAtom {
                    id: h_idx,
                    residue_name: "MOL".to_string(),
                    residue_index: 0,
                    element: "H".to_string(),
                    atom_type: "H".to_string(),
                    position: [
                        p.x + offset[0] * 1.1 + h_noise_x, 
                        p.y + offset[1] * 1.1 + h_noise_y, 
                        p.z + offset[2] * 1.1 + h_noise_z
                    ].into(),
                    charge: 0.0,
                    formal_charge: 0.0,
                    chain_index: 0,
                });
                
                bonds.push(MyBond { atom_i: i, atom_j: h_idx, order: 1.0 });
            }
        }
    }

    // 4. VSEPR + UFF Optimization
    let vsepr = vsepr_rs::VseprOptimizer::new();
    vsepr.optimize(&mut atoms, &bonds);

    // Create temporary MoleculeTemplate to use its UFF helper
    let mut tmpl = MoleculeTemplate { atoms, bonds };
    let mut uff_sys = tmpl.as_uff_system();
    let uff = uff_relax::UffOptimizer::new(500, 0.1);
    uff.optimize(&mut uff_sys);
    tmpl.update_from_uff_atoms(&uff_sys.atoms);

    Ok(tmpl)
}

/// Resolves user-provided heavy atom indices (based on SMILES order) to actual atom IDs 
/// (including hydrogens) and identifies appropriate leaving groups (hydrogens).
/// 
/// Returns (head_atom_id, tail_atom_id, head_leaving_id, tail_leaving_id)
pub fn resolve_polymer_indices(
    tmpl: &MoleculeTemplate,
    head_heavy_idx: Option<usize>,
    tail_heavy_idx: Option<usize>,
) -> Result<(usize, usize, Option<usize>, Option<usize>)> {
    // 1. Check for "R" atoms (Explicit connection markers like * in RadonPy)
    let r_atoms: Vec<&MyAtom> = tmpl.atoms.iter().filter(|a| a.element == "R").collect();
    
    if r_atoms.len() >= 2 {
        let r1 = r_atoms[0];
        let r2 = r_atoms[1];
        
        // Find heavy atoms connected to these R atoms
        let find_neighbor = |r_id: usize| -> Option<usize> {
            for b in &tmpl.bonds {
                if b.atom_i == r_id && tmpl.atoms[b.atom_j].element != "H" { return Some(b.atom_j); }
                if b.atom_j == r_id && tmpl.atoms[b.atom_i].element != "H" { return Some(b.atom_i); }
            }
            None
        };

        if let (Some(h_id), Some(t_id)) = (find_neighbor(r1.id), find_neighbor(r2.id)) {
            return Ok((h_id, t_id, Some(r1.id), Some(r2.id)));
        }
    }

    // 2. Fallback to old behavior if no R atoms found
    let heavy_atoms: Vec<&MyAtom> = tmpl.atoms.iter().filter(|a| a.element != "H" && a.element != "R").collect();
    
    if heavy_atoms.is_empty() {
        return Err(anyhow!("No heavy atoms found in template"));
    }

    // Default to first and last heavy atom if not specified
    let h_idx = head_heavy_idx.unwrap_or(0);
    let t_idx = tail_heavy_idx.unwrap_or(heavy_atoms.len() - 1);

    if h_idx >= heavy_atoms.len() || t_idx >= heavy_atoms.len() {
        return Err(anyhow!("Head/Tail index out of bounds. Found {} heavy atoms.", heavy_atoms.len()));
    }

    let head_atom_id = heavy_atoms[h_idx].id;
    let tail_atom_id = heavy_atoms[t_idx].id;

    // Helper to find a connected hydrogen
    let find_connected_h = |atom_id: usize| -> Option<usize> {
        for bond in &tmpl.bonds {
            let other = if bond.atom_i == atom_id { Some(bond.atom_j) }
                       else if bond.atom_j == atom_id { Some(bond.atom_i) }
                       else { None };
            
            if let Some(other_id) = other {
                if tmpl.atoms[other_id].element == "H" {
                    return Some(other_id);
                }
            }
        }
        None
    };

    let head_leaving = find_connected_h(head_atom_id);
    let tail_leaving = find_connected_h(tail_atom_id);

    // If we can't find a leaving group, and it's a polymer, it might fail later.
    // But we let the builder handle the error if indices are missing.
    
    Ok((head_atom_id, tail_atom_id, head_leaving, tail_leaving))
}