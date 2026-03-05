use super::system::System;
use super::types::{Bond, MoleculeTemplate};
use glam::DVec3;
use rand::seq::SliceRandom;
use std::io::Write;

pub fn place_molecules(
    system: &mut System,
    instances: Vec<(MoleculeTemplate, String, usize)>,
    box_size: [f64; 3],
) {
    let n_grid = (instances.len() as f64).powf(1.0/3.0).ceil() as usize;
    let mut grid_indices = Vec::new();
    for i in 0..n_grid { for j in 0..n_grid { for k in 0..n_grid {
        grid_indices.push((i, j, k));
    }}}
    let mut rng = rand::thread_rng();
    grid_indices.shuffle(&mut rng);

    let dx = box_size[0] / n_grid as f64;
    let dy = box_size[1] / n_grid as f64;
    let dz = box_size[2] / n_grid as f64;

    let mut atom_id = 0;
    let mut last_report = 0;
    let report_interval = 2000;
    let collision_threshold_sq = 2.5 * 2.5;

    println!("Starting molecule placement...");
    let _ = std::io::stdout().flush();
    
    let mut used_grid = vec![false; grid_indices.len()];
    let mut any_overlap = false;

    for (idx, (tmpl, res, c_idx)) in instances.into_iter().enumerate() {
        let mut placed = false;
        let mut grid_search_offset = 0;
        let mut cells_checked = 0;

        // Pre-calculate COM to rotate around it
        let tmpl_com = tmpl.atoms.iter().fold(DVec3::ZERO, |acc, a| acc + a.position) / tmpl.atoms.len() as f64;

        while cells_checked < 15 && (idx + grid_search_offset) < grid_indices.len() {
            let grid_idx = (idx + grid_search_offset) % grid_indices.len();
            grid_search_offset += 1;
            if used_grid[grid_idx] { continue; }
            cells_checked += 1;

            let (gi, gj, gk) = grid_indices[grid_idx];
            let origin = DVec3::new(gi as f64 * dx, gj as f64 * dy, gk as f64 * dz);

            for _rotation_attempt in 0..8 {
                let q = crate::core::linalg::random_quaternion();
                let quat = glam::DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize();

                // Collision Check (Heavy Atoms)
                let mut collision = false;
                for a_new in tmpl.atoms.iter().filter(|a| a.element != "H") {
                    let pos_new = quat * (a_new.position - tmpl_com) + origin;
                    
                    for a_exist in system.atoms.iter().filter(|a| a.element != "H") {
                        if system.dist_sq_pbc(&pos_new, &a_exist.position) < collision_threshold_sq {
                            collision = true;
                            break;
                        }
                    }
                    if collision { break; }
                }

                if !collision {
                    add_to_system(system, &tmpl, &res, c_idx, idx + 1, quat, origin, tmpl_com, &mut atom_id);
                    used_grid[grid_idx] = true;
                    placed = true;
                    break;
                }
            }
            if placed { break; }
        }

        if !placed {
            any_overlap = true;
            println!("Warning: Possible overlap detected for molecule {} ({}).", idx + 1, res);
            let _ = std::io::stdout().flush();
            
            // Fallback to first available grid cell
            if let Some(first_free) = used_grid.iter().position(|&used| !used) {
                let (gi, gj, gk) = grid_indices[first_free];
                let origin = DVec3::new(gi as f64 * dx, gj as f64 * dy, gk as f64 * dz);
                add_to_system(system, &tmpl, &res, c_idx, idx + 1, glam::DQuat::IDENTITY, origin, tmpl_com, &mut atom_id);
                used_grid[first_free] = true;
            }
        }

        if atom_id - last_report >= report_interval {
            println!("... {} atoms generated", atom_id);
            let _ = std::io::stdout().flush();
            last_report = atom_id;
        }
    }

    if any_overlap {
        println!("!!! Warning: Initial placement failed to resolve all overlaps. Consider lower density. !!!");
        let _ = std::io::stdout().flush();
    }
    println!("Finished placement (Total {} atoms).", atom_id);
    let _ = std::io::stdout().flush();
}

fn add_to_system(
    system: &mut System,
    tmpl: &MoleculeTemplate,
    res: &str,
    c_idx: usize,
    res_idx: usize,
    quat: glam::DQuat,
    origin: DVec3,
    tmpl_com: DVec3,
    atom_id: &mut usize,
) {
    let start = *atom_id;
    
    for atom in &tmpl.atoms {
        let mut new_a = atom.clone();
        new_a.id = *atom_id; 
        new_a.residue_name = res.to_string(); 
        new_a.residue_index = res_idx;
        new_a.chain_index = c_idx;
        new_a.position = quat * (new_a.position - tmpl_com) + origin;
        new_a.charge = atom.charge;
        new_a.formal_charge = atom.formal_charge;
        system.wrap_position(&mut new_a.position);
        system.add_atom(new_a); 
        *atom_id += 1;
    }
    for b in &tmpl.bonds {
        system.add_bond(Bond { 
            atom_i: start + b.atom_i, 
            atom_j: start + b.atom_j, 
            order: b.order 
        });
    }
}
