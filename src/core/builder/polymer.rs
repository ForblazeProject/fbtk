use super::config::{PolymerParams, Tacticity};
use super::types::{Bond, MoleculeTemplate};
use anyhow::Result;
use glam::DVec3;
use rand::Rng;

pub fn generate_chain(monomer: &MoleculeTemplate, params: &PolymerParams) -> Result<MoleculeTemplate> {
    let mut chain_atoms = Vec::new(); 
    let mut chain_bonds = Vec::new();
    let n_mon = monomer.atoms.len();
    let head_idx = params.head_index.unwrap_or(0);
    let tail_idx = params.tail_index.unwrap_or(n_mon-1);
    let p_head = monomer.atoms[head_idx].position;
    let p_tail = monomer.atoms[tail_idx].position;
    let mut shift = p_tail - p_head;
    let mag = shift.length();
    let buffer = 1.54;
    if mag < 0.1 { shift.x = buffer; } else { shift *= 1.0 + (buffer/mag); }

    // Tacticity Handling
    let tacticity = params.tacticity.as_ref().unwrap_or(&Tacticity::Isotactic);
    let use_complex_logic = *tacticity != Tacticity::Isotactic;

    let local_frame = if use_complex_logic {
        // 1. Identify Backbone (Shortest Path from Head to Tail)
        let mut adj = vec![Vec::new(); n_mon];
        for b in &monomer.bonds {
            adj[b.atom_i].push(b.atom_j);
            adj[b.atom_j].push(b.atom_i);
        }
        let mut parent = vec![None; n_mon];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(head_idx);
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            if u == tail_idx { found = true; break; }
            for &v in &adj[u] {
                if v != head_idx && parent[v].is_none() {
                    parent[v] = Some(u); queue.push_back(v);
                }
            }
        }
        let mut bb_set = std::collections::HashSet::new();
        if found {
            let mut curr = tail_idx;
            bb_set.insert(curr);
            while let Some(p) = parent[curr] { curr = p; bb_set.insert(curr); }
        } else { bb_set.insert(head_idx); bb_set.insert(tail_idx); }

        // 2. Calculate Side Chain Center of Mass
        let (side_sum, side_count) = monomer.atoms.iter().enumerate()
            .filter(|(i, _)| !bb_set.contains(i))
            .fold((DVec3::ZERO, 0), |(sum, count), (_, a)| (sum + a.position, count + 1));
        
        let side_com = if side_count > 0 { side_sum / side_count as f64 } else { p_head + DVec3::new(1.0, 0.0, 0.0) };

        // 3. Define Local Frame (Z: Head->Tail, X: Toward Side COM)
        let ez = (p_tail - p_head).normalize();
        let v_side = side_com - p_head;
        let ex = (v_side - ez * v_side.dot(ez)).normalize();
        let ey = ez.cross(ex);
        Some(glam::DMat3::from_cols(ex, ey, ez))
    } else { None };

    let mut atom_cnt = 0;
    let mut prev_tail: Option<usize> = None;
    let mut rng = rand::thread_rng();

    for m in 0..params.degree {
        let mut l2g = vec![None; n_mon];
        let origin = shift * m as f64;

        // Determine if we should flip this unit (Mirror)
        let flip = match tacticity {
            Tacticity::Isotactic => false,
            Tacticity::Syndiotactic => m % 2 == 1,
            Tacticity::Atactic => rng.gen_bool(0.5),
        };

        for i in 0..n_mon {
            let is_head_leaving = Some(i) == params.head_leaving_index;
            let is_tail_leaving = Some(i) == params.tail_leaving_index;

            // 1. Remove if it's a junction point
            if is_head_leaving && m > 0 { continue; }
            if is_tail_leaving && m < params.degree - 1 { continue; }

            // 2. Handle R markers at ends (convert to H)
            let mut atom = monomer.atoms[i].clone();
            if atom.element == "R" {
                if (is_head_leaving && m == 0) || (is_tail_leaving && m == params.degree - 1) {
                    atom.element = "H".to_string();
                    atom.atom_type = "H".to_string();
                } else {
                    continue;
                }
            }

            atom.id = atom_cnt; 
            atom.residue_index = m + 1;
            
            // Transform Position
            if let Some(frame) = local_frame {
                let mut local_p = frame.transpose() * (atom.position - p_head);
                if flip { local_p.x *= -1.0; }
                atom.position = (frame * local_p) + origin + p_head;
            } else {
                atom.position += origin;
            }

            chain_atoms.push(atom); 
            l2g[i] = Some(atom_cnt); 
            atom_cnt += 1;
        }
        for b in &monomer.bonds {
            if let (Some(gi), Some(gj)) = (l2g[b.atom_i], l2g[b.atom_j]) {
                chain_bonds.push(Bond { atom_i: gi, atom_j: gj, order: b.order });
            }
        }
        if let Some(pt) = prev_tail {
            if let Some(ch) = l2g[params.head_index.unwrap_or( head_idx )] {
                chain_bonds.push(Bond { atom_i: pt, atom_j: ch, order: 1.0 });
            }
        }
        prev_tail = l2g[params.tail_index.unwrap_or( tail_idx )];
    }

    let mut tmpl = MoleculeTemplate { atoms: chain_atoms, bonds: chain_bonds };
    tmpl.assign_partial_charges();

    // Step 2: Relax the final template (adjusting polymer junctions)
    let uff = uff_relax::UffOptimizer::new(500, 0.1);
    let mut uff_sys = tmpl.as_uff_system();
    uff.optimize(&mut uff_sys);
    tmpl.update_from_uff_atoms(&uff_sys.atoms);

    Ok(tmpl)
}
