use glam::{DVec3, DMat3};
use anyhow::Result;
use super::types::{Atom, Bond};

#[derive(Debug, Clone)]
pub struct System {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub cell: DMat3,
    pub inv_cell: DMat3,
}

impl System {
    pub fn new(cell_mat: [[f64; 3]; 3]) -> Self {
        let cell = DMat3::from_cols_array_2d(&cell_mat);
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
            cell,
            inv_cell: cell.inverse(),
        }
    }

    pub fn add_atom(&mut self, atom: Atom) { self.atoms.push(atom); }
    pub fn add_bond(&mut self, bond: Bond) { self.bonds.push(bond); }

    pub fn get_mic_vector(&self, p1: &DVec3, p2: &DVec3) -> DVec3 {
        let dr_real = *p1 - *p2;
        let s = self.inv_cell * dr_real;
        let f_diff_pbc = DVec3::new(
            s.x - s.x.round(),
            s.y - s.y.round(),
            s.z - s.z.round(),
        );
        self.cell * f_diff_pbc
    }

    pub fn dist_sq_pbc(&self, p1: &DVec3, p2: &DVec3) -> f64 {
        let v = self.get_mic_vector(p1, p2);
        v.length_squared()
    }

    pub fn wrap_position(&self, pos: &mut DVec3) {
        let mut s = self.inv_cell * (*pos);
        s.x = s.x.rem_euclid(1.0);
        s.y = s.y.rem_euclid(1.0);
        s.z = s.z.rem_euclid(1.0);
        *pos = self.cell * s;
    }

    pub fn get_all_distances(&self, mic: bool) -> Vec<f64> {
        use rayon::prelude::*;
        let n = self.atoms.len();
        let mut dists = vec![0.0; n * n];
        dists.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let pi = &self.atoms[i].position;
            for j in 0..n {
                if i == j { row[j] = 0.0; }
                else {
                    if mic { row[j] = self.get_mic_vector(pi, &self.atoms[j].position).length(); }
                    else {
                        row[j] = (*pi - self.atoms[j].position).length();
                    }
                }
            }
        });
        dists
    }

    pub fn get_center_of_mass(&self) -> DVec3 {
        use rayon::prelude::*;
        let (sum_m_pos, total_m) = self.atoms.par_iter().map(|a| {
            let m = crate::core::elements::get_atomic_mass(&a.element);
            (a.position * m, m)
        }).reduce(|| (DVec3::ZERO, 0.0), |a, b| {
            (a.0 + b.0, a.1 + b.1)
        });
        if total_m > 0.0 { sum_m_pos / total_m } else { DVec3::ZERO }
    }

    pub fn get_angles(&self, indices: &[[usize; 3]], mic: bool) -> Vec<f64> {
        use rayon::prelude::*;
        indices.par_iter().map(|&[i, j, k]| {
            let v1 = if mic { self.get_mic_vector(&self.atoms[i].position, &self.atoms[j].position) }
                     else { self.atoms[i].position - self.atoms[j].position };
            let v2 = if mic { self.get_mic_vector(&self.atoms[k].position, &self.atoms[j].position) }
                     else { self.atoms[k].position - self.atoms[j].position };
            v1.angle_between(v2).to_degrees()
        }).collect()
    }

    pub fn get_dihedrals(&self, indices: &[[usize; 4]], mic: bool) -> Vec<f64> {
        use rayon::prelude::*;
        indices.par_iter().map(|&[i, j, k, l]| {
            let b1 = if mic { self.get_mic_vector(&self.atoms[j].position, &self.atoms[i].position) } else { self.atoms[j].position - self.atoms[i].position };
            let b2 = if mic { self.get_mic_vector(&self.atoms[k].position, &self.atoms[j].position) } else { self.atoms[k].position - self.atoms[j].position };
            let b3 = if mic { self.get_mic_vector(&self.atoms[l].position, &self.atoms[k].position) } else { self.atoms[l].position - self.atoms[k].position };
            
            let n1 = b1.cross(b2);
            let n2 = b2.cross(b3);
            let m1 = n1.cross(b2);
            let x = n1.dot(n2);
            let y = m1.dot(n2) / b2.length();
            y.atan2(x).to_degrees()
        }).collect()
    }

    pub fn wrap(&mut self) {
        let n = self.atoms.len();
        for i in 0..n {
            let mut pos = self.atoms[i].position;
            self.wrap_position(&mut pos);
            self.atoms[i].position = pos;
        }
    }

    pub fn unwrap(&mut self) {
        if self.atoms.is_empty() { return; }
        let n = self.atoms.len();
        let mut visited = vec![false; n];
        let mut adj = vec![Vec::new(); n];
        for b in &self.bonds {
            adj[b.atom_i].push(b.atom_j);
            adj[b.atom_j].push(b.atom_i);
        }
        for i in 0..n {
            if visited[i] { continue; }
            let mut stack = vec![i]; visited[i] = true;
            while let Some(u) = stack.pop() {
                let pu = self.atoms[u].position;
                for &v in &adj[u] {
                    if !visited[v] {
                        let dr_mic = self.get_mic_vector(&self.atoms[v].position, &pu);
                        self.atoms[v].position = pu + dr_mic;
                        visited[v] = true; stack.push(v);
                    }
                }
            }
        }
    }

    pub fn get_neighbor_list(&self, cutoff: f64) -> Vec<(usize, usize, f64)> {
        use rayon::prelude::*;
        let n = self.atoms.len();
        (0..n).into_par_iter().map(|i| {
            let mut res = Vec::new();
            for j in i+1..n {
                let d2 = self.dist_sq_pbc(&self.atoms[i].position, &self.atoms[j].position);
                if d2 < cutoff*cutoff { res.push((i, j, d2.sqrt())); }
            }
            res
        }).flatten().collect()
    }

    pub fn get_volume(&self) -> f64 {
        self.cell.determinant().abs()
    }

    pub fn get_total_mass(&self) -> f64 {
        use rayon::prelude::*;
        self.atoms.par_iter().map(|a| {
            crate::core::elements::get_atomic_mass(&a.element)
        }).sum()
    }

    pub fn get_density(&self) -> f64 {
        let vol = self.get_volume();
        if vol < 1e-9 { return 0.0; }
        let mass_amu = self.get_total_mass();
        let avogadro = 0.602214076; // (amu/A^3) / (g/cm^3)
        mass_amu / (vol * avogadro)
    }

    pub fn as_uff_system(&self) -> uff_relax::System {
        let (atoms, bonds) = super::relax::convert_to_uff_data(&self.atoms, &self.bonds);
        let uff_cell = uff_relax::UnitCell::new_triclinic(self.cell);
        uff_relax::System::new(atoms, bonds, uff_cell)
    }

    pub fn update_from_uff_system(&mut self, uff_sys: &uff_relax::System) {
        for (i, atom) in uff_sys.atoms.iter().enumerate() {
            self.atoms[i].position = atom.position;
        }
    }

    pub fn stack(&mut self, other: &System, axis: usize) -> Result<()> {
        if axis > 2 { return Err(anyhow::anyhow!("Invalid axis")); }
        
        let offset_shift = self.cell.col(axis)[axis];
        let atom_offset = self.atoms.len();

        for other_atom in &other.atoms {
            let mut new_atom = other_atom.clone();
            new_atom.id += atom_offset;
            let mut pos = new_atom.position;
            match axis {
                0 => pos.x += offset_shift,
                1 => pos.y += offset_shift,
                2 => pos.z += offset_shift,
                _ => {}
            }
            new_atom.position = pos;
            self.atoms.push(new_atom);
        }

        for other_bond in &other.bonds {
            let mut new_bond = other_bond.clone();
            new_bond.atom_i += atom_offset;
            new_bond.atom_j += atom_offset;
            self.bonds.push(new_bond);
        }

        let mut cols = [self.cell.col(0), self.cell.col(1), self.cell.col(2)];
        cols[axis][axis] += other.cell.col(axis)[axis];
        self.cell = DMat3::from_cols(cols[0], cols[1], cols[2]);
        self.inv_cell = self.cell.inverse();

        Ok(())
    }

    pub fn assign_partial_charges(&mut self) {
        use gasteiger_rs::GasteigerSolver;
        let solver = GasteigerSolver::default();
        let target_total: f64 = self.atoms.iter().map(|a| a.formal_charge as f64).sum();
        let charges = solver.compute_charges(&self.atoms, &self.bonds);
        
        let current_total: f64 = charges.iter().sum::<f64>();
        let diff = target_total - current_total;
        let n_atoms = self.atoms.len();
        let correction = if n_atoms > 0 { diff / n_atoms as f64 } else { 0.0 };

        for (i, q) in charges.iter().enumerate() {
            self.atoms[i].charge = (*q as f64) + correction;
        }
    }
}
