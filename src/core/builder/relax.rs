use super::system::System;
use uff_relax::{UffOptimizer};

pub struct RelaxParams {
    pub steps: usize,
    pub threshold: f64,
    pub verbose: bool,
    pub num_threads: usize,
    pub cutoff: f64,
    pub history_size: usize,
}

impl Default for RelaxParams {
    fn default() -> Self {
        Self {
            steps: 1000,
            threshold: 1.0,
            verbose: false,
            num_threads: 0,
            cutoff: 6.0,
            history_size: 10,
        }
    }
}

pub fn minimize(system: &mut System, params: RelaxParams) {
    let n_atoms = system.atoms.len();
    if n_atoms == 0 { return; }

    // Convert to UFF system
    let mut uff_sys = system.as_uff_system();

    // Initialize optimizer
    let mut optimizer = UffOptimizer::new(params.steps, params.threshold);
    optimizer.verbose = params.verbose;
    optimizer.num_threads = params.num_threads;
    optimizer.cutoff = params.cutoff;
    optimizer.history_size = params.history_size;
    
    // Perform optimization
    optimizer.optimize(&mut uff_sys);

    // Update original system with optimized coordinates
    system.update_from_uff_system(&uff_sys);
}



pub fn convert_to_uff_data(atoms: &[super::types::Atom], bonds: &[super::types::Bond]) -> (Vec<uff_relax::Atom>, Vec<uff_relax::Bond>) {

    let u_atoms = atoms.iter().map(|a| {

        let z = crate::core::elements::get_atomic_number(&a.element);

        uff_relax::Atom::new(z, a.position)

    }).collect();



    let u_bonds = bonds.iter().map(|b| {

        uff_relax::Bond {

            atom_indices: (b.atom_i, b.atom_j),

            order: b.order as f32,

        }

    }).collect();



    (u_atoms, u_bonds)

}
