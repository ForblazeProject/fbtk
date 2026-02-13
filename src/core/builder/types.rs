use glam::DVec3;
use vsepr_rs::traits::{AtomTrait, BondTrait};

#[derive(Debug, Clone)]
pub struct Atom {
    pub id: usize,
    pub residue_name: String,
    pub residue_index: usize,
    pub element: String,
    pub atom_type: String,
    pub position: DVec3,
    pub charge: f64,
    pub chain_index: usize,
}

impl AtomTrait for Atom {
    fn get_position(&self) -> [f64; 3] { self.position.to_array() }
    fn set_position(&mut self, pos: [f64; 3]) { self.position = DVec3::from_array(pos); }
    fn atomic_number(&self) -> usize {
        crate::core::elements::get_atomic_number(&self.element)
    }
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom_i: usize,
    pub atom_j: usize,
    pub order: f64,
}

impl BondTrait for Bond {
    fn get_atom_indices(&self) -> (usize, usize) {
        (self.atom_i, self.atom_j)
    }
    fn get_bond_order(&self) -> f32 {
        self.order as f32
    }
}

#[derive(Debug, Clone)]
pub struct MoleculeTemplate {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

impl MoleculeTemplate {
    pub fn as_uff_system(&self) -> uff_relax::System {
        let (atoms, bonds) = super::relax::convert_to_uff_data(&self.atoms, &self.bonds);
        let uff_cell = uff_relax::UnitCell::new_none();
        uff_relax::System::new(atoms, bonds, uff_cell)
    }

    pub fn update_from_uff_atoms(&mut self, uff_atoms: &[uff_relax::Atom]) {
        for (i, atom) in uff_atoms.iter().enumerate() {
            self.atoms[i].position = atom.position;
        }
    }
}
