use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::core::selection::AtomInfo;

pub struct GromacsTrajectory {
    reader: BufReader<File>,
    atom_info: Vec<AtomInfo>,
}

impl GromacsTrajectory {
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        let mut title = String::new();
        reader.read_line(&mut title)?;
        if title.is_empty() { return Err(anyhow!("Empty GRO file")); }

        let mut n_str = String::new();
        reader.read_line(&mut n_str)?;
        let num_atoms = n_str.trim().parse::<usize>()?;

        let mut atom_info = Vec::with_capacity(num_atoms);
        for i in 0..num_atoms {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line.len() < 20 { return Err(anyhow!("Malformed GRO line at atom {}", i)); }
            
            let resname = line[5..10].trim().to_string();
            let atom_name = line[10..15].trim().to_string();
            let element = guess_element(&atom_name);

            atom_info.push(AtomInfo {
                index: i,
                element,
                resname,
                atom_type: 0,
            });
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        Ok(Self {
            reader,
            atom_info,
        })
    }

    pub fn get_atom_info(&self) -> Vec<AtomInfo> {
        self.atom_info.clone()
    }
}

/// Robustly guess element from atom name by checking against the periodic table.
fn guess_element(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() { return "X".to_string(); }

    // List of all valid chemical elements (alphabetical order for search)
    let elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ];

    let name_up = name.to_uppercase();

    // 1. Try first 2 characters (e.g., "CL", "FE", "Si")
    if name_up.len() >= 2 {
        let first_two = &name_up[0..2];
        for &el in elements.iter() {
            if el.to_uppercase() == first_two {
                // Found a 2-character match
                return el.to_string();
            }
        }
    }

    // 2. Try first 1 character (e.g., "C1", "OW", "H")
    if name_up.len() >= 1 {
        let first_one = &name_up[0..1];
        for &el in elements.iter() {
            if el.to_uppercase() == first_one {
                return el.to_string();
            }
        }
    }

    "X".to_string()
}

impl Iterator for GromacsTrajectory {
    type Item = (Vec<[f64; 3]>, [[f64; 3]; 3]);

    fn next(&mut self) -> Option<Self::Item> {
        let mut title = String::new();
        if self.reader.read_line(&mut title).ok()? == 0 { return None; }

        let mut n_str = String::new();
        self.reader.read_line(&mut n_str).ok()?;
        let num_atoms = n_str.trim().parse::<usize>().ok()?;

        let mut positions = Vec::with_capacity(num_atoms);
        for _ in 0..num_atoms {
            let mut line = String::new();
            self.reader.read_line(&mut line).ok()?;
            if line.len() < 44 { continue; }
            
            let x = line[20..28].trim().parse::<f64>().ok()? * 10.0;
            let y = line[28..36].trim().parse::<f64>().ok()? * 10.0;
            let z = line[36..44].trim().parse::<f64>().ok()? * 10.0;
            positions.push([x, y, z]);
        }

        let mut box_line = String::new();
        self.reader.read_line(&mut box_line).ok()?;
        let box_parts: Vec<&str> = box_line.split_whitespace().collect();
        if box_parts.len() < 3 { return None; }

        let lx = box_parts[0].parse::<f64>().ok()? * 10.0;
        let ly = box_parts[1].parse::<f64>().ok()? * 10.0;
        let lz = box_parts[2].parse::<f64>().ok()? * 10.0;
        
        let cell = [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]];
        Some((positions, cell))
    }
}