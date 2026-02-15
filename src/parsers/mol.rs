use anyhow::{Result, anyhow};
use crate::core::builder::types::{Atom, Bond, MoleculeTemplate};

pub fn parse_mol(content: &str) -> Result<MoleculeTemplate> {
    // Treat the entire content as a single MOL block. 
    // If it contains SDF separators ($$$$), we only take the first part.
    let mol_block = content.split("$$$$").next().unwrap_or("");
    let mut lines = mol_block.lines();
    
    // Header (Title, Info, Comment)
    let title = lines.next().unwrap_or("").trim().to_string();
    let _info = lines.next().unwrap_or("");
    let _comment = lines.next().unwrap_or("");
    
    let counts_line = lines.next().ok_or_else(|| anyhow!("Unexpected EOF in MOL counts line"))?;
    if counts_line.contains("V3000") {
        return Err(anyhow!("V3000 is not supported yet, please use V2000"));
    }
    if counts_line.len() < 6 {
        return Err(anyhow!("Invalid MOL counts line: '{}'", counts_line));
    }
    
    let num_atoms = counts_line[0..3].trim().parse::<usize>().unwrap_or(0);
    let num_bonds = counts_line[3..6].trim().parse::<usize>().unwrap_or(0);
    
    let mut atoms = Vec::with_capacity(num_atoms);
    for i in 0..num_atoms {
        let line = lines.next().ok_or_else(|| anyhow!("Unexpected EOF in atoms block"))?;
        if line.len() < 34 { continue; }
        let x = line[0..10].trim().parse::<f64>()?;
        let y = line[10..20].trim().parse::<f64>()?;
        let z = line[20..30].trim().parse::<f64>()?;
        let symbol = line[31..34].trim().to_string();
        
        atoms.push(Atom {
            id: i,
            residue_name: title.clone(),
            residue_index: 0,
            element: symbol,
            atom_type: "XX".to_string(), // Default
            position: [x, y, z].into(),
            charge: 0.0,
            chain_index: 0,
        });
    }
    
    let mut bonds = Vec::with_capacity(num_bonds);
    for _ in 0..num_bonds {
        let line = lines.next().ok_or_else(|| anyhow!("Unexpected EOF in bonds block"))?;
        if line.len() < 6 { continue; }
        let id1 = line[0..3].trim().parse::<usize>().unwrap_or(0);
        let id2 = line[3..6].trim().parse::<usize>().unwrap_or(0);
        let order = if line.len() >= 9 {
            line[6..9].trim().parse::<f64>().unwrap_or(1.0)
        } else { 1.0 };

        if id1 > 0 && id2 > 0 {
            bonds.push(Bond {
                atom_i: id1 - 1,
                atom_j: id2 - 1,
                order,
            });
        }
    }
    
    Ok(MoleculeTemplate { atoms, bonds })
}

pub fn write_mol(tmpl: &MoleculeTemplate, title: &str) -> String {
    let mut out = String::new();
    // Header
    out.push_str(title);
    out.push('\n');
    out.push_str("  FBTK-v0.9.1\n\n");
    
    // Counts line
    out.push_str(&format!("{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        tmpl.atoms.len(), tmpl.bonds.len()));
    
    // Atoms block
    for atom in &tmpl.atoms {
        out.push_str(&format!("{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            atom.position[0], atom.position[1], atom.position[2], atom.element));
    }
    
    // Bonds block
    for bond in &tmpl.bonds {
        let order = if (bond.order - 1.5).abs() < 0.1 { 4 } // Aromatic in MDL
                    else { bond.order.round() as usize };
        out.push_str(&format!("{:>3}{:>3}{:>3}  0  0  0  0\n",
            bond.atom_i + 1, bond.atom_j + 1, order));
    }
    
    out.push_str("M  END\n");
    out
}
