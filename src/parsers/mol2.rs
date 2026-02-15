use anyhow::{Result};
use crate::core::builder::types::{Atom, Bond, MoleculeTemplate};
use crate::core::builder::system::System;

#[derive(Debug, Default)]
struct Mol2Data {
    name: String,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    cell: Option<[[f64; 3]; 3]>,
}

pub fn parse_mol2(content: &str) -> Result<MoleculeTemplate> {
    let data = parse_mol2_internal(content)?;
    Ok(MoleculeTemplate {
        atoms: data.atoms,
        bonds: data.bonds,
    })
}

pub fn parse_mol2_as_system(content: &str) -> Result<System> {
    let data = parse_mol2_internal(content)?;
    let cell = data.cell.unwrap_or([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]);
    let mut sys = System::new(cell);
    sys.atoms = data.atoms;
    sys.bonds = data.bonds;
    Ok(sys)
}

fn parse_mol2_internal(content: &str) -> Result<Mol2Data> {
    let mut data = Mol2Data::default();
    let mut state = "";
    
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        
        if line.starts_with("@<TRIPOS>") {
            state = &line[9..];
            continue;
        }
        
        match state {
            "MOLECULE" => {
                if data.name.is_empty() {
                    data.name = line.to_string();
                }
            }
            "ATOM" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 6 {
                    let id = parts[0].parse::<usize>()? - 1;
                    let x = parts[2].parse::<f64>()?;
                    let y = parts[3].parse::<f64>()?;
                    let z = parts[4].parse::<f64>()?;
                    let atom_type = parts[5].to_string();
                    let element = atom_type.split('.').next().unwrap_or("XX").to_string();
                    
                    data.atoms.push(Atom {
                        id,
                        residue_name: data.name.clone(),
                        residue_index: 0,
                        element,
                        atom_type,
                        position: [x, y, z].into(),
                        charge: if parts.len() >= 9 { parts[8].parse().unwrap_or(0.0) } else { 0.0 },
                        chain_index: 0,
                    });
                }
            }
            "BOND" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let id1 = parts[1].parse::<usize>()? - 1;
                    let id2 = parts[2].parse::<usize>()? - 1;
                    let order_str = parts[3];
                    let order = match order_str {
                        "1" => 1.0,
                        "2" => 2.0,
                        "3" => 3.0,
                        "ar" | "am" => 1.5,
                        _ => 1.0,
                    };
                    data.bonds.push(Bond { atom_i: id1, atom_j: id2, order });
                }
            }
            "CRYSIN" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 6 {
                    let a = parts[0].parse::<f64>()?;
                    let b = parts[1].parse::<f64>()?;
                    let c = parts[2].parse::<f64>()?;
                    let alpha = parts[3].parse::<f64>()?.to_radians();
                    let beta = parts[4].parse::<f64>()?.to_radians();
                    let gamma = parts[5].parse::<f64>()?.to_radians();
                    
                    // Convert parameters to matrix
                    let ax = a;
                    let bx = b * gamma.cos();
                    let by = b * gamma.sin();
                    let cx = c * beta.cos();
                    let cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin();
                    let cz = (c * c - cx * cx - cy * cy).sqrt();
                    
                    data.cell = Some([
                        [ax, 0.0, 0.0],
                        [bx, by, 0.0],
                        [cx, cy, cz],
                    ]);
                }
            }
            _ => {}
        }
    }
    Ok(data)
}

pub fn write_mol2(system: &System) -> String {
    write_mol2_full(system, true)
}

pub fn write_mol2_full(system: &System, include_crysin: bool) -> String {
    let mut out = String::new();
    out.push_str("@<TRIPOS>MOLECULE
");
    out.push_str("FBTK_System
");
    out.push_str(&format!("{:>5} {:>5} {:>5} {:>5} {:>5}
", system.atoms.len(), system.bonds.len(), 1, 0, 0));
    out.push_str("SMALL
USER_CHARGES

");
    
    out.push_str("@<TRIPOS>ATOM
");
    for atom in &system.atoms {
        let mut type_name = atom.atom_type.clone();
        if type_name == "XX" { type_name = atom.element.clone(); }
        out.push_str(&format!("{:>7} {:<8} {:>10.4} {:>10.4} {:>10.4} {:<8} {:>5} {:<8} {:>10.4}
",
            atom.id + 1, atom.element, atom.position[0], atom.position[1], atom.position[2],
            type_name, 1, "RES", atom.charge
        ));
    }
    
    out.push_str("@<TRIPOS>BOND
");
    for (i, bond) in system.bonds.iter().enumerate() {
        let order_str = match bond.order {
            o if (o - 1.5).abs() < 0.1 => "ar",
            o if (o - 2.0).abs() < 0.1 => "2",
            o if (o - 3.0).abs() < 0.1 => "3",
            _ => "1",
        };
        out.push_str(&format!("{:>6} {:>6} {:>6} {:>3}
", i + 1, bond.atom_i + 1, bond.atom_j + 1, order_str));
    }
    
    if include_crysin {
        // CRYSIN
        let cell = system.cell;
        let va = glam::DVec3::new(cell.col(0).x, cell.col(0).y, cell.col(0).z);
        let vb = glam::DVec3::new(cell.col(1).x, cell.col(1).y, cell.col(1).z);
        let vc = glam::DVec3::new(cell.col(2).x, cell.col(2).y, cell.col(2).z);
        
        let a = va.length();
        let b = vb.length();
        let c = vc.length();
        let alpha = (vb.dot(vc) / (b * c)).acos().to_degrees();
        let beta = (va.dot(vc) / (a * c)).acos().to_degrees();
        let gamma = (va.dot(vb) / (a * b)).acos().to_degrees();
        
        out.push_str("@<TRIPOS>CRYSIN
");
        out.push_str(&format!("{:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} 1 1
", a, b, c, alpha, beta, gamma));
    }
    
    out
}
