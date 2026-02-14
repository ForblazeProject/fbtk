use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::{Result, anyhow};
use ndarray::{Array3};

pub struct LammpsFrame {
    pub timestep: u64,
    pub num_atoms: usize,
    pub box_bounds: [[f64; 2]; 3], // [xlo, xhi], [ylo, yhi], [zlo, zhi]
    pub tilt: [f64; 3],           // xy, xz, yz
    pub atom_types: Vec<usize>,
    pub positions: Vec<[f64; 3]>,
}

pub struct LammpsTrajectory {
    pub frames: Vec<LammpsFrame>,
}

pub fn parse_dump_file(path: &str) -> Result<LammpsTrajectory> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut frames = Vec::new();

    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 { break; }
        let line = line.trim();

        if line.starts_with("ITEM: TIMESTEP") {
            let mut ts_line = String::new();
            reader.read_line(&mut ts_line)?;
            let timestep = ts_line.trim().parse()?;

            let mut n_line = String::new(); // ITEM: NUMBER OF ATOMS
            reader.read_line(&mut n_line)?;
            let mut n_val_line = String::new();
            reader.read_line(&mut n_val_line)?;
            let num_atoms: usize = n_val_line.trim().parse()?;

            let mut box_item_line = String::new(); // ITEM: BOX BOUNDS
            reader.read_line(&mut box_item_line)?;
            
            let mut x_line = String::new();
            reader.read_line(&mut x_line)?;
            let mut y_line = String::new();
            reader.read_line(&mut y_line)?;
            let mut z_line = String::new();
            reader.read_line(&mut z_line)?;

            let parse_box = |s: &str| -> Result<(f64, f64, f64)> {
                let v: Vec<f64> = s.split_whitespace().map(|x| x.parse().unwrap_or(0.0)).collect();
                if v.len() >= 3 { Ok((v[0], v[1], v[2])) }
                else if v.len() >= 2 { Ok((v[0], v[1], 0.0)) }
                else { Err(anyhow!("Invalid box bounds")) }
            };

            let (xlo, xhi, xy) = parse_box(&x_line)?;
            let (ylo, yhi, xz) = parse_box(&y_line)?;
            let (zlo, zhi, yz) = parse_box(&z_line)?;

            let mut atoms_item_line = String::new(); // ITEM: ATOMS ...
            reader.read_line(&mut atoms_item_line)?;
            let parts: Vec<&str> = atoms_item_line.split_whitespace().collect();
            let type_idx = parts.iter().position(|&s| s == "type").ok_or(anyhow!("No type col"))? - 2;
            let x_idx = parts.iter().position(|&s| s == "x" || s == "xs" || s == "xu" || s == "xsu").ok_or(anyhow!("No x col"))? - 2;
            let y_idx = x_idx + 1;
            let z_idx = x_idx + 2;

            let is_scaled = parts.iter().any(|&s| s == "xs" || s == "xsu");

            let mut atom_types = vec![0; num_atoms];
            let mut positions = vec![[0.0; 3]; num_atoms];

            for i in 0..num_atoms {
                let mut a_line = String::new();
                reader.read_line(&mut a_line)?;
                let a_parts: Vec<&str> = a_line.split_whitespace().collect();
                atom_types[i] = a_parts[type_idx].parse()?;
                let p = [
                    fast_float::parse(a_parts[x_idx])?,
                    fast_float::parse(a_parts[y_idx])?,
                    fast_float::parse(a_parts[z_idx])?,
                ];
                positions[i] = p;
            }

            // Unscale if necessary
            if is_scaled {
                for p in &mut positions {
                    let zs = p[2];
                    let ys = p[1];
                    let xs = p[0];
                    p[2] = zlo + zs * (zhi - zlo);
                    p[1] = ylo + ys * (yhi - ylo) + zs * yz;
                    p[0] = xlo + xs * (xhi - xlo) + ys * xy + zs * xz;
                }
            }

            frames.push(LammpsFrame {
                timestep,
                num_atoms,
                box_bounds: [[xlo, xhi], [ylo, yhi], [zlo, zhi]],
                tilt: [xy, xz, yz],
                atom_types,
                positions,
            });
        }
    }

    Ok(LammpsTrajectory { frames })
}

pub fn traj_to_ndarray(traj: &LammpsTrajectory) -> (Array3<f64>, Array3<f64>) {
    let n_frames = traj.frames.len();
    let n_atoms = traj.frames[0].num_atoms;
    
    let mut pos_array = Array3::zeros((n_frames, n_atoms, 3));
    let mut cell_array = Array3::zeros((n_frames, 3, 3));

    for (f_idx, frame) in traj.frames.iter().enumerate() {
        for a_idx in 0..n_atoms {
            pos_array[[f_idx, a_idx, 0]] = frame.positions[a_idx][0];
            pos_array[[f_idx, a_idx, 1]] = frame.positions[a_idx][1];
            pos_array[[f_idx, a_idx, 2]] = frame.positions[a_idx][2];
        }

        let b = frame.box_bounds;
        let t = frame.tilt;
        // LAMMPS H-matrix:
        // h = [ xhi-xlo, 0, 0 ]
        //     [ xy, yhi-ylo, 0 ]
        //     [ xz, yz, zhi-zlo ]
        // We use the convention where cell vectors are rows
        cell_array[[f_idx, 0, 0]] = b[0][1] - b[0][0];
        cell_array[[f_idx, 1, 0]] = t[0]; // xy
        cell_array[[f_idx, 1, 1]] = b[1][1] - b[1][0];
        cell_array[[f_idx, 2, 0]] = t[1]; // xz
        cell_array[[f_idx, 2, 1]] = t[2]; // yz
        cell_array[[f_idx, 2, 2]] = b[2][1] - b[2][0];
    }

    (pos_array, cell_array)
}

pub fn get_atom_info(frame: &LammpsFrame) -> Vec<crate::core::selection::AtomInfo> {
    frame.atom_types.iter().enumerate().map(|(i, &t)| {
        crate::core::selection::AtomInfo {
            index: i,
            element: format!("{}", t), // Default to type string
            resname: "UNK".to_string(),
            atom_type: t,
        }
    }).collect()
}
