use numpy::ndarray::{ArrayView2, ArrayView3};
use rayon::prelude::*;

pub struct RdfParams {
    pub r_max: f64,
    pub n_bins: usize,
}

pub struct RdfResult {
    pub r_axis: Vec<f64>,
    pub g_r: Vec<f64>,
}

/// Main RDF computation function.
/// Selects the best strategy (Cell List or Brute Force) per frame.
pub fn compute_rdf_core(
    positions: ArrayView3<f64>,
    cells: ArrayView3<f64>,
    indices_i: &[usize],
    indices_j: &[usize],
    params: RdfParams,
) -> RdfResult {
    let n_frames = positions.shape()[0];
    let dr = params.r_max / params.n_bins as f64;
    let r_max_sq = params.r_max * params.r_max;

    // Pre-check if indices_i and indices_j are physically the same array (pointer check not easy here, so value check)
    // Actually, we can just pass a flag if they are the same to optimize later.
    // For now, let's keep it simple.

    let global_histogram: Vec<u64> = (0..n_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let mut local_hist = vec![0u64; params.n_bins];
            let pos_frame = positions.index_axis(numpy::ndarray::Axis(0), frame_idx);
            let cell = cells.index_axis(numpy::ndarray::Axis(0), frame_idx);

            let cell_mat = glam::DMat3::from_cols_array_2d(&[
                [cell[[0, 0]], cell[[0, 1]], cell[[0, 2]]],
                [cell[[1, 0]], cell[[1, 1]], cell[[1, 2]]],
                [cell[[2, 0]], cell[[2, 1]], cell[[2, 2]]],
            ]);

            // Check if cell is orthogonal (off-diagonals are close to 0)
            let is_orthogonal = cell[[0, 1]].abs() < 1e-6 && cell[[0, 2]].abs() < 1e-6 &&
                                cell[[1, 0]].abs() < 1e-6 && cell[[1, 2]].abs() < 1e-6 &&
                                cell[[2, 0]].abs() < 1e-6 && cell[[2, 1]].abs() < 1e-6;

            let min_l = cell[[0, 0]].min(cell[[1, 1]]).min(cell[[2, 2]]);
            
            if is_orthogonal && min_l >= params.r_max {
                compute_frame_cell_list(
                    &mut local_hist, 
                    &pos_frame, 
                    &cell_mat, 
                    indices_i, 
                    indices_j, 
                    params.r_max, 
                    dr, 
                    r_max_sq
                );
            } else {
                compute_frame_brute_force(
                    &mut local_hist, 
                    &pos_frame, 
                    &cell_mat, 
                    indices_i, 
                    indices_j, 
                    params.r_max, 
                    dr
                );
            }
            local_hist
        })
        .reduce(
            || vec![0u64; params.n_bins],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        );

    // Normalization (Standard RDF normalization)
    let mut g_r = Vec::with_capacity(params.n_bins);
    let mut r_axis = Vec::with_capacity(params.n_bins);
    
    let mut total_volume = 0.0;
    for i in 0..n_frames {
        let c = cells.index_axis(numpy::ndarray::Axis(0), i);
        // volume = |det(h)|
        let vol = c[[0,0]] * (c[[1,1]]*c[[2,2]] - c[[1,2]]*c[[2,1]])
                - c[[0,1]] * (c[[1,0]]*c[[2,2]] - c[[1,2]]*c[[2,0]])
                + c[[0,2]] * (c[[1,0]]*c[[2,1]] - c[[1,1]]*c[[2,0]]);
        total_volume += vol.abs();
    }
    
    let avg_volume = total_volume / n_frames as f64;
    let rho_j = indices_j.len() as f64 / avg_volume;
    let n_i = indices_i.len() as f64;
    let n_frames_f64 = n_frames as f64;

    for bin in 0..params.n_bins {
        let r_inner = bin as f64 * dr;
        let r_outer = (bin + 1) as f64 * dr;
        let shell_vol = 4.0 / 3.0 * std::f64::consts::PI * (r_outer.powi(3) - r_inner.powi(3));
        let count = global_histogram[bin] as f64;
        let expected = rho_j * shell_vol * n_i * n_frames_f64;
        let value = if expected > 0.0 { count / expected } else { 0.0 };
        g_r.push(value);
        r_axis.push(r_inner + dr * 0.5);
    }

    RdfResult { r_axis, g_r }
}

/// Brute-force calculation (O(N^2)) with robust Triclinic MIC.
fn compute_frame_brute_force(
    hist: &mut Vec<u64>,
    pos: &ArrayView2<f64>,
    cell: &glam::DMat3,
    indices_i: &[usize],
    indices_j: &[usize],
    r_max: f64,
    dr: f64,
) {
    let inv_cell = cell.inverse();
    let n_bins = hist.len();

    for &i in indices_i {
        let p_i = glam::DVec3::new(pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]);
        
        for &j in indices_j {
            if i == j { continue; }
            let p_j = glam::DVec3::new(pos[[j, 0]], pos[[j, 1]], pos[[j, 2]]);
            let dr_vec = p_j - p_i;
            
            // MIC using glam
            let f_diff = inv_cell * dr_vec;
            let f_diff_pbc = glam::DVec3::new(
                f_diff.x - f_diff.x.round(),
                f_diff.y - f_diff.y.round(),
                f_diff.z - f_diff.z.round(),
            );
            let r_mic = *cell * f_diff_pbc;
            let d = r_mic.length();
            
            if d < r_max {
                let bin = (d / dr) as usize;
                if bin < n_bins { hist[bin] += 1; }
            }
        }
    }
}

/// Cell List calculation (O(N)) for Orthogonal Box.
fn compute_frame_cell_list(
    hist: &mut Vec<u64>,
    pos: &ArrayView2<f64>,
    cell: &glam::DMat3,
    indices_i: &[usize],
    indices_j: &[usize],
    r_max: f64,
    dr: f64,
    r_max_sq: f64,
) {
    let lx = cell.col(0).x;
    let ly = cell.col(1).y;
    let lz = cell.col(2).z;

    let nx = (lx / r_max).floor() as usize;
    let ny = (ly / r_max).floor() as usize;
    let nz = (lz / r_max).floor() as usize;
    
    if nx == 0 || ny == 0 || nz == 0 {
        return compute_frame_brute_force(hist, pos, cell, indices_i, indices_j, r_max, dr);
    }

    let cx = lx / nx as f64;
    let cy = ly / ny as f64;
    let cz = lz / nz as f64;

    let mut cells_vec: Vec<Vec<usize>> = vec![Vec::new(); nx * ny * nz];
    for &atom_idx in indices_j {
        let mut x = pos[[atom_idx, 0]] % lx; if x < 0.0 { x += lx; }
        let mut y = pos[[atom_idx, 1]] % ly; if y < 0.0 { y += ly; }
        let mut z = pos[[atom_idx, 2]] % lz; if z < 0.0 { z += lz; }

        let ix = ((x / cx).floor() as usize).min(nx - 1);
        let iy = ((y / cy).floor() as usize).min(ny - 1);
        let iz = ((z / cz).floor() as usize).min(nz - 1);
        cells_vec[ix * ny * nz + iy * nz + iz].push(atom_idx);
    }

    let n_bins = hist.len();
    for &i in indices_i {
        let p_i = glam::DVec3::new(pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]);
        let mut x = p_i.x % lx; if x < 0.0 { x += lx; }
        let mut y = p_i.y % ly; if y < 0.0 { y += ly; }
        let mut z = p_i.z % lz; if z < 0.0 { z += lz; }

        let ix_i = ((x / cx).floor() as usize).min(nx - 1);
        let iy_i = ((y / cy).floor() as usize).min(ny - 1);
        let iz_i = ((z / cz).floor() as usize).min(nz - 1);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let ix_n = (ix_i as isize + dx).rem_euclid(nx as isize) as usize;
                    let iy_n = (iy_i as isize + dy).rem_euclid(ny as isize) as usize;
                    let iz_n = (iz_i as isize + dz).rem_euclid(nz as isize) as usize;
                    let flat_idx = ix_n * ny * nz + iy_n * nz + iz_n;
                    
                    for &j in &cells_vec[flat_idx] {
                        if i == j { continue; }
                        let p_j = glam::DVec3::new(pos[[j, 0]], pos[[j, 1]], pos[[j, 2]]);
                        let mut dr_vec = p_i - p_j;
                        dr_vec.x -= lx * (dr_vec.x / lx).round();
                        dr_vec.y -= ly * (dr_vec.y / ly).round();
                        dr_vec.z -= lz * (dr_vec.z / lz).round();

                        let d2 = dr_vec.length_squared();
                        if d2 < r_max_sq {
                            let d = d2.sqrt();
                            let bin = (d / dr) as usize;
                            if bin < n_bins { hist[bin] += 1; }
                        }
                    }
                }
            }
        }
    }
}