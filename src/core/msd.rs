use ndarray::{Array2, ArrayView3};
use rayon::prelude::*;
use glam::DVec3;

pub struct MsdResult {
    pub time: Vec<f64>,
    pub msd_total: Vec<f64>,
    pub msd_x: Vec<f64>,
    pub msd_y: Vec<f64>,
    pub msd_z: Vec<f64>,
}

/// Compute Windowed MSD for selected atoms with automatic unwrapping.
pub fn compute_msd_core(
    positions: ArrayView3<f64>,
    cells: ArrayView3<f64>,
    indices: &[usize],
    max_lag: usize,
    dt: f64,
) -> MsdResult {
    let n_frames = positions.shape()[0];
    let n_atoms = indices.len();
    
    // 1. Unwrap coordinates (remove PBC jumps)
    let mut unwrapped = Array2::<f64>::zeros((n_frames, n_atoms * 3));

    // First frame initialization
    let pos0 = positions.index_axis(ndarray::Axis(0), 0);
    for (idx, &atom_idx) in indices.iter().enumerate() {
        unwrapped[[0, idx * 3 + 0]] = pos0[[atom_idx, 0]];
        unwrapped[[0, idx * 3 + 1]] = pos0[[atom_idx, 1]];
        unwrapped[[0, idx * 3 + 2]] = pos0[[atom_idx, 2]];
    }

    // Sequential unwrapping
    for t in 1..n_frames {
        let pos_curr = positions.index_axis(ndarray::Axis(0), t);
        let cell_prev = cells.index_axis(ndarray::Axis(0), t - 1);
        
        let lx = cell_prev[[0, 0]];
        let ly = cell_prev[[1, 1]];
        let lz = cell_prev[[2, 2]];
        
        for (idx, &atom_idx) in indices.iter().enumerate() {
            let curr_x = pos_curr[[atom_idx, 0]];
            let curr_y = pos_curr[[atom_idx, 1]];
            let curr_z = pos_curr[[atom_idx, 2]];

            let prev_x = unwrapped[[t - 1, idx * 3 + 0]];
            let prev_y = unwrapped[[t - 1, idx * 3 + 1]];
            let prev_z = unwrapped[[t - 1, idx * 3 + 2]];

            unwrapped[[t, idx * 3 + 0]] = unwrap(curr_x, prev_x, lx);
            unwrapped[[t, idx * 3 + 1]] = unwrap(curr_y, prev_y, ly);
            unwrapped[[t, idx * 3 + 2]] = unwrap(curr_z, prev_z, lz);
        }
    }

    // 2. Compute MSD (Windowed Average)
    let valid_max_lag = if max_lag == 0 { n_frames - 1 } else { max_lag.min(n_frames - 1) };
    
    let results: Vec<(f64, f64, f64, f64)> = (0..=valid_max_lag)
        .into_par_iter()
        .map(|lag| {
            if lag == 0 {
                return (0.0, 0.0, 0.0, 0.0);
            }

            let mut sum_sq_x = 0.0;
            let mut sum_sq_y = 0.0;
            let mut sum_sq_z = 0.0;
            
            let t0_max = n_frames - lag;
            for t0 in 0..t0_max {
                let t1 = t0 + lag;
                for idx in 0..n_atoms {
                    let base = idx * 3;
                    let p0 = DVec3::new(unwrapped[[t0, base + 0]], unwrapped[[t0, base + 1]], unwrapped[[t0, base + 2]]);
                    let p1 = DVec3::new(unwrapped[[t1, base + 0]], unwrapped[[t1, base + 1]], unwrapped[[t1, base + 2]]);
                    let dr = p1 - p0;

                    sum_sq_x += dr.x * dr.x;
                    sum_sq_y += dr.y * dr.y;
                    sum_sq_z += dr.z * dr.z;
                }
            }

            let count = t0_max * n_atoms;
            if count > 0 {
                let inv = 1.0 / count as f64;
                (lag as f64 * dt, sum_sq_x * inv, sum_sq_y * inv, sum_sq_z * inv)
            } else {
                (lag as f64 * dt, 0.0, 0.0, 0.0)
            }
        })
        .collect();

    let mut time = Vec::with_capacity(results.len());
    let mut msd_x = Vec::with_capacity(results.len());
    let mut msd_y = Vec::with_capacity(results.len());
    let mut msd_z = Vec::with_capacity(results.len());
    let mut msd_total = Vec::with_capacity(results.len());

    for (t, mx, my, mz) in results {
        time.push(t);
        msd_x.push(mx);
        msd_y.push(my);
        msd_z.push(mz);
        msd_total.push(mx + my + mz);
    }

    MsdResult { time, msd_total, msd_x, msd_y, msd_z }
}

#[inline(always)]
fn unwrap(curr: f64, prev: f64, box_len: f64) -> f64 {
    let diff = curr - prev;
    if diff > box_len * 0.5 {
        curr - box_len
    } else if diff < -box_len * 0.5 {
        curr + box_len
    } else {
        curr
    }
}
