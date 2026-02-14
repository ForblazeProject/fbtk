use glam::{DMat3, DVec3, DQuat};
use ndarray::{Array1, Array2, ArrayView2};

/// Manual inversion of a 3x3 matrix using glam.
pub fn invert_3x3(m: &ArrayView2<f64>) -> Option<Array2<f64>> {
    let mat = DMat3::from_cols_array_2d(&[
        [m[[0, 0]], m[[0, 1]], m[[0, 2]]],
        [m[[1, 0]], m[[1, 1]], m[[1, 2]]],
        [m[[2, 0]], m[[2, 1]], m[[2, 2]]],
    ]).transpose(); // ndarray is row-major, glam is column-major

    if mat.determinant().abs() < 1e-12 {
        return None;
    }

    let inv = mat.inverse();
    let mut out = Array2::zeros((3, 3));
    let rows = inv.transpose().to_cols_array_2d(); // Convert back to row-major
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = rows[i][j];
        }
    }
    Some(out)
}

/// Matrix-vector multiplication (y = Ax) using glam.
pub fn mat_vec_mul(a: &ArrayView2<f64>, x: &[f64]) -> Array1<f64> {
    let mat = DMat3::from_cols_array_2d(&[
        [a[[0, 0]], a[[0, 1]], a[[0, 2]]],
        [a[[1, 0]], a[[1, 1]], a[[1, 2]]],
        [a[[2, 0]], a[[2, 1]], a[[2, 2]]],
    ]).transpose();
    let vec = DVec3::from_slice(x);
    let res = mat * vec;
    Array1::from_vec(res.to_array().to_vec())
}

/// Rotate a point using a quaternion [w, x, y, z] using glam.
pub fn rotate_point(p: [f64; 3], q: [f64; 4]) -> [f64; 3] {
    let vec = DVec3::from_array(p);
    // glam::DQuat uses [x, y, z, w] order in from_array but [w, x, y, z] was provided? 
    // Let's check my previous rotate_point implementation.
    // Previous: [qw, qx, qy, qz] = q. 
    // DQuat::from_xyzw(qx, qy, qz, qw)
    let quat = DQuat::from_xyzw(q[1], q[2], q[3], q[0]);
    (quat * vec).to_array()
}

/// Generate a uniform random unit quaternion using glam.
pub fn random_quaternion() -> [f64; 4] {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let u1: f64 = rng.r#gen();
    let u2: f64 = rng.r#gen();
    let u3: f64 = rng.r#gen();

    let q = DQuat::from_xyzw(
        (1.0 - u1).sqrt() * (2.0 * std::f64::consts::PI * u2).cos(),
        u1.sqrt() * (2.0 * std::f64::consts::PI * u3).sin(),
        u1.sqrt() * (2.0 * std::f64::consts::PI * u3).cos(),
        (1.0 - u1).sqrt() * (2.0 * std::f64::consts::PI * u2).sin(),
    );
    // Return as [w, x, y, z] to match previous convention
    [q.w, q.x, q.y, q.z]
}