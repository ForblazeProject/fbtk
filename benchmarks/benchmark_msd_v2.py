import numpy as np
import time
from ase.io import read as ase_read
import fbtk
import MDAnalysis as mda
from MDAnalysis.analysis import msd

def benchmark_msd_v2():
    input_file = 'test_data/traj.lammpstrj' # 01_atomic
    type_id = 1
    max_lag = 50 
    dt = 1.0
    
    print(f"=== Benchmarking MSD calculation (01_atomic) ===")
    
    # --- FBTK (Rust) ---
    print("\n--- FBTK (Rust) ---")
    traj_ase = ase_read(input_file, format='lammps-dump-text', index=':')
    positions = np.array([a.get_positions() for a in traj_ase])
    cells = np.array([a.get_cell() for a in traj_ase])
    indices = np.where(traj_ase[0].get_array('type') == type_id)[0].astype(np.uint64)
    
    start_calc = time.time()
    res_time, res_total, res_x, res_y, res_z = fbtk.compute_msd(
        positions, cells, indices, max_lag, dt
    )
    end_calc = time.time()
    print(f"Calculation (Rust): {end_calc - start_calc:.4f}s")
    
    # --- MDAnalysis ---
    print("\n--- MDAnalysis ---")
    u = mda.Universe(input_file, format='LAMMPSDUMP', dt=dt)
    sel = u.select_atoms(f"type {type_id}")
    
    # We DON'T unwrap MDA here because it requires bonds.
    # INSTEAD, we check if MDA's result is significantly different.
    msd_analyzer = msd.EinsteinMSD(sel, msd_type='xyz', fft=True)
    msd_analyzer.run()
    
    mda_total_slice = msd_analyzer.results.timeseries[:max_lag+1]
    
    # --- Comparison ---
    print("\n=== Comparison (Lag 0 to 10) ===")
    print("   Lag | FBTK Total | MDA Total  | Diff")
    print("-------|------------|------------|-------")
    for i in range(11):
        diff = res_total[i] - mda_total_slice[i]
        print(f" {i:5d} | {res_total[i]:10.4f} | {mda_total_slice[i]:10.4f} | {diff:+.4f}")

if __name__ == "__main__":
    benchmark_msd_v2()
