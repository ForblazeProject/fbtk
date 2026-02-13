import numpy as np
import time
from ase.io import read as ase_read
import fbtk
import MDAnalysis as mda
from MDAnalysis.analysis import msd

def benchmark_msd():
    input_file = 'test_data/traj_ps.lammpstrj'
    type_id = 1
    max_lag = 5 
    dt = 1.0 # ps
    
    print(f"=== Benchmarking Windowed MSD calculation ===")
    print(f"Input: {input_file}")
    
    # --- FBTK (Rust) ---
    print("\n--- FBTK (Rust) ---")
    start_load = time.time()
    traj_ase = ase_read(input_file, format='lammps-dump-text', index=':')
    positions = np.array([a.get_positions() for a in traj_ase])
    cells = np.array([a.get_cell() for a in traj_ase])
    
    try:
        atom_types = traj_ase[0].get_array('type')
    except KeyError:
        atom_types = np.ones(len(traj_ase[0])) 
    
    indices = np.where(atom_types == type_id)[0].astype(np.uint64)
    end_load = time.time()
    
    print(f"Data loading (ASE): {end_load - start_load:.4f}s")
    
    start_calc = time.time()
    res_time, res_total, res_x, res_y, res_z = fbtk.compute_msd(
        positions, cells, indices, max_lag, dt
    )
    end_calc = time.time()
    fbtk_time = end_calc - start_calc
    print(f"Calculation (Rust): {fbtk_time:.4f}s")
    
    # --- MDAnalysis ---
    print("\n--- MDAnalysis ---")
    u = mda.Universe(input_file, format='LAMMPSDUMP', dt=dt)
    sel = u.select_atoms(f"type {type_id}")
    
    # MDAnalysis needs unwrapping to be correct.
    # We apply the transformation to unwrap the trajectory
    # from MDAnalysis import transformations
    # workflow = [transformations.unwrap(u.atoms)]
    # u.trajectory.add_transformations(*workflow)
    
    msd_analyzer = msd.EinsteinMSD(sel, msd_type='xyz', fft=True)
    msd_analyzer.run()
    
    mda_total_slice = msd_analyzer.results.timeseries[:max_lag+1]
    
    # --- Comparison ---
    print("\n=== Comparison Results ===")
    rmse = np.sqrt(np.mean((res_total - mda_total_slice)**2))
    print(f"RMSE (Total MSD): {rmse:.6f}")
    
    print("\n   Lag | FBTK Total | MDA Total  | Diff")
    print("-------|------------|------------|-------")
    for i in range(len(res_total)):
        diff = res_total[i] - mda_total_slice[i]
        print(f" {i:5d} | {res_total[i]:10.4f} | {mda_total_slice[i]:10.4f} | {diff:+.4f}")

if __name__ == "__main__":
    benchmark_msd()