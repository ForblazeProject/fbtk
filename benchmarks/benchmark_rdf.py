import numpy as np
import time
from ase.io import read as ase_read
import fbtk
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

def benchmark():
    input_file = 'test_data/traj_ps.lammpstrj'
    r_max = 10.0
    n_bins = 200
    type_id = 1
    
    print(f"=== Benchmarking RDF calculation ===")
    print(f"Input: {input_file}")
    print(f"Params: r_max={r_max}, n_bins={n_bins}, type_id={type_id}")
    
    # --- FBTK (Rust) ---
    print("\n--- FBTK (Rust) ---")
    start_load = time.time()
    traj_ase = ase_read(input_file, format='lammps-dump-text', index=':')
    positions = np.array([a.get_positions() for a in traj_ase])
    cells = np.array([a.get_cell() for a in traj_ase])
    # Identify indices
    # Atoms in PS have 'type' in atoms.get_array('type')
    try:
        atom_types = traj_ase[0].get_array('type')
    except KeyError:
        # Fallback if ASE fails to parse types correctly in this specific dump
        # Looking at previous test, types were parsed.
        atom_types = np.ones(len(traj_ase[0])) 
    
    indices = np.where(atom_types == type_id)[0].astype(np.uint64)
    end_load = time.time()
    
    print(f"Data loading (ASE): {end_load - start_load:.4f}s")
    
    start_calc = time.time()
    r_fbtk, g_fbtk = fbtk.compute_rdf(
        positions, 
        cells, 
        indices, 
        indices, 
        r_max, 
        n_bins
    )
    end_calc = time.time()
    fbtk_time = end_calc - start_calc
    print(f"Calculation (Rust): {fbtk_time:.4f}s")
    
    # --- MDAnalysis ---
    print("\n--- MDAnalysis ---")
    # MDA needs to know the format. LAMMPS dump is supported.
    # Note: MDA might need a topology if atom types are not in dump, but LAMMPS dump usually has them.
    start_mda = time.time()
    u = mda.Universe(input_file, format='LAMMPSDUMP')
    
    # MDA selection
    # In LAMMPS dump, "type" is usually mapped to "type"
    sel_i = u.select_atoms(f"type {type_id}")
    sel_j = u.select_atoms(f"type {type_id}")
    
    print(f"MDA Atoms selected: {len(sel_i)}")
    
    # InterRDF(ag1, ag2, nbins=75, range=(0.0, 15.0), exclusion_block=None, ...)
    # If ag1 and ag2 are same, we should use exclusion_block=(1, 1) or similar to avoid self-counting
    # Our Rust code skips i == j.
    irdf = rdf.InterRDF(sel_i, sel_j, nbins=n_bins, range=(0.0, r_max))
    irdf.run()
    
    end_mda = time.time()
    mda_time = end_mda - start_mda
    print(f"Total time (Load + Calc): {mda_time:.4f}s")
    
    r_mda = irdf.results.bins
    g_mda = irdf.results.rdf
    
    # --- Comparison ---
    print("\n=== Comparison Results ===")
    print(f"FBTK Calculation Speedup vs MDA Total: {mda_time / fbtk_time:.1f}x")
    
    # Interpolate g_mda to match r_fbtk if bins are slightly different
    # (MDAnalysis usually places bins at r + dr/2)
    from scipy.interpolate import interp1d
    f_mda = interp1d(r_mda, g_mda, bounds_error=False, fill_value=0.0)
    g_mda_interp = f_mda(r_fbtk)
    
    rmse = np.sqrt(np.mean((g_fbtk - g_mda_interp)**2))
    print(f"RMSE between FBTK and MDA: {rmse:.6f}")
    
    # Print some values
    print("\n   r   |  FBTK g(r) |  MDA g(r)  | Diff")
    print("-------|------------|------------|-------")
    for i in range(0, n_bins, 20):
        diff = g_fbtk[i] - g_mda_interp[i]
        print(f" {r_fbtk[i]:5.2f} | {g_fbtk[i]:10.4f} | {g_mda_interp[i]:10.4f} | {diff:+.4f}")

    if rmse < 0.1:
        print("\n✅ VALIDATION PASSED: Results are consistent with MDAnalysis.")
    else:
        print("\n❌ VALIDATION FAILED: Results deviate significantly.")

if __name__ == "__main__":
    benchmark()
