import numpy as np
from ase.io import read
import fbtk
import time

def test_rdf():
    print("Loading trajectory...")
    traj = read('test_data/traj.lammpstrj', index=':')
    n_frames = len(traj)
    n_atoms = len(traj[0])
    
    # Prepare data for Rust
    positions = np.zeros((n_frames, n_atoms, 3))
    cells = np.zeros((n_frames, 3, 3))
    
    for i, atoms in enumerate(traj):
        positions[i] = atoms.get_positions()
        cells[i] = atoms.get_cell()
    
    # Indices for RDF (H-H in this case as all atoms are H)
    indices_i = np.arange(n_atoms, dtype=np.uint64)
    indices_j = np.arange(n_atoms, dtype=np.uint64)
    
    r_max = 6.0
    n_bins = 100
    
    print(f"Computing RDF for {n_frames} frames, {n_atoms} atoms...")
    start = time.time()
    r, g_r = fbtk.compute_rdf(
        positions, 
        cells, 
        indices_i, 
        indices_j, 
        r_max, 
        n_bins
    )
    end = time.time()
    
    print(f"Calculation took {end - start:.4f} seconds.")
    
    # Simple ASCII plot-like output
    print("\nRDF (g(r)) Preview:")
    print(" r (A) | g(r)")
    print("-------|-------")
    # Show some points where g(r) is expected to be non-zero
    for i in range(0, n_bins, 10):
        print(f" {r[i]:5.2f} | {g_r[i]:7.3f}")

    # Basic validation: g(r) should be ~1.0 at large r for a fluid/gas
    print(f"\ng(r) at r_max: {g_r[-1]:.3f}")

if __name__ == "__main__":
    test_rdf()
