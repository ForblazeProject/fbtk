import numpy as np
from ase.io import read
import fbtk
import time

def test_ps_rdf():
    print("Loading PS trajectory (1st frame)...")
    # Using format='lammps-dump-text' to be explicit
    # Read only the first frame for initial test
    atoms = read('test_data/traj_ps.lammpstrj', format='lammps-dump-text', index=0)
    
    n_atoms = len(atoms)
    print(f"Atoms: {n_atoms}")
    
    # Get atom types from ASE (it stores LAMMPS types in atoms.get_array('type'))
    # If not present, we might need to access it differently
    try:
        atom_types = atoms.get_array('type')
    except KeyError:
        # Fallback: maybe they are mapped to elements? 
        # For now let's just use all atoms if type is unknown
        print("Warning: 'type' array not found. Using all atoms.")
        atom_types = np.ones(n_atoms)

    # Prepare data for Rust (1 frame, N atoms, 3 coords)
    positions = atoms.get_positions().reshape(1, n_atoms, 3)
    cells = np.array(atoms.get_cell()).reshape(1, 3, 3)
    
    # Let's compute RDF for Type 1 vs Type 1 (e.g., C-C)
    indices_i = np.where(atom_types == 1)[0].astype(np.uint64)
    indices_j = np.where(atom_types == 1)[0].astype(np.uint64)
    
    print(f"Indices I (Type 1): {len(indices_i)}")
    print(f"Indices J (Type 1): {len(indices_j)}")
    
    r_max = 10.0
    n_bins = 200
    
    print(f"Computing RDF (Type 1 - Type 1)...")
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
    
    print("\nRDF (g(r)) Preview (Type 1 - Type 1):")
    for i in range(0, n_bins, 20):
        print(f" {r[i]:5.2f} | {g_r[i]:7.3f}")

if __name__ == "__main__":
    test_ps_rdf()
