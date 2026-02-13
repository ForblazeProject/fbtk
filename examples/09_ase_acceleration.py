import fbtk
import numpy as np
from ase.build import bulk

def run_ase_acceleration():
    print("--- FBTK ASE Acceleration: Triclinic MIC ---")
    
    # 1. Create a problematic Triclinic Supercell in ASE
    # Primitive FCC Copper is highly non-orthogonal
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=False)
    supercell = atoms * (3, 3, 3)
    
    # 2. Convert to FBTK
    system = fbtk.from_ase(supercell)
    print(f"System: {len(supercell)} atoms")
    print(f"Cell Matrix:\n{supercell.cell}\n")
    
    # 3. Fast Distance Matrix (mic=True)
    import time
    t0 = time.time()
    d_fbtk = system.get_all_distances(mic=True)
    print(f"FBTK Distance Calculation (mic=True): {time.time()-t0:.4f}s")
    
    # 4. Fast Neighbor List
    t0 = time.time()
    neighbors = system.get_neighbor_list(cutoff=3.0)
    print(f"FBTK Neighbor Search (O(N) Cell List): {time.time()-t0:.4f}s")
    print(f"  Found {len(neighbors)} pairs.")
    
    # 5. Fast Angles
    # Pick first 10 triplets
    indices = [[i, i+1, i+2] for i in range(10)]
    angles = system.get_angles(indices, mic=True)
    print(f"First 5 angles: {angles[:5]}")

if __name__ == "__main__":
    run_ase_acceleration()
