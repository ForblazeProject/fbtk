import fbtk
from ase import Atoms
import numpy as np

def test_smart_analysis():
    print("--- Testing Smart Analysis with Queries ---")
    
    # 1. Create a dummy trajectory (2 frames of a simple system)
    # Frame 1
    atoms1 = Atoms('OHH', 
                  positions=[[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]], 
                  cell=[5, 5, 5], pbc=True)
    # Frame 2 (displaced)
    atoms2 = Atoms('OHH', 
                  positions=[[0.1, 0.1, 0.1], [0.1, 0.1, 1.1], [0.1, 1.1, 0.1]], 
                  cell=[5, 5, 5], pbc=True)
    
    traj = [atoms1, atoms2]
    print(f"Created trajectory with {len(traj)} frames, {len(atoms1)} atoms each.")

    # 2. Test RDF with query "O-H"
    print("\nTesting RDF query='O-H'...")
    r, g_r = fbtk.compute_rdf(traj, query="O-H", r_max=2.0, n_bins=10)
    print(f"RDF result shape: {len(g_r)}")
    print(f"First few g(r) values: {g_r[:3]}")

    # 3. Test RDF with self-query "O"
    print("\nTesting RDF query='O' (Self-RDF)...")
    r, g_r = fbtk.compute_rdf(traj, query="O", r_max=2.0, n_bins=10)
    print(f"Self-RDF result shape: {len(g_r)}")

    # 4. Test MSD with query "index 0:2"
    print("\nTesting MSD query='index 0:2'...")
    res = fbtk.compute_msd(traj, query="index 0:2", dt=0.5)
    print(f"MSD result keys: {res.keys()}")
    print(f"MSD values: {res['msd']}")

    print("\nAll smart analysis tests passed!")

if __name__ == "__main__":
    try:
        test_smart_analysis()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
