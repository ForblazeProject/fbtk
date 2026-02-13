import fbtk
import numpy as np
from ase.build import bulk
import time

def test_triclinic_comprehensive():
    print("=== Comprehensive Triclinic Verification: fbtk vs ASE ===")
    
    # 1. Setup: 2x2x2 Supercell of primitive FCC Copper (Triclinic)
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=False)
    supercell = atoms * (2, 2, 2)
    system = fbtk.from_ase(supercell)
    n = len(supercell)
    print(f"System: {n} atoms, Cell:\n{supercell.cell}\n")

    # 2. Center of Mass
    print("[1. Center of Mass]")
    com_f = system.get_center_of_mass()
    com_a = supercell.get_center_of_mass()
    diff_com = np.linalg.norm(np.array(com_f) - com_a)
    print(f"  Diff: {diff_com:.2e}")

    # 3. Angles (pick some indices)
    print("\n[2. Angles (mic=True)]")
    # Copper crystal doesn't have bonds, but we can pick indices
    indices_a = [[0, 1, 2], [3, 4, 5]]
    ang_f = system.get_angles(indices_a, mic=True)
    ang_a = [supercell.get_angle(*idx, mic=True) for idx in indices_a]
    diff_ang = np.max(np.abs(np.array(ang_f) - np.array(ang_a)))
    print(f"  Max Diff: {diff_ang:.2e} deg")

    # 4. Dihedrals
    print("\n[3. Dihedrals (mic=True)]")
    indices_d = [[0, 1, 2, 3], [4, 5, 6, 7]]
    dih_f = system.get_dihedrals(indices_d, mic=True)
    dih_a = [supercell.get_dihedral(*idx, mic=True) for idx in indices_d]
    dih_diff = np.abs(np.array(dih_f) - np.array(dih_a))
    dih_diff = np.minimum(dih_diff, 360.0 - dih_diff)
    print(f"  Max Diff: {np.max(dih_diff):.2e} deg")

    # 5. Neighbor List
    print("\n[4. Neighbor List (mic=True)]")
    cutoff = 3.0
    nl_f = system.get_neighbor_list(cutoff)
    
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list('ijd', supercell, cutoff)
    
    print(f"  fbtk pairs: {len(nl_f)}")
    print(f"  ASE pairs:  {len(i)//2}")
    
    # Verify distance values match
    if len(nl_f) > 0:
        f_dists = sorted([p[2] for p in nl_f])
        a_dists = sorted(d)[::2] # i<j only
        if len(f_dists) == len(a_dists):
            dist_diff = np.max(np.abs(np.array(f_dists) - np.array(a_dists)))
            print(f"  Max Distance Diff: {dist_diff:.2e}")
        else:
            print("  Warning: Pair count mismatch!")

if __name__ == "__main__":
    test_triclinic_comprehensive()
