import fbtk
import time
import numpy as np
import os

def benchmark_ase_replacements():
    print("=== Benchmark: fbtk vs ASE (Comprehensive) ===")
    
    L = 30.0
    recipe = f"""
system:
  density: 0.8
  cell_shape: [{L}, {L}, {L}]
components:
  - name: "styrene"
    role: "polymer"
    input: {{ smiles: "CC(c1ccccc1)" }}
    polymer_params: {{ degree: 30, n_chains: 10 }}
"""
    with open("bench_ase.yaml", "w") as f: f.write(recipe)
    builder = fbtk.Builder()
    builder.load_recipe("bench_ase.yaml")
    builder.build()
    system = builder.get_system()
    atoms = system.to_ase()
    n = len(atoms)
    print(f"System: {n} atoms\n")

    # 1. Center of Mass
    print("[1. Center of Mass]")
    t0 = time.time()
    com_f = system.get_center_of_mass()
    t_f = time.time() - t0
    t0 = time.time()
    com_a = atoms.get_center_of_mass()
    t_a = time.time() - t0
    print(f"  fbtk: {t_f:.6f}s, ASE: {t_a:.6f}s")
    print(f"  Diff: {np.linalg.norm(np.array(com_f)-com_a):.2e}")

    # 2. Angles & Dihedrals
    print("\n[2. Angles & Dihedrals]")
    bonds = system.get_bonds()
    adj = {}
    for i_atom, j_atom, order in bonds:
        i, j = i_atom - 1, j_atom - 1
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    
    angles_idx = []
    for j in adj:
        neighs = adj[j]
        if len(neighs) >= 2:
            for k in range(len(neighs)):
                for l in range(k+1, len(neighs)):
                    angles_idx.append([neighs[k], j, neighs[l]])
                    if len(angles_idx) >= 500: break
        if len(angles_idx) >= 500: break
    
    dihedrals_idx = []
    for j in adj:
        for k in adj[j]:
            for i in adj[j]:
                if i == k: continue
                for l in adj[k]:
                    if l == j or l == i: continue
                    dihedrals_idx.append([i, j, k, l])
                    if len(dihedrals_idx) >= 500: break
                if len(dihedrals_idx) >= 500: break
            if len(dihedrals_idx) >= 500: break
        if len(dihedrals_idx) >= 500: break

    t0 = time.time()
    ang_f = system.get_angles(angles_idx)
    t_f = time.time() - t0
    
    ang_a_valid = []
    valid_angles_idx = []
    for idx in angles_idx:
        try:
            val = atoms.get_angle(*idx, mic=True)
            ang_a_valid.append(val)
            valid_angles_idx.append(idx)
        except: continue
    t_a = 0 # Measuring loop overhead isn't the point, just accuracy
    
    # Recalculate fbtk only for valid ones for accuracy check
    ang_f_valid = system.get_angles(valid_angles_idx)
    print(f"  Angles ({len(valid_angles_idx)}): fbtk: {t_f:.6f}s")
    print(f"  Max Diff: {np.max(np.abs(np.array(ang_f_valid) - np.array(ang_a_valid))):.2e} deg")
    
    t0 = time.time()
    dih_f = system.get_dihedrals(dihedrals_idx)
    t_f = time.time() - t0
    
    dih_a_valid = []
    valid_dih_idx = []
    for idx in dihedrals_idx:
        try:
            val = atoms.get_dihedral(*idx, mic=True)
            dih_a_valid.append(val)
            valid_dih_idx.append(idx)
        except: continue
    
    dih_f_valid = system.get_dihedrals(valid_dih_idx)
    dih_diff = np.abs(np.array(dih_f_valid) - np.array(dih_a_valid))
    dih_diff = np.minimum(dih_diff, 360.0 - dih_diff)
    print(f"  Dihedrals ({len(valid_dih_idx)}): fbtk: {t_f:.6f}s")
    print(f"  Max Diff: {np.max(dih_diff):.2e} deg")

    # 3. Neighbor List
    print("\n[3. Neighbor List]")
    cutoff = 5.0
    t0 = time.time()
    nl_f = system.get_neighbor_list(cutoff)
    t_f = time.time() - t0
    from ase.neighborlist import neighbor_list
    t0 = time.time()
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    t_a = time.time() - t0
    print(f"  fbtk: {t_f:.6f}s, ASE: {t_a:.6f}s (Speedup: {t_a/t_f:.1f}x)")
    print(f"  Pairs found: fbtk: {len(nl_f)}, ASE: {len(i)//2}")

    os.remove("bench_ase.yaml")

if __name__ == "__main__":
    benchmark_ase_replacements()
