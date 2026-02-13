import fbtk
import time
import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list

# 1. 系を作成 (銅の 10x10x10 supercell = 4,000 原子)
print("Creating system (4,000 atoms)...")
atoms = bulk('Cu', 'fcc', a=3.614) * (10, 10, 10)

# --- get_all_distances の比較 ---
print("\n--- Benchmarking: get_all_distances(mic=True) ---")
start = time.time()
dists_ase = atoms.get_all_distances(mic=True)
t_ase = time.time() - start
print(f"ASE:  {t_ase:.4f} sec")

start = time.time()
system = fbtk.from_ase(atoms)
dists_fbtk = system.get_all_distances(mic=True)
t_fbtk = time.time() - start
print(f"FBTK: {t_fbtk:.4f} sec (Speedup: {t_ase/t_fbtk:.1f}x)")

if not np.allclose(dists_ase, dists_fbtk):
    diff = np.abs(dists_ase - dists_fbtk)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Mismatch! Max diff: {np.max(diff):.6f}")
    print(f"At index {idx}: ASE={dists_ase[idx]:.6f}, FBTK={dists_fbtk[idx]:.6f}")
    # Don't exit yet, check neighbor list
else:
    print("Consistency Check: OK")

# --- get_neighbor_list の比較 ---
print("\n--- Benchmarking: neighbor_list (cutoff=5.0 A) ---")
cutoff = 5.0
start = time.time()
i, j, d = neighbor_list('ijd', atoms, cutoff)
t_ase_nl = time.time() - start
print(f"ASE:  {t_ase_nl:.4f} sec")

start = time.time()
system = fbtk.from_ase(atoms)
nl_fbtk = system.get_neighbor_list(cutoff)
t_fbtk_nl = time.time() - start
print(f"FBTK: {t_fbtk_nl:.4f} sec (Speedup: {t_ase_nl/t_fbtk_nl:.1f}x)")
