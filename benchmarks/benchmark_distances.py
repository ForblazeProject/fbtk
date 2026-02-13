import fbtk
import time
import numpy as np
import os

def benchmark_distances():
    print("--- Benchmark: fbtk vs ASE (get_all_distances) ---")
    
    # ~5,000 atoms
    # 1 monomer = 16 atoms. DP=30 * 10 chains = 300 * 16 = 4800 atoms
    L = 30.0
    recipe = f"""
system:
  density: 0.8
  cell_shape: [{L}, {L}, {L}]
components:
  - name: "styrene"
    role: "polymer"
    input: {{ smiles: "CC(c1ccccc1)" }}
    polymer_params:
      degree: 30
      n_chains: 10
      head_index: 1
      tail_index: 0
      head_leaving_index: 11
      tail_leaving_index: 8
"""
    recipe_path = "temp_bench.yaml"
    with open(recipe_path, "w") as f: f.write(recipe)
    
    builder = fbtk.Builder()
    builder.load_recipe(recipe_path)
    builder.build()
    
    system = builder.get_system()
    atoms = system.to_ase()
    n = len(atoms)
    print(f"System Size: {n} atoms")

    # 1. fbtk (Rust + Rayon)
    # Using 4 threads (default)
    print("\n[fbtk] Calculating distances...")
    t0 = time.time()
    d_fbtk = system.get_all_distances()
    t_fbtk = time.time() - t0
    print(f"  Time: {t_fbtk:.4f} seconds")

    # 2. ASE (Python/NumPy)
    # Warning: 32k^2 is 1 billion elements, ~8GB of float64. 
    if n > 15000:
        print("\n[ASE] System is very large (~8GB RAM). Proceeding...")
    
    t0 = time.time()
    try:
        d_ase = atoms.get_all_distances(mic=True)
        t_ase = time.time() - t0
        print(f"  Time: {t_ase:.4f} seconds")
        
        # Verify results
        diff = np.max(np.abs(d_fbtk - d_ase))
        print(f"\nVerification: Max difference = {diff:.6e}")
        print(f"Speedup: {t_ase / t_fbtk:.1f}x")
    except MemoryError:
        print("\n[ASE] Failed due to MemoryError")
    except Exception as e:
        print(f"\n[ASE] Error: {e}")

    if os.path.exists(recipe_path):
        os.remove(recipe_path)

if __name__ == "__main__":
    benchmark_distances()