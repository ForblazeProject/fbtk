import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import os

def benchmark_large_polymer():
    print("--- Benchmark: Large Polymer Amorphous Cell ---")
    
    # 1. Monomer
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(c1ccccc1)"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # Identify connection points
    # (Assuming indices based on styrene structure)
    head_idx, tail_idx = 1, 0
    head_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(head_idx).GetNeighbors() if n.GetSymbol() == "H")
    tail_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(tail_idx).GetNeighbors() if n.GetSymbol() == "H")

    builder = fbtk.Builder()
    builder.add_rdkit_mol("styrene", mol)
    
    dp = 100
    n_chains = 20
    recipe_path = "test_data/benchmark_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write(f"""
system:
  density: 0.9
  cell_shape: [60.0, 60.0, 60.0]
components:
  - name: "styrene"
    role: "polymer"
    input: {{ smiles: "CC(c1ccccc1)" }}
    polymer_params:
      degree: {dp}
      n_chains: {n_chains}
      head_index: {head_idx}
      tail_index: {tail_idx}
      head_leaving_index: {head_leave}
      tail_leaving_index: {tail_leave}
""")

    builder.load_recipe(recipe_path)
    
    # 2. Build Time
    start_build = time.time()
    system = builder.build()
    end_build = time.time()
    print(f"Build Time ({system.n_atoms} atoms): {end_build - start_build:.4f} seconds")

    # 3. Relax Time (The heavy part)
    steps = 1000
    print(f"Running relaxation ({steps} steps) in parallel...")
    start_relax = time.time()
    builder.relax(steps)
    end_relax = time.time()
    
    print(f"Relaxation Time: {end_relax - start_relax:.4f} seconds")
    print(f"Average time per step: {(end_relax - start_relax)/steps*1000:.2f} ms")

    # Final check
    relaxed_system = builder.get_system()
    # Check if anything broke (minimal check)
    if relaxed_system.n_atoms > 0:
        print("Final structure check passed.")

    os.remove(recipe_path)

if __name__ == "__main__":
    benchmark_large_polymer()
