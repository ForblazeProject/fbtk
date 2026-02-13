import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import time

def final_high_density_benchmark():
    print("--- Final Benchmark: High-Density Polystyrene (30k atoms, Density 1.0) ---")
    
    # 1. Prepare Monomer (Styrene)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(c1ccccc1)"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    head_idx, tail_idx = 1, 0
    head_leave, tail_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(head_idx).GetNeighbors() if n.GetSymbol() == "H"), next(n.GetIdx() for n in mol.GetAtomWithIdx(tail_idx).GetNeighbors() if n.GetSymbol() == "H")

    builder = fbtk.Builder()
    builder.add_rdkit_mol("styrene", mol)
    
    # 2. Setup Recipe for ~30,000 atoms
    dp = 100
    n_chains = 20 # 20 chains * 100 units * ~15 atoms/unit approx 30,000 atoms
    
    # Box size for density 1.0 (approx calculation)
    # Mass of 2000 styrene units approx 208,000 g/mol
    # Volume at density 1.0 approx 345,000 A^3 -> L approx 70A
    L = 70.0 

    recipe_path = "test_data/final_benchmark.yaml"
    with open(recipe_path, "w") as f:
        f.write(f"""
system:
  density: 0.8
  cell_shape: [{L}, {L}, {L}]
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
"""    )

    builder.load_recipe(recipe_path)
    
    # 3. Build
    print("Building system...")
    system = builder.build()
    print(f"Build complete: {system.n_atoms} atoms in {L}x{L}x{L} box.")

    # 4. Relax with Fmax output (Rust will print this)
    print(f"Starting relaxation (Using Defaults)...")
    start = time.time()
    builder.relax() # Use defaults: steps=500, tolerance=50.0, delta=0.01
    end = time.time()
    
    print(f"\nTotal Relaxation Time: {end - start:.2f} seconds")
    
    import os
    os.remove(recipe_path)

if __name__ == "__main__":
    final_high_density_benchmark()
