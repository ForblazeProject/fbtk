import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ase import Atoms

def verify_collision_reduction():
    print("--- Verification: Collision Reduction via Relaxation ---")
    
    # 1. Prepare Styrene Monomer
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(c1ccccc1)"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    head_idx, tail_idx = 1, 0
    head_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(head_idx).GetNeighbors() if n.GetSymbol() == "H")
    tail_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(tail_idx).GetNeighbors() if n.GetSymbol() == "H")

    builder = fbtk.Builder()
    builder.add_rdkit_mol("styrene", mol)
    
    # 小規模な系: 重合度20, 2本
    dp = 20
    n_chains = 2
    
    recipe_path = "test_data/verify_relax.yaml"
    with open(recipe_path, "w") as f:
        f.write(f"""
system:
  density: 0.5
  cell_shape: [30.0, 30.0, 30.0]
components:
  - name: \"styrene\"
    role: \"polymer\"
    input: {{ smiles: \"CC(c1ccccc1)\" }}
    polymer_params:
      degree: {dp}
      n_chains: {n_chains}
      head_index: {head_idx}
      tail_index: {tail_idx}
      head_leaving_index: {head_leave}
      tail_leaving_index: {tail_leave}
"""    )

    builder.load_recipe(recipe_path)
    
    # 2. Build Initial Structure
    builder.build()
    atoms_init = builder.get_system().to_ase()
    
    def count_collisions(atoms, threshold=1.2):
        # 1-2, 1-3 などの近接ペアを除外するため、非結合ペアの距離のみを確認するのが理想ですが、
        # ここでは単純に全ペア距離のうち threshold 未満の数をカウントします。
        # 結合長(C-H ~1.1, C-C ~1.5)を考慮し、1.0A未満を深刻な衝突とみなします。
        dists = atoms.get_all_distances(mic=True)
        # 対角成分（自分自身）を除外
        dists = dists[~np.eye(dists.shape[0], dtype=bool)]
        collisions = np.sum(dists < threshold) // 2 # ペアなので2で割る
        min_d = np.min(dists)
        return collisions, min_d

    c_init, min_init = count_collisions(atoms_init, threshold=1.0)
    print(f"Before Relaxation:")
    print(f"  Total Atoms: {len(atoms_init)}")
    print(f"  Collisions (< 1.0A): {c_init}")
    print(f"  Minimum Distance: {min_init:.4f} A")

    # 3. Perform Relaxation
    print("\nRunning relaxation...")
    builder.relax(500, 50.0, 0.01) # Position arguments
    
    # 4. Post-Relaxation Check
    atoms_relaxed = builder.get_system().to_ase()
    c_relax, min_relax = count_collisions(atoms_relaxed, threshold=1.0)
    
    print(f"\nAfter Relaxation:")
    print(f"  Collisions (< 1.0A): {c_relax}")
    print(f"  Minimum Distance: {min_relax:.4f} A")

    if c_relax < c_init or min_relax > min_init:
        print("\nResult: SUCCESS. Collisions reduced and minimum distance increased.")
    else:
        print("\nResult: FAILED. No improvement observed.")

    import os
    os.remove(recipe_path)

if __name__ == "__main__":
    verify_collision_reduction()
