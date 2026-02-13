import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def test_polymer_polymerization():
    print("--- Testing Polymer Polymerization with Leaving Atoms ---")
    
    # 1. Create Styrene monomer with RDKit
    smiles = "CC(c1ccccc1)" # Styrene
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    n_atoms_monomer = mol.GetNumAtoms()
    print(f"Monomer: {n_atoms_monomer} atoms (including H)")

    # 2. Identify indices for connection (Head: C1, Tail: C2)
    # For CC(c1ccccc1), let's assume:
    # Atom 0: CH2 (Tail Carbon)
    # Atom 1: CH  (Head Carbon)
    # We find H atoms attached to them to remove
    head_idx = 1
    tail_idx = 0
    
    head_leaving = -1
    tail_leaving = -1
    
    for neighbor in mol.GetAtomWithIdx(head_idx).GetNeighbors():
        if neighbor.GetSymbol() == "H":
            head_leaving = neighbor.GetIdx()
            break
            
    for neighbor in mol.GetAtomWithIdx(tail_idx).GetNeighbors():
        if neighbor.GetSymbol() == "H":
            tail_leaving = neighbor.GetIdx()
            break

    print(f"Head C: {head_idx}, Leaving H: {head_leaving}")
    print(f"Tail C: {tail_idx}, Leaving H: {tail_leaving}")

    # 3. Setup Builder
    builder = fbtk.Builder()
    builder.add_rdkit_mol("styrene", mol)
    
    degree = 10
    n_chains = 1
    
    recipe_path = "test_data/polymer_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write(f"""
system:
  density: 0.5
  cell_shape: [50.0, 50.0, 50.0]
components:
  - name: "styrene"
    role: "polymer"
    input:
      smiles: "CC(c1ccccc1)"
    polymer_params:
      degree: {degree}
      n_chains: {n_chains}
      head_index: {head_idx}
      tail_index: {tail_idx}
      head_leaving_index: {head_leaving}
      tail_leaving_index: {tail_leaving}
""")

    # 4. Build and Verify
    builder.load_recipe(recipe_path)
    system = builder.build()
    
    # Calculation:
    # (16 atoms * 10 monomers) - (2 atoms removed * 9 connections) 
    # = 160 - 18 = 142
    expected_atoms = (n_atoms_monomer * degree) - (2 * (degree - 1))
    
    print(f"Build result: {system.n_atoms} atoms")
    print(f"Expected: {expected_atoms} atoms")

    if system.n_atoms == expected_atoms:
        print("Success: Atom count matches perfectly!")
    else:
        print("Failure: Atom count mismatch.")

    # 5. Check coordinates via ASE (optional visualization check)
    atoms = system.to_ase()
    atoms.write("test_data/polystyrene_10.xyz")
    print("Saved to test_data/polystyrene_10.xyz")

    # Cleanup
    os.remove(recipe_path)

if __name__ == "__main__":
    test_polymer_polymerization()
