import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os

def test_rdkit_ase_integration():
    print("--- Testing RDKit to fbtk to ASE integration ---")
    
    # 1. Create a "smart" molecule with RDKit
    smiles = "CCO" # Ethanol
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol) # Add Hydrogens (Intellectual part)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # Generate 3D coordinates
    
    print(f"RDKit Molecule: {mol.GetNumAtoms()} atoms (including H)")

    # 2. Register to fbtk Builder
    builder = fbtk.Builder()
    builder.add_rdkit_mol("ethanol", mol)
    
    # 3. Create a simple recipe
    recipe_path = "test_data/test_rdkit_recipe.yaml"
        with open(recipe_path, "w") as f:
            f.write("""
    system:
      density: 0.8
      cell_shape: [15.0, 15.0, 15.0]
    components:
      - name: "ethanol"
        count: 5
        role: "molecule"
        input:
          smiles: "CCO"
    """)
    # 4. Build system
    builder.load_recipe(recipe_path)
    system = builder.build()
    print(f"Build successful: {system.n_atoms} total atoms")

    # 5. Convert to ASE and check
    atoms = system.to_ase()
    print(f"ASE conversion: {len(atoms)} atoms, cell: {atoms.get_cell().lengths()}")
    
    # Verify positions are not all zero (meaning 3D info was preserved)
    pos = atoms.get_positions()
    print(f"Sample position of first atom: {pos[0]}")
    
    if np.any(pos != 0):
        print("Success: 3D coordinates preserved.")
    else:
        print("Error: All positions are zero!")

    # 6. Save as XYZ via ASE to verify it's a valid Atoms object
    atoms.write("test_data/built_system.xyz")
    print(f"Saved to test_data/built_system.xyz")

    # Cleanup
    os.remove(recipe_path)

if __name__ == "__main__":
    try:
        test_rdkit_ase_integration()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
