import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def run_quickstart():
    print("--- FBTK Quick Start: Building Ethanol Bulk ---")
    
    # 1. Prepare Monomer with RDKit
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # 2. Builder Setup
    builder = fbtk.Builder()
    builder.add_rdkit_mol("ethanol", mol)
    
    # 3. Create Recipe (10 molecules in a 20A box)
    recipe_path = "quickstart_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write("""
system:
  density: 0.8
  cell_shape: [15.0, 15.0, 15.0]
components:
  - name: "ethanol"
    role: "solvent"
    input: { smiles: "CCO" }
    count: 20
""")
    
    builder.load_recipe(recipe_path)
    
    # 4. Build and Relax
    print("Building system...")
    builder.build()
    
    print("Relaxing structure...")
    builder.relax(steps=200)
    
    # 5. Export
    system = builder.get_system()
    atoms = system.to_ase()
    atoms.write("ethanol_bulk.xyz")
    
    print(f"Done! Created {system.n_atoms} atoms. Output saved to ethanol_bulk.xyz")
    
    if os.path.exists(recipe_path):
        os.remove(recipe_path)

if __name__ == "__main__":
    run_quickstart()
