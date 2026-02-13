from rdkit import Chem
from rdkit.Chem import AllChem
import fbtk

# 1. RDKit で構造を生成 (Ethanol)
mol = Chem.MolFromSmiles("CCO") 
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# 2. FBTK の Molecule として取り込む
custom_mol = fbtk.Molecule.from_rdkit(mol, name="Custom")

# 3. Builder でパッキング
builder = fbtk.Builder(box_size=[30, 30, 30])
builder.add_molecule(custom_mol, count=100)
system = builder.build()

print(f"Hybrid build: {system.n_atoms} atoms.")
