import fbtk
import sys
from unittest.mock import MagicMock

# RadonPy がない環境でもテストできるよう Mock を作成
mock_radonpy = MagicMock()
sys.modules["radonpy"] = mock_radonpy
sys.modules["radonpy.core"] = MagicMock()

# RDKit で代用して RadonPy から渡された Mol をシミュレート
from rdkit import Chem
from rdkit.Chem import AllChem
radon_mol = Chem.MolFromSmiles("CCO")
radon_mol = Chem.AddHs(radon_mol)
AllChem.EmbedMolecule(radon_mol)

# --- ここからドキュメントのコード ---
# 2. FBTK に渡して高速パッキング
fbtk_mol = fbtk.Molecule.from_rdkit(radon_mol, name="RadonPoly")

builder = fbtk.Builder(density=0.5)
builder.add_molecule(fbtk_mol, count=50)
system = builder.build()

# 3. FBTK で高速初期緩和
system.relax(steps=100)

# 4. ASE Atoms に戻す
atoms = system.to_ase()
print(f"RadonPy hybrid workflow success: {len(atoms)} atoms")
