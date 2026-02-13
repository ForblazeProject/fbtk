import fbtk
import os

# 1. 分子の定義
ethanol = fbtk.Molecule.from_smiles("CCO", name="Ethanol")

# 2. システムの構築
builder = fbtk.Builder(box_size=[30, 30, 30])
builder.add_molecule(ethanol, count=200)
system = builder.build()

# 3. 初期緩和
system.relax()

# 4. ASEへ出力
atoms = system.to_ase()
print(f"Built {len(atoms)} atoms.")
atoms.write("examples/out_02_01.xyz")
