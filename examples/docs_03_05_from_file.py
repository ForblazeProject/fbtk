import fbtk
import os

# テスト用のMOLファイルパス
mol_path = "test_data/ethanol.mol"

# --- ドキュメントのコードの動作検証 ---
# ファイルから分子テンプレートを読み込み
# (実際には test_data/ethanol.mol が存在することを確認済み)
mol_template = fbtk.Molecule.from_file(mol_path)

builder = fbtk.Builder(density=0.8)
builder.add_molecule(mol_template, count=100)
system = builder.build()

print(f"Successfully built system from file: {system.n_atoms} atoms.")