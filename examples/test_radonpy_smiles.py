import fbtk

# RadonPy 形式の SMILES (ポリスチレン)
# '*' は接続点
smiles = "*C(C*)c1ccccc1"

builder = fbtk.Builder(density=0.5)
# * を自動認識するので、追加のインデックス指定は不要
builder.add_polymer(
    name="PS",
    smiles=smiles,
    count=5,
    degree=10
)

system = builder.build()
print(f"RadonPy-style SMILES test: {system.n_atoms} atoms.")

# 重合が正しく行われていれば、1ユニットあたり 8(重原子) + 8(水素) = 16原子? 
# 元の SMILES は C8H8 + 2(*) = 18原子。
# 10量体で端の処理を含めると原子数が計算通りか確認。
print(f"Atoms per chain: {system.n_atoms / 5}")
