import fbtk

builder = fbtk.Builder(density=0.9)

# ポリスチレン 10 量体を 50 鎖作成
builder.add_polymer(
    name="PS",
    smiles="[*]CC([*])c1ccccc1", # [*] は重合ポイント
    count=50, 
    degree=10
)

system = builder.build()
print(f"Polymer build: {system.n_atoms} atoms.")
