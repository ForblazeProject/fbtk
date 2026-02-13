import fbtk

builder = fbtk.Builder(box_size=[40, 40, 40])

# 複数種類の低分子を異なる個数で追加
water = fbtk.Molecule.from_smiles("O", name="WAT")
ethanol = fbtk.Molecule.from_smiles("CCO", name="EtOH")

builder.add_molecule(water, count=800)
builder.add_molecule(ethanol, count=200)

system = builder.build()
print(f"Mixed system: {system.n_atoms} atoms.")
