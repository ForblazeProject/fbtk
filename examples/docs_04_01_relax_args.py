import fbtk
ethanol = fbtk.Molecule.from_smiles("CCO")
builder = fbtk.Builder(box_size=[20, 20, 20])
builder.add_molecule(ethanol, count=50)
system = builder.build()

# relaxの引数指定テスト
system.relax(
    steps=100,
    threshold=1.0,
    num_threads=4
)
print("Relax with args finished.")
