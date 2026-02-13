import fbtk

builder = fbtk.Builder()
builder.load_recipe("examples/docs_03_03_recipe.yaml")
system = builder.build()
print(f"Recipe build: {system.n_atoms} atoms.")
