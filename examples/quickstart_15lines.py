import fbtk
import os

# 1. SMILESから分子トポロジを作成
ethanol = fbtk.Molecule.from_smiles("CCO", name="Ethanol")

# 2. ユニットセルの構築 (Density 0.789 g/cm3)
builder = fbtk.Builder(density=0.789)
builder.add_molecule(ethanol, count=200)
system = builder.build()

print(f"Initial n_atoms: {system.n_atoms}")

# 3. 高速初期緩和
system.relax(steps=100)
print("Relaxation finished.")

# 4. ASEへの変換と保存
try:
    atoms = system.to_ase()
    print(f"Successfully converted to ASE Atoms: {len(atoms)} atoms")
    atoms.write("test_data/quickstart_bulk.xyz")
    print("Saved to test_data/quickstart_bulk.xyz")
except ImportError:
    print("ASE not installed, skipping conversion.")
except Exception as e:
    print(f"Error: {e}")
