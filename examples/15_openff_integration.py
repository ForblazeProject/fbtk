"""
Example: High-precision OpenFF Integration with FBTK (Clean Version)
"""
import fbtk
import numpy as np

try:
    from openff.toolkit import Molecule, ForceField
    from openff.interchange import Interchange
    from openff.units import unit
except ImportError:
    print("Error: OpenFF toolkit or interchange not found.")
    exit(1)

# 1. OpenFF で電荷付きモノマーを準備
print("--- 1. Preparing monomer in OpenFF ---")
off_mol = Molecule.from_smiles("CCO")
try:
    off_mol.assign_partial_charges("am1bcc")
except:
    off_mol.assign_partial_charges("gasteiger")
off_charges = off_mol.partial_charges.m
print("OpenFF Charges (first 3):")
print(off_charges[:3])

# 2. FBTK へインポート
print("\n--- 2. Importing into FBTK ---")
fbtk_monomer = fbtk.Molecule.from_openff(off_mol)
fbtk_charges = fbtk_monomer.get_charges()
print("FBTK Charges (first 3):")
print(fbtk_charges[:3])

# 3. システムの構築
print("\n--- 3. Building system in FBTK ---")
builder = fbtk.Builder()
builder.add_molecule(fbtk_monomer, count=8) # 8分子
builder.set_density(0.1)
system = builder.build()
cell_diag = np.diag(system.cell)
print("System built.")
print("Atoms:", system.n_atoms)
print("Cell diagonal (Angstrom):", cell_diag)

# 4. OpenFF Interchange への書き出し (力場適用)
print("\n--- 4. Exporting to OpenFF Interchange ---")
# ForceField を指定して Interchange を作成
# セルと電荷が自動的に同期されます
interchange = system.to_openff(forcefield="openff-2.1.0.offxml")
print("Interchange created successfully.")
print("Interchange Box (nm):")
print(interchange.box)

# 5. FBTK への再インポートと確認
print("\n--- 5. Re-importing to FBTK for validation ---")
re_system = fbtk.System.from_openff(interchange)
re_cell_diag = np.diag(re_system.cell)
re_charges = re_system.get_charges()

print("Re-imported Cell diagonal (Angstrom):")
print(re_cell_diag)
print("Re-imported Charges (first 3):")
print(re_charges[:3])

# 検証
if np.allclose(cell_diag, re_cell_diag):
    print("\nSUCCESS: Cell information is synchronized!")
if np.allclose(fbtk_charges[:len(re_charges)//8], re_charges[:len(re_charges)//8]):
    print("SUCCESS: Partial charges are preserved!")

print("\nExample completed.")