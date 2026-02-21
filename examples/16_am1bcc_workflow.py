"""
Example: Using OpenFF-calculated AM1-BCC charges in FBTK
"""
import fbtk
import numpy as np

try:
    from openff.toolkit import Molecule, ForceField
    from openff.interchange import Interchange
except ImportError:
    print("Error: OpenFF toolkit or interchange not found.")
    exit(1)

# 1. OpenFF 側でモノマーに対して AM1-BCC 電荷を計算
print("--- 1. Calculating AM1-BCC charges in OpenFF ---")
off_mol = Molecule.from_smiles("CCO") # Ethanol

try:
    off_mol.assign_partial_charges("am1bcc")
    print("AM1-BCC calculation successful.")
except Exception as e:
    print("AM1-BCC calculation failed (AmberTools might be missing).")
    print("Falling back to Gasteiger for this demonstration...")
    off_mol.assign_partial_charges("gasteiger")

am1bcc_charges = off_mol.partial_charges.m
print("Calculated Charges (first 3):")
print(am1bcc_charges[:3])

# 2. 電荷を保持したまま FBTK.Molecule に変換
print("\n--- 2. Importing into FBTK with charges ---")
fbtk_monomer = fbtk.Molecule.from_openff(off_mol)

# 3. FBTK でシステムを構築
print("\n--- 3. Building system in FBTK ---")
builder = fbtk.Builder()
builder.add_molecule(fbtk_monomer, count=50)
builder.set_density(0.5)
system = builder.build()
print("System built.")
print("Total Atoms:", system.n_atoms)

# 4. OpenFF Interchange への書き出し
print("\n--- 4. Exporting to Interchange (Charges Preserved) ---")
interchange = system.to_openff(forcefield="openff-2.1.0.offxml")
print("Interchange created.")

# 5. 確認
print("\n--- 5. Verification ---")
coll = interchange.collections["Electrostatics"]
# Resolve first charge manually
temp_q = {}
for k, v in coll.key_map.items():
    temp_q[k.atom_indices[0]] = float(coll.potentials[v].parameters["charge"].m)
final_charges = [temp_q[i] for i in range(len(temp_q))]

print("Final charge in simulation (first 3):")
print(final_charges[:3])
print("Original pre-calculated charges (first 3):")
print(list(am1bcc_charges[:3]))

if np.allclose(final_charges[:9], am1bcc_charges, atol=1e-5):
    print("\nSUCCESS: AM1-BCC charges were perfectly preserved through FBTK packing!")

print("\nWorkflow completed.")