import fbtk
import os

# 1. Molecule verification
print("Testing Molecule.to_file...")
mol = fbtk.Molecule.from_smiles("CCO", name="Ethanol")

mol.to_file("test_mol.mol")
mol.to_file("test_mol.mol2")

print("Checking test_mol.mol content:")
with open("test_mol.mol", "r") as f:
    content = f.read()
    lines = content.splitlines()
    print("  -> Header: " + lines[0])
    if "M  END" in content:
        print("  -> Found M  END (OK)")

print("\nChecking test_mol.mol2 content:")
with open("test_mol.mol2", "r") as f:
    content = f.read()
    if "@<TRIPOS>CRYSIN" in content:
        print("  -> ERROR: Found CRYSIN in Molecule.mol2")
    else:
        print("  -> CRYSIN not found (OK)")

# 2. System verification
print("\nTesting System.to_file...")
builder = fbtk.Builder()
builder.add_molecule_smiles("water", count=10, smiles="O")
system = builder.build() # Returns PySystem

system.to_file("test_sys.mol2")

print("Checking test_sys.mol2 content:")
with open("test_sys.mol2", "r") as f:
    content = f.read()
    if "@<TRIPOS>CRYSIN" in content:
        print("  -> Found CRYSIN in System.mol2 (OK)")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "@<TRIPOS>CRYSIN" in line:
                print("  -> CRYSIN values: " + lines[i+1])
    else:
        print("  -> ERROR: CRYSIN not found in System.mol2")
