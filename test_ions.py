import fbtk

def test_ion(smiles, name):
    print(f"Testing SMILES: {smiles} ({name})")
    mol = fbtk.Molecule.from_smiles(smiles, name=name)
    pos = mol.get_positions()
    
    # We can't directly get charge from PyMolecule yet (not exposed to Python)
    # But we can save to mol2 and check the content.
    mol.to_file(f"test_{name}.mol2")
    
    with open(f"test_{name}.mol2", "r") as f:
        content = f.read()
        atom_lines = [l for l in content.splitlines() if "RES" in l]
        for line in atom_lines:
            print(f"  Atom: {line}")

test_ion("[Li+]", "Li")
test_ion("C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)(F)", "TFSI")
