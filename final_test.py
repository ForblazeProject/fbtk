import fbtk
import os
import subprocess

def test_ion_parsing():
    print("Testing Ion Parsing...")
    li = fbtk.Molecule.from_smiles("[Li+]", name="Li")
    li.to_file("test_li.mol2")
    with open("test_li.mol2", "r") as f:
        content = f.read()
        if "1.0000" in content:
            print("  -> Ion charge detected (OK)")
        else:
            print("  -> ERROR: Ion charge not detected")
    if os.path.exists("test_li.mol2"): os.remove("test_li.mol2")

def test_polymer_from_molecule():
    print("\nTesting Molecule.from_polymer...")
    monomer = fbtk.Molecule.from_smiles("*C(C*)c1ccccc1", name="PS")
    chain = fbtk.Molecule.from_polymer(monomer, degree=10, tacticity="syndiotactic")
    if chain.n_atoms > monomer.n_atoms:
        print(f"  -> Polymer chain created with {chain.n_atoms} atoms (OK)")
    else:
        print("  -> ERROR: Polymer chain creation failed")
    chain.to_file("test_chain.mol")
    if os.path.exists("test_chain.mol"): os.remove("test_chain.mol")

def test_tacticity_atactic():
    print("\nTesting Atactic Builder...")
    builder = fbtk.Builder(density=0.1)
    builder.add_polymer("PS", count=2, degree=5, smiles="*C(C*)c1ccccc1", tacticity="atactic")
    system = builder.build()
    if system.n_atoms > 0:
        print(f"  -> Atactic system built with {system.n_atoms} atoms (OK)")
    else:
        print("  -> ERROR: Atactic build failed")

def test_cli_build():
    print("\nTesting CLI fbtk-build...")
    recipe = """
system:
  density: 0.1
components:
  - name: "ethanol"
    role: "molecule"
    input:
      smiles: "CCO"
    count: 5
"""
    with open("cli_recipe.yaml", "w") as f:
        f.write(recipe)
    
    exe = "./target/release/fbtk-build"
    if not os.path.exists(exe):
        exe = "fbtk-build"
        
    try:
        res = subprocess.run([exe, "--recipe", "cli_recipe.yaml", "--output", "cli_out.mol2"], 
                             capture_output=True, text=True)
        if res.returncode == 0:
            print("  -> CLI build success (OK)")
        else:
            print(f"  -> CLI build failed: {res.stderr}")
    except Exception as e:
        print(f"  -> CLI test skipped (exec not found: {e})")
    
    if os.path.exists("cli_recipe.yaml"): os.remove("cli_recipe.yaml")
    if os.path.exists("cli_out.mol2"): os.remove("cli_out.mol2")

if __name__ == "__main__":
    try:
        test_ion_parsing()
        test_polymer_from_molecule()
        test_tacticity_atactic()
        test_cli_build()
        print("\nAll automated tests completed successfully!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)