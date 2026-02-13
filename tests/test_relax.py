import fbtk
import numpy as np
from ase import Atoms

def test_relaxation():
    print("--- Testing Relaxation (3D Expansion & Bond Preservation) ---")
    
    # 1. Build a simple linear molecule system (Butane: CCCC)
    # Using SMILES will initially place atoms on a straight line (Z=0)
    builder = fbtk.Builder()
    recipe_path = "test_data/relax_test_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write("""
system:
  density: 0.5
  cell_shape: [30.0, 30.0, 30.0]
components:
  - name: "butane"
    role: "molecule"
    input:
      smiles: "CCCC"
    count: 1
""")
    builder.load_recipe(recipe_path)
    
    # Build (Initial state: Linear)
    system_initial = builder.build()
    pos_init = system_initial.get_positions()
    print(f"Initial Z-range: {np.ptp(pos_init[:, 2]):.4f} (Expected near 0)")
    
    # 2. Run Relaxation
    print("Running relaxation (2000 steps)...")
    builder.relax(2000)
    
    # 3. Check results
    system_relaxed = builder.get_system() # Use get_system, NOT build()
    pos_relaxed = system_relaxed.get_positions()
    
    z_span = np.ptp(pos_relaxed[:, 2])
    print(f"Relaxed Z-range: {z_span:.4f} (Expected > 0.5 if expanded to 3D)")
    
    # 4. Check bond lengths (to ensure bonds didn't break)
    # Butane C-C bond is ~1.54A. Let's check the first bond in the first molecule.
    p1, p2 = pos_relaxed[0], pos_relaxed[1]
    dist = np.linalg.norm(p1 - p2)
    print(f"Sample C-C bond length after relax: {dist:.4f} A (Expected ~1.5 A)")

    if z_span > 0.1 and 1.0 < dist < 2.0:
        print("Success: System expanded to 3D and bonds preserved.")
    else:
        print("Failure: Check coordinates or bond logic.")

    import os
    os.remove(recipe_path)

if __name__ == "__main__":
    try:
        test_relaxation()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
