import fbtk
import numpy as np

def test_relaxation_bulk():
    print("--- Testing Relaxation (Bulk System) ---")
    
    builder = fbtk.Builder()
    recipe_path = "test_data/relax_bulk_recipe.yaml"
    with open(recipe_path, "w") as f:
        # High density to force collisions
        f.write("""
system:
  density: 0.8
  cell_shape: [20.0, 20.0, 20.0]
components:
  - name: "ethanol"
    role: "molecule"
    input:
      smiles: "CCO"
    count: 50
""")
    builder.load_recipe(recipe_path)
    
    # 1. Build
    builder.build()
    system_init = builder.get_system()
    pos_init = system_init.get_positions()
    print(f"Initial atoms: {system_init.n_atoms}")
    print(f"Initial Z-range: {np.ptp(pos_init[:, 2]):.4f} (Expected near 20.0 since it is packed randomly)")
    
    # Actually, random packing already gives 3D. 
    # The real problem we want to solve is REDUCING OVERLAPS.
    # Let's measure minimum distance between non-bonded atoms.
    
    def min_dist(pos):
        # Very naive O(N^2) for testing
        min_d = 100.0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d: min_d = d
        return min_d

    d_min_init = min_dist(pos_init)
    print(f"Initial Min Distance: {d_min_init:.4f} A (Might be very small)")
    
    # 2. Relax
    print("Running relaxation (500 steps)...")
    builder.relax(500)
    
    # 3. Check
    system_relaxed = builder.get_system()
    pos_relaxed = system_relaxed.get_positions()
    d_min_relaxed = min_dist(pos_relaxed)
    print(f"Relaxed Min Distance: {d_min_relaxed:.4f} A (Expected to increase)")

    if d_min_relaxed > d_min_init or d_min_relaxed > 1.0:
        print("Success: Minimum distance increased (Overlaps removed).")
    else:
        print("Failure: Overlaps remain.")

    import os
    os.remove(recipe_path)

if __name__ == "__main__":
    try:
        test_relaxation_bulk()
    except Exception as e:
        print(f"Test failed: {e}")
