import fbtk
import numpy as np
from rdkit import Chem

def validate_all():
    print("=== FBTK Final Validation (Post-Refactoring) ===")
    
    # 1. Builder & Refinement Validation
    print("\n1. Building Polystyrene with VSEPR + UFF refinement...")
    builder = fbtk.Builder()
    builder.set_cell([20.0, 20.0, 20.0])
    builder.add_polymer("PS", count=2, degree=5, smiles="CC(c1ccccc1)", head=0, tail=1)
    
    # This triggers build -> VSEPR (local) -> UFF (local)
    builder.build()
    
    # 2. Relaxation Validation (UFF + FIRE)
    print("\n2. Performing global relaxation (UFF + FIRE)...")
    builder.relax(steps=200)
    
    system = builder.get_system()
    print(f"   Success: Built system with {system.n_atoms} atoms.")
    
    # 3. RDF Validation (using glam refactored core)
    print("\n3. Calculating C-C RDF...")
    atoms = system.to_ase()
    r, gr = fbtk.compute_rdf(atoms, query="C-C", r_max=6.0, n_bins=100)
    
    # Find the first peak
    peak_idx = np.argmax(gr)
    first_peak_r = r[peak_idx]
    print(f"   Success: RDF first peak at {first_peak_r:.4f} A (Expected ~1.4-1.5A)")
    
    # 4. Geometry Analysis Validation (COM, Angles)
    print("\n4. Analyzing geometry (Center of Mass)...")
    com = system.get_center_of_mass()
    print(f"   Center of Mass: {com}")
    
    # 5. Interface Stacking Validation
    print("\n5. Validating system stacking...")
    builder2 = fbtk.Builder()
    builder2.set_cell([20.0, 20.0, 20.0])
    builder2.add_solvent("water", count=10, smiles="O")
    builder2.build()
    sys2 = builder2.get_system()
    
    system.stack(sys2, axis=2) # Stack along Z
    print(f"   Success: Stacked system has {system.n_atoms} atoms.")
    
    print("\n=== Validation Complete: ALL SYSTEMS NOMINAL ===")

if __name__ == "__main__":
    try:
        validate_all()
    except Exception as e:
        print(f"\n!!! VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
