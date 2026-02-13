import fbtk
from rdkit import Chem

def run_polymer_example():
    print("--- FBTK Polymer Synthesis: Polystyrene (Smart Connection) ---")
    
    # Initialize Builder
    builder = fbtk.Builder()
    
    # Set target density
    builder.set_density(0.8)
    
    # Add Polystyrene (PS) using the new simplified API
    builder.add_polymer(
        name="styrene",
        count=2,
        degree=10,
        smiles="CC(c1ccccc1)",
        head=0,
        tail=1
    )
    
    # Build the system
    print("Building system...")
    builder.build()
    
    # Structural relaxation (FIRE algorithm)
    # MUST be called before get_system() if you want the relaxed structure
    print("Performing relaxation (500 steps)...")
    builder.relax(steps=500)
    
    # Get the resulting system (Relaxed)
    system = builder.get_system()
    print(f"Successfully built and relaxed polymer system.")
    print(f"Total atoms: {system.n_atoms}")
    
    # Save to SDF using RDKit to preserve bond information
    print("Converting to RDKit and saving to SDF...")
    rdmol = system.to_rdkit()
    
    writer = Chem.SDWriter("polystyrene_10.sdf")
    writer.write(rdmol)
    writer.close()
    
    # Also save XYZ for comparison if needed
    system.to_ase().write("polystyrene_10.xyz")
    print("Output saved to polystyrene_10.sdf and .xyz")

if __name__ == "__main__":
    run_polymer_example()
