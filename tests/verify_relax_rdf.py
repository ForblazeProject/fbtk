import fbtk
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ase import Atoms
import os

def verify_relax_with_rdf():
    print("--- Verification: Distance Check vs RDF Analysis ---")
    
    # 1. Prepare Styrene Monomer
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(c1ccccc1)"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    head_idx, tail_idx = 1, 0
    head_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(head_idx).GetNeighbors() if n.GetSymbol() == "H")
    tail_leave = next(n.GetIdx() for n in mol.GetAtomWithIdx(tail_idx).GetNeighbors() if n.GetSymbol() == "H")

    builder = fbtk.Builder()
    builder.add_rdkit_mol("styrene", mol)
    
    # System: DP=20, 2 chains (~640 atoms)
    dp = 20
    n_chains = 2
    L = 30.0
    
    recipe_path = "test_data/verify_rdf.yaml"
    with open(recipe_path, "w") as f:
        f.write(f"""
system:
  density: 0.5
  cell_shape: [{L}, {L}, {L}]
components:
  - name: \"styrene\"
    role: \"polymer\"
    input: {{ smiles: \"CC(c1ccccc1)\" }}
    polymer_params:
      degree: {dp}
      n_chains: {n_chains}
      head_index: {head_idx}
      tail_index: {tail_idx}
      head_leaving_index: {head_leave}
      tail_leaving_index: {tail_leave}
""" )

    builder.load_recipe(recipe_path)
    
    # 2. Build Initial Structure
    print("Building system...")
    builder.build()
    system_init = builder.get_system()
    atoms_init = system_init.to_ase()
    
    # 3. Direct Distance Check (ASE)
    def check_min_dists(atoms):
        dists = atoms.get_all_distances(mic=True)
        # Exclude self-distances (diagonal)
        dists = dists[~np.eye(dists.shape[0], dtype=bool)]
        return np.min(dists), np.sum(dists < 1.0) // 2

    min_init, col_init = check_min_dists(atoms_init)
    
    # 4. RDF Analysis (fbtk)
    # Using a high-resolution RDF to see short-range details
    def get_short_range_rdf(atoms_obj):
        # Select all atoms via index query
        n = len(atoms_obj)
        query = f"index 0:{n}-index 0:{n}"
        r, gr = fbtk.compute_rdf(atoms_obj, query=query, r_max=5.0, n_bins=100)
        # Find the first non-zero g(r) index
        first_nonzero_idx = np.where(gr > 0)[0]
        if len(first_nonzero_idx) > 0:
            onset_r = r[first_nonzero_idx[0]]
        else:
            onset_r = 5.0
        return r, gr, onset_r

    r_init, gr_init, onset_init = get_short_range_rdf(atoms_init)

    print(f"\n[Initial State]")
    print(f"  ASE Minimum Distance: {min_init:.4f} A")
    print(f"  ASE Collisions (<1.0A): {col_init}")
    print(f"  RDF Onset Distance:    {onset_init:.4f} A")

    # 5. Perform Relaxation
    print("\nRunning relaxation...")
    builder.relax(500, 50.0, 0.01)
    
    # 6. Post-Relaxation Check
    system_relaxed = builder.get_system()
    atoms_relaxed = system_relaxed.to_ase()
    
    min_relax, col_relax = check_min_dists(atoms_relaxed)
    r_relax, gr_relax, onset_relax = get_short_range_rdf(atoms_relaxed)

    print(f"\n[Relaxed State]")
    print(f"  ASE Minimum Distance: {min_relax:.4f} A")
    print(f"  ASE Collisions (<1.0A): {col_relax}")
    print(f"  RDF Onset Distance:    {onset_relax:.4f} A")

    # Final Consistency Check
    # Onset of RDF should be very close to the Minimum Distance found by ASE
    diff_init = abs(min_init - onset_init)
    diff_relax = abs(min_relax - onset_relax)
    
    print(f"\n--- Consistency Analysis ---")
    print(f"  Init Diff (ASE vs RDF):  {diff_init:.6f}")
    print(f"  Relax Diff (ASE vs RDF): {diff_relax:.6f}")

    if diff_relax < 0.1: # Allow small binning error
        print("\nSUCCESS: RDF analysis confirms the distance check results.")
    else:
        print("\nWARNING: Discrepancy between distance check and RDF.")

    os.remove(recipe_path)

if __name__ == "__main__":
    verify_relax_with_rdf()
