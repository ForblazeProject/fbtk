import fbtk
from ase.build import bulk
import numpy as np

print("Validating RDF: Copper FCC crystal")
atoms = bulk('Cu', 'fcc', a=3.614) * (4, 4, 4)
r, g_r = fbtk.compute_rdf(atoms, query="element Cu", r_max=10.0, n_bins=100)

targets = [2.55, 3.65, 4.45]
expected = [17.33, 4.23, 7.28]

print("\nComputed RDF Peaks:")
for i, t in enumerate(targets):
    idx = np.argmin(np.abs(r - t))
    val = g_r[idx]
    exp = expected[i]
    diff = abs(val - exp) / exp
    print(f"  r ~ {r[idx]:.2f} A, g(r) = {val:.2f} (Expected: {exp}, Diff: {diff:.1%})")
    assert diff < 0.05, f"Peak {i+1} mismatch!"

print("\nVerification: SUCCESS!")