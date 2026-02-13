import fbtk
from ase.build import bulk

# ASE Atomsを生成
atoms = bulk('Cu', 'fcc', a=3.6) * (5, 5, 5)

# ASEオブジェクトをそのまま渡して解析 (自己相関)
r, g_r = fbtk.compute_rdf(atoms, query="element Cu")
print(f"RDF computed, bins: {len(g_r)}")
