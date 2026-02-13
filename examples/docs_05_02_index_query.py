import fbtk
from ase.build import bulk

# クエリテスト用の系
atoms = bulk('Cu', 'fcc', a=3.6) * (10, 1, 1) # 40原子

# 1. 範囲指定クエリ
# 前半 20 原子と後半 20 原子の間の相関
# 現状の実装は "index A-B" 形式をサポートしているか確認
try:
    # 10原子の系に合わせて修正 (0-4 と 5-9)
    r, g_r = fbtk.compute_rdf(atoms, query="index 0 to 4 with index 5 to 9")
    print(f"Index query success, bins: {len(g_r)}")
except Exception as e:
    print(f"Index query failed: {e}")
