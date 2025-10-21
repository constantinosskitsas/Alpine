import numpy as np
import pandas as pd

# Load original feature matrices
F1 = np.loadtxt('./Data/data/allmv_tmdb/allmv_tmdb_s_feat.txt', dtype=float)
F2 = np.loadtxt('./Data/data/allmv_tmdb/allmv_tmdb_t_feat.txt', dtype=float)

# Load CSVs (replace with your actual file paths)
csv2 = pd.read_csv("./Data/Full-dataset/attribute/am-tdattr1.csv", header=None).iloc[:, 1:].to_numpy()
csv1 = pd.read_csv("./Data/Full-dataset/attribute/am-tdattr2.csv", header=None).iloc[:, 1:].to_numpy()

def compare_features(name, arr1, arr2):
    print(f"--- Comparing {name} ---")
    print("Shape 1:", arr1.shape, "Shape 2:", arr2.shape)

    if arr1.shape != arr2.shape:
        print("⚠️ Shapes differ!")
        return

    if np.array_equal(arr1, arr2):
        print("✅ Arrays are exactly the same")
    else:
        diff_count = np.sum(arr1 != arr2)
        print(f"⚠️ Arrays differ in {diff_count} entries")

# Compare
compare_features("CSV1 vs F1", csv1, F1)
compare_features("CSV2 vs F2", csv2, F2)
