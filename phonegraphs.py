import numpy as np
import os

def generate_and_save_permutations(n, num_runs=5, folder="Data/data/phone"):
    """
    Generates `num_runs` random permutations of size n.
    Saves each permutation in a separate file.
    """
    os.makedirs(folder, exist_ok=True)

    for i in range(num_runs):
        P = np.random.permutation(n)
        filename = os.path.join(folder, f"P_run{i}.txt")
        np.savetxt(filename, P, fmt='%d')
        print(f"Saved permutation {i} to {filename}")



generate_and_save_permutations(1000)