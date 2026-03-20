from openpyxl import Workbook
import numpy as np
import torch
def save_pairs_to_txt(list1, list2, filename="pairs.txt"):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length to form pairs")

    with open(filename, "w") as f:
        for a, b in zip(list1, list2):
            f.write(f"{a},{b}\n")
    print(f"Saved {len(list1)} pairs to {filename}")

def save_high_similarity_pairs(P, threshold=0.50, filename="high_similarity_pairs.txt"):
    # Ensure NumPy array
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    elif not isinstance(P, np.ndarray):
        raise TypeError("P must be a NumPy array or a PyTorch tensor")

    pairs = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            score = float(P[i, j])
            if score > threshold:
                pairs.append((i, j, score))

    # Sort by score descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Save to text file
    with open(filename, "w") as f:
        for i, j, score in pairs:
            f.write(f"{i},{j},{score:.6f}\n")

    print(f"Saved {len(pairs)} pairs with similarity > {threshold} to {filename}")
    return pairs

def save_list_to_txt(lst, filename):
    with open(filename, "w") as f:
        for item in lst:
            f.write(str(item) + "\n")
    print(f"List saved to {filename}")

def save_matrix_to_excel(P, filename="permutation_matrix.xlsx"):
    # Ensure NumPy array
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    elif not isinstance(P, np.ndarray):
        raise TypeError("P must be a NumPy array or a PyTorch tensor")

    wb = Workbook()
    ws = wb.active

    # Write each value into its cell
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ws.cell(row=i+1, column=j+1, value=float(P[i, j]))

    wb.save(filename)
    print(f"Matrix saved to {filename}")