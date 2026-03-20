import scipy
import torch
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

def PermHungarian(M):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    return None, row_ind,col_ind

def convertToPermHungarian(M, n1, n2):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)
    P = np.zeros((n2, n1))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, row_ind,col_ind

def convertToPermGreedy(M, n1, n2):
    n = len(M)
    indices = torch.argsort(M.flatten())
    row_done = np.zeros(n)
    col_done = np.zeros(n)

    P = np.zeros((n, n))
    ans = []
    for i in range(n*n):
        cur_row = int(indices[n*n - 1 - i]/n)
        cur_col = int(indices[n*n - 1 - i]%n)
        if (row_done[cur_row] == 0) and (col_done[cur_col] == 0):
            P[cur_row][cur_col] = 1
            row_done[cur_row] = 1
            col_done[cur_col] = 1
            if (cur_row >= n1) or (cur_col >= n2):
                continue
            ans.append((cur_row, cur_col))
    return P, ans

def convertToPermHungarian2new(row_ind, col_ind, n, m):
    P = torch.zeros((n,m), dtype = torch.float64)
    #A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    ans = []
    n = max(len(row_ind), len(col_ind))
    for i in range(n):
        #P[row_ind[i]][col_ind[i]] = 1
        #if (row_ind[i] >= n) or (col_ind[i] >= m):
        #    continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def superfast(l2,k):
    n = l2.shape[0]
    rows = set()
    rows1=[]
    cols = set()
    cols1=[]
    vals = np.argsort(l2, axis=None)[::-1]
    for val in vals:
        x, y = np.unravel_index(val, l2.shape)

        if x in rows or y in cols:

            continue
        rows.add(x)
        cols.add(y)
        rows1.append(x)
        cols1.append(y)
        if (len(rows)>=k):
            break
    return rows1, cols1

def EVAL_new(A, B, P, k, Hung=True,initial_node=0, gt_G1=None, gt_G2=None):
    hr=0
    if Hung:
        # --- Hungarian assignment mode ---
        row_ind, col_ind = linear_sum_assignment(P,maximize=True)
        selected_pairs = list(zip(row_ind, col_ind))
        pair_costs = P[row_ind, col_ind]
        sorted_pairs = sorted(zip(selected_pairs, pair_costs), key=lambda x: x[1], reverse=True)
        top_n_pairs = sorted_pairs[:k]
        top_n_row_indices = [pair[0][0] for pair in top_n_pairs]
        top_n_col_indices = [pair[0][1] for pair in top_n_pairs]
        selected_pairs = [(pair, cost) for pair, cost in top_n_pairs]
        # hr=0
    else:
        # --- Connectivity-aware greedy selection ---
        selected_pairs, top_n_row_indices, top_n_col_indices,hr = select_connected_pairs(A, B, P, k,initial_node)
        #selected_pairs, top_n_row_indices, top_n_col_indices=select_connected_pairs_reverse(A,B,P,k)
        # hr=0
    # Build tensors
    A_tensor = torch.tensor(nx.to_numpy_array(A), dtype=torch.float64)
    B_tensor = torch.tensor(nx.to_numpy_array(B), dtype=torch.float64)
    A_sub = A_tensor[top_n_col_indices, :][:, top_n_col_indices]
    B_sub = B_tensor[top_n_row_indices, :][:, top_n_row_indices]
    #save_pairs_to_txt(top_n_row_indices,top_n_col_indices,"pairs-Selected")
    # Build subgraph networks
    G_A_sub = nx.from_numpy_array(A_sub.numpy())
    G_B_sub = nx.from_numpy_array(B_sub.numpy())
    correct_pairs = 0
    correct_pairs1=0
    for u, v in zip(top_n_col_indices, top_n_row_indices):
        for gu, gv in zip(gt_G1,gt_G2):
            if int(u) == int(gu) and int(v) == int(gv):
                correct_pairs += 1
                #break
    for gu, gv in zip(gt_G1, gt_G2):
    # Look up the probability in P
        prob = P[int(gv), int(gu)]
    
    # If similarity is above 0.5, count it
        if prob > 0.5:
            correct_pairs1 += 1
    
    local_1 = 0
    local_2 = 0
    for u in top_n_col_indices:
        for gu in gt_G1:
            if int(u) == int(gu):
                local_1 += 1
    
    for v in top_n_row_indices:
        for gv in gt_G2:
            if int(v) == int(gv):
                local_2 += 1
        
    fro_norm_A = torch.norm(A_sub, p='fro').item()
    fro_norm_B = torch.norm(B_sub, p='fro').item()
    fro_norm_diff = torch.norm(A_sub - B_sub, p='fro').item()
    return fro_norm_diff,hr,correct_pairs,local_1,local_2
    return {
        "pairs": selected_pairs,
        "fro_norm_A": fro_norm_A,
        "fro_norm_B": fro_norm_B,
        "fro_norm_diff": fro_norm_diff
    }

def EVAL_new_diff(A, B, P, k, Hung=True,initial_node=0,return_n_components=False):
    if Hung:
        # --- Hungarian assignment mode ---
        row_ind, col_ind = linear_sum_assignment(P,maximize=True)
        selected_pairs = list(zip(row_ind, col_ind))
        pair_costs = P[row_ind, col_ind]
        sorted_pairs = sorted(zip(selected_pairs, pair_costs), key=lambda x: x[1], reverse=True)
        top_n_pairs = sorted_pairs[:k]
        top_n_row_indices = [pair[0][0] for pair in top_n_pairs]
        top_n_col_indices = [pair[0][1] for pair in top_n_pairs]
        # selected_pairs = [(pair, cost) for pair, cost in top_n_pairs]
        selected_pairs = [(pair, cost) for pair, cost in top_n_pairs if cost > 0]
        print("Number of selected pairs with positive cost:", len(selected_pairs), "out of", k)
        hr=0
    else:
        # --- Connectivity-aware greedy selection ---
        selected_pairs, top_n_row_indices, top_n_col_indices,hr = select_connected_pairs(A, B, P, k,initial_node)

    # Build tensors
    A_tensor = torch.tensor(nx.to_numpy_array(A), dtype=torch.float64)
    B_tensor = torch.tensor(nx.to_numpy_array(B), dtype=torch.float64)
    A_sub = A_tensor[top_n_col_indices, :][:, top_n_col_indices]
    B_sub = B_tensor[top_n_row_indices, :][:, top_n_row_indices]

    # Build subgraph networks
    G_A_sub = nx.from_numpy_array(A_sub.numpy())
    G_B_sub = nx.from_numpy_array(B_sub.numpy())

    # Connectivity diagnostics
    num_components_A = nx.number_connected_components(G_A_sub)
    num_components_B = nx.number_connected_components(G_B_sub)
    is_connected_A = (num_components_A == 1)
    is_connected_B = (num_components_B == 1)
    print("A_sub: connected =", is_connected_A, ", components =", num_components_A)
    print("B_sub: connected =", is_connected_B, ", components =", num_components_B)

    # Frobenius norms
    fro_norm_A = torch.norm(A_sub, p='fro').item()
    fro_norm_B = torch.norm(B_sub, p='fro').item()
    fro_norm_diff = torch.norm(A_sub - B_sub, p='fro').item()
    if return_n_components:
        return fro_norm_diff,hr,num_components_A,num_components_B
    else:
        return fro_norm_diff,hr
    return {
        "pairs": selected_pairs,
        "fro_norm_A": fro_norm_A,
        "fro_norm_B": fro_norm_B,
        "fro_norm_diff": fro_norm_diff
    }

def select_connected_pairs(A, B, P, k,initial_node=0):
    # Convert to numpy arrays
    Rei=False
    if (initial_node==-1):
        initial_node=0
        poS_ini=0
        Rei=True

    if not isinstance(A, np.ndarray):
        A = nx.to_numpy_array(A)
    if not isinstance(B, np.ndarray):
        B = nx.to_numpy_array(B)
    hr=0
    # Initial candidates from Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(P,maximize=True)
    candidates = [(i, j, P[i,j]) for i,j in zip(row_ind, col_ind)]
    candidates = [(i, j, P[i,j], pos) for pos, (i, j) in enumerate(zip(row_ind, col_ind))]
    #save_pairs_to_txt(row_ind,col_ind,"pairs-Hung")
    # Sort ascending by cost
    candidates.sort(key=lambda x: x[2],reverse=True)

    selected = []
    rows, cols = list(), list()
    count_problem = 0

    while len(selected) < k and len(candidates) > 0:
        # Find the best feasible candidate
        added = False
        for idx, (i, j, cost, pos) in enumerate(candidates):
            if len(selected) == 0:
                # First node always accepted
                if (idx==initial_node):
                    selected.append((i, j, cost))
                    rows.append(i)
                    cols.append(j)
                    candidates.pop(idx)
                    added = True
                    break
                continue

            test_rows = rows + [i]
            test_cols= cols+[j]
            A_sub = A[np.ix_(test_cols, test_cols)]
            B_sub = B[np.ix_(test_rows, test_rows)]
            G_A = nx.from_numpy_array(A_sub)
            G_B = nx.from_numpy_array(B_sub)

            if nx.is_connected(G_A) and nx.is_connected(G_B):
                selected.append((i, j, cost))
                rows.append(i)
                cols.append(j)
                candidates.pop(idx)
                added = True
                if (pos>hr):
                    hr=pos
                break
                    
        if not added:
            # No candidate maintains connectivity, stop or relax
            print("problem")
            count_problem += 1
            hr=-1
            if (Rei) and count_problem < 10:
                poS_ini=poS_ini+1
                initial_node=poS_ini
                hr=0
                selected = []
                rows, cols = list(), list()
                candidates = [(i, j, P[i,j], pos) for pos, (i, j) in enumerate(zip(row_ind, col_ind))]
                candidates.sort(key=lambda x: x[2],reverse=True)

            else:
                break
    return selected, list(rows), list(cols),hr