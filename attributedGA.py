from sinkhorn import greenkhorn,sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
import numpy as np
import math
import torch
from numpy import linalg as LA
from pred import convertToPermHungarian,eucledian_dist,feature_extraction1,feature_extraction,convertToPermHungarian2new
import networkx as nx
from GradP.gradp import gradPMain
import time
import os
from help_functions import read_graph,read_real_graph, read_list
from resultsfolder import generate_new_id,create_new_folder,get_max_previous_id 
import pandas as pd
import scipy 
from SlotaAlign.SlotaAlign_main import SlotaA
from REGAL.regal import RegalATT
from HTC.main import HTC_main
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)
folderall = 'data3_'
experimental_folder=f'./{folderall}/res/'
new_id = generate_new_id(get_max_previous_id(experimental_folder))
experimental_folder=f'./{folderall}/res/_{new_id}/'   
def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Time:', time_diff)
    print()     
def PermHungarian(M):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    return _ ,row_ind,col_ind
def pad_numpy_rows(arr, target_rows):
    current_rows = arr.shape[0]
    if current_rows < target_rows:
        padding = np.zeros((target_rows - current_rows, arr.shape[1]))
        arr = np.vstack((arr, padding))
    return arr
def compare_features(F1, F2, A_to_B):
    l2_diffs = []
    cos_sims = []

    for a, b in enumerate(A_to_B):   # <-- works if A_to_B is a list or 1D numpy array
        f1 = F1[a]
        f2 = F2[b]

        # L2 distance
        l2 = np.linalg.norm(f1 - f2)
        l2_diffs.append(l2)

        # Cosine similarity
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-9)
        cos = np.dot(f1, f2) / denom
        cos_sims.append(cos)

    print("=== Ground Truth Feature Comparison ===")
    print(f"Average L2 distance: {np.mean(l2_diffs):.6f}")
    print(f"Median L2 distance: {np.median(l2_diffs):.6f}")
    print(f"Average cosine similarity: {np.mean(cos_sims):.6f}")
    print(f"Median cosine similarity: {np.median(cos_sims):.6f}")
    print(f"Min/Max cosine similarity: {np.min(cos_sims):.6f} / {np.max(cos_sims):.4f}")
    print(f"distance{np.sum(l2_diffs):.6f}")
    return l2_diffs, cos_sims 

def add_noise_per_row(features, noise_fraction=0.1):
    noisy_features = features.copy()
    n_rows, n_cols = noisy_features.shape
    num_noisy_per_row = int(n_cols * noise_fraction)

    for i in range(n_rows):
        zero_indices = np.random.choice(n_cols, num_noisy_per_row, replace=False)
        noisy_features[i, zero_indices] = 0

    return noisy_features

# Apply to F1 and F2
def Alpine_pp_labels(A,B,feat, K, niter,A1,weight=1):
    m = len(A)
    n = len(B)
    I_p = torch.zeros((m,m+1),dtype = torch.float64)
    for i in range(m):
        I_p[i,i] = 1
    Pi=torch.ones((m+1,n),dtype = torch.float64)
    feat1 = torch.tensor(feat, dtype=torch.float64, device=Pi.device)
    Pi[:-1,:] *= 1/n
    Pi[-1,:] *= (n-m)/n
    reg = 1.0
    mat_ones = torch.ones((m+1, n), dtype = torch.float64)
    ones_ = torch.ones(n, dtype = torch.float64)
    ones_augm_ = torch.ones(m+1, dtype = torch.float64)
    ones_augm_[-1] = n-m
    gamma=1
    A0 = torch.mean(np.abs(feat1))
    dd=1
    degrees = A.sum(dim=1)
# Average degree = mean of all degrees
    avg_degree = degrees.mean()
    degrees1=B.sum(dim=1)
    avg_degree1 = degrees1.mean()
    if (avg_degree<3 or avg_degree1<3):
        dd=2
    for i in range(10):
        for it in range(1, 11):
            deriv=(-4*I_p.T@(A-I_p@Pi@B@Pi.T@I_p.T)@I_p@Pi@B)*dd+i*(mat_ones - 2*Pi)+K
            S0 = deriv.abs().mean().item()  # PyTorch version
            gamma_a = gamma * S0 / (A0+0.0001 )
            deriv = deriv + gamma_a*feat1
            q=sinkhorn(ones_augm_, ones_, deriv, reg,method="sinkhorn",maxIter = 500, stopThr = 1e-9) 
            alpha = (2 / float(2 + it) )    
            Pi[:m,:n] = Pi[:m,:n] + alpha * (q[:m,:n] - Pi[:m,:n])
    Pi=Pi[:-1]
    P2,row_ind,col_ind = convertToPermHungarian(Pi, n, m)
    forbnorm = LA.norm(A - I_p[:,:m].T@P2@B@P2.T@I_p[:,:m], 'fro')**2
    return Pi, forbnorm,row_ind,col_ind
def AlpineL(Gq, Gt,f1=None,f2=None, mu=1, niter=10, weight=2):
    n1 = Gq.number_of_nodes()
    n2 = Gt.number_of_nodes()
    n = max(n1, n2)
    for node in nx.isolates(Gq):
        Gq.add_edge(node, node)
    for node in nx.isolates(Gt):
        Gt.add_edge(node, node)
        
    Gq.add_node(n1)
    Gq.add_edge(n1,n1)
    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    feat = eucledian_dist(f1,f2,n)
    zeros_row = np.zeros((1, feat.shape[1]))
    feat=np.vstack([feat, zeros_row])
    
# Append it to feat
    
    #weight=1
    if (weight==2):
        F1 = feature_extraction1(Gq)
        F2 = feature_extraction1(Gt) 
    else:
        F1 = feature_extraction(Gq)
        F2 = feature_extraction(Gt)
    D = eucledian_dist(F1,F2,n)
    D = torch.tensor(D, dtype = torch.float64)
    P, forbnorm,row_ind,col_ind = Alpine_pp_labels(A[:n1,:n1], B,feat, mu*D, niter,A)
    _, ans=convertToPermHungarian2new(row_ind,col_ind, n1, n2)
    list_of_nodes = []
    for el in ans: list_of_nodes.append(el[1])
    return ans, list_of_nodes, forbnorm    
iters=1
tun=[1,6,10,12,14]
tuns=["Alpine","REGAL","GradP","SlotaA","HTC"]
tun=[1]
tuns=["AlpineNF"]
nL=["testing"]
foldernames=['douban','allmv_tmdb','acm_dblp','fb_tw','ppi']
n_G2 = [1118,5712,9872,1043,1767] #s
n_G=[3906,6010,9916,1043,1767] #t
gt_size=[1118,5174,6325,1043,1767]
foldernames=['douban','allmv_tmdb','acm_dblp','ppi']
n_G2 = [1118,5712,9872,1767] #s
n_G=   [3906,6010,9916,1767] #t
gt_size=[1118,5174,6325,1767]
attrN=[True,True,True,True]


for k in range(0,len(foldernames)):
        #G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        G = read_real_graph(n = n_G[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_edge.txt')
        print(G)
        DGS=G.number_of_nodes()
    # Get the number of edges
        DGES = G.number_of_edges()       
        for _ in nL: 
        #for noiseL in nL: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}/{ptun}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}/{ptun}'
                file_A_results = open(f'{folder1}/Thesis_results.txt', 'w')
                file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                F2 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_feat.txt', dtype=float)  # shape: (n1, k)
                F1 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_feat.txt', dtype=float)  # shape: (n2, k)
                #diffs = B_feat[:, None, :] - A_feat[None, :, :]   # shape: (n1, n2, k)
                #X = np.abs(diffs).sum(axis=2)  # shape: (n1, n2)
                #Feat = np.linalg.norm(diffs, axis=2)  # shape: (n1, n2)
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
                #print(f'Size of subgraph: {n_Q}')
                
                F2=F2
                F1=F1
                

# Split into two arrays
                if (foldernames[k]=="douban"):
                    csv2 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr1.csv", header=None).iloc[:, 1:].to_numpy()
                    csv1 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr2.csv", header=None).iloc[:, 1:].to_numpy()
                    #two versions exists of douban attributes
                    #we chose the one which is harder
                for iter in range(iters):
                    if (foldernames[k]=="douban"):
                        F2=csv2
                        F1=csv1
                    #you have to do that because the features have ID making them 
                    #giving ground truth information
                    if (foldernames[k]=="acm_dblp"):
                        data = np.load(f'JOENA/datasets/ACM-DBLP_0.2.npz')
                        F2=data['x2']
                        F1=data['x1']

                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph.txt'
                    file_nodes = f'{folder_}/nodes.txt'
                    #Q_real = read_list(file_nodes)
                    #G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
                    G_Q = read_real_graph(n = n_G2[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_edge.txt')
                    pairs = []
                    with open(f'./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True.txt', "r") as f:
                        for line in f:
                            a, b = line.strip().split()
                            pairs.append((int(a), int(b)))
                    max_A = n_G2[k]
                    max_B = n_G2[k]
                    true1=False
                    true2=False
                    #if douban/dblp/fb_tw no+1 -allmv_tdmbwith +1
                    if (foldernames[k]=="allmv_tmdb"):
                        max_A=max_A+1
                        max_B=max_B+1
                    A_to_B = [-1] * (max_A)
                    for a, b in pairs:
                        if (a>=max_A):
                            true1=True
                        else:
                         A_to_B[a] = b
# 3️⃣ Build B→A mapping with -1 for missing
                    B_to_A = [-1] * (max_B)
                    for a, b in pairs:
                        if (b>=max_B):
                            true2=True
                        else:
                            B_to_A[b] = a
                    print(true1,true2)
                    QGS=G_Q.number_of_nodes()
                    QGES = G_Q.number_of_edges()
                    print(np.shape(F2))
                    print("G_Q",G_Q.number_of_edges())
                    print("G",G.number_of_edges())
                    start = time.time()
                    F1_c  = F1.copy()
                    F2_c  = F2.copy()

# Target number of rows
                    n_rows = G_Q.number_of_nodes()

# Pad F1 and F2
                    F1_c = pad_numpy_rows(F1_c, n_rows)
                    n_rows = G.number_of_nodes()
                    F2_c = pad_numpy_rows(F2_c, n_rows)
                    #compare_features(F1,F2,A_to_B)
                    #compare_features(F1,F2,B_to_A)
                    if(tun[ptun]==1):
                        print("Alpine")
                        mun=0.1
                        mun=0
                        _, list_of_nodes, forb_norm = AlpineL(G_Q.copy(), G.copy(),F1,F2,mun,weight=2)
                    elif(tun[ptun]==10):
                        print("GradAlignP")
                        list_of_nodes, forb_norm = gradPMain(G_Q.copy(), G.copy(),F1.copy(),F2.copy())
                    elif(tun[ptun]==6):
                            print("Regal")
                            _, list_of_nodes, forb_norm = RegalATT(G_Q.copy(), G.copy(),F1.copy(),F2.copy())   
                    elif(tun[ptun]==12):
                        forb_norm=1
                        print("SlotaAlign")
                        similarity = SlotaA(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),foldernames[k])
                        similarity=similarity.T #maybe PPI not? to check.
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])   
                    
                    elif tun[ptun] == 14:
                        forb_norm=1
                        print("HTC")
                        #if foldernames[k] in ["acm_dblp","ppi","cora"]:
                        #    print("in")
                        #    data_GT1 = data_GT[:, [1, 0]]  # swap columns
                         #   
                        #else:
                        #    data_GT1=data_GT
                        ratio=0 
                        data_GT1=None   
                        similarity = HTC_main(foldernames[k], ratio, data_GT1, f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_orca.txt', f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_orca.txt', src_laps_name=f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_laps.pth', trg_laps_name=f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_laps.pth')
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity.T
                        print('htc shape: ', similarity.shape)
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    
                    
                    else:
                        print("Error")
                        exit()
                    end = time.time()
                    subgraph = G.subgraph(list_of_nodes)
                    PGS=subgraph.number_of_nodes()
                    PGES = subgraph.number_of_edges()
                    isomorphic=False
                    if(forb_norm==0):
                        isomorphic=True
                    time_diff = end - start
                    file_nodes_pred = open(f'{folder1_}/{tuns[ptun]}.txt','w')
                    for node in list_of_nodes: file_nodes_pred.write(f'{node}\n')
                    spec_norm=0
                    #accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
                    accuracy1 = np.sum(np.array(A_to_B)==np.array(list_of_nodes))/gt_size[k]
                    accuracy2 = np.sum(np.array(B_to_A)==np.array(list_of_nodes))/gt_size[k]
                    print("ACC 1 or 2?",accuracy1,accuracy2)
                    print(np.sum(np.array(A_to_B)==np.array(list_of_nodes)))
                    accuracy=0
                    if ({foldernames[k]}=="douban" or{foldernames[k]}=="allmv_tmdb" ):
                        accuracy=accuracy2
                    else:
                        accuracy=accuracy1
                    with open("differences.txt", "w") as f:
                        f.write("Differences A_to_B:\n")
                        f.write("\n\nDifferences B_to_A:\n")
                        f.write("\n\nAccuracy A_to_B: {:.4f}\n".format(accuracy1))
                        f.write("Accuracy B_to_A: {:.4f}\n".format(accuracy2))
                    file_A_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy1} {accuracy2} {time_diff} {isomorphic}\n')
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,isomorphic)          
            print('\n')
        print('\n\n')

