from sinkhorn import greenkhorn,sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
import numpy as np
import math
import torch
from numpy import linalg as LA
from pred import convertToPermHungarian2new,AlpineL
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

iters=5
tun=[1,6,10,12,14]
tuns=["Alpine","REGAL","GradP","SlotaA","HTC"]
nL=["testing"]
foldernames=['douban','allmv_tmdb','acm_dblp','fb_tw','ppi']
n_G2 = [1118,5712,9872,1043,1767] #s
n_G=[3906,6010,9916,1043,1767] #t
gt_size=[1118,5174,6325,1043,1767]
attrN=[True,True,True,True,True]
seed=1

for k in range(0,len(foldernames)):
        G = read_real_graph(n = n_G[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_edge.txt')
        print(G)
        DGS=G.number_of_nodes()
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
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
                    #two versions exists of douban attributes
                    #we chose the one which is harder
                for iter in range(iters):
                    #you have to do that because the features have ID making them 
                    #giving ground truth information

                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph.txt'
                    file_nodes = f'{folder_}/nodes.txt'
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
                    #if -allmv_tdmbwith +1
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
                    start = time.time()
                    F1_c  = F1.copy()
                    F2_c  = F2.copy()
                    n_rows = G_Q.number_of_nodes()
                    F1_c = pad_numpy_rows(F1_c, n_rows)
                    n_rows = G.number_of_nodes()
                    F2_c = pad_numpy_rows(F2_c, n_rows)
                    if(tun[ptun]==1):
                        print("Alpine")
                        mun=0.1
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
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])   
                    
                    elif tun[ptun] == 14:
                        forb_norm=1
                        print("HTC")
                        ratio=0 
                        data_GT1=None   
                        similarity = HTC_main(foldernames[k], ratio, data_GT1, f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_orca.txt', f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_orca.txt', src_laps_name=f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_laps.pth', trg_laps_name=f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_laps.pth')
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"or foldernames[k]=="acm_dblp"):
                            similarity=similarity.T
                        P2, row_ind, col_ind = PermHungarian(similarity)
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

                    time_diff = end - start
                    file_nodes_pred = open(f'{folder1_}/{tuns[ptun]}.txt','w')
                    for node in list_of_nodes: file_nodes_pred.write(f'{node}\n')
                    spec_norm=0
                    accuracy1 = np.sum(np.array(A_to_B)==np.array(list_of_nodes))/gt_size[k]
                    accuracy2 = np.sum(np.array(B_to_A)==np.array(list_of_nodes))/gt_size[k]
                    #print("ACC 1 or 2?",accuracy1,accuracy2)
                    accuracy=max(accuracy1,accuracy2)
                    
                    with open("differences.txt", "w") as f:
                        f.write("Differences A_to_B:\n")
                        f.write("\n\nDifferences B_to_A:\n")
                        f.write("\n\nAccuracy A_to_B: {:.4f}\n".format(accuracy1))
                        f.write("Accuracy B_to_A: {:.4f}\n".format(accuracy2))
                    file_A_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy1} {accuracy2} {time_diff} {isomorphic}\n')
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,False)          
            print('\n')
        print('\n\n')

