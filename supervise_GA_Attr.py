import networkx as nx
import torch
import numpy as np
from numpy import linalg as LA
import ot
import scipy
import time
import os
import pandas as pd
from help_functions import read_real_graph, read_list
from resultsfolder import generate_new_id,get_max_previous_id 
from JOENA.Joena_main import JOENA
from SlotaAlign.SlotaAlign_main import SlotaA
from NextAlign.NextA import NextAlign
from Parrot.parrot_main import Parrot
from REGAL.regal import RegalATT
from pred import Alpine_supervised
from GradP.gradp import gradPMain
from superviseutil import synchronize_features,seed_link,synchronize_graphs
from hung_utils import PermHungarian,convertToPermHungarian2new,convertToPermHungarian
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
def load_permutation(run_index, folder="Data/data/phone"):
    """
    Loads a single permutation by index.
    """
    filename = os.path.join(folder, f"P_run{run_index}.txt")
    P = np.loadtxt(filename, dtype=int)
    return P



iters=5
tun=[1,10,11,13,14]
tuns=["Alpine","GradP","Joena","NextAlign","Parrot"]
nL=["testing"]
foldernames=['douban','allmv_tmdb','acm_dblp','fb_tw','ppi','cora','foursquare','phone']
n_G2 = [1118,5712,9872,1043,1767,2708,5120,1000] #s
n_G=   [3906,6010,9916,1043,1767,2708,5313,1003] #t
gt_size=[1118,5174,6325,1043,1767,2708,1609,1000]
attrN=[True,True,True,False,True,True,False,False]
ratioN=[0.05,0.10,0.15,0.20]
ratio=0.05

Perm=None
for k in range(0,len(foldernames)):
        G = read_real_graph(n = n_G[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_edge.txt')
        print(G)
        DGS=G.number_of_nodes()
        DGES = G.number_of_edges()       
        for ratio in ratioN: 
        #for noiseL in nL: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}/{ptun}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}/{ptun}'
                file_A_results = open(f'{folder1}/Thesis_results.txt', 'w')
                file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
                if (foldernames[k]=="foursquare" or foldernames[k]=="phone"or foldernames[k]=="fb_tw"):
                        F1=np.zeros((n_G2[k],1))
                        F2=np.zeros((n_G[k],1))
                else:
                    F2 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_feat.txt', dtype=float)  # shape: (n1, k)
                    F1 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_feat.txt', dtype=float)  # shape: (n2, k)
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
                F2=F2
                F1=F1                
                if (foldernames[k]=="douban"):
                    csv2 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr1.csv", header=None).iloc[:, 1:].to_numpy()
                    csv1 = pd.read_csv(f"./Data/Full-dataset/attribute/{foldernames[k]}attr2.csv", header=None).iloc[:, 1:].to_numpy()
                if (foldernames[k]=="acm_dblp"):
                    data = np.load(f'JOENA/datasets/ACM-DBLP_0.2.npz')
                    csv2=data['x2']
                    csv1=data['x1']
                for iter in range(iters):
                    
                    if (foldernames[k]=="douban" or foldernames[k]=="acm_dblp"):
                        F2=csv2
                        F1=csv1
                    #you have to do that because the features have ID making them 
                    #giving ground truth information
                    if (foldernames[k]=="fb_tw"):
                        F2=F2*0
                        F1=F1*0
                    data_GT = np.loadtxt(f"./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True_{ratio}_{iter}.txt", dtype=int)
                    #F2=csv2
                    #F1=csv1
                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph.txt'
                    file_nodes = f'{folder_}/nodes.txt'
                    G_Q = read_real_graph(n = n_G2[k], name_ = f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_edge.txt')
                    if (foldernames[k]=="phone"):                      
                        Perm=  load_permutation(iter)
                        A1_perm=nx.to_numpy_array(G_Q, dtype=int)
                        A_perm = A1_perm[Perm][:, Perm]
                        G_Q = nx.from_numpy_array(A_perm)
                    pairs = []
                    if (foldernames[k]=="phone"):
                        with open(f'./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True.txt', "r") as f:
                            for line in f:
                                a, b = line.strip().split()
                                a = int(a)       # source node
                                b = int(b)       # target node (original)
                                a_perm = Perm[a]
                                pairs.append((a_perm, b))
                    else:
                        with open(f'./Data/data/{foldernames[k]}/{foldernames[k]}_ground_True.txt', "r") as f:
                            for line in f:
                                a, b = line.strip().split()
                                pairs.append((int(a), int(b)))
# ✅ Find the max node ID in each graph
                    max_A = n_G2[k]#max(a for a, b in pairs)
                    max_B = n_G2[k]#max(b for a, b in pairs)

# 2️⃣ Build A→B mapping with -1 for missing
                    true1=False
                    true2=False
                    #if allmv_tmdbwith +1
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
                    anchors_G = data_GT[:, 1].tolist()
                    anchors_GQ = data_GT[:, 0].tolist()

                    if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb" or foldernames[k]=="foursquare"  ):
                        anchors_G = data_GT[:, 0].tolist()
                        anchors_GQ = data_GT[:, 1].tolist()
                    if(foldernames[k]=="phone"):
                        anchors_G = [Perm[i] for i in data_GT[:, 0]]
                        anchors_GQ = data_GT[:, 1].tolist()
                    G1=G
                    G1_Q=G_Q
                    print("Size of anchors_G:", len(anchors_G))
                    print("Size of anchors_GQ:", len(anchors_GQ))
                    if (foldernames[k]!="foursquare" and foldernames[k]!="phone"and foldernames[k]!="fb_tw"):
                        
                        F2_n,F1_n=synchronize_features(F2,F1,anchors_G,anchors_GQ)
                    else:
                        F2_n=F2
                        F1_n=F1
                    start = time.time()
                    if(tun[ptun]==1):
                        print("Alpine")
                        mun=0.1                       
                        if(attrN[k]==False and foldernames[k]!="foursquare"):
                            mun=1
                            if (foldernames[k]=="phone"):
                                mun=8
                        G1,G1_Q=seed_link(anchors_G,anchors_GQ,G,G_Q)
                        _, list_of_nodes, forb_norm = Alpine_supervised(G1_Q.copy(), G1.copy(),F1_n,F2_n,anchors_GQ,anchors_G,mun,weight=2)                    
                    elif(tun[ptun]==6):
                            print("Regal")
                            _, list_of_nodes, forb_norm = RegalATT(G_Q.copy(), G.copy(),F1_n,F2_n)      
                    elif(tun[ptun]==10):
                        print("GradAlignP")
                        list_of_nodes, forb_norm = gradPMain(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),anchors_GQ=anchors_GQ,anchors_G=anchors_G)
                    elif(tun[ptun]==11):
                        forb_norm=1
                        print("JOENA")

                        if foldernames[k] in ["acm_dblp","ppi","cora","phone"]:
                            print("in")
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                            #data_GT1=data_GT
                            
                        else:
                            data_GT1=data_GT
                        if foldernames[k] in ["phone"]:
                            data_GT1[:, 1] = Perm[data_GT1[:, 1]]     
                        similarity = JOENA(foldernames[k],ratio,attrN[k],data_GT1,Perm)
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity.T
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    
                    elif(tun[ptun]==12):
                        forb_norm=1
                        print("SlotaAlign")
                        similarity = SlotaA(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),foldernames[k])
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    elif(tun[ptun]==13):
                        forb_norm=1
                        print("NextAlign")
                        if foldernames[k] not in ["acm_dblp","ppi","cora"]:
                            print("in")
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                            #data_GT1=data_GT                            
                        else:
                            data_GT1=data_GT
                        if foldernames[k] in ["phone"]:
                            data_GT1[:, 0] = Perm[data_GT1[:, 0]]         
                        similarity = NextAlign(foldernames[k],attrN[k],data_GT1,Perm)
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity                        
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    elif(tun[ptun]==14):
                        forb_norm=1
                        print("Parrot")
                        if foldernames[k] in ["acm_dblp","ppi","cora"]:
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                        else:
                            data_GT1=data_GT
                            if foldernames[k] in ["phone"]:
                                data_GT1[:, 0] = Perm[data_GT1[:, 0]] 
                        similarity = Parrot(foldernames[k],G1_Q.copy(),G1.copy(),F1_n.copy(),F2_n.copy() ,data_GT1)
                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity                
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])                    
                    
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
                    accuracy1 = np.sum(np.array(A_to_B)==np.array(list_of_nodes))/gt_size[k]
                    accuracy2 = np.sum(np.array(B_to_A)==np.array(list_of_nodes))/gt_size[k]
                    accuracy=0
                    if ({foldernames[k]}=="douban" or{foldernames[k]}=="allmv_tmdb" ):
                        accuracy=accuracy2
                    else:
                        accuracy=accuracy1
                    print("ACC 1 or 2?",accuracy1,accuracy2)
                    print(np.sum(np.array(A_to_B)==np.array(list_of_nodes)))
                    with open("differences.txt", "w") as f:
                        f.write("Differences A_to_B:\n")
                        f.write("\n\nDifferences B_to_A:\n")
                        f.write("\n\nAccuracy A_to_B: {:.4f}\n".format(accuracy1))
                        f.write("Accuracy B_to_A: {:.4f}\n".format(accuracy2))
                    file_A_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy1} {accuracy2} {time_diff} {isomorphic}\n')
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,isomorphic)          
            print('\n')
        print('\n\n')

