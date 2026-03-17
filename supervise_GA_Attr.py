import networkx as nx
import torch
import numpy as np
from pred import convertToPermHungarian,Alpine_supervised
from pred import convertToPermHungarian2new
import ot
import scipy
from numpy import linalg as LA
from GradP.gradp import gradPMain
import time
import os
from help_functions import read_graph
from help_functions import read_real_graph, read_list
from resultsfolder import generate_new_id,create_new_folder,get_max_previous_id 
import pandas as pd
from JOENA.Joena_main import JOENA
from SlotaAlign.SlotaAlign_main import SlotaA
from NextAlign.NextA import NextAlign
from Parrot.parrot_main import Parrot
from REGAL.regal import RegalATT
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
def PermHungarian(M):

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    return _ ,row_ind,col_ind
def synchronize_features(F1, F2, anchors_G, anchors_G_Q, direction="F1_to_F2"):
    """
    Synchronize features along ground truth anchors.
    
    Parameters
    ----------
    F1 : np.ndarray, shape (n1, k)
        Feature matrix of graph G
    F2 : np.ndarray, shape (n2, k)
        Feature matrix of graph G_Q
    anchors_G : list[int]
        Anchor node indices in G (for F1)
    anchors_G_Q : list[int]
        Corresponding anchor node indices in G_Q (for F2)
    direction : str
        Either "F1_to_F2" or "F2_to_F1" indicating copy direction
    
    Returns
    -------
    F1_new, F2_new : np.ndarray
        Feature matrices after synchronization
    """
    # Make copies
    F1_new = F1.copy()
    F2_new = F2.copy()

    if direction == "F1_to_F2":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F2_new[u_q] = F1_new[u]
    elif direction == "F2_to_F1":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F1_new[u] = F2_new[u_q]
    else:
        raise ValueError("direction must be 'F1_to_F2' or 'F2_to_F1'")
    
    return F1_new, F2_new
  
def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end='\t')
    return G1, G2

def synchronize_graphs(G, G_Q, anchors_G, anchors_G_Q):
    """
    Create new graphs G1 and G1_Q where edges between aligned anchors
    are synchronized across G and G_Q.
    
    Parameters
    ----------
    G : nx.Graph
        Original graph G
    G_Q : nx.Graph
        Original graph G_Q
    anchors_G : list[int]
        List of nodes in G (anchors)
    anchors_G_Q : list[int]
        List of corresponding nodes in G_Q (same length as anchors_G)
    
    Returns
    -------
    G1, G1_Q : nx.Graph
        Synchronized graphs
    """

    # Build mapping dicts between anchors
    G_to_GQ = dict(zip(anchors_G, anchors_G_Q))
    GQ_to_G = dict(zip(anchors_G_Q, anchors_G))

    # Start with copies of the original graphs
    G1 = G.copy()
    G1_Q = G_Q.copy()
    counter=0
    counter1=0
    anchor_nodes = set(G_to_GQ.keys())
    anchor_edges = [(u, v) for u, v in G.edges() if u in anchor_nodes and v in anchor_nodes]
    print("Number of edges among anchors in G:", len(anchor_edges))

    # --- Step 1: Transfer edges from G to G_Q ---
    for u, v in G.edges():
        if u in G_to_GQ and v in G_to_GQ:
            u_q, v_q = G_to_GQ[u], G_to_GQ[v]
            if not G1_Q.has_edge(u_q, v_q):
                G1_Q.add_edge(u_q, v_q)
                counter=counter+1
            else:
                counter1=counter1+1

    # --- Step 2: Transfer edges from G_Q to G ---
    for u_q, v_q in G_Q.edges():
        if u_q in GQ_to_G and v_q in GQ_to_G:
            u, v = GQ_to_G[u_q], GQ_to_G[v_q]
            if not G1.has_edge(u, v):
                G1.add_edge(u, v)
                counter=counter+1
            else:
                counter1=counter1+1
    return G1, G1_Q



iters=5
tun=[1,10,11,12]
tuns=["Alpine","Regal","GradP","JOENA","SlotAlign","NextAlign","Parrot"]
#SUperivtuns=["Alpine",","GradP","JOENA",","NextAlign","Parrot"]
#Supervitun=[1,10,11,13,14]

tun=[1,10,11,13,14]
tuns=["Alpine","GradP","Joena","NextAlign","Parrot"]
nL=["testing"]
foldernames=['douban','allmv_tmdb','acm_dblp','fb_tw','ppi','cora','foursquare','phone']
n_G2 = [1118,5712,9872,1043,1767,2708,5120,1000] #s
n_G=   [3906,6010,9916,1043,1767,2708,5313,1003] #t
gt_size=[1118,5174,6325,1043,1767,2708,1609,1000]
attrN=[True,True,True,False,True,True,False,False]


ratio=0.05

#douban
#foldernames=['ppi']
#n_G2 = [1767] #s
#n_G=[1767] #t
#gt_size=[1767]
tun=[1]
tuns=["Alpine"]
#attrN=[True]
Perm=None
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
                
                if (foldernames[k]=="foursquare" or foldernames[k]=="phone"or foldernames[k]=="fb_tw"):
                        F1=np.zeros((n_G2[k],1))
                        F2=np.zeros((n_G[k],1))
                else:
                    F2 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_t_feat.txt', dtype=float)  # shape: (n1, k)
                    F1 = np.loadtxt(f'./Data/data/{foldernames[k]}/{foldernames[k]}_s_feat.txt', dtype=float)  # shape: (n2, k)
                #diffs = B_feat[:, None, :] - A_feat[None, :, :]   # shape: (n1, n2, k)
                #X = np.abs(diffs).sum(axis=2)  # shape: (n1, n2)
                #Feat = np.linalg.norm(diffs, axis=2)  # shape: (n1, n2)
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
                F2=F2
                F1=F1
                

# Split into two arrays
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
                    #Q_real = read_list(file_nodes)
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
                    #if douban/dblp/fb_tw no+1 -allmv_tmdbwith +1
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
                    #A = nx.adjacency_matrix(G_Q).todense()
                    QGS=G_Q.number_of_nodes()
                    QGES = G_Q.number_of_edges()
                    anchors_G = data_GT[:, 1].tolist()
                    anchors_GQ = data_GT[:, 0].tolist()
                    #G1,G1_Q=synchronize_graphs(G,G_Q,anchors_G,anchors_GQ)
                    print(np.shape(F2))
                    print("G_Q",G_Q.number_of_edges())
                    print("G",G.number_of_edges())
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
                        #F2_n*=0
                    #F1_n*=0
                    start = time.time()
                    #compare_features(F1,F2,A_to_B)
                    #compare_features(F1,F2,B_to_A)
                    if(tun[ptun]==1):
                        print("Alpine")
                        mun=0.1
                        #mun=0
                        #if(foldernames[k]=="fb_tw" or foldernames[k]=="foursquare" or foldernames[k]=="phone"):
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
                        #cora,douban,fb-tw,phone,allmv_tmdb,ppi  works
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    
                    elif(tun[ptun]==12):
                        forb_norm=1
                        print("SlotaAlign")
                        similarity = SlotaA(G_Q.copy(), G.copy(),F1.copy(),F2.copy(),foldernames[k])
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        #P2, row_ind, col_ind = convertToPermHungarian(similarity, QGS, n_G[k])
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
                        
                        #cora,douban,fb-tw,phone,allmv_tmdb,ppi  works
                        P2, row_ind, col_ind = PermHungarian(similarity)
                        _, ans=convertToPermHungarian2new(row_ind,col_ind, QGS, n_G[k])
                        list_of_nodes = []
                        for el in ans: list_of_nodes.append(el[1])
                    elif(tun[ptun]==14):
                        forb_norm=1
                        print("Parrot")
                        if foldernames[k] in ["acm_dblp","ppi","cora"]:
                            print("in")
                            data_GT1 = data_GT[:, [1, 0]]  # swap columns
                        else:
                            data_GT1=data_GT
                            if foldernames[k] in ["phone"]:
                                data_GT1[:, 0] = Perm[data_GT1[:, 0]] 
                        similarity = Parrot(foldernames[k],G1_Q.copy(),G1.copy(),F1_n.copy(),F2_n.copy() ,data_GT1)
                        #similarity = Parrot(foldernames[k],G1.copy(),G1_Q.copy(),F2_n,F1_n,data_GT1)

                        if (foldernames[k]=="douban" or foldernames[k]=="allmv_tmdb"or foldernames[k]=="foursquare"or foldernames[k]=="cora"or foldernames[k]=="phone"):
                            similarity=similarity
                        
                        #cora,douban,fb-tw,phone,allmv_tmdb,ppi  works
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

