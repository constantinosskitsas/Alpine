from itertools import product
import pathlib
import networkx as nx
import os
import sys
import numpy as np
import time
import torch
from resultsfolder import generate_new_id,create_new_folder,get_max_previous_id 
from scipy.optimize import linear_sum_assignment
from ssSGM import ssSGM_simulation
os.environ["MKL_NUM_THREADS"] = "36"
torch.set_num_threads(36)
from hung_utils import PermHungarian,EVAL_new_diff
from save_util import save_pairs_to_txt,save_high_similarity_pairs,save_list_to_txt,save_matrix_to_excel
from pred import AlpineTopk,AlpineTorchTopk,AlpinePlusTopk,Martian

def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Spec_norm:', spec_norm)
    print('----> Time:', time_diff)
    print('----> Isomorphic:', isomorphic)
    print()  

def readGraph(filename, N):
    file1 = open(filename)
    Lines = file1.readlines()
    count = 0
    for line in Lines:
        if count == 0:
            g = nx.Graph()
            for i in range(N):
                g.add_node(i)
        else:
            edge = list(map(float, list(line[0:-1].split(",")[0:2])))
            if (edge[0] <= (N - 1)) and (edge[1] <= (N - 1)):
                g.add_edge(edge[0], edge[1])
        count += 1
    return g


    #from larger to smaller
graph_name_and_size = [
    ("celegans", 453),
    ("village", 372),
    ("malaria", 307),
    ("jazz", 198),
    ("football", 115)
]


graphs = {
    name: readGraph(f"raw_data/{name}/edges.csv", N=size)
    for (name, size) in graph_name_and_size
}

algorithms_name = {
    2: 'Alpine',
    3: 'Alpine++',
    4: 'Martian',
    8: 'ssSGM'
}

ksizes=[0.6,0.7,0.8,0.9]
ksizes=[0.9]
algorithms=["Alpine","TOP-K","AlpineLP","TOP-KLP"]
ptune=[2,3,4,6,7,5]
ptune=[4]
ptune=[2, 3, 4,8]
algorithms=["Penalty"]
folderall = 'data3_topk'
experimental_folder=f'./{folderall}/res/'
new_id = generate_new_id(get_max_previous_id(experimental_folder))
experimental_folder=f'./{folderall}/res/_{new_id}'
pathlib.Path(experimental_folder).mkdir(parents=True, exist_ok=True)
n_run = 1
for i in range(len(graph_name_and_size) - 1):
    for j in range(i + 1, len(graph_name_and_size)):
        (name_G_A, size_G_A) = graph_name_and_size[j]
        (name_G_B, size_G_B) = graph_name_and_size[i]
        GAname=name_G_A
        GBname=name_G_B
        file_A_results = open(f'{experimental_folder}/{GAname}vs{GBname}.txt', 'a')
        file_A_results.write(f'G1 G2 K Algo FROB FROB1 FROB2 FROB3 FROB4 FROB5 HR HR1 HR2 HR3 HR4 HR5 TIME n_component_G1 n_component_G2\n')        
        file_A_results.close()
        G = graphs[name_G_B].copy()
        G_Q = graphs[name_G_A].copy()
        SG=G_Q.number_of_nodes()
        SG1=G.number_of_nodes()
        start1 = time.time()
        n_Q=SG
        print(nx.is_connected(G))
        print(nx.is_connected(G_Q))
        for k, _ in product(ksizes, range(n_run)):
            n_Q=int(k*SG)
            for tun in ptune:
                start = time.time()
                if(tun==1):
                    print("Alpine")
                    P,P1, list_of_nodes = AlpineTopk(G.copy(),G_Q.copy(),n_Q,t1=10)
                elif(tun==2):
                    start1 = time.time()
                    P,P1, list_of_nodes  = P,P1, list_of_nodes = AlpineTorchTopk(G.copy(),G_Q.copy(),n_Q,t1=10)
                    end1 = time.time()
                    AlpTime=end1-start1
                elif(tun==3):
                    print("k-Size")
                    P,P1, list_of_nodes  = AlpinePlusTopk(G.copy(),G_Q.copy(),n_Q,t1=10)
                elif(tun==4):
                    print("k-Penalty")
                    P,P1, list_of_nodes  = Martian(G.copy(),G_Q.copy(),n_Q,t1=10)
                elif(tun==8):
                    print("ssSGM")
                    P = ssSGM_simulation(nx.to_numpy_array(G_Q.copy()), nx.to_numpy_array(G.copy()), 0, n_Q, 30, 0.001)
                else:
                    print("NO given algorithm ID")
                    exit()   
                end=time.time()
                isomorphic=False
                time_diff = end - start  
                print("k size", n_Q)
                num_components_A, num_components_B = 0, 0
                if (tun==8):
                    print(type(G), type(G_Q), type(P))
                    forb_norm,hr,num_components_A,num_components_B=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=True,initial_node=-1, return_n_components=True)
                    forb_norm1,hr1 = 0, 0
                    forb_norm2,hr2 = 0, 0
                    forb_norm3,hr3 = 0, 0
                    forb_norm4,hr4 = 0, 0
                    forb_norm5,hr5 = 0, 0
                else:
                    forb_norm,hr=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=-1)
                    forb_norm1,hr1=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=0)
                    forb_norm2,hr2=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=1)
                    forb_norm3,hr3=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=2)
                    forb_norm4,hr4=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=3)
                    forb_norm5,hr5=EVAL_new_diff(G.copy(),G_Q.copy(),P,n_Q,Hung=False,initial_node=4)

                print(forb_norm,hr,forb_norm1,hr1,forb_norm2,hr2,forb_norm3,hr3,forb_norm4,hr4)
                file_A_results = open(f'{experimental_folder}/{GAname}vs{GBname}.txt', 'a')
                file_A_results.write(f'{name_G_A} {name_G_B} {n_Q} {algorithms_name[tun]} {forb_norm}  {forb_norm1}  {forb_norm2}  {forb_norm3}  {forb_norm4} {forb_norm5} {hr} {hr1} {hr2} {hr3} {hr4} {hr5} {time_diff} {num_components_A} {num_components_B}\n') 
                file_A_results.close()
                printR(tun,forb_norm,0,0,time_diff,isomorphic)   
                time.sleep(2)     
        print('\n')
    print('\n\n')

    