import networkx as nx
import os
import sys
import numpy as np
import time
import torch
from resultsfolder import generate_new_id,create_new_folder,get_max_previous_id 
import json
from subgraph import create_sub_graph_ground_truth
import matplotlib.pyplot as plt
from itertools import product
import pickle
from ssSGM import ssSGM_simulation
import copy
from pred import AlpineTopk,AlpineTorchTopk,AlpinePlusTopk,Martian
from hung_utils import EVAL_new
os.environ["MKL_NUM_THREADS"] = "36"
np.random.seed(1)
torch.set_num_threads(36)
# from openpyxl import Workbook
from save_util import save_pairs_to_txt,save_high_similarity_pairs,save_list_to_txt,save_matrix_to_excel


def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False,local=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Localization:', local)
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

ksizes=[0.6, 0.7, 0.8, 0.9]
ksizes=[0.6]
algorithms_name = {
    2: 'Alpine',
    3: 'Alpine++',
    4: 'Martian',
    8: 'ssSGM'
}
choice_name = {
    0: 'uniform',
}

ptune = [2,3,4,8]
algorithms=["Penalty"]
folderall = 'data3_topk'
experimental_folder=f'./{folderall}/res/'
new_id = generate_new_id(get_max_previous_id(experimental_folder))
experimental_folder=f'./{folderall}/res/_{new_id}'
os.makedirs(f'{experimental_folder}', exist_ok=True)

beta=1.0
uniform_only=True

file_A_results = open(f'{experimental_folder}/results.txt', 'a')
file_A_results.write(f'G1,G2,K,Algo,FROB,FROB1,FROB2,FROB3,FROB4,FROB5,HR,HR1,HR2,HR3,HR4,HR5,TIME,CRP,CRP1,CRP2,CRP3,CRP4,CRP5,trial,local_g1,local_g2,local_g1_1,local_g2_1,local_g1_2,local_g2_2,local_g1_3,local_g2_3,local_g1_4,local_g2_4,local_g1_5,local_g2_5,initialization\n')        
file_A_results.close()
for i in range(len(graph_name_and_size) - 1):
    for j in range(i + 1, len(graph_name_and_size)):
        (name_G_A, size_G_A) = graph_name_and_size[j]
        (name_G_B, size_G_B) = graph_name_and_size[i]
        GAname=name_G_A
        GBname=name_G_B        
        G = graphs[name_G_B]
        G_Q = graphs[name_G_A]
        SG=G_Q.number_of_nodes()
        start1 = time.time()
        n_Q=SG
        for k, trial in product(ksizes, range(1)):
            n_Q=int(k*SG)
            if os.path.exists(f'./data3_topk/synthetic_data_martian/normal/{GAname}vs{GBname}_{k}_{trial}.json'):
                print('True')
                with open(f'./data3_topk/synthetic_data_martian/normal/{GAname}vs{GBname}_{k}_{trial}.json', 'r') as fin_graph:
                    data_syn = json.load(fin_graph)
                    gt_G1 = data_syn['ordered_selected_nodes_G1']
                    gt_G2 = data_syn['ordered_selected_nodes_G2']
                    new_G, new_G_Q, info = create_sub_graph_ground_truth(G, G_Q, n_Q, gt_G1=gt_G1, gt_G2=gt_G2)
            else:

                new_G, new_G_Q, info = create_sub_graph_ground_truth(G, G_Q, n_Q)
                gt_G1 = info['ordered_selected_nodes_G1']
                gt_G2 = info['ordered_selected_nodes_G2']
                print(n_Q)
                
                with open(f'./data3_topk/synthetic_data_martian/normal/{GAname}vs{GBname}_{k}_{trial}.pkl', 'wb') as fout_graph:
                    data_syn = {
                        'G1': new_G,
                        'G2': new_G_Q,
                        'gt_G1': gt_G1,
                        'gt_G2': gt_G2,
                    }
                    pickle.dump(data_syn, fout_graph)
            
            for (tun, choice_init) in product(ptune, range(10)):
                P_init = None
                start = time.time()
                if(tun==1):
                    print("Alpine")
                    start1 = time.time()
                    P,P1, list_of_nodes = AlpineTopk(new_G.copy(),new_G_Q.copy(),n_Q,t1=10)
                    end1 = time.time()
                elif(tun==2):
                    print("Alpine")
                    start1 = time.time()
                    P,P1, list_of_nodes  = P,P1, list_of_nodes = AlpineTorchTopk(new_G.copy(),new_G_Q.copy(),n_Q,t1=10)
                    end1 = time.time()
                    AlpTime=end1-start1
                elif(tun==3):
                    print("Alpine+")
                    P,P1, list_of_nodes  = AlpinePlusTopk(new_G.copy(),new_G_Q.copy(),n_Q,t1=10)
                elif(tun==4):
                    print("Martian")
                    P,P1, list_of_nodes  = Martian(new_G.copy(),new_G_Q.copy(),n_Q,t1=10)
                elif(tun==8):
                    print("ssSGM")
                    P = ssSGM_simulation(nx.to_numpy_array(new_G_Q), nx.to_numpy_array(new_G), 0, n_Q, 30, 0.001)
                else:
                    print("NO given algorithm ID")
                    exit()   
                end=time.time()
                isomorphic=False
                if (tun==2):
                    time_diff=AlpTime                
                else:
                    time_diff = end - start  
                print("k size", n_Q)
                if (tun in [2,3,4,5,6,7]):
                    forb_norm,hr,correct_pairs,local_1,local_2=EVAL_new(new_G.copy(),new_G_Q.copy(),P,n_Q,Hung=False,initial_node=-1, gt_G1=gt_G1, gt_G2=gt_G2)
                elif (tun==8):
                    forb_norm,hr,correct_pairs,local_1,local_2=EVAL_new(new_G.copy(),new_G_Q.copy(),P,n_Q,Hung=True,initial_node=-1, gt_G1=gt_G1, gt_G2=gt_G2)
                file_A_results = open(f'{experimental_folder}/results.txt', 'a')
                file_A_results.write(f'{name_G_A},{name_G_B},{n_Q},{algorithms_name[tun]},{forb_norm},{0},{0},{0},{0},{0},{hr},{0},{0},{0},{0},{0},{time_diff},{correct_pairs},{0},{0},{0},{0},{0},{trial},{local_1},{local_2},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{choice_name[choice_init]}\n') 
                file_A_results.close()
                time.sleep(2)    
        print('\n')
    print('\n\n')

    