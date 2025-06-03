import numpy as np
from pred import convex_initSM, align_SM, align_new, Alpine, Fugal
from help_functions import read_graph
import torch
import scipy
import networkx as nx
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
from help_functions import read_real_graph, read_list
import time
import os
from aa import generate_new_id,create_new_folder,get_max_previous_id 
from k_sol import kPMatch,Aeval,topkEval

os.environ["MKL_NUM_THREADS"] = "30"
torch.set_num_threads(30)

plotall = False

folderall = 'data3_topk'
foldernames = [ 'arenas','netscience', 'multimanga', 'highschool', 'voles']
n_G = [ 1133,379, 1004, 327, 712]
foldernames = [ 'netscience','highschool', 'multimanga', 'voles']
n_G = [ 379,327, 1004,712]
#foldernames = [  'netscience','highschool','multimanga', 'voles']
#n_G = [  379,327, 1004,712]
iters =5
#percs = [(i+1)/10 for i in range(0,9)]
percs=[0.1,0.2,0.3]
tuns=["Atorch","Atorch-k","Atorch-k1","Atorch-k3","Atorch-k5","Atorch-k7","Atorch-k9"]
tun=[2,3,4,5,6,7,8]
tuns=["Atorch-k","Atorch-k1","Atorch-k3"]
tun=[3,4,5]
tuns=["Alpine","Atorch-k"]
tun=[2,3]

def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Spec_norm:', spec_norm)
    print('----> Time:', time_diff)
    print('----> Isomorphic:', isomorphic)
    print()     

experimental_folder=f'./{folderall}/res/'
new_id = generate_new_id(get_max_previous_id(experimental_folder))
experimental_folder=f'./{folderall}/res/_{new_id}/'   
DGS=0
DGES=0
QGS=0
QGES=0
PGS=0
PGES=0         
for k in range(0,len(foldernames)):
        G_Max = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        DGS=G_Max.number_of_nodes()

# Get the number of edges
        DGES = G_Max.number_of_edges()
        
        #perc=percs[0]
        for perc in percs: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}/TK/{int(perc*100)}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}/{int(perc*100)}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}/{int(perc*100)}'
                file_A_results = open(f'{folder1}/SizeTest_results{tuns[ptun]}.txt', 'w')
                file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                n_Q = int(perc*G_Max.number_of_nodes())

                print(f'Size of subgraph: {n_Q}')
                for iter in range(iters):
                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    #folder_=foldernames1[k]
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph_KG.txt'
                    file_nodes = f'{folder_}/nodes_KG.txt'
                    Q_real = read_list(file_nodes)
                    Q_1=read_list(f'{folder_}/nodes_G1.txt')
                    Q_2=read_list(f'{folder_}/nodes_G2.txt')
                    intersection_Qreal_Q1 = set(Q_real) & set(Q_1)
                    intersection_Qreal_Q2 = set(Q_real) & set(Q_2)
                    intersection_Q1_Q2 = set(Q_1) & set(Q_2)
                    intersection_all = set(Q_real) & set(Q_1) & set(Q_2)

# Print lengths
                    print("Intersection of Q_real and Q_1:", len(intersection_Qreal_Q1))
                    print("Intersection of Q_real and Q_2:", len(intersection_Qreal_Q2))
                    print("Intersection of Q_1 and Q_2:", len(intersection_Q1_Q2))
                    print("Intersection of all three:", len(intersection_all))
                    print(f'Reading subgraph at {file_subgraph}')
                    print(f'Reading alignment at {file_nodes}')
                    node_colors = []
                    for node in G_Max.nodes():
                        if node in Q_real:
                            node_colors.append('red')  # Common nodes
                        elif node in Q_1:
                            node_colors.append('blue')  # Nodes in Q1
                        elif node in Q_2:
                            node_colors.append('green')  # Nodes in Q2
                        else:
                            node_colors.append('gray')  # Other nodes
# Draw the graph        
#                           
                    plt.figure(figsize=(8, 6))
                    pos = nx.spring_layout(G_Max, seed=42)  # Layout for better visualization

                    nx.draw(G_Max, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=300, font_size=8)

# Show the plot
                    plt.title("Graph Visualization with Q_real (Red), Q_1 (Blue), Q_2 (Green)")
                    plt.show() 

                    sys.exit()   
            print('\n')
        print('\n\n')

