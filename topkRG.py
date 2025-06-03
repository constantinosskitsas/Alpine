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
percs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
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
G_1 = read_real_graph(n = n_G[2], name_ = f'./raw_data/{foldernames[2]}.txt')
G_2 = read_real_graph(n = n_G[3], name_ = f'./raw_data/{foldernames[3]}.txt')
DGS=G_1.number_of_nodes()

# Get the number of edges
DGES = G_1.number_of_edges()
        
        #perc=percs[0]
for perc in percs: 
    for ptun in range(len(tun)): 
            os.makedirs(f'{experimental_folder}{foldernames[0]}/{int(perc*100)}', exist_ok=True)
            folder1=f'./{experimental_folder}/{foldernames[0]}/{int(perc*100)}'
            file_A_results = open(f'{folder1}/SizeTest_results{tuns[ptun]}.txt', 'w')
            file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
            file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
            n_Q = int(perc*G_2.number_of_nodes())

# Get the number of edges
            start = time.time()
            if(tun[ptun]==1):
                print("Alpine")
                _, list_of_nodes, forb_norm = Alpine(G_1.copy(), G_2.copy(),mu=1,weight=2)
            elif(tun[ptun]==2):
                print("Alpine-Torch")
                P,P1, list_of_nodes  = kPMatch(G_1.copy(),G_2.copy(),n_Q,t1=1)
            elif(tun[ptun]==3):
                print("k-Size")
                P,P1, list_of_nodes  = kPMatch(G_1.copy(),G_2.copy(),n_Q,t1=3)
            elif(tun[ptun]==4):
                print("k-Size")
                P,P1, list_of_nodes  = kPMatch(G_1.copy(),G_2.copy(), n_Q,t1=2)
                    
            else:
                print("NO given algorithm ID")
                exit()
            end = time.time()

            isomorphic=False                    
            time_diff = end - start

            forb_norm1=[-100,-100,-100,-100,-100]
                        #forb_norm=Aeval(G.copy(),G_Q1.copy(),P,G_Q.number_of_nodes())
            forb_norm,forb_norm1[0],forb_norm1[1]=topkEval(G_2.copy(),G_1.copy(),G_2.copy(),P,n_Q)
            forb_norm1[2],forb_norm1[3],forb_norm1[4]=topkEval(G_2.copy(),G_1.copy(),G_2.copy(),P1,n_Q)
                    #accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/1000
                    #len(Q_real)
            spec_norm=0
            accuracy=0
            file_A_results.write(f'{DGS} {QGS} {PGS} {PGES} {accuracy} {forb_norm} {forb_norm1[0]} {forb_norm1[1]} {forb_norm1[2]} {forb_norm1[3]} {forb_norm1[4]}{time_diff}\n')
            printR(tuns[ptun],forb_norm,0,0,time_diff,isomorphic)            
    print('\n')
print('\n\n')

