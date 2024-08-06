import numpy as np
from pred import convex_initSM, align_SM, align_new, algo_fusbal
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
from conealign import coneGAM
from sgwl import SGWLSA
from grampa import Grampa
from REGAL.regal import Regal
#git push -f origin main

#QAP
#ACCURACY
#SPECTRUM NORM

#FUSBAL RESULTS
#QAP ACCYRACY SPECTRUM

#FUGAL RESULTS
#QAP ACCYRACY SPECTRUM TIME

#REAL SPECTRUM
#FUSBAL SPECTRU
#FUGAL SPECTRUM

os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)

plotall = False

folderall = 'data3_'


foldernames = [ 'arenas','netscience', 'multimanga', 'highschool', 'voles']
#foldernames = ['arenas']
#n_G=[379]
n_G = [ 1133,379, 1004, 327, 712]
#n_G = [1133]
iters =2
percs = [(i+1)/10 for i in range(0,9)]
#percs =[0.5]
#percs = [0.2]
#tun = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
#tuns = ["1","0_9","0_8","0_7","0_6","0_5","0_4","0_3","0_2","0_1"]
#tun=[0.5,0,1]
#tun=[1,0]
#tun=[0,1]
#tun=[1,2,3,4]
#tun=[1,2,3,4,5,6,7]
#tun=[1,3,4,5,6]
tun=[1,2,3,4,5,6]
tun=[1,7]
#tuns=["NO_FUGAL","FUGAL","FUGALX1_5","FUGALX2","FUGALM"]
#tuns=["FunPGA","cone","SGWL","fugal","Grampa","Regal","FunPGA_D"]
#tuns=["FunPGA","FunPGA_D"]
#tuns=["FunPGA","SGWL","fugal","Grampa","Regal"]
#tuns=["FunPGA","Cone","SGWL","fugal","Grampa","Regal"]
tuns=["fugal","FunPGA_D"]
#nL=["_Noise5","_Noise10","_Noise15","_Noise20","_Noise25"]
#nL=[]
#tuns=["Cone"]
#tuns=["FunPGA","Grampa","Regal"]
#tuns=["algo_fusbal"]
#tuns=["Fusbal","FusbalNew"]
#tuns=["Normal","Degree_Penalty"]
def printR(name,forb_norm,accuracy,spec_norm,time_diff,isomorphic=False):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Spec_norm:', spec_norm)
    print('----> Time:', time_diff)
    print('----> Isomorphic:', isomorphic)
    print()     


def plotres(eigv_G_Q,eigv_G_pred,eigv_G_fugal):
    plt.plot(eigv_G_Q, color = 'b', label = 'Real')
    plt.plot(eigv_G_pred, color = 'r', label = 'Fusbal')
    plt.plot(eigv_G_fugal, color = 'g', label = 'Fugal')
    plt.title('Spectrum')
    plt.legend()
    plt.show()
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
        G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        print(G)
        DGS=G.number_of_nodes()

# Get the number of edges
        DGES = G.number_of_edges()
        
        #perc=percs[0]
        for perc in percs: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}/{int(perc*100)}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}/{int(perc*100)}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}/{int(perc*100)}'
                file_fusbal_results = open(f'{folder1}/fusbal_Tresults{tuns[ptun]}.txt', 'w')
                file_fusbal_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_fusbal_spectrum = open(f'{folder1}/fusbal_Tspectrum{tuns[ptun]}.txt', 'w')
                n_Q = int(perc*G.number_of_nodes())
                print(f'Size of subgraph: {n_Q}')
                for iter in range(iters):
                    folder_ = f'{folder}/{iter}'
                    folder1_ = f'{folder1}/{iter}'
                    os.makedirs(f'{folder1_}', exist_ok=True)
                    file_subgraph = f'{folder_}/subgraph.txt'
                    file_nodes = f'{folder_}/nodes.txt'
                    Q_real = read_list(file_nodes)
                    
                    G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
                    A = nx.adjacency_matrix(G_Q).todense()
                    QGS=G_Q.number_of_nodes()
                    QGES = G_Q.number_of_edges()
                    L = np.diag(np.array(np.sum(A, axis = 0)))
                    eigv_G_Q, _ = linalg.eig(L - A)
                    idx = eigv_G_Q.argsort()[::]   
                    eigv_G_Q = eigv_G_Q[idx]
                    for el in eigv_G_Q: file_real_spectrum.write(f'{el} ')
                    file_real_spectrum.write(f'\n')
                    start = time.time()
                    if(tun[ptun]==1):
                        #algo_fusbal
                        print("FUNPGA")
                        #_, list_of_nodes, forb_norm = align_SM(G_Q.copy(), G.copy())
                        _, list_of_nodes, forb_norm = algo_fusbal(G_Q.copy(), G.copy(),mu=1,weight=1)
                    elif(tun[ptun]==2):
                        print("Cone")
                        _, list_of_nodes, forb_norm = coneGAM(G_Q.copy(), G.copy())
                    elif(tun[ptun]==3):
                        print("SGWL")
                        _, list_of_nodes, forb_norm = SGWLSA(G_Q.copy(), G.copy())
                    elif(tun[ptun]==4):
                        print("fugal")
                        _, list_of_nodes, forb_norm = align_new(G_Q.copy(), G.copy(),weight=1)
                    elif(tun[ptun]==5):
                        print("Grampa")
                        _, list_of_nodes, forb_norm = Grampa(G_Q.copy(), G.copy())
                    elif(tun[ptun]==6):
                        print("Regal")
                        _, list_of_nodes, forb_norm = Regal(G_Q.copy(), G.copy())
                    elif(tun[ptun]==7):
                        print("FUNPGA_D")
                        _, list_of_nodes, forb_norm = algo_fusbal(G_Q.copy(), G.copy(),mu=1,weight=2)        
                    else:
                        print("Error")
                        exit()
                    #_, list_of_nodes, forb_norm = align_SM(G_Q.copy(), G.copy(), mu=1,weight=1)
                    #_, list_of_nodes, forb_norm = align_SM(G_Q.copy(), G.copy(), mu=1)
                    #_, list_of_nodes, forb_norm = align_new(G_Q.copy(), G.copy(),weight=tun[ptun])
                    end = time.time()
                    subgraph = G.subgraph(list_of_nodes)
                    
                    PGS=subgraph.number_of_nodes()
                    PGES = subgraph.number_of_edges()
                    #isomorphic = nx.is_isomorphic(subgraph, G_Q.copy())
                    isomorphic=False
                    if(forb_norm==0):
                        isomorphic=True
                    time_diff = end - start
                    file_nodes_pred = open(f'{folder1_}/{tuns[ptun]}.txt','w')
                    for node in list_of_nodes: file_nodes_pred.write(f'{node}\n')
                    
                    A = nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes)).todense()
                    L = np.diag(np.array(np.sum(A, axis = 0)))
                    eigv_G_pred, _ = linalg.eig(L - A)
                    idx = eigv_G_pred.argsort()[::]   
                    eigv_G_pred = eigv_G_pred[idx]
                    for el in eigv_G_pred: file_fusbal_spectrum.write(f'{el} ')
                    file_fusbal_spectrum.write(f'\n')
                    spec_norm = LA.norm(eigv_G_Q - eigv_G_pred)**2
                    accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
                    file_fusbal_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy} {spec_norm} {time_diff} {isomorphic}\n')
                    #printR("FusbalT",forb_norm,accuracy,spec_norm,time_diff,isomorphic)
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,isomorphic)
                #if plotall:
                #    plotres(eigv_G_Q,eigv_G_pred,eigv_G_fugal)                
            print('\n')
        print('\n\n')



sys.exit()


plotall = True

for i in range(5):
    G = read_graph(f'./Data/G_{i}.txt')
    Gsmall = read_graph(f'./Data/Gsmall_{i}.txt')

    eigv_Gsmall, _ = linalg.eig(nx.adjacency_matrix(Gsmall).todense())
    idx = eigv_Gsmall.argsort()[::]   
    eigv_Gsmall = eigv_Gsmall[idx]

    _, list_of_nodes, forbnorm = align_SM(Gsmall.copy(), G.copy())
    eigv_G_induced, _ = linalg.eig(nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes[0:Gsmall.number_of_nodes()])).todense())
    idx = eigv_G_induced.argsort()[::]   
    eigv_G_induced = eigv_G_induced[idx]
    print(list_of_nodes)
    print(f'-----> {forbnorm}')

    _, list_of_nodes2, forbnorm = align_new(Gsmall.copy(), G.copy())
    eigv_G_induced2, _ = linalg.eig(nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes2[0:Gsmall.number_of_nodes()])).todense())
    idx = eigv_G_induced2.argsort()[::]   
    eigv_G_induced2 = eigv_G_induced2[idx]
    print(list_of_nodes2)
    print(f'-----> {forbnorm}')

    if plotall:
        plt.plot(eigv_Gsmall, color = 'b', label = 'Real')
        plt.plot(eigv_G_induced, color = 'r', label = 'Fusbal')
        plt.plot(eigv_G_induced2, color = 'g', label = 'Fugal')
        plt.title('Spectrum')
        plt.legend()
        plt.show()
