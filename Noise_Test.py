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
from conealign import coneGAM
from sgwl import SGWLSA
from grampa import Grampa
from REGAL.regal import Regal
from MDS import MDSGA
from GradP import gradp
from Grad import grad
from GradP.gradp import gradPMain
from mcmc.mc import mcAlign
os.environ["MKL_NUM_THREADS"] = "38"
torch.set_num_threads(38)

plotall = False

folderall = 'data3_'


foldernames = [ 'arenas','netscience', 'multimanga', 'highschool', 'voles']
n_G = [ 1133,379, 1004, 327, 712]
#foldernames = [ 'highschool']
#n_G = [327]
iters =50
percs = [(i+1)/10 for i in range(0,10)]
percs =[0.5]
#tun=[1,2,3,4,5,6,7]
#tuns=["Alpine","Cone","SGWL","Alpine_Dummy","Grampa","Regal","MDS"]
#nL=["_Noise5","_Noise10","_Noise15","_Noise20","_Noise25"]
tun=[1,2,3,4,5,6,7,8,9,10]
#tun=[8]
tuns=["Alpine","Cone","SGWL","Alpine_Dummy","Grampa","Regal","MDS","Fugal","MC","gradP"]
#tuns=["Fugal"]

nL=["_Noise5","_Noise10","_Noise15","_Noise20","_Noise25"]
#nL=["_Noise5"]
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
        G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        print(G)
        DGS=G.number_of_nodes()

# Get the number of edges
        DGES = G.number_of_edges()
        
        perc=percs[0]
        for noiseL in nL: 
            for ptun in range(len(tun)): 
                folder = f'./{folderall}/{foldernames[k]}{noiseL}/{int(perc*100)}'
                os.makedirs(f'{experimental_folder}{foldernames[k]}{noiseL}/{int(perc*100)}', exist_ok=True)
                folder1=f'./{experimental_folder}/{foldernames[k]}{noiseL}/{int(perc*100)}'
                file_A_results = open(f'{folder1}/NoiseTest_results{tuns[ptun]}.txt', 'w')
                file_A_results.write(f'DGS DGES QGS QGES PGS PGES forb_norm accuracy spec_norm time isomorphic \n')
                
                file_real_spectrum = open(f'{folder1}/real_Tspectrum{tuns[ptun]}.txt', 'w')
                file_A_spectrum = open(f'{folder1}/A_Tspectrum{tuns[ptun]}.txt', 'w')
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
                        print("Alpine")
                        _, list_of_nodes, forb_norm = Alpine(G_Q.copy(), G.copy(),mu=1,weight=2)
                    elif(tun[ptun]==2):
                        print("Cone")
                        _, list_of_nodes, forb_norm = coneGAM(G_Q.copy(), G.copy())
                    elif(tun[ptun]==3):
                        print("SGWL")
                        _, list_of_nodes, forb_norm = SGWLSA(G_Q.copy(), G.copy())
                    elif(tun[ptun]==4):
                        print("Alpine_Dummy")
                        _, list_of_nodes, forb_norm = align_new(G_Q.copy(), G.copy(),weight=1)
                    elif(tun[ptun]==5):
                        print("Grampa")
                        _, list_of_nodes, forb_norm = Grampa(G_Q.copy(), G.copy())
                    elif(tun[ptun]==6):
                        print("Regal")
                        _, list_of_nodes, forb_norm = Regal(G_Q.copy(), G.copy())     
                    elif(tun[ptun]==7):
                        print("MDS")
                        _, list_of_nodes, forb_norm = MDSGA(G_Q.copy(), G.copy())
                    elif(tun[ptun]==8):
                        print("Fugal")
                        _,list_of_nodes, forb_norm = Fugal(G_Q.copy(), G.copy())
                    elif(tun[ptun]==9):
                        print("mcmc")
                        list_of_nodes, forb_norm = mcAlign(G_Q.copy(), G.copy(),Q_real)
                    elif(tun[ptun]==10):
                        print("GradAlignP")
                        list_of_nodes, forb_norm = gradPMain(G_Q.copy(), G.copy())
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
                    A = nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes)).todense()
                    L = np.diag(np.array(np.sum(A, axis = 0)))
                    eigv_G_pred, _ = linalg.eig(L - A)
                    idx = eigv_G_pred.argsort()[::]   
                    eigv_G_pred = eigv_G_pred[idx]
                    for el in eigv_G_pred: file_A_spectrum.write(f'{el} ')
                    file_A_spectrum.write(f'\n')
                    #spec_norm = LA.norm(eigv_G_Q - eigv_G_pred)**2
                    spec_norm=0
                    accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
                    file_A_results.write(f'{DGS} {DGES} {QGS} {QGES} {PGS} {PGES} {forb_norm} {accuracy} {spec_norm} {time_diff} {isomorphic}\n')
                    printR(tuns[ptun],forb_norm,accuracy,spec_norm,time_diff,isomorphic)          
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
