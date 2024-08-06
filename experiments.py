import numpy as np
from pred import convex_initSM, align_SM, align_new
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

os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

plotall = False

folderall = 'data3_'

#foldernames = ['arenas', 'netscience', 'multimanga', 'highschool', 'voles']
#n_G = [1133, 379, 1004, 327, 712]
foldernames = [ 'netscience']
n_G = [ 379]
iters = 10
percs = [(i+1)/10 for i in range(2,4)]
def printR(name,forb_norm,accuracy,spec_norm,time_diff):
    print('---- ',name, '----')
    print('----> Forb_norm:', forb_norm)
    print('----> Accuracy:', accuracy)
    print('----> Spec_norm:', spec_norm)
    print('----> Time:', time_diff)
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
#create_new_folder(experimental_folder, new_id)  
experimental_folder=f'./{folderall}/res/_{new_id}/'       
for k in range(0,len(foldernames)):
        G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        print(G)
        for perc in percs: 
            folder = f'./{folderall}/{foldernames[k]}/{int(perc*100)}'
            

    # Create the entire folder structure recursively
            
            folder1 = f'{experimental_folder}{foldernames[k]}/{int(perc*100)}'
            os.makedirs(f'{experimental_folder}{foldernames[k]}/{int(perc*100)}', exist_ok=True)
            print(folder1)
            file_fusbal_results = open(f'{folder1}/fusbal_results.txt', 'w')
            file_fusbal_results.write(f'forb_norm accuracy spec_norm time\n')
            file_fugal_results = open(f'{folder1}/fugal_results.txt', 'w')
            file_fugal_results.write(f'forb_norm accuracy spec_norm time\n')
            
            file_real_spectrum = open(f'{folder1}/real_spectrum.txt', 'w')
            file_fusbal_spectrum = open(f'{folder1}/fusbal_spectrum.txt', 'w')
            file_fugal_spectrum = open(f'{folder1}/fugal_spectrum.txt', 'w')

            file_fusbal_nf_results = open(f'{folder1}/fusbal_nf_results.txt', 'w')
            file_fusbal_nf_results.write(f'forb_norm accuracy spec_norm time\n')
            file_fugal_nf_results = open(f'{folder1}/fugal_nf_results.txt', 'w')
            file_fugal_nf_results.write(f'forb_norm accuracy spec_norm time\n')
        
            file_fusbal_nf_spectrum = open(f'{folder1}/fusbal_nf_spectrum.txt', 'w')
            file_fugal_nf_spectrum = open(f'{folder1}/fugal_nf_spectrum.txt', 'w')
            
            n_Q = int(perc*G.number_of_nodes())
            print(f'Size of subgraph: {n_Q}')
            for iter in range(iters):
                folder_ = f'{folder}/{iter}'
                folder_1 = f'{folder1}/{iter}'
                file_subgraph = f'{folder_}/subgraph.txt'
                file_nodes = f'{folder_}/nodes.txt'
                Q_real = read_list(file_nodes)
                
                G_Q= read_real_graph(n = n_Q, name_ = file_subgraph)
                A = nx.adjacency_matrix(G_Q).todense()

                L = np.diag(np.array(np.sum(A, axis = 0)))
                eigv_G_Q, _ = linalg.eig(L - A)
                idx = eigv_G_Q.argsort()[::]   
                eigv_G_Q = eigv_G_Q[idx]
                file_real_spectrum.write(f'{str(eigv_G_Q)[1:-1]}\n')
                for el in eigv_G_Q: file_real_spectrum.write(f'{el} ')
                file_real_spectrum.write(f'\n')
                
                start = time.time()
                _, list_of_nodes, forb_norm = align_SM(G_Q.copy(), G.copy())
                end = time.time()
                time_diff = end - start
                os.makedirs(folder_1, exist_ok=True)
                file_nodes_pred = open(f'{folder_1}/nodes_fusbal.txt','w')
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
                file_fusbal_results.write(f'{forb_norm} {accuracy} {spec_norm} {time_diff}\n')
                printR("Fusbal",forb_norm,accuracy,spec_norm,time_diff)

                start = time.time()
                _, list_of_nodes2, forb_norm = align_new(G_Q.copy(), G.copy())
                end = time.time()
                time_diff = end - start
                file_nodes_fugal = open(f'{folder_1}/nodes_fugal.txt','w')
                for node in list_of_nodes2: file_nodes_fugal.write(f'{node}\n')
                A = nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes2)).todense()
                L = np.diag(np.array(np.sum(A, axis = 0)))
                eigv_G_fugal, _ = linalg.eig(L - A)
                idx = eigv_G_fugal.argsort()[::]   
                eigv_G_fugal = eigv_G_fugal[idx]
                for el in eigv_G_fugal: file_fugal_spectrum.write(f'{el} ')
                file_fugal_spectrum.write(f'\n')
                spec_norm = LA.norm(eigv_G_Q - eigv_G_fugal)**2
                accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes2))/len(Q_real)
                file_fugal_results.write(f'{forb_norm} {accuracy} {spec_norm} {time_diff}\n')
                printR("FUGAL",forb_norm,accuracy,spec_norm,time_diff)  

                start = time.time()
                _, list_of_nodes, forb_norm = align_SM(G_Q.copy(), G.copy(), mu=0)
                end = time.time()
                time_diff = end - start
                file_nodes_pred = open(f'{folder_1}/nodes_fusbal_nf.txt','w')
                for node in list_of_nodes: file_nodes_pred.write(f'{node}\n')
                A = nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes)).todense()
                L = np.diag(np.array(np.sum(A, axis = 0)))
                eigv_G_pred, _ = linalg.eig(L - A)
                idx = eigv_G_pred.argsort()[::]   
                eigv_G_pred = eigv_G_pred[idx]
                for el in eigv_G_pred: file_fusbal_nf_spectrum.write(f'{el} ')
                file_fusbal_nf_spectrum.write(f'\n')
                spec_norm = LA.norm(eigv_G_Q - eigv_G_pred)**2
                accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
                file_fusbal_nf_results.write(f'{forb_norm} {accuracy} {spec_norm} {time_diff}\n')
                printR("Fusbal_nf",forb_norm,accuracy,spec_norm,time_diff)

                start = time.time()
                _, list_of_nodes2, forb_norm = align_new(G_Q.copy(), G.copy(), mu=0)
                end = time.time()
                time_diff = end - start
                file_nodes_fugal = open(f'{folder_1}/nodes_fugal_nf.txt','w')
                for node in list_of_nodes2: file_nodes_fugal.write(f'{node}\n')
                A = nx.adjacency_matrix(nx.induced_subgraph(G, list_of_nodes2)).todense()
                L = np.diag(np.array(np.sum(A, axis = 0)))
                eigv_G_fugal, _ = linalg.eig(L - A)
                idx = eigv_G_fugal.argsort()[::]   
                eigv_G_fugal = eigv_G_fugal[idx]
                for el in eigv_G_fugal: file_fugal_nf_spectrum.write(f'{el} ')
                file_fugal_nf_spectrum.write(f'\n')
                spec_norm = LA.norm(eigv_G_Q - eigv_G_fugal)**2
                accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes2))/len(Q_real)
                file_fugal_nf_results.write(f'{forb_norm} {accuracy} {spec_norm} {time_diff}\n')
                printR("FUGAL_nf",forb_norm,accuracy,spec_norm,time_diff)


                if plotall:
                    plotres(eigv_G_Q,eigv_G_pred,eigv_G_fugal)                
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
