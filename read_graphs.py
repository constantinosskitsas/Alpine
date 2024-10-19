import pandas as pd
import networkx as nx
import numpy as np
import random
from help_functions import read_real_graph
import sys
import os
random.seed(10)

folderall = 'data3_'
#the code will not work for disconected graphs
def refill_e(edges, n, amount):
    if amount == 0:
        return edges
    # print(edges)
    # ee = np.sort(edges).tolist()
    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        # _ee = np.sort(_e).tolist()
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if not(_ee in ee) and _e[0] != _e[1]:
            # ee.append(_ee)
            ee.add(_ee)
            new_e.append(_e)
            check = 0
            # print(f"refill - {len(new_e)}/{amount}")
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)

def random_walk_(G, n_Q):
    # List of nodes in the original graph
    nodes = list(G.nodes())
    print(f'Size of subgraph: {n_Q}')
    # Randomly choose a starting node
    start_node = random.choice(nodes)
    # Perform a random walk to select nodes for the subgraph
    subgraph_nodes = [start_node]
    current_node = start_node
    stuck_counter=0
    restart_all=0
    while len(subgraph_nodes) < n_Q:
        stuck_counter+=1
        restart_all+=1
        
        
        if (restart_all==n_Q*100):
            start_node = random.choice(nodes)
            subgraph_nodes = [start_node]
            current_node = start_node
            restart_all=0
            stuck_counter=0
        # Perform a random walk step
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break  # If no neighbors, break out of the loop
        next_node = random.choice(neighbors)
        if (not (next_node in subgraph_nodes)):
            subgraph_nodes.append(next_node)
            current_node = next_node
            stuck_counter=0
        elif(stuck_counter==20):
            current_node=random.choice(subgraph_nodes)
            stuck_counter=0
    return subgraph_nodes

def dfs(G, n_Q):
    while True:
        selected_sources = set()
        source_node = random.randint(0, G.number_of_nodes()-1)
        if source_node not in selected_sources: break
    selected_sources.add(source_node)
    return list(nx.dfs_preorder_nodes(G, source = source_node))[0:n_Q]    

def generate_data(foldernames, n_G, iters, percs,alg):
    n=80
    #for k in range(len(foldernames)):
    for k in range(1):
        #G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        G = read_real_graph(n, name_ = f'./raw_data/random/subgraph_DG_{n}.txt')
        print(n, G)
        selected_sources = set()
        for perc in percs: 
            n_Q = int(perc*G.number_of_nodes())
            print(f'Size of subgraph: {n_Q}')
            for iter in range(iters):
                if (alg=="DFS"):
                    dfs_traversal_G=dfs(G,n_Q)
                elif(alg=="RW"):
                    dfs_traversal_G=random_walk_(G,n_Q)
                elif(alg=="AS"):
                    dfs_traversal_G= list(G.nodes())
                else: return
                random.shuffle(dfs_traversal_G)
                map_id_to_index = {}
                for i in range(n_Q): map_id_to_index[dfs_traversal_G[i]] = i
                #folder_ = f'./{folderall}/{foldernames[k]}/{int(perc*100)}/{iter}'
                #if not os.path.exists(folder_): os.makedirs(folder_)
                file_subgraph = open(f'./raw_data/random/subgraph_QG_{n}.txt','w')
                for i in range(n_Q):
                    for j in range(i, n_Q):
                        if G.has_edge(dfs_traversal_G[i],dfs_traversal_G[j]):
                            file_subgraph.write(f'{map_id_to_index[dfs_traversal_G[i]]} {map_id_to_index[dfs_traversal_G[j]]}\n')
                file_nodes = open(f'./raw_data/random/nodes_QG_{n}.txt','w')
                for node in dfs_traversal_G: file_nodes.write(f'{node}\n')
        print('\n\n')

def generate_data_noise(foldernames, n_G, iters, percs,alg):
    for k in range(len(foldernames)):
        G = read_real_graph(n = n_G[k], name_ = f'./raw_data/{foldernames[k]}.txt')
        print(foldernames[k], G)
        
        selected_sources = set()
        for perc in percs: 
            n_Q = int(perc*G.number_of_nodes())
            print(f'Size of subgraph: {n_Q}')
            for iter in range(iters):
                G1 = nx.Graph()
                if (alg=="DFS"):
                    dfs_traversal_G=dfs(G,n_Q)
                elif(alg=="RW"):
                    dfs_traversal_G=random_walk_(G,n_Q)
                else: return
                random.shuffle(dfs_traversal_G)
                map_id_to_index = {}
                for i in range(n_Q): map_id_to_index[dfs_traversal_G[i]] = i
                folder_ = f'./{folderall}/{foldernames[k]}_Noise25/{int(perc*100)}/{iter}'
                if not os.path.exists(folder_): os.makedirs(folder_)
                file_subgraph = open(f'{folder_}/subgraph.txt','w')
                for i in range(n_Q):
                    for j in range(i, n_Q):
                        if G.has_edge(dfs_traversal_G[i],dfs_traversal_G[j]):
                            #file_subgraph.write(f'{map_id_to_index[dfs_traversal_G[i]]} {map_id_to_index[dfs_traversal_G[j]]}\n')
                            G1.add_edge(map_id_to_index[dfs_traversal_G[i]],map_id_to_index[dfs_traversal_G[j]])
                file_nodes = open(f'{folder_}/nodes.txt','w')
                for node in dfs_traversal_G: file_nodes.write(f'{node}\n')
                Src_e = np.array(G1.edges)
                n = np.amax(Src_e) + 1
                nedges = Src_e.shape[0]
                Src_e = remove_e(Src_e, 0.25)
                #Src_e = refill_e(Src_e, n, nedges - Src_e.shape[0])
                G1 = nx.from_edgelist(Src_e)
                for edge in G1.edges():
                    file_subgraph.write(f"{edge[0]} {edge[1]}\n")
        print('\n\n')

def remove_e(edges, noise, no_disc=True, until_connected=False):
    ii = 0
    while True:
        ii += 1
        print(f"##<{ii}>##")

        if no_disc:
            bin_count = np.bincount(edges.flatten())
            rows_to_delete = []
            for i, edge in enumerate(edges):
                if np.random.sample(1)[0] < noise:
                    e, f = edge
                    if bin_count[e] > 1 and bin_count[f] > 1:
                        bin_count[e] -= 1
                        bin_count[f] -= 1
                        rows_to_delete.append(i)
            new_edges = np.delete(edges, rows_to_delete, axis=0)
        else:
            new_edges = edges[np.random.sample(edges.shape[0]) >= noise]

        graph = nx.Graph(new_edges.tolist())
        graph_cc = len(max(nx.connected_components(graph), key=len))
        print(graph_cc, np.amax(edges)+1)
        graph_connected = graph_cc == np.amax(edges) + 1
        # if not graph_connected:
        #     break
        if graph_connected or not until_connected:
            break
    return new_edges


def generate_LGN(dataset,size,iters,noiseL):
    G = read_real_graph(size, name_ = f'./raw_data/{dataset}.txt')
    for i in range(iters):
        G1=G.copy()
        Src_e = np.array(G1.edges)
        Src_e = remove_e(Src_e, noiseL/100)
        G1 = nx.from_edgelist(Src_e)
        if not os.path.exists(f'./data3_/{dataset}raw/'): os.makedirs(f'./data3_/{dataset}raw/')
        file_subgraph = open(f'./data3_/{dataset}raw/{noiseL}_{i}.txt','w')
        for edge in G1.edges():
            file_subgraph.write(f"{edge[0]} {edge[1]}\n")
foldernames = ['arenas', 'celegans', 'netscience', 'multimanga', 'highschool', 'voles']
n_G = [1133, 453, 379, 1004, 327, 712]

foldernames = ['cit-DBLP']
#n_G = [12591]
iters = 1
#foldernames=["DG_"]
percs = [(i+1)/10 for i in range(9,10)]
percs=[0.5]
#generate_data_noise(foldernames, n_G, iters, percs,"RW")
#generate_data_noise(foldernames, n_G, iters, percs,"RW")
generate_data(foldernames, n_G, iters, percs,"RW")
#generate_LGN("highschool",327,iters,25)
n=80
#for i in range(7):
#    G = nx.erdos_renyi_graph(n, 0.1)
 #   file_subgraph = open(f'raw_data/random/subgraph_DG_{n}.txt','w')
 #   for edge in G.edges():
 #       file_subgraph.write(f"{edge[0]} {edge[1]}\n")
#    print(n)
#    n=n*2
    