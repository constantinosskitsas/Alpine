import os
from collections import defaultdict
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import time
import torch
from pred import convex_initSM, align_SM, align_new,algo_fusbal,standardize_nodes
def read_graph_from_file(file_path):
    nodes = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:  # Line should have two parts for an edge
                node1 = int(parts[0])
                node2 = int(parts[1])
                nodes.add(node1)
                nodes.add(node2)
    return nodes


def statistics(folder_path,nodes_num):
    node_size_distribution = defaultdict(int)
    for i in range(nodes_num):  # From 0 to 2000 inclusive
        file_name = f"AIDS_D{i}.txt"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            nodes = read_graph_from_file(file_path)
            node_size_distribution[len(nodes)] += 1
            if (len(nodes)<=16):
                print(file_path)
        else:
            print(f"File {file_name} does not exist")

    # Print the node size distribution
    print("Node size distribution:")
    for size, count in sorted(node_size_distribution.items()):
        print(f"Graphs with {size} nodes: {count}")

def Subgraph_GT(qnum,nodes_num,pathQ,pathG,output_file):
    with open(output_file, 'a') as f:
        for i in range(qnum):
            QG = nx.read_edgelist(pathQ+str(i)+".txt")
            CN=[]
            for j in range(nodes_num):
                DG = nx.read_edgelist(pathG+str(j)+".txt")
                matcher1 = GraphMatcher(DG, QG)
                is_isomorphic = matcher1.subgraph_is_isomorphic()
                if(is_isomorphic==True):
                    CN.append(j)
            print(i,len(CN))
            f.write(f"QID {i} -> {CN}\n")
        
def parse_file_to_matrix(file_path):
    matrix = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove the newline character at the end
            line = line.strip()
            # Split the line at the '->' symbol
            if '->' in line:
                qid_part, numbers_part = line.split('->')
                # Extract the list of numbers, removing the square brackets and spaces
                numbers_str = numbers_part.strip()[1:-1]
                # Convert the numbers into a list of integers
                numbers = list(map(int, numbers_str.split(',')))
                # Add the list of numbers to the matrix
                matrix.append(numbers)
    
    return matrix


# Usage example

def standardize_nodes1(G):
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)

# Function to read graph from an edge list string
def read_graph_from_edgelist_string(edgelist_string):
    from io import StringIO
    edge_list = StringIO(edgelist_string)
    G = nx.read_edgelist(edge_list, nodetype=int)
    return G




def main():
    os.environ["MKL_NUM_THREADS"] = "37"
    torch.set_num_threads(37)
    folder_path = 'Dataset_SM'  # Directory where the files are located
    nodes_num=2001
    pathQ="Dataset_SM/Query_S16_"
    pathG="Dataset_SM/AIDS_D"
    corrFile="GT/AIDS_GT16.txt"
    #statistics(folder_path,nodes_num)
    #Subgraph_GT(200,nodes_num,pathQ,pathG,corrFile)
    file_path = 'path_to_your_file.txt'
    data = parse_file_to_matrix(corrFile)
    #print(data[0])
    counter=0
    counterC=0
    




# Check if G_Q is a subgraph isomorphic to G

    for k in range (200):
        G_Q=nx.read_edgelist(pathQ+str(k)+".txt")
        G_Q1 = read_graph_from_edgelist_string(pathQ+str(k)+".txt")
        G_Q1 = standardize_nodes1(G_Q1)
        #G_Q = standardize_nodes(G_Q)
        for i in (data[0]):
                #for node in G_Q.nodes():
                  #  print(f'Node: {node}, Type: {type(node)}')
            G=nx.read_edgelist(pathG+str(i)+".txt")
            G1 = read_graph_from_edgelist_string(pathG+str(i)+".txt")
            G1 = standardize_nodes1(G1)
            start1 = time.time()
            matcher1 = GraphMatcher(G1, G_Q1)
            #G = standardize_nodes(G)
            counter+=1
            #matcher1 = GraphMatcher(G_Q.copy(), G.copy())
            isom=matcher1.subgraph_is_isomorphic()
            
            end1 = time.time()
            print(isom,end1-start1)
            start = time.time()
            _, list_of_nodes, forb_norm = algo_fusbal(G_Q.copy(), G.copy())
            end = time.time()
            if (forb_norm==0):
                counterC+=1
            print("QID",k ," out of G_Q ",counter, "we found ",counterC," that is percentage",counterC/counter*100, "time :",end-start, "frob",forb_norm)
        print("QID",k ," out of G_Q ",counter, "we found ",counterC," that is percentage",counterC/counter*100)
    print("out of G_Q ",counter, "we found ",counterC," that is percentage",counterC/counter*100)
if __name__ == "__main__":
    main()