import networkx as nx
import numpy as np
def read_list(filename):
    list_nodes = []
    with open(filename) as file:
        for line in file:
            linesplit = line[:-1].split(' ')
            list_nodes.append(int(linesplit[0]))
    return list_nodes


def read_graph(filename):
    G = nx.Graph()
    with open(filename) as file:
        for line in file:
            linesplit = line[:-1].split(' ')
            if len(linesplit) == 1:
                n = int(linesplit[0])
                for i in range(n): G.add_node(i)
            else:
                u = int(linesplit[0])
                v = int(linesplit[1])
                G.add_edge(u, v)
    return G

def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    #print("Just checking",nx.is_directed(G))
    return np.array(G.edges)

def read_real_graph(n, name_):
    print(f'Making {name_} graph...')
    filename = open(f'{name_}', 'r')
    lines = filename.readlines()
    G = nx.Graph()
    for i in range(n): G.add_node(i)
    for line in lines:
        u_v = (line[:-1].split(' '))
        u = int(u_v[0])
        v = int(u_v[1])
        G.add_edge(u, v)
    return G   