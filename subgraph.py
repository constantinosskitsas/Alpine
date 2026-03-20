import numpy as np
import networkx as nx
import random

def induced_subgraph_with_ids(adj_matrix, num_nodes):
    G = nx.from_numpy_array(adj_matrix)
    
    # Randomly select N nodes
    selected_nodes = random.sample(G.nodes(), num_nodes)
    
    # Create the induced subgraph
    subgraph = G.subgraph(selected_nodes)
    
    # Convert the subgraph to adjacency matrix
    subgraph_adj_matrix = nx.to_numpy_array(subgraph)
    
    return subgraph_adj_matrix, selected_nodes, list(subgraph.nodes())

def connected_subgraph_of_size_with_ids_random_walk(adj_matrix, size, walk_length=100):
    G = nx.from_numpy_array(adj_matrix)
    
    # List of nodes in the original graph
    nodes = list(G.nodes())
    
    # Randomly choose a starting node
    start_node = random.choice(nodes)
    
    # Perform a random walk to select nodes for the subgraph
    subgraph_nodes = [start_node]
    current_node = start_node
    while len(subgraph_nodes) < size:
        # Perform a random walk step
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break  # If no neighbors, break out of the loop
        next_node = random.choice(neighbors)
        if (not (next_node in subgraph_nodes)):
            subgraph_nodes.append(next_node)
        current_node = next_node
    
    # Extract the subgraph from the original graph
    subgraph = G.subgraph(subgraph_nodes)
    
    # Get the IDs of the selected nodes in the original graph
    subgraph_node_ids = list(subgraph.nodes())
    
    # Create a mapping between node IDs in the original graph and subgraph
    node_id_mapping = {subgraph_node_id: original_node_id for subgraph_node_id, original_node_id in zip(subgraph_node_ids, subgraph_nodes)}
    #node_id_mapping=lig()
    # Create subgraph adjacency matrix
    subgraph_adj_matrix = nx.to_numpy_array(subgraph)
    
    return subgraph_adj_matrix, subgraph_node_ids, node_id_mapping

def connected_subgraph_of_size_with_ids(adj_matrix, size):
    G = nx.from_numpy_array(adj_matrix)(adj_matrix)
    
    # List of nodes in the original graph
    nodes = list(G.nodes())
    
    # Randomly choose a starting node
    start_node = random.choice(nodes)
    
    # Initialize a list to store the nodes of the connected subgraph
    subgraph_nodes = [start_node]
    
    # Perform BFS to grow the subgraph until it reaches the desired size
    while len(subgraph_nodes) < size:
        current_node = subgraph_nodes.pop(0)
        neighbors = list(G.neighbors(current_node))
        random.shuffle(neighbors)  # Shuffle neighbors to randomly select next node
        for neighbor in neighbors:
            if neighbor not in subgraph_nodes:
                subgraph_nodes.append(neighbor)
            if len(subgraph_nodes) == size:
                break
    
    # Extract the subgraph from the original graph
    subgraph = G.subgraph(subgraph_nodes)
    
    # Get the IDs of the selected nodes
    subgraph_node_ids = list(subgraph.nodes())
    
    return nx.to_numpy_array(subgraph), subgraph_node_ids

def create_sub_graph_ground_truth(G1: nx.Graph, G2: nx.Graph, n_Q, shuffle='random', method='random walk', p=0.0, gt_G1=None, gt_G2=None):
    from copy import deepcopy
    G = deepcopy(G1)
    G_Q = deepcopy(G2)
    node_G = []
    node_G_Q = []
    
    if gt_G1 is None or gt_G2 is None:
        while len(node_G) < n_Q:
            node_G = connected_subgraph_of_size_with_original_ids(G, n_Q, method, p)
        while len(node_G_Q) < n_Q: 
            node_G_Q = connected_subgraph_of_size_with_original_ids(G_Q, n_Q, method, p)
    else:
        node_G = gt_G1
        node_G_Q = gt_G2
    assert len(node_G) == n_Q
    assert len(node_G_Q) == n_Q
    assert nx.is_connected(nx.subgraph(G, node_G))
    assert nx.is_connected(nx.subgraph(G_Q, node_G_Q))
    
    G_new_edges = []
    G_Q_new_edges = []
    for u in range(n_Q):
        for v in range(n_Q):
            if G.has_edge(node_G[u], node_G[v]) and (not G_Q.has_edge(node_G_Q[u], node_G_Q[v])):
                G_Q.add_edge(node_G_Q[u], node_G_Q[v])
                G_Q_new_edges.append((node_G_Q[u], node_G_Q[v]))
            if (not G.has_edge(node_G[u], node_G[v])) and G_Q.has_edge(node_G_Q[u], node_G_Q[v]):
                G.add_edge(node_G[u], node_G[v])
                G_new_edges.append((node_G[u], node_G[v]))
    
    info = {

        'ordered_selected_nodes_G1': node_G,
        'ordered_selected_nodes_G2': node_G_Q, 
    }

    return G, G_Q, info

def connected_subgraph_of_size_with_original_ids(G: nx.Graph, size: int, method='random walk', p=0.0):
    if method=='random walk':
        return connected_subgraph_of_size_with_original_ids_random_walk(G, size)
    elif method=='bfs':
        return connected_subgraph_of_size_with_original_ids_bfs(G, size, p)
    
def connected_subgraph_of_size_with_original_ids_bfs(G: nx.Graph, size, p):
    
    # List of nodes in the original graph
    nodes = list(G.nodes())
    
    # Randomly choose a starting node
    start_node = random.choice(nodes)
    
    # Perform a random walk to select nodes for the subgraph
    subgraph_nodes = [start_node]
    queue = [start_node]
    current_node = start_node
    visit = set()
    visit.add(start_node)
    while len(queue) > 0 and len(subgraph_nodes) < size:
        # Perform a random walk step
        current_node = queue.pop(0)        
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            continue  # If no neighbors, break out of the loop
        for neighbor in neighbors:
            if len(subgraph_nodes) >= size:
                break
            if neighbor in visit:
                continue
            r = np.random.rand()
            if r > p:
                queue.append(neighbor)
                visit.add(neighbor)
                subgraph_nodes.append(neighbor)
    
    return subgraph_nodes

def connected_subgraph_of_size_with_original_ids_random_walk(G: nx.Graph, size):
    
    # List of nodes in the original graph
    nodes = list(G.nodes())
    
    # Randomly choose a starting node
    start_node = int(random.choice(nodes))
    
    # Perform a random walk to select nodes for the subgraph
    subgraph_nodes = [start_node]
    current_node = start_node
    while len(subgraph_nodes) < size:
        # Perform a random walk step
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break  # If no neighbors, break out of the loop
        next_node = random.choice(neighbors)
        if (not (next_node in subgraph_nodes)):
            subgraph_nodes.append(int(next_node))
        current_node = next_node
    
    return subgraph_nodes


    G = nx.from_numpy_array(adj_matrix)
    
    # List of nodes in the original graph
    nodes = list(G.nodes())
    
    # Randomly choose a starting node
    start_node = random.choice(nodes)
    
    # Initialize a list to store the nodes of the connected subgraph
    subgraph_nodes = [start_node]
    
    # Perform BFS to grow the subgraph until it reaches the desired size
    while len(subgraph_nodes) < size:
        current_node = subgraph_nodes.pop(0)
        neighbors = list(G.neighbors(current_node))
        random.shuffle(neighbors)  # Shuffle neighbors to randomly select next node
        for neighbor in neighbors:
            if neighbor not in subgraph_nodes:
                subgraph_nodes.append(neighbor)
            if len(subgraph_nodes) == size:
                break
    
    # Extract the subgraph from the original graph
    subgraph = G.subgraph(subgraph_nodes)
    
    # Get the IDs of the selected nodes
    subgraph_node_ids = list(subgraph.nodes())
    
    return nx.to_numpy_array(subgraph), subgraph_node_ids
