import numpy as np
import networkx as nx
import random

def induced_subgraph_with_ids(adj_matrix, num_nodes):
    G = nx.from_numpy_matrix(adj_matrix)
    
    # Randomly select N nodes
    selected_nodes = random.sample(G.nodes(), num_nodes)
    
    # Create the induced subgraph
    subgraph = G.subgraph(selected_nodes)
    
    # Convert the subgraph to adjacency matrix
    subgraph_adj_matrix = nx.to_numpy_array(subgraph)
    
    return subgraph_adj_matrix, selected_nodes, list(subgraph.nodes())

def connected_subgraph_of_size_with_ids_random_walk(adj_matrix, size, walk_length=100):
    G = nx.from_numpy_matrix(adj_matrix)
    
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
    G = nx.from_numpy_matrix(adj_matrix)
    
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

# Example adjacency matrix representing a graph
adjacency_matrix = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

desired_size = 8  # Desired size of the connected subgraph
G = nx.connected_watts_strogatz_graph(20, 10, 0.1)

# Convert the graph to an adjacency matrix
adjacency_matrix = nx.to_numpy_array(G)
connected_subgraph_matrix, subgraph_node_ids, node_id_mapping = connected_subgraph_of_size_with_ids_random_walk(adjacency_matrix, desired_size)
connected_subgraph_matrix1, subgraph_node_ids1, node_id_mapping1 =induced_subgraph_with_ids(adjacency_matrix,desired_size)
print("Original Ajc matrix")
print(adjacency_matrix)
print("Connected Subgraph of Size", desired_size, ":")
print("Adjacency Matrix:")
print(connected_subgraph_matrix)
print("Node IDs in Subgraph:", subgraph_node_ids)
print("Node ID Mapping (Subgraph Node ID -> Original Node ID):", node_id_mapping)

print("Original Ajc matrix")
print(adjacency_matrix)
print("Connected Subgraph of Size", desired_size, ":")
print("Adjacency Matrix:")
print(connected_subgraph_matrix1)
print("Node IDs in Subgraph:", subgraph_node_ids1)
print("Node ID Mapping (Subgraph Node ID -> Original Node ID):", node_id_mapping1)
desired_size = connected_subgraph_matrix1.shape[0]

# Generate a random permutation
permutation = np.random.permutation(desired_size)
print(permutation)
#permutation=np.array(list(range(0,8,1)))
#print(permutation)
#permutation[0]=1
#permutation[1]=0
#print(permutation)
permutation_matrix = np.zeros((desired_size, desired_size), dtype=int)

# Set values to 1 for the permutation indices
permutation_matrix[np.arange(desired_size), permutation] = 1
# Permute rows and columns of the adjacency matrix
permuted_connected_subgraph_matrix = connected_subgraph_matrix1[permutation][:, permutation]
permuted_connected_subgraph_matrix1=permutation_matrix@connected_subgraph_matrix1@permutation_matrix.T
print(permuted_connected_subgraph_matrix1)
print("test")
print(permuted_connected_subgraph_matrix)
print(permuted_connected_subgraph_matrix1==permuted_connected_subgraph_matrix)
reverse_permutation = np.argsort(permutation)

# Permute rows and columns of the permuted adjacency matrix to recover the original adjacency matrix
recovered_adjacency_matrix = permuted_connected_subgraph_matrix[reverse_permutation][:, reverse_permutation]
recovered_adjacency_matrix1=permutation_matrix.T@permuted_connected_subgraph_matrix@permutation_matrix
#print(recovered_adjacency_matrix)
print(recovered_adjacency_matrix1==connected_subgraph_matrix1)