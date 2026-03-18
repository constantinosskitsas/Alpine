import numpy as np
import networkx as nx
def synchronize_features(F1, F2, anchors_G, anchors_G_Q, direction="F1_to_F2"):
    """
    Synchronize features along ground truth anchors.
    
    Parameters
    ----------
    F1 : np.ndarray, shape (n1, k)
        Feature matrix of graph G
    F2 : np.ndarray, shape (n2, k)
        Feature matrix of graph G_Q
    anchors_G : list[int]
        Anchor node indices in G (for F1)
    anchors_G_Q : list[int]
        Corresponding anchor node indices in G_Q (for F2)
    direction : str
        Either "F1_to_F2" or "F2_to_F1" indicating copy direction
    
    Returns
    -------
    F1_new, F2_new : np.ndarray
        Feature matrices after synchronization
    """
    # Make copies
    F1_new = F1.copy()
    F2_new = F2.copy()

    if direction == "F1_to_F2":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F2_new[u_q] = F1_new[u]
    elif direction == "F2_to_F1":
        for u, u_q in zip(anchors_G, anchors_G_Q):
            F1_new[u] = F2_new[u_q]
    else:
        raise ValueError("direction must be 'F1_to_F2' or 'F2_to_F1'")
    
    return F1_new, F2_new

def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end='\t')
    return G1, G2

def synchronize_graphs(G, G_Q, anchors_G, anchors_G_Q):
    """
    Create new graphs G1 and G1_Q where edges between aligned anchors
    are synchronized across G and G_Q.
    
    Parameters
    ----------
    G : nx.Graph
        Original graph G
    G_Q : nx.Graph
        Original graph G_Q
    anchors_G : list[int]
        List of nodes in G (anchors)
    anchors_G_Q : list[int]
        List of corresponding nodes in G_Q (same length as anchors_G)
    
    Returns
    -------
    G1, G1_Q : nx.Graph
        Synchronized graphs
    """

    # Build mapping dicts between anchors
    G_to_GQ = dict(zip(anchors_G, anchors_G_Q))
    GQ_to_G = dict(zip(anchors_G_Q, anchors_G))

    # Start with copies of the original graphs
    G1 = G.copy()
    G1_Q = G_Q.copy()
    counter=0
    counter1=0
    anchor_nodes = set(G_to_GQ.keys())
    anchor_edges = [(u, v) for u, v in G.edges() if u in anchor_nodes and v in anchor_nodes]
    print("Number of edges among anchors in G:", len(anchor_edges))

    # --- Step 1: Transfer edges from G to G_Q ---
    for u, v in G.edges():
        if u in G_to_GQ and v in G_to_GQ:
            u_q, v_q = G_to_GQ[u], G_to_GQ[v]
            if not G1_Q.has_edge(u_q, v_q):
                G1_Q.add_edge(u_q, v_q)
                counter=counter+1
            else:
                counter1=counter1+1

    # --- Step 2: Transfer edges from G_Q to G ---
    for u_q, v_q in G_Q.edges():
        if u_q in GQ_to_G and v_q in GQ_to_G:
            u, v = GQ_to_G[u_q], GQ_to_G[v_q]
            if not G1.has_edge(u, v):
                G1.add_edge(u, v)
                counter=counter+1
            else:
                counter1=counter1+1
    return G1, G1_Q

