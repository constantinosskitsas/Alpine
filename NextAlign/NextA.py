import torch
import dgl
from NextAlign.utils.utils import *
from NextAlign.utils.rwr_scoring import rwr_scores
from NextAlign.utils.test import test,compute_full_similarity
from NextAlign.utils.node2vec import load_walks
from NextAlign.model.model import Model
from NextAlign.model.negative_sampling import negative_sampling_exact
from NextAlign.dataset.data import Train_Data
import networkx as nx
import argparse
import time, os
import pandas as pd
from torch.utils.data import DataLoader
def make_args():
    

    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--dim', type=int, default=128, help='dimension of output embeddings.')
    parser.add_argument('--num_layer', type=int, default=1, help='number of layers.')
    parser.add_argument('--ratio', type=float, default=0.2, help='training ratio.')
    parser.add_argument('--coeff1', type=float, default=1.0, help='coefficient for within-network link prediction loss.')
    parser.add_argument('--coeff2', type=float, default=1.0, help='coefficient for anchor link prediction loss.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs.')
    parser.add_argument('--batch_size', type=int, default=300, help='batch_size.')
    parser.add_argument('--walks_num', type=int, default=100,
                        help='length of walk per user node.')
    parser.add_argument('--N_steps', type=int, default=10,
                        help='burn-in iteration.')
    parser.add_argument('--N_negs', type=int, default=20,
                        help='number of negative samples per anchor node.')
    parser.add_argument('--p', type=int, default=1,
                        help='return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=int, default=1,
                        help='inout hyperparameter. Default is 1.')
    parser.add_argument('--walk_length', type=int, default=80,
                    help='Length of walk per source. Default is 80.')
    parser.add_argument('--num_walks', type=int, default=10,
                    help='Number of walks per source. Default is 10.')
    parser.add_argument('--dataset', type=str, default='new_ACM-DBLP', help='dataset name.')
    parser.add_argument('--gpu', type=int, default=0, help='cuda number.')
    parser.add_argument('--dist', type=str, default='L1', help='distance for scoring.')
    return parser.parse_args()
def load_data_from_txt(dataset_dir, use_attr=True,Perm=None):
    """
    Load dataset directly from .txt files (same output as load_data from .npz).

    :param dataset_dir: folder path containing edge and feature files
    :param p: training ratio (for naming consistency)
    :param use_attr: whether to load node attributes
    :param dtype: data type for attributes
    :return:
        edge_index1, edge_index2, x1, x2, anchor_links, test_pairs
    """
    dtype=np.float32
    # --- Load edge indices ---
    edge_index1 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_t_edge.txt', dtype=np.int64)
    edge_index2 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_s_edge.txt', dtype=np.int64)
    if (dataset_dir=="phone"):
        edge_index2=Perm[edge_index2]
    # Ensure shape consistency (transpose to match .npz format)
    if edge_index1.ndim == 1:
        edge_index1 = edge_index1[None, :]  # handle 1-line files
    if edge_index2.ndim == 1:
        edge_index2 = edge_index2[None, :]
    edge_index1 = edge_index1# shape (2, num_edges)
    edge_index2 = edge_index2

    # --- Load node attributes (if used) ---
    if use_attr:
        x1 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_t_feat.txt', dtype=np.float32)  # shape: (n1, k)
        x2 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_s_feat.txt', dtype=np.float32)  # shape: (n2, k)
    else:
        x1, x2 = None, None
    if (dataset_dir=="douban"):
        x1 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr1.csv", header=None).iloc[:, 1:].to_numpy()
        x2 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr2.csv", header=None).iloc[:, 1:].to_numpy()
    #print(f"✅ Loaded dataset from {dataset_dir} (p={p:.1f})")
    return edge_index1, edge_index2, x1, x2

def compare(nameA, A, nameB, B):
    #print(f"\n===== Comparing {nameA}  vs  {nameB} =====")

    # Convert to numpy for comparison
    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy()
    else:
        A_np = np.array(A)

    if isinstance(B, torch.Tensor):
        B_np = B.cpu().numpy()
    else:
        B_np = np.array(B)

    #print(f"{nameA} dtype:", A_np.dtype)
    #print(f"{nameB} dtype:", B_np.dtype)

    #print(f"{nameA} shape:", A_np.shape)
    #print(f"{nameB} shape:", B_np.shape)

    # Check if identical
    same_shape = (A_np.shape == B_np.shape)
    same_values = np.array_equal(A_np, B_np)

    #print("Same shape?:", same_shape)
    #print("Same values?:", same_values)

    # If not identical, show difference summary
    if same_shape and not same_values:
        diff = A_np != B_np
        print("Number of differing elements:", diff.sum())

        # print first few differing entries
        idx = np.argwhere(diff)
        #print("First 10 differences (index, A, B):")
        for i in idx[:10]:
            idx_tuple = tuple(i)
            #print(idx_tuple, A_np[idx_tuple], B_np[idx_tuple])


def NextAlign(dataset,use_attr,anchor_links,Perm=None):
    args = make_args()
    edge_index2,edge_index1,x2,x1=load_data_from_txt(dataset,use_attr,Perm)
    anchor_nodes1, anchor_nodes2 = anchor_links[:, 0], anchor_links[:, 1]
    anchor_links2= anchor_nodes2

    G1, G2 = nx.Graph(), nx.Graph()
    if (use_attr):
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)
        max1=x1.shape[0]
        max2=np.max(edge_index1)
        G1.add_nodes_from(np.arange(max(max1,max2))) 
        max1=x2.shape[0]
        max2=np.max(edge_index2)
        G2.add_nodes_from(np.arange(max(max1,max2))) 
    #edge_index1A, edge_index2A, x1A, x2A, anchor_linksA, _ = load_data(f'NextAlign/dataset/foursquare-twitter_0.2.npz',0.2,False)
    #compare("edge_index1", edge_index1, "edge_index1A", edge_index1A)
    #compare("edge_index2", edge_index2, "edge_index2A", edge_index2A)
    #compare("x1", x1, "x1A", x1A)
    #compare("x2", x2, "x2A", x2A)
    #edge_index1,edge_index2,x1,x2=edge_index1A, edge_index2A, x1A, x2A
    #print(max(anchor_linksA[:, 0]),max(anchor_nodes1))
    #print(max(anchor_linksA[:, 1]),max(anchor_nodes2))
    #anchor_nodes1, anchor_nodes2 = anchor_linksA[:, 0], anchor_linksA[:, 1]
    #anchor_links2= anchor_nodes2
    
    
    G1.add_edges_from(edge_index1)
    G2.add_edges_from(edge_index2)
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    #print(n1,n2)
    #print(max(anchor_nodes1),max(anchor_nodes2))
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1
    for edge in G2.edges():
        G2[edge[0]][edge[1]]['weight'] = 1

    ################################################################################################
    # run node2vec or load from existing file for positive context pairs
    t0 = time.time()
        # run node2vec from scratch
    walks1 = load_walks(G1, args.p, args.q, args.num_walks, args.walk_length)
    walks2 = load_walks(G2, args.p, args.q, args.num_walks, args.walk_length)
    context_pairs1 = extract_pairs(walks1, anchor_nodes1)
    context_pairs2 = extract_pairs(walks2, anchor_nodes2)
    context_pairs1, context_pairs2 = balance_inputs(context_pairs1, context_pairs2)
    ################################################################################################
    # run random walk with restart or load from existing file for pre-positioning
    t0 = time.time()
    rwr_score1, rwr_score2 = rwr_scores(G1, G2, anchor_links)
    print("Anchor nodes:", len(anchor_nodes1))
    print("Context pairs G1:", len(context_pairs1))
    print("Context pairs G2:", len(context_pairs2))
    ################################################################################################
    # Set initial relative positions
    position_score1, position_score2 = anchor_emb(G1, G2, anchor_links)
    for node in G1.nodes:
        if node not in anchor_nodes1:
            position_score1[node] += rwr_score1[node]
    for node in G2.nodes:
        if node not in anchor_nodes2:
            position_score2[node] += rwr_score2[node]
    x1 = (position_score1, x1) if use_attr else position_score1
    x2 = (position_score2, x2) if use_attr else position_score2
    #print('Finished initial relative positioning in %.2f seconds' % (time.time() - t0))

    ################################################################################################
    # merge input networks into a world-view network
    t0 = time.time()
    node_mapping1 = np.arange(G1.number_of_nodes()).astype(np.int64)
    edge_index, edge_types, x, node_mapping2 = merge_graphs(edge_index1, edge_index2, x1, x2, anchor_links)
    #print('Finished merging networks in %.2f seconds' % (time.time() - t0))

    # input node features: (one-hot encoding, position, optional - node attributes)
    x1 = np.arange(len(x[0]), dtype=np.int64) if use_attr else np.arange(len(x), dtype=np.int64)
    x2 = x[0].astype(np.float32) if use_attr else x.astype(np.float32)
    x = (x1, x2, x[1]) if use_attr else (x1, x2)

    args.device = 'cpu'
    #torch.manual_seed(args.seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(args.seed)
    #    args.device = 'cuda:%d' % args.gpu

    landmark = torch.from_numpy(anchor_nodes1).to(args.device)
    num_nodes = x[0].shape[0]
    num_attrs = x[2].shape[1] if use_attr else 0
    num_anchors = x[1].shape[1]

    model = Model(num_nodes, args.dim, landmark, args.dist, num_anchors=num_anchors, num_attrs=num_attrs)

    ################################################################################################
    # to device
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = dgl.graph((edge_index.T[0], edge_index.T[1]), device=args.device)
    x1 = torch.from_numpy(x[0]).to(args.device)
    x2 = torch.from_numpy(x[1]).to(args.device)
    x = (x1, x2, torch.from_numpy(x[2]).to(args.device)) if use_attr else (x1, x2)
    edge_types = torch.from_numpy(edge_types).to(args.device)
    node_mapping1 = torch.from_numpy(node_mapping1).to(args.device)
    node_mapping2 = torch.from_numpy(node_mapping2).to(args.device)

    ################################################################################################
    # prepare training data
    dataset = Train_Data(context_pairs1, context_pairs2)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_size = len(data_loader)

    ################################################################################################
    # start training
    pn_examples1, pn_examples2, pnc_examples1, pnc_examples2 = [], [], [], []
    t_neg_sampling, t_get_emb, t_loss, t_model = 0, 0, 0, 0
    total_loss = 0

    topk = [1, 10, 30, 50, 100]
    max_hits = np.zeros(len(topk), dtype=np.float32)
    max_hit_10, max_hit_30, max_epoch = 0, 0, 0

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(data_loader):
            nodes1, nodes2 = data
            nodes1 = nodes1.to(args.device)
            nodes2 = nodes2.to(args.device)
            anchor_nodes1 = nodes1[:, 0].reshape((-1,))
            pos_context_nodes1 = nodes1[:, 1].reshape((-1,))
            anchor_nodes2 = nodes2[:, 0].reshape((-1,))
            pos_context_nodes2 = nodes2[:, 1].reshape((-1,))
            # forward pass
            t0 = time.time()
            out_x = model(g, x, edge_types)
            t_model += (time.time() - t0)

            t0 = time.time()
            context_pos1_emb = out_x[node_mapping1[pos_context_nodes1]]
            context_pos2_emb = out_x[node_mapping2[pos_context_nodes2]]
            #print("here")
            pn_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                            'p_n', 'g1')
            pn_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                            'p_n', 'g2')
            pnc_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                                'p_nc', 'g1', node_mapping2=node_mapping2)
            pnc_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                                'p_nc', 'g2', node_mapping2=node_mapping1)

            t_neg_sampling += (time.time() - t0)

            # get node embeddings
            t0 = time.time()

            pn_examples1 = torch.from_numpy(pn_examples1).reshape((-1,)).to(args.device)
            pn_examples2 = torch.from_numpy(pn_examples2).reshape((-1,)).to(args.device)
            pnc_examples1 = torch.from_numpy(pnc_examples1).reshape((-1,)).to(args.device)
            pnc_examples2 = torch.from_numpy(pnc_examples2).reshape((-1,)).to(args.device)

            anchor1_emb = out_x[node_mapping1[anchor_nodes1]]
            anchor2_emb = out_x[node_mapping2[anchor_nodes2]]
            context_neg1_emb = out_x[node_mapping1[pn_examples1]]
            context_neg2_emb = out_x[node_mapping2[pn_examples2]]
            anchor_neg1_emb = out_x[node_mapping2[pnc_examples1]]
            anchor_neg2_emb = out_x[node_mapping1[pnc_examples2]]

            input_embs = (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb,
                        context_neg2_emb, anchor_neg1_emb, anchor_neg2_emb)

            t_get_emb += (time.time() - t0)

            # compute loss
            t0 = time.time()
            loss1, loss2 = model.loss(input_embs)
            total_loss = args.coeff1 * loss1 + args.coeff2 * loss2
            t_loss += (time.time() - t0)

            #("Epoch:{}, Iteration:{}, Training loss:{}, Loss1:{},"
            #    " Loss2:{}".format(epoch + 1, i + 1, round(total_loss.item(), 4), round(loss1.item(), 4),
            #                        round(loss2.item(), 4)))

            # backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_t_model = round(t_model / ((epoch+1) * data_loader_size), 2)
        avg_t_neg_sampling = round(t_neg_sampling / ((epoch+1) * data_loader_size), 2)
        avg_t_get_emb = round(t_get_emb / ((epoch+1) * data_loader_size), 2)
        avg_t_loss = round(t_loss / ((epoch+1) * data_loader_size), 2)
        time_cost = [avg_t_model, avg_t_neg_sampling, avg_t_get_emb, avg_t_loss]
    similarity= compute_full_similarity(model, g, x, edge_types,
                            node_mapping1, node_mapping2,
                            args.dist)
    return similarity
        #train_hits = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, anchor_links, anchor_links2, args.dist)
        #hits = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, test_pairs, anchor_links2, args.dist, 'testing')
        #print("Epoch:{}, Training loss:{}, Train_Hits:{},  Test_Hits:{}, Time:{}".format(
        #    epoch+1, round(total_loss.item(), 4), train_hits, hits, time_cost))

        






