
import torch
import numpy as np
from HTC.MyNet import *
from HTC.utils import *
# from algorithms.network_alignment_model import NetworkAlignmentModel
from torch import optim
import torch.nn.functional as F
import time
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
# from input.dataset import Dataset
#from utils.graph_utils import load_gt
# import utils.graph_utils as graph_utils
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from abc import ABCMeta
import orbit_count
from argparse import ArgumentParser
import os
import json
import pandas as pd
import networkx as nx

class HTC:
    __metaclass__ = ABCMeta
    def __init__(self, src_laps, trg_laps, src_feat, trg_feat, groundtruth, args):
        # super().__init__(source_dataset, target_dataset)
        # self.src_data = source_dataset
        # self.trg_data = target_dataset
        self.gt = groundtruth
        #print('num of groundtruth: %d'%len(self.gt))
        self.args = args
        self.src_feat = torch.Tensor(src_feat)
        self.trg_feat = torch.Tensor(trg_feat)
        # self.src_A, self.trg_A, self.src_feat, self.trg_feat = get_elements(source_dataset, target_dataset)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src_feat = self.src_feat.to(self.device)
        self.trg_feat = self.trg_feat.to(self.device)
        # if self.args.first_run:
        #     src_goms = orca2gom(self.args.source_dataset, self.src_A)
        #     self.src_laps = torch.Tensor(gom2lap(src_goms)).to(self.device)
        #     trg_goms = orca2gom(self.args.target_dataset, self.trg_A)
        #     self.trg_laps = torch.Tensor(gom2lap(trg_goms)).to(self.device)
        #     torch.save(self.src_laps, args.source_dataset + '/src_laps.pt')
        #     torch.save(self.trg_laps, args.target_dataset + '/trg_laps.pt')
        # else:
        #     print('loading orbit_laplacian matrices...')
        #     self.src_laps = torch.load(args.source_dataset + '/src_laps.pt').to(self.device)
        #     self.trg_laps = torch.load(args.target_dataset + '/trg_laps.pt').to(self.device)
        self.src_laps = src_laps
        self.trg_laps = trg_laps
        

        print('the shape of laps: ', self.src_laps.shape)
        self.num_node_s = self.src_feat.shape[0]
        self.num_node_t = self.trg_feat.shape[0]
        self.num_feat = self.src_feat.shape[1]
        print('attribute dimension: %d'%self.num_feat)
        self.num_hid1 = args.hid_dim
        self.num_hid2 = args.hid_dim
        self.utrain_epoch = args.num_utrn
        self.utrain_lr = args.ulr
        self.ftune_epoch = args.num_ftune
        self.ftune_lr = args.flr
        self.ftune_alpha = args.alpha
        self.k = args.k

    def align(self):

        myNet = MyNet(self.num_node_s, self.num_node_t, self.num_feat, self.num_hid1, self.num_hid2, self.args.p).to(self.device)

        myNet = self.unsupervised_train(myNet)

        myNet, count_max = self.trusted_refine(myNet)

        S_MyAlign = self.weighted_integration(myNet, count_max)

        return S_MyAlign

    def unsupervised_train(self, myNet):
        myNet.train()
        utrain_optimizer = optim.Adam(myNet.parameters(), lr=self.utrain_lr)
        rec_loss = Reconstruction_loss()
        loss_recorder = np.zeros((len(self.src_laps), self.utrain_epoch+1))
        for epoch in range(self.utrain_epoch):
            for i in range(len(self.src_laps)):
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
                src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
                trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
                trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
                loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
                loss_recorder[i, epoch] = loss_st
                print('epoch %d | loss_%d: %.4f' % (epoch, i, loss_st))
                utrain_optimizer.zero_grad()
                loss_st.backward()
                utrain_optimizer.step()
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
            src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
            trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
            trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
            loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
            loss_recorder[i, epoch+1] = loss_st
            print('epoch %d | loss_%d: %.4f' % (epoch+1, i, loss_st))
        return myNet

    def trusted_refine(self, myNet):
        if self.ftune_epoch>0:
            print('doing refinement')
            count_max = torch.zeros(len(self.src_laps))
            tune_flag = torch.ones(len(self.src_laps))
        else:
            count_max = torch.ones(len(self.src_laps))/len(self.src_laps)
            return myNet, count_max
        for epoch in range(self.ftune_epoch):
            if tune_flag.sum() == 0:
                print('done refinement')
                break
            for i in range(len(self.src_laps)):
                if tune_flag[i] == False:
                    print('epoch %d_%d: undo' % (epoch, i))
                    continue
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                csls = CSLS(src_output.detach(), trg_output.detach(), self.k)
                index_r = torch.argmax(csls, dim = 0)
                index_c = torch.argmax(csls, dim = 1)
                count = 0
                qs = torch.ones(len(self.src_laps[i])).to(self.device)
                qt = torch.ones(len(self.trg_laps[i])).to(self.device)
                for j in range(len(index_r)):
                    if j == index_c[index_r[j]]:
                        count += 1
                        qs[index_r[j]] *= self.ftune_alpha
                        qt[j] *= self.ftune_alpha
                qs = qs.reshape(-1, 1)
                qt = qt.reshape(-1, 1)
                if count > count_max[i]:
                    count_max[i] = count
                    self.src_laps[i] = (qs * (self.src_laps[i] * qs).t()).t()
                    self.trg_laps[i] = (qt * (self.trg_laps[i] * qt).t()).t()
                else:
                    tune_flag[i] = False
                print('epoch %d_%d: mutual closest pairs: %d' % (epoch, i, count))
        return myNet, count_max

    def weighted_integration(self, myNet, count_max):
        myNet.eval()
        total_count = sum(count_max)
        count_max = count_max / total_count
        score = torch.zeros((self.num_node_s, self.num_node_t)).to(self.device)
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            s = CSLS(src_output.detach(), trg_output.detach(), self.k)
            score += count_max[i] * s
        score = score.detach().cpu().numpy()
        return score
    
def make_args():
    parser_MyAlign = ArgumentParser()
    parser_MyAlign.add_argument('--first_run', action = "store_true", default = False)
    parser_MyAlign.add_argument('--cuda', action = "store_true")
    parser_MyAlign.add_argument('--gm', action = "store_true", default = False)
    parser_MyAlign.add_argument('--hid_dim', type=int, default=500)
    parser_MyAlign.add_argument('--num_utrn', type=int, default=15)
    parser_MyAlign.add_argument('--num_ftune', type=int, default=25)
    parser_MyAlign.add_argument('--k', type=int, default=40)
    parser_MyAlign.add_argument('--ulr', type=float, default=0.01)
    parser_MyAlign.add_argument('--flr', type=float, default=0.01)
    parser_MyAlign.add_argument('--alpha', type=float, default=1.1)
    parser_MyAlign.add_argument('--p', type=float, default=0.5)


    args = parser_MyAlign.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args
def make_args_orig():
    parser_MyAlign = ArgumentParser()
    parser_MyAlign.add_argument('--first_run', action = "store_true", default = False)
    parser_MyAlign.add_argument('--cuda', action = "store_true")
    parser_MyAlign.add_argument('--gm', action = "store_true", default = False)
    parser_MyAlign.add_argument('--hid_dim', type=int, default=200)
    parser_MyAlign.add_argument('--num_utrn', type=int, default=10)
    parser_MyAlign.add_argument('--num_ftune', type=int, default=20)
    parser_MyAlign.add_argument('--k', type=int, default=20)
    parser_MyAlign.add_argument('--ulr', type=float, default=0.01)
    parser_MyAlign.add_argument('--flr', type=float, default=0.01)
    parser_MyAlign.add_argument('--alpha', type=float, default=1.1)
    parser_MyAlign.add_argument('--p', type=float, default=0.5)

    args = parser_MyAlign.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args
def load_data_from_txt(dataset_dir, p, use_attr=True, dtype=np.float32):
    """
    Load dataset directly from .txt files (same output as load_data from .npz).

    :param dataset_dir: folder path containing edge and feature files
    :param p: training ratio (for naming consistency)
    :param use_attr: whether to load node attributes
    :param dtype: data type for attributes
    :return:
        edge_index1, edge_index2, x1, x2, anchor_links, test_pairs
    """

    # --- Load edge indices ---
    edge_index1 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_t_edge.txt', dtype=np.int64)
    edge_index2 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_s_edge.txt', dtype=np.int64)

    # Ensure shape consistency (transpose to match .npz format)
    if edge_index1.ndim == 1:
        edge_index1 = edge_index1[None, :]  # handle 1-line files
    if edge_index2.ndim == 1:
        edge_index2 = edge_index2[None, :]
    edge_index1 = edge_index1# shape (2, num_edges)
    edge_index2 = edge_index2
    
    if edge_index1.shape[0] == 2:
        edge_index1 = edge_index1.T
    if edge_index2.shape[0] == 2:
        edge_index2 = edge_index2.T

    # --- Load node attributes (if used) ---
    if use_attr:
        x1 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_t_feat.txt', dtype=np.float32)  # shape: (n1, k)
        x2 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_s_feat.txt', dtype=np.float32)  # shape: (n2, k)
    else:
        x1, x2 = None, None
    if (dataset_dir=="acm_dblp"):
                        data = np.load(f'JOENA/datasets/ACM-DBLP_0.2.npz')
                        F2=data['x2']
                        F1=data['x1']
    if (dataset_dir=="douban"):
        x1 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr1.csv", header=None).iloc[:, 1:].to_numpy()
        x2 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr2.csv", header=None).iloc[:, 1:].to_numpy()
    print(f"✅ Loaded dataset from {dataset_dir} (p={p:.1f})")
    return edge_index1, edge_index2, x1, x2

def build_nx_graph(edge_index, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    #if x is not None:
        #G.add_nodes_from(np.arange(x.shape[0]))
    max1=0
    if x is not None:
        max1 = x.shape[0]
    max2 = np.max(edge_index)

    G.add_nodes_from(np.arange(max(max1,max2)))    
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G

def HTC_main(dataset, ratio, GT, src_name, trg_name, src_laps_name, trg_laps_name):
    args = make_args()
    if os.path.exists(f"HTC/settings/{dataset}.json"):
        print(f"Loading settings from settings/{dataset}.json")
        with open(f"HTC/settings/{dataset}.json", 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                print(f"Setting {key} to {value}")
                setattr(args, key, value)
    else:
        print(f"Using default arguments from command line")
    #assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    
    edge_index1, edge_index2, x1, x2 = load_data_from_txt(dataset, ratio)
    
    if os.path.exists(src_laps_name) and os.path.exists(trg_laps_name):
        src_laps = torch.load(src_laps_name).to(device)
        trg_laps = torch.load(trg_laps_name).to(device)
    else:
        G1 = build_nx_graph(edge_index1, x1)
        G1.remove_edges_from(nx.selfloop_edges(G1))
        adj1 = nx.adjacency_matrix(G1)
        
        

        if os.path.exists(src_name):
            src_orbit_counts = np.loadtxt(src_name)
        else:
            src_orbit_counts = orbit_count.edge_orbit_counts(G1, graphlet_size=4)
            np.savetxt(src_name, src_orbit_counts.astype(int), fmt='%i')
        
        edges = np.array(list(G1.edges()))
        del G1
        src_goms = orca2gom(edges, src_orbit_counts, adj1)
        #src_goms = orca2gom(edge_index1, src_orbit_counts, adj1)
        del adj1
        
        G2 = build_nx_graph(edge_index2, x2)
        G2.remove_edges_from(nx.selfloop_edges(G2))

        adj2 = nx.adjacency_matrix(G2)
        if os.path.exists(trg_name):
            trg_orbit_counts = np.loadtxt(trg_name)
        else:
            trg_orbit_counts = orbit_count.edge_orbit_counts(G2, graphlet_size=4)
            np.savetxt(trg_name, trg_orbit_counts.astype(int), fmt='%i')
        edges = np.array(list(G2.edges()))
        del G2
        trg_goms = orca2gom(edges, trg_orbit_counts, adj2)
        del adj2
        print('compute laps')
        src_laps = torch.Tensor(gom2lap(src_goms)).to(device)
        trg_laps = torch.Tensor(gom2lap(trg_goms)).to(device)
        torch.save(src_laps, src_laps_name)
        torch.save(trg_laps, trg_laps_name)
        print('done laps')
    
    model = HTC(src_laps, trg_laps, x1, x2, GT, args)
    return model.align()
    
    
