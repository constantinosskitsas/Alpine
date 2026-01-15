import json
from Parrot.utils import *
from Parrot.args import *
from Parrot.parrot import parrot
import numpy as np
import networkx as nx
import torch
def Parrot(dataset,Gq,Gt,x1,x2,data_GT):
    args = make_args()
    with open(f"Parrot/settings/{dataset}.json", "r") as f:
        settings = json.load(f)

    graph1, graph2 = settings["graph1"], settings["graph2"]
    use_attr = settings["use_attr"]
    rwrIter = settings["rwrIter"]
    rwIter = settings["rwIter"]
    alpha = settings["alpha"]
    beta = settings["beta"]
    gamma = settings["gamma"]
    inIter = settings["inIter"]
    outIter = settings["outIter"]
    l1 = settings["l1"]
    l2 = settings["l2"]
    l3 = settings["l3"]
    l4 = settings["l4"]
    adj1 = nx.to_numpy_array(Gq)
    adj2 = nx.to_numpy_array(Gt)
    n1=np.shape(adj1)[0]
    n2=np.shape(adj2)[0]
    adj1 = torch.from_numpy(adj1).int()
    adj2 = torch.from_numpy(adj2).int()
    x1 = torch.from_numpy(x1).to(torch.float64) if x1 is not None else None
    x2 = torch.from_numpy(x2).to(torch.float64) if x2 is not None else None
    #gnd = torch.from_numpy(gnd).long()
    print(x1.shape)
    H = np.zeros((n2, n1), dtype=int)
    for i2, i1 in data_GT:
        H[i2, i1] = 1
    H = torch.from_numpy(H).int()
    #adj1, adj2, x1, x2, gnd, H = load_data(f"Parrot/datasets/Douban.mat", graph1, graph2, True)
    S, W, res = parrot(args.dataset, adj1, adj2, x1, x2, H, rwrIter, rwIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4, device=args.device)
    return S
