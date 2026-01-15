import os.path

#from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time
import json

from JOENA.args import *
from JOENA.utils import *
from JOENA.model import *
import numpy as np
import os
import pandas as pd
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

    # --- Load node attributes (if used) ---
    if use_attr:
        x1 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_t_feat.txt', dtype=np.float32)  # shape: (n1, k)
        x2 = np.loadtxt(f'./Data/data/{dataset_dir}/{dataset_dir}_s_feat.txt', dtype=np.float32)  # shape: (n2, k)
    else:
        x1, x2 = None, None
    if (dataset_dir=="douban"):
        x1 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr1.csv", header=None).iloc[:, 1:].to_numpy()
        x2 = pd.read_csv(f"./Data/Full-dataset/attribute/{dataset_dir}attr2.csv", header=None).iloc[:, 1:].to_numpy()
    if (dataset_dir=="acm_dblp"):
        data = np.load(f'JOENA/datasets/ACM-DBLP_0.2.npz')
        x1=data['x2']
        x2=data['x1']
    print(f"✅ Loaded dataset from {dataset_dir} (p={p:.1f})")
    return edge_index1, edge_index2, x1, x2
def JOENA(dataset,ratio,use_attr,anchor_links):
    args = make_args()
    if os.path.exists(f"JOENA/settings/{dataset}.json"):
        print(f"Loading settings from settings/{dataset}.json")
        with open(f"JOENA/settings/{dataset}.json", 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                print(f"Setting {key} to {value}")
                setattr(args, key, value)
    else:
        print(f"Using default arguments from command line")
    print("Loading data...", end=" ")
    print(use_attr)
    edge_index1,edge_index2,x1,x2=load_data_from_txt(dataset,ratio,use_attr)
    #edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data(f"JOENA/datasets/Douban",ratio,
    #                                                                      use_attr, dtype=np.float64)

    #edge_index1 = list(map(tuple, edge_index1.T))
    #edge_index2 = list(map(tuple, edge_index2.T))
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]

    G1, G2 = build_nx_graph(edge_index1, anchor1, x1), build_nx_graph(edge_index2, anchor2, x2)
    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, dataset, ratio, dtype=np.float64)
    if x1 is None:
        x1 = rwr1
    else:
        x1 = np.concatenate([x1, rwr1], axis=1)
    if x2 is None:
        x2 = rwr2
    else:
        x2 = np.concatenate([x2, rwr2], axis=1)

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)

    # build PyG Data objects
    G1_tg = build_tg_graph(edge_index1, x1, rwr1, dtype=torch.float64).to(device)
    G2_tg = build_tg_graph(edge_index2, x2, rwr2, dtype=torch.float64).to(device)
    n1, n2 = G1_tg.x.shape[0], G2_tg.x.shape[0]
    args.gw_weight = args.alpha / (1 - args.alpha) * min(n1, n2) ** 0.5

    #out_dir = 'logs'
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
    #writer = SummaryWriter(save_path(args.dataset, out_dir, args.use_attr))

    max_hits_list = defaultdict(list)
    max_mrr_list = []
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        model = MLP(input_dim=G1_tg.x.shape[1],
                    hidden_dim=args.hidden_dim,
                    output_dim=args.out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FusedGWLoss(G1_tg, G2_tg, anchor1, anchor2,
                                gw_weight=args.gw_weight,
                                gamma_p=args.gamma_p,
                                init_threshold_lambda=args.init_threshold_lambda,
                                in_iter=args.in_iter,
                                out_iter=args.out_iter,
                                total_epochs=args.epochs).to(device)

        print("Training...")
        max_hits = defaultdict(int)
        max_mrr = 0
        print(args.epochs)
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            optimizer.zero_grad()
            out1, out2 = model(G1_tg, G2_tg)
            loss, similarity, threshold_lambda = criterion(out1=out1, out2=out2)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}', end=', ')
        #hits, _ = compute_metrics(-similarity, test_pairs)    
        return similarity*1
            # testing
            #with torch.no_grad():
            #    model.eval()
            #    hits, mrr = compute_metrics(-similarity, test_pairs)
            #    s_entropy = torch.sum(-similarity * torch.log(similarity))
            #    end = time.time()
            #    print(f's_entropy: {s_entropy:.4f}, threshold_lambda: {threshold_lambda * n1 * n2:.4f}, '
            #          f'{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}')
#
            #    max_mrr = max(max_mrr, mrr.cpu().item())
            #    for key, value in hits.items():
            #        max_hits[key] = max(max_hits[key], value.cpu().item())
#
             #   writer.add_scalar('Loss', loss.item(), epoch)
            #    writer.add_scalar('MRR', mrr, epoch)
            #    for key, value in hits.items():
             #       writer.add_scalar(f'Hits/Hits@{key}', value, epoch)

        #for key, value in max_hits.items():
        #    max_hits_list[key].append(value)
        #max_mrr_list.append(max_mrr)

    #    print("")

    #max_hits = {}
    #max_hits_std = {}
    #for key, value in max_hits_list.items():
    #    hits_list = np.array([val for val in value])
    #    max_hits[key] = hits_list.mean()
    #    max_hits_std[key] = hits_list.std()
    #max_mrr = np.array(max_mrr_list).mean()
    #max_mrr_std = np.array(max_mrr_list).std()

    #hparam_dict = {
    #    'dataset': args.dataset,
    #    'use_attr': args.use_attr,
    #    'epochs': args.epochs,
    #    'lr': args.lr,
    #    'alpha': args.alpha,
    #    'gamma_p': args.gamma_p,
    #    'threshold_lambda': threshold_lambda.cpu().item(),
    #}
    #writer.add_hparams(hparam_dict, {'hparam/MRR': max_mrr,
    #                                 'hparam/std_MRR': max_mrr_std,
    #                                 **{f'hparam/Hits@{key}': value for key, value in max_hits.items()},
    #                                 **{f'hparam/std_Hits@{key}': value for key, value in max_hits_std.items()}})
