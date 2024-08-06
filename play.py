CUDA_INDEX = 1
NAME = 'DBLP'
CLASSES = 8
import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '../../pyged/lib')


import os
import pickle
import random
import time

import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
import torch.optim
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm

from greed_main.greed_main.neuro import config, datasets, metrics, models, train, utils, viz
import pyged

from importlib import reload
reload(config)
reload(datasets)
reload(metrics)
reload(models)
reload(pyged)
reload(train)
reload(utils)
reload(viz)

model = models.NormSEDModel(8, CLASSES, 64, 64)
model.load_state_dict(torch.load(f'./greed-data-and-models/runlogs/{NAME}-FS/1633411903.8613322/best_model.pt', map_location='cpu'))

nn_model = models.NeuralSiameseModel(8, CLASSES, 64, 64)
nn_model.load_state_dict(torch.load(f'./greed-data-and-models/runlogs/{NAME}-NN-FS/1633341624.735046/best_model.pt', map_location='cpu'))

dual_model = models.DualNormSEDModel(8, CLASSES, 64, 64)
dual_model.load_state_dict(torch.load(f'./greed-data-and-models/runlogs/{NAME}-Dual-FS/1633375855.20642/best_model.pt', map_location='cpu'))

inner_test_set, _ = torch.load(f'./greed-data-and-models/data/{NAME}/inner_test.pt', map_location='cpu')
#inner_queries, inner_targets, _, _ = inner_test_set
#inner_test_set, _ = torch.load(f'./greed-data-and-models/data/{NAME}/inner_test.pt', map_location='cpu')
inner_targets=torch.load(f'./greed-data-and-models/data/{NAME}/inner_targets_modified.pt', map_location='cpu')
inner_queries=torch.load(f'./greed-data-and-models/data/{NAME}/inner_queries_modified.pt', map_location='cpu')
#print(inner_queries)
print(len(inner_queries))
#print(inner_queries[0].x)
for x in inner_queries:
    print(len(x))
    print(x)
    break

new_label = 1  # Change this to the desired label

# Modify the labels in the node features tensor (x)
#for i in range(len(inner_queries)):
#    num_nodes = inner_queries[i].x.size(0)  # Get the number of nodes
#    inner_queries[i].x = torch.full((num_nodes, 1), new_label, dtype=torch.float)

# Verify by printing the features of the first graph
print(inner_queries[0])
# Save the modified dataset back if needed
#torch.save((inner_targets, _), f'./greed-data-and-models/data/{NAME}/inner_targets_modified.pt')


#print(len(inner_targets),"targets")
config.n_workers=1
tic = time.time()
#inner_pred = model.predict_inner(inner_queries, inner_targets, batch_size=4096)
toc = time.time()
#torch.save(inner_pred, f'./preds/inner_pred.pt')
print(f'NeuroGSim prediction time: {toc-tic:.3f}s')

