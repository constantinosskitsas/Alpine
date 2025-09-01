# Alpine
## **Introduction**
Several applications call to align the nodes of two graphs in a way
that minimizes a distance function. In practicality, the graphs to
be aligned often have unequal orders (i.e., numbers of vertices)
and no auxiliary labels or attributes; we refer to this problem as
partial unlabeled graph alignment. Some proposals to address this
problem add dummy nodes to the smaller graph to even the orders
and align the ensuing graphs or employ embeddings such as GNNs,
which yield ad hoc node representations. Unfortunately, as we show,
an optimal solution to equal-order graph alignment using dummy
nodes does not imply an optimal solution to partial graph alignment.
To address this deficiency, in this paper, we propose Alpine, a
Partial Unlabeled Grap h Ali gn ment algorithm that solely peruses the
graphs’ adjacency matrices, guided by a tailored objective function
inspired by best-of-breed shape matching techniques and a state-of-
the-art optimization method. Extensive experiments demonstrate
that Alpine consistently surpasses state-of-the-art graph alignment
methods in solution quality across all benchmark datasets.

## Execution

### Required Libraries
numpy, torch, scipy, networkx, matplotlib, tqdm, scikit-learn, pandas

### How to run experiments :
```shell
python PartialTest.py #: This will run Size Ratio Experiment (Figure 5)
python NoiseTest.py #: This will run Noise Experiment (Figure 6)
```

## Reference

Please cite our work in your publications if it helps your research:

```
inproceedings{10.1145/3711896.3736839,
author = {Petsinis, Petros and Skitsas, Konstantinos and Ranu, Sayan and Mottin, Davide and Karras, Panagiotis},
title = {Alpine: Partial Unlabeled Graph Alignment},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3736839},
doi = {10.1145/3711896.3736839},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {2315–2325},
numpages = {11},
keywords = {partial graph alignment, quadratic assignment problem},
location = {Toronto ON, Canada},
series = {KDD '25}
}

```
