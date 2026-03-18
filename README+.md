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
python supervise_GA_Attr.py #: This will run supervision experiment (Figure X)
python attributed_GA.py #: This will run Attributed Experiment (Figure X)
```
### Example with real graphs:
Please see Example_LSG.ipynb. to run example with real graphs and how to run the algorithm.

## Reference

Please cite our work in your publications if it helps your research:

```
under submission

```
