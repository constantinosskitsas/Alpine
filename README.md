# Alpine
## **Introduction**
The necessity to align two graphs, minimizing a structural distance
metric, is prevalent in biology, chemistry, recommender systems,
and social network analysis. Due to the problem’s NP-hardness,
prevailing graph alignment methods follow an attributed and la-
beled approach, for equally-size graphs, solving a linear assignment
problem over intermediary graph representations. However, in re-
ality we often phase unrestricted scenarios, where we seek to align
nodes of unequally-size graphs with no additional label or attributed
information provided. Prior graph alignment methods distort the un-
restricted scenario, and are hence predisposed to miss high-quality
solutions, by either adding dummy nodes to match the graph sizes
or extracting graph-embedding to solve the assignment problem.
To address these limitations, in this paper, we propose Alpine; a
P a rtial Unlabeled Gra ph Ali gn me nt algorithm, inspired by shape
matching, that maps nodes of smaller graph 𝐻 (V𝐻 , E𝐻 ) to nodes
of a larger graph 𝐺 (V𝐺 , E𝐺 ), |V𝐻 | < |V𝐺 |, by purely operating
on their adjacency matrices with no external labeled or attributed
information. Extensive experimentation demonstrates that Alpine
is more efficient and consistently surpasses state-of-the-art graph
alignment methods using dummy nodes, in both accuracy and
edge-disagreements, across all benchmark datasets.


## Required Libraries
numpy, torch, scipy, networkx, matplotlib, tqdm, scikit-learn, pandas