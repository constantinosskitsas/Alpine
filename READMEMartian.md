# Alpine
## **Introduction**
...

## Execution
The code is written in Python

### Required Libraries
numpy, torch, scipy, networkx, matplotlib, tqdm, scikit-learn, pandas


#### Partially mutual graph alignment experiments:

For the synthetic experiments with injected ground truth
1. Set the _ksizes_ parameter for ground truth size and the _ptune_ for algorithm choice in _synthetic.py_
2. Run the script:
   ```shell
   python synthetic.py
   ```
Please note that in the first run, the newly injected graphs will be generated and stored in .json files. Subsequent runs will use the cached graphs.

For the experiment of the malaria-football pair in our paper, use the script:
```shell
python synthetic_hard.py
```

To run the experiment with original graphs:
```shell
python diffgraph.py
```
## Reference

Please cite our work in your publications if it helps your research:

```
Paper under submission
```

