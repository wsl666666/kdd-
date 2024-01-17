# DynGL: Dynamic Graphs Learning Library

## ```DynGraph``` maintain the structure of dynamic graphs based on graph events.
Internally, we maintain the graph structure using ```scipy.sparse.csr_matrix()``` and store/update the node/edge changes w.r.t. the following events:
- Edge-level events:
    - Edge structure event. (e.g., add/delete edge weights)
    - Edge feature event. (e.g., override edge feature)
    - Edge label event. (e.g., override edge label for edge type prediction)
- Node-level events:
    - Node feature event.(e.g., override node tagged feature)
    - Node label event. (e.g., override node label for node type prediction)
- ```DynGraphLoader``` base class to generate snapshot-based dynamic graphs. 
- Callback function interface for external graph algorithm, which can be triggered by the following events:
    - before/after individual consolidated edge structure event (e.g., add/delete edge weights)
    - before/after individual consolidated node event 
    - before/after whole consolidated node events for one snapshot.
    - before/after whole consolidated edge events for one snapshot.


## ```DynPageRank``` contains incremental pagerank algorithms for dynamic graphs:
- Power Iteration
- Forward Push and Dynamic Adjustment
- ISTA for L-1 regularized optimizaiton

# Get Started: 
## 1. Set up environment

Recommend to use [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for faster conda environment solving.
```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Environment setup (included in ```./scripts/setup.sh```):
```
conda create --name DynGL python=3.8
conda activate DynGL

/*we use cuda 11.6*/
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install scipy==1.9.3
pip install dataclasses_json
pip install numba==0.56.4
pip install overrides==7.3.1
pip install pandas
pip install matplotlib
pip install ogb==1.3.5
pip install torchsummary
#torch_geometric.datasets may need torch_sparse, please select the
#corresponding cuda version. We use cuda 11.6.

pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
# pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
# pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric

```
## 2. Run sanity check 
```
# cd to current project
<USER>@<SERVER>:~$ conda activate DynGL
(DynGL) <USER>@<SERVER>:~$ cd ~/git_project/DynMixer/src

# run three PPR algorithms and see l-1 error compared to Power Iteration Algo. 

(DynGL) <USER>@<SERVER>:~/git_project/DynMixer/src$ python integrate_dyngl_ppr_test.py
==========power_iter (as gold result)==========
-- ppr l1 error:
[0. 0. 0. ... 0. 0.]
==========ista==========
-- ppr l1 error:
[1.2649567e-06 1.2802094e-06 1.2660086e-06, ..., 2.4550536e-06 2.4590636e-06]
==========forward_push==========
-- ppr l1 error:
[1.0096576e-06 1.0818850e-06 1.0030747e-06, ..., 5.2593518e-06 5.3968042e-06]
```


# Mixer(MixRank)-in-progress: Mixing PageRank for Node Classification 
- Inspired from MixHop, which directly aggreagates features from K-hop neighbors 
- MixRank -> aggreagate neighbors based on high/low PageRank



# TODOs:
- ~~Debug with CiteSeer graph (have disconnected components)~~
- Add Dyn ISTA (restart ISTA is 2x faster than others.)
- ~~Compare Dyn/Restart PwrIter~~
- ~~Add Classifier (Mixer)~~
- Add ppr speed comparison scripts