# Building Modelling using GNN
- the most accurate model is defined in file `gcn2.py`

## gcn2.py
- uses GNN-RNN architecture and was able to achieve on par performance to a typical mlp (RNN) based model
- gcn2 uses a GCN for generating a representation of the graph nodes, and the nodes are passed into a RNN layer

## gnn model/training pipeline
1. `data_collect.py`: collects system-excited (randomly set actuation value) -> the data is not in graph format
2. `data_scale.py`: data dimension goes over the scikitlearn.StandardScaler capability
3. `data_process.py`: loop over the collected data, embed the vector data in graph format
4. `gcn2.py`: include PyG dataloader + model + training code
