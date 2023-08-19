
import math
import os
import sys
import time
import gc

import numpy as np
import pickle
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin


class ThermoStaticGraphLoader(object):
    def __init__(self, data_file_path):
        super(ThermoStaticGraphLoader, self).__init__()
        self.data_file_path = data_file_path

    def _read_data(self):
        pf = open(self.data_file_path, 'rb')
        self.gnn_data = pickle.load(pf)
        pf.close()
        self.edge_index = torch.tensor([[2,3,0,2,3,0,2,1],
                                        [3,2,2,0,0,3,1,2]], dtype=torch.long)
        self.edge_weights = torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float)

        self.x = self.gnn_data[0]
        self.y = self.gnn_data[1]
        del self.gnn_data
        gc.collect()

        self.snapshot_count = len(self.x)

    def get_dataset(self):
        self._read_data()
        dataset = StaticGraphTemporalSignal(
            edge_index=self.edge_index,
            edge_weight=self.edge_weights,
            features=self.x,
            targets=self.y
        )
        self.gnn_data = []
        gc.collect()
        return dataset

loader =  ThermoStaticGraphLoader('./data/gnn_data.pt')
dataset = loader.get_dataset()

print(next(iter(dataset)))
print(next(iter(dataset)).y)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print("Number of train buckets: ", len((train_dataset.features)))
print("Number of test buckets: ", len((test_dataset.features)))


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, hidden_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=hidden_size,
                           periods=periods)
        # Equals single-shot prediction
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.recur1 = torch.nn.RNN(hidden_size,hidden_size)
        self.batch1 = torch.nn.BatchNorm1d(hidden_size)
        self.recur2 = torch.nn.RNN(hidden_size, hidden_size)
        self.batch2 = torch.nn.BatchNorm1d(hidden_size)
        self.out = torch.nn.Linear(hidden_size, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = self.drop1(h)
        h = F.relu(h)
        # h, _ = self.recur1(h)
        # h = self.batch1(h)
        h = self.linear1(h)
        h = F.relu(h)
        # h, _ = self.recur2(h)
        # h = self.batch2(h)
        # h = F.relu(h)
        h = self.out(h)
        return h

device = torch.device('cpu')
model = TemporalGNN(node_features=8, periods=20, hidden_size=15).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

subset = 1000

OUTDOOR = 0
GROUND = 1
LIVING = 2
ATTIC = 3

model.train()
print('Running training')
for epoch in range(10000):
    loss = 0
    step = 0
    for snapshot in train_dataset:
        y_hat = model(snapshot.x, snapshot.edge_index)
        y_hat = y_hat[LIVING]
        #print('y_hat', y_hat, y_hat.shape)
        loss = loss + loss_fn(y_hat, np.squeeze(snapshot.y))
        step += 1
        if step > subset:
            break

    loss = loss / (step + 1)
    loss = torch.sqrt(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
