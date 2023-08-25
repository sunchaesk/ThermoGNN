
from tqdm import tqdm
import os
import sys
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric as pyg
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal

class ThermoStaticGraphNoChunk(object):
    '''
    residence building thermal modelling dataset loader
    '''
    def __init__(self, data_file_path):
        super(ThermoStaticGraphNoChunk, self).__init__()
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

        self.y = [[x] for x in self.y]
        self.y = np.array(self.y)

        del self.gnn_data
        gc.collect()

    def get_dataset(self):
        self._read_data() # propagate self.gnn_data
        dataset = StaticGraphTemporalSignal(
            edge_index=self.edge_index,
            edge_weight=self.edge_weights,
            features=self.x,
            targets=self.y
        )
        self.get_data = [] # prob clears memory?
        return dataset

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]

class ThermoStaticGraphChunk(object):
    def __init__(self, data_file_path, chunk_size):
        super(ThermoStaticGraphChunk, self).__init__()
        self.data_file_path = data_file_path
        self.chunk_size = chunk_size

    def _read_data(self):
        pf = open(self.data_file_path, 'rb')
        self.gnn_data = pickle.load(pf)
        pf.close()
        self.edge_index = torch.tensor([[2,3,0,2,3,0,2,1],
                                        [3,2,2,0,0,3,1,2]], dtype=torch.long)
        self.edge_weights = torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float)

        self.x = self.gnn_data[0]
        self.y = self.gnn_data[1]

        self.y = [[x] for x in self.y]
        self.y = np.array(self.y)

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
        # del self.gnn_data
        # gc.collect()
        return dataset

class GCN_RNN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(GCN_RNN, self).__init__()
        self.gcn= GCNConv(node_features, hidden_channels)
        self.recur1 = nn.RNN(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)
        self.zone_labels = {'OUTDOOR': 0,
                            'GROUND': 1,
                            'LIVING': 2,
                            'ATTIC': 3}

    def forward(self, x, edge_index):
        y = self.gcn(x, edge_index)
        y = F.relu(y)

        # y will not have dim: {|V|, F_out}
        y = y[self.zone_labels['LIVING']]

        # y = self.recur1(y)
        y, _ = self.recur1(y)
        y = F.relu(y)
        y = self.out(y)
        return y

# thermo_data = ThermoStaticGraphNoChunk('./data/gnn_training_data_no_chunk.pt')
# dataset = thermo_data.get_dataset()

# thermo_data_chunk = ThermoStaticGraphChunk('./data/gnn_training_data_chunk.pt', 20)
thermo_data_chunk = ThermoStaticGraphChunk('./data/gnn_training_data_no_chunk.pt', 20)
dataset_chunk = thermo_data_chunk.get_dataset()


train_dataset, test_dataset = temporal_signal_split(dataset_chunk, train_ratio=0.15)
# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.15)
print(next(enumerate(train_dataset)), next(enumerate(train_dataset)))
# print(next(enumerate(train_dataset))[0].shape)
print('DATASET READY')

model = GCN_RNN(node_features=8, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
loss_fn = nn.MSELoss()
model.train()
best_loss = float('inf')

OUTDOOR = 0
GROUND = 1
LIVING = 2
ATTIC = 3
labels = [OUTDOOR, GROUND, LIVING, ATTIC]
num_nodes = len(labels)

batch_size = 7000
for epoch in tqdm(range(10000)):
    count = 0
    total_loss = 0
    loss = 0
    for time, snapshot in enumerate(train_dataset):
        count += 1
        if count >= batch_size:
            break
        y_hat = model(snapshot.x, snapshot.edge_index) #NOTE

        loss = loss_fn(y_hat, torch.unsqueeze(snapshot.y, 0))
        loss = torch.sqrt(loss)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = total_loss / (count + 1)
    print("Epoch {} train MSE: {:.4f}".format(epoch, epoch_loss.item()))
    if epoch_loss.item() < best_loss:
        best_loss = epoch_loss.item()

    if epoch % 5 == 0 and epoch != 0:
        # test
        print('TESTING')
        model.eval()
        test_loss = 0
        test_count = 0
        ground_truth = []
        model_prediction = []
        for time, snapshot in enumerate(test_dataset):
            test_count += 1
            if test_count >= 5000:
                break

            y_hat = model(snapshot.x, snapshot.edge_index)
            loss_temp = loss_fn(y_hat, torch.unsqueeze(snapshot.y, 0))

            model_prediction.append(y_hat.item())
            ground_truth.append(snapshot.y.item())

            loss += torch.sqrt(loss_temp)
        loss = test_loss / (test_count + 1)
        #loss = loss.item()

        # plot the test result
        start = 300
        end = 1300
        x = list(range(end - start))
        ground_truth_plot = ground_truth[start:end]
        model_prediction_plot = model_prediction[start:end]
        # print('ground', ground_truth_plot)
        # print('model:', model_prediction_plot)
        plt.plot(x, ground_truth_plot, 'r-', label="ground truth")
        plt.plot(x, model_prediction_plot, 'b-', label='model prediction')

        plt.legend()
        plt.title('LOSS:{}'.format(loss),)
        print('y_hat_dim:', y_hat, len(y_hat))
        show = True
        if show:
            plt.show()
