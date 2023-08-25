import tqdm
import os
import sys
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN

class ThermoStaticGraphNoChunk(object):
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

        #sys.exit(1)
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

thermo_data = ThermoStaticGraphNoChunk('./data/gnn_training_data_no_chunk.pt')
dataset = thermo_data.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.15)
print(type(train_dataset.features[0]), type(test_dataset.targets[0]))
print(train_dataset.features[0])
print(train_dataset.targets[0])
print('DATASET READY')

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(RecurrentGCN, self).__init__()
        self.recur1 = TGCN(node_features, hidden_channels,improved=True)
        self.recur2 = TGCN(hidden_channels, hidden_channels, improved=True)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.b_norm1 = nn.BatchNorm1d(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)
        self.rnn1 = nn.RNN(hidden_channels, 1)

    def forward(self, x, edge_index):
        h = self.recur1(x, edge_index)
        # print('h:', h, h.shape)
        y = F.relu(h)
        # print('y:', y, y.shape)
        # sys.exit(1)
        #h = F.dropout(y, p=0.5, training=self.training)

        # h = self.recur2(h, edge_index)
        # h = F.relu(h)
        # h = F.dropout(h, p=0.5, training=self.training)
        y = h

        # y = self.linear1(y)
        # y = self.b_norm1(y)
        # y = F.relu(y)
        # y, _ = self.rnn1(h)
        # y = self.b_norm1(y)
        # y = F.relu(y)
        y = self.linear2(y)
        return y

model = RecurrentGCN(node_features=8, hidden_channels=300)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()
model.train()
losses = []
best_loss = float('inf')

OUTDOOR = 0
GROUND = 1
LIVING = 2
ATTIC = 3
labels = [OUTDOOR, GROUND, LIVING, ATTIC]
num_nodes = len(labels)

mask = torch.zeros(num_nodes, dtype=torch.bool)
mask[LIVING] = True

batch_size = 7000
for epoch in tqdm.tqdm(range(10000)):
    count = 0
    total_loss = 0
    loss = 0
    for time, snapshot in enumerate(train_dataset):
        count += 1
        if count >= batch_size:
            break
        y_hat = model(snapshot.x, snapshot.edge_index)
        # print('y_hat:', y_hat)
        # sys.exit(1)
        # print('y_hat:', y_hat[mask], y_hat[mask].shape)
        # print('y:', snapshot.y, snapshot.y.shape)
        # sys.exit(1)

        # loss = loss_fn(y_hat[mask], snapshot.y)
        loss = loss_fn(y_hat[mask], torch.unsqueeze(snapshot.y, 0))
        loss = torch.sqrt(loss)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = total_loss / (count + 1)
    print("Epoch {} train MSE: {:.4f}".format(epoch, epoch_loss.item()))
    if epoch_loss.item() < best_loss:
        best_loss = epoch_loss.item()

    if epoch % 1 == 0:
        # test
        print('TESTING')
        model.eval()
        test_loss = 0
        test_count = 0
        ground_truth = []
        model_prediction = []
        other_stuff1 = []
        other_stuff2 = []
        for time, snapshot in enumerate(test_dataset):
            test_count += 1
            if test_count >= 5000:
                break

            y_hat = model(snapshot.x, snapshot.edge_index)
            loss_temp = loss_fn(y_hat[mask], torch.unsqueeze(snapshot.y, 0))

            model_prediction.append(y_hat[LIVING].item())
            ground_truth.append(snapshot.y.item())
            other_stuff1.append(y_hat[0].item())
            other_stuff2.append(y_hat[1].item())

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

        plt.plot(x, other_stuff1[start:end])
        plt.plot(x, other_stuff2[start:end])

        plt.legend()
        plt.title('LOSS:{}'.format(loss))
        show = True
        if show:
            plt.show()
