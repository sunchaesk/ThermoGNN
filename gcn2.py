
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
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
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
        self.bnorm1 = nn.BatchNorm1d(hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)
        self.zone_labels = {'OUTDOOR': 0,
                            'GROUND': 1,
                            'LIVING': 2,
                            'ATTIC': 3}

    def forward(self, x, edge_index):
        y = self.gcn(x, edge_index) # ret torch.Size([4,25,20])
        y = F.relu(y)

        # y will not have dim: {|V|, F_out}
        y = y.permute(1,2,0)
        y = y[self.zone_labels['LIVING']] # ret torch.Size([25,20])

        y = torch.transpose(y, 0, 1)

        y, _ = self.recur1(y)
        y = self.bnorm1(y)
        y = F.relu(y)
        y = self.out(y)
        return y

def overlapping_sequences(input_list, sublist_length):
    overlapping_sublists = []
    for i in range(len(input_list) - sublist_length + 1):
        sublist = input_list[i:i + sublist_length]
        overlapping_sublists.append(sublist)
    return overlapping_sublists

# thermo_data = ThermoStaticGraphNoChunk('./data/gnn_training_data_no_chunk.pt')
# dataset = thermo_data.get_dataset()

# thermo_data_chunk = ThermoStaticGraphChunk('./data/gnn_training_data_chunk.pt', 20)
thermo_data_chunk = ThermoStaticGraphChunk('./data/gnn_training_data_no_chunk.pt', 20)
dataset_chunk = thermo_data_chunk.get_dataset()


train_dataset, test_dataset = temporal_signal_split(dataset_chunk, train_ratio=0.15)

batch_size = 10

print(type(train_dataset))
train_list = list(train_dataset)
num_samples = len(train_list)
test_list = list(test_dataset)
num_test = len(test_list)

print(train_dataset[0], type(train_dataset[0]))
print('complete')

epoch = 0
best_loss = float('inf')

model = GCN_RNN(node_features=8, hidden_channels=300)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

steps = 0
check_every = 50000
while True:
    total_loss = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_loss = 0
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        x_batch = []
        edge_indices_batch = []
        y_batch = []

        for i in range(start_idx, end_idx):
            snapshot = train_list[i]
            x_batch.append(snapshot.x)
            edge_indices = snapshot.edge_index
            edge_indices_with_self_loops, _ = add_self_loops(edge_indices, num_nodes=len(snapshot.x))  # Unpack the tuple
            edge_indices_batch.append(edge_indices_with_self_loops)
            y_batch.append(snapshot.y)

        steps = end_idx

        x_batch = torch.stack(x_batch)
        edge_indices_batch = torch.stack(edge_indices_batch, dim=1)
        y_batch = torch.stack(y_batch)

        y_hat_batch = model(x_batch, edge_indices_batch)

        # print('--------------')
        # print(y_hat_batch, y_hat_batch.shape)
        # print(y_batch, y_batch.shape)
        # print('--------------')

        batch_loss = loss_fn(y_hat_batch, y_batch)
        batch_loss = torch.sqrt(batch_loss)

        temp_batch_loss = batch_loss.item()

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if steps % check_every and steps != 0:
            epoch_loss = total_loss / (check_every + 1)
            #print("Current Step {} train MSE: {:.4f}".format(steps, temp_batch_loss))
            # if epoch_loss < best_loss:
            #     best_loss = epoch_loss

    # plotting
    #print('Testing')
    model.eval()
    total_test_loss = 0
    test_count = num_test
    ground_truth = []
    model_prediction = []


    num_batches = (num_test + batch_size - 1)  // batch_size

    test_iterations = 200

    for batch_idx in range(test_iterations):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        test_x_batch = []
        test_edge_indices_batch = []
        test_y_batch = []

        for i in range(start_idx, end_idx):
            snapshot = test_list[i]
            test_x_batch.append(snapshot.x)
            edge_indices = snapshot.edge_index
            edge_indices_with_self_loops, _ = add_self_loops(edge_indices, num_nodes=len(snapshot.x))  # Unpack the tuple
            test_edge_indices_batch.append(edge_indices_with_self_loops)
            #y_batch.append(snapshot.y)
            test_y_batch.append(snapshot.y)
            ground_truth.append(snapshot.y)


        #print('test:', test_x_batch[-1], len(test_x_batch))

        test_x_batch = torch.stack(test_x_batch)
        test_x_batch = test_x_batch.squeeze()
        #test_x_batch = torch.stack(torch.tensor(test_x_batch))
        test_edge_indices_batch = torch.stack(test_edge_indices_batch, dim=1)

        test_y_hat_batch = model(test_x_batch, test_edge_indices_batch)

        model_prediction.extend(test_y_hat_batch.tolist())

        test_loss = loss_fn(test_y_hat_batch, torch.tensor(test_y_batch).squeeze(0))
        test_loss = torch.sqrt(test_loss)

        total_test_loss += test_loss.item()
        #print('TOTAL TEST LOSS +:', total_test_loss)


    start = 300
    end = 1300
    x = list(range(end - start))
    ground_truth_plot = ground_truth[start:end]
    model_prediction_plot = model_prediction[start:end]
    plt.plot(x, ground_truth_plot, 'r-', label='ground truth')
    plt.plot(x, model_prediction_plot, 'b', label='model prediction')

    plt.legend()

    ret = total_test_loss / (batch_size * test_iterations)
    # print('---------------')
    # print('total test loss', total_test_loss)
    # print('batch size', batch_size)
    # print('test iterations', test_iterations)
    # print('test loss', ret)
    # print('---------------')

    plt.title('LOSS {}'.format(ret))
    print('TEST LOSS:', ret)
    show = False
    if show:
        plt.show()

# for epoch in range(10000):
#     total_loss = 0

#     for batch in tqdm(train_loader):
#         x_batch = []
#         edge_indices_batch = []
#         y_batch = []

#         for snapshot in batch:
#             x_batch.append(snapshot.x)
#             edge_indices_batch.append(snapshot.edge_index)
#             y_batch.append(snapshot.y)

#         x_batch = torch.stack(x_batch)
#         edge_indices_batch = torch.stack(edge_indices_batch)
#         y_batch = torch.stack(y_batch)

#         y_hat_batch = model(x_batch, edge_indices_batch)

#         loss = loss_fn(y_hat_batch, y_batch.unsqueeze(1))  # Add an extra dimension for batch
#         loss = torch.sqrt(loss)

#         total_loss += loss.item()

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     epoch_loss = total_loss / len(train_loader)
#     print("Epoch {} train MSE: {:.4f}".format(epoch, epoch_loss))
#     if epoch_loss < best_loss:
#         best_loss = epoch_loss

#     if epoch % 5 == 0 and epoch != 0:
#         # test
#         print('TESTING')
#         model.eval()
#         test_loss = 0
#         test_count = 0
#         ground_truth = []
#         model_prediction = []
#         for time, snapshot in enumerate(test_dataset):
#             test_count += 1
#             if test_count >= 5000:
#                 break

#             y_hat = model(snapshot.x, snapshot.edge_index)
#             loss_temp = loss_fn(y_hat, torch.unsqueeze(snapshot.y, 0))

#             model_prediction.append(y_hat.item())
#             ground_truth.append(snapshot.y.item())

#             loss += torch.sqrt(loss_temp)
#         loss = test_loss / (test_count + 1)
#         #loss = loss.item()

#         # plot the test result
#         start = 300
#         end = 1300
#         x = list(range(end - start))
#         ground_truth_plot = ground_truth[start:end]
#         model_prediction_plot = model_prediction[start:end]
#         # print('ground', ground_truth_plot)
#         # print('model:', model_prediction_plot)
#         plt.plot(x, ground_truth_plot, 'r-', label="ground truth")
#         plt.plot(x, model_prediction_plot, 'b-', label='model prediction')

#         plt.legend()
#         plt.title('LOSS:{}'.format(loss),)
#         print('y_hat_dim:', y_hat, len(y_hat))
#         show = True
#         if show:
#             plt.show()
