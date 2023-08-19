
import tqdm

import os
import sys
import time as ttt

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import DCRNN, A3TGCN, TGCN
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from itertools import cycle

from buffer import Buffer
import pickle

class ThermoModelDatasetLoader(object):
    '''
    residence building thermal modelling dataset loader
    '''
    def __init__(self, data_file_path):
        super(ThermoModelDatasetLoader, self).__init__()
        self.data_file_path = data_file_path

    def _read_data(self):
        pf = open(self.data_file_path, 'rb')
        self.gnn_data = pickle.load(pf)
        pf.close()
        self.edge_index = torch.tensor([[2,3,0,2,3,0,2,1],
                                        [3,2,2,0,0,3,1,2]], dtype=torch.long)
        # self.edge_index = torch.tensor([[2,3],
        #                        [3,2],
        #                        [0,2],
        #                        [2,0],
        #                        [3,0],
        #                        [0,3],
        #                        [2,1],
        #                        [1,2]], dtype=torch.long)
        self.edge_weights = torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float)

        self.x = self.gnn_data[0]
        self.y = self.gnn_data[1]
        self.snapshot_count = len(self.x)

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



def GNN_data_process():
    buf = Buffer()
    p_f = open('./data/save_training_data.pt', 'rb')
    buf = pickle.load(p_f)
    p_f.close()
    print('buf')

    # process to graph format
    # Node 0: Outdoor
    # Node 1: Ground
    # Node 2: Indoor
    # - [temp, humidity, hour, day of week, diffuse solar sum, humidity]

    # graph node indices
    x_list = []
    y_list = []

    OUTDOOR = 0
    GROUND = 1
    LIVING = 2
    ATTIC = 3

    ZONE_TEMP = 0
    ZONE_HUMIDITY = 1
    DIRECT_SOLAR = 2
    HORIZONTAL_INFRARED = 3
    DIFFUSE_SOLAR = 4
    HOUR = 5
    DAY_OF_WEEK = 6
    ACTION = 7
    # edge_index = torch.tensor([[2,3],
    #                            [3,2],
    #                            [0,2],
    #                            [2,0],
    #                            [3,0],
    #                            [0,3],
    #                            [2,1],
    #                            [1,2]], dtype=torch.long)

    for buffer_episode in tqdm.tqdm(buf.buffer):
        #for curr_data in buffer_episode:
        for i in range(0,len(buffer_episode)):
            x = [] # feature matrix -> shape [num_nodes, num_node_features]

            curr_data = buffer_episode[i]
            curr_timestep = curr_data[0]
            # compute outdoor stuff: 0 diffuse solar?
            outdoor_feature = []
            outdoor_feature.append(curr_timestep[0])
            outdoor_feature.append(curr_timestep[4])
            outdoor_feature.append(curr_timestep[27])
            outdoor_feature.append(curr_timestep[28])
            outdoor_feature.append(0) # outdoor has 0 direct solar?
            outdoor_feature.append(curr_timestep[30])
            outdoor_feature.append(curr_timestep[31])
            outdoor_feature.append(0) # action
            x.append(outdoor_feature)

            ground_feature = []
            ground_feature.append(curr_timestep[3])
            ground_feature.append(0) # ground humidity is set to 0
            ground_feature.append(curr_timestep[27])
            ground_feature.append(curr_timestep[28])
            ground_feature.append(0) # ground doesn't have diffuse solar
            ground_feature.append(curr_timestep[30])
            ground_feature.append(curr_timestep[31])
            ground_feature.append(0) # action
            x.append(ground_feature)

            living_feature = []
            living_feature.append(curr_timestep[1])
            living_feature.append(curr_timestep[5])
            living_feature.append(curr_timestep[27])
            living_feature.append(curr_timestep[28])
            living_diffuse_solar = 0
            for index in [7,8,9,10,11,12,13,14,19,20,21,22,23,24,25,26]:
                living_diffuse_solar += curr_timestep[index]
            living_feature.append(living_diffuse_solar)
            living_feature.append(curr_timestep[30])
            living_feature.append(curr_timestep[31])
            living_feature.append(curr_timestep[-1]) # action
            x.append(living_feature)

            attic_feature = []
            attic_feature.append(curr_timestep[2])
            attic_feature.append(curr_timestep[6])
            attic_feature.append(curr_timestep[27])
            attic_feature.append(curr_timestep[28])
            attic_diffuse_solar = 0
            for index in [15,16,17,18]:
                attic_diffuse_solar += curr_timestep[index]
            attic_feature.append(attic_diffuse_solar)
            attic_feature.append(curr_timestep[30])
            attic_feature.append(curr_timestep[31])
            attic_feature.append(0) # action
            x.append(attic_feature)

            x_list.append(x)

            y_list.append(np.array([curr_data[1]], dtype='double'))

    # StandardScaler
    x_list = np.array(x_list)
    x_scaler = StandardScaler()
    x_list = x_scaler.fit_transform(x_list.reshape(-1, x_list.shape[-1])).reshape(x_list.shape)
    x_list = x_list.tolist()

    GNN_training_pf = open('./data/gnn_processed_training_data.pt', 'wb')
    pickle.dump([x_list, np.array(y_list, dtype='double')], GNN_training_pf)
    GNN_training_pf.close()
    print('DATA processing completed')


#GNN_data_process()
# b_process = False
# if b_process:
#     GNN_data_process()
#     gnn_data_pf = open('./data/gnn_processed_training_data.pt', 'rb')
#     gnn_training_data = pickle.load(gnn_data_pf)
#     gnn_data_pf.close()
# else:
#     gnn_data_pf = open('./data/gnn_processed_training_data.pt', 'rb')
#     gnn_training_data = pickle.load(gnn_data_pf)
#     gnn_data_pf.close()

# Pytorch geometric temporal
thermo_data = ThermoModelDatasetLoader('./data/gnn_processed_training_data.pt')
gnn_training_dataset = thermo_data.get_dataset()

train_dataset, test_dataset = temporal_signal_split(gnn_training_dataset, train_ratio=0.15)
print(type(train_dataset.features[0]), type(test_dataset.targets[0]))
print(train_dataset.features[0])
print(train_dataset.targets[0])
print('DATASET READY')


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(RecurrentGCN, self).__init__()
        self.recurrent1 = TGCN(node_features, hidden_channels)
        self.recurrent2 = TGCN(hidden_channels, hidden_channels)
        self.linear1  = nn.Linear(hidden_channels, hidden_channels)
        self.b_norm1 = nn.BatchNorm1d(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        h = self.recurrent1(x, edge_index)
        y = F.relu(h)
        # h = self.recurrent2(h, edge_index)
        # print('h:', h)
        #h = self.recurrent2()
        # y = F.relu(h)
        h = F.dropout(y, p=0.5, training=self.training)
        #h = self.recurrent2(x, edge_index)
        #h = F.relu(h)
        y = self.linear1(y)
        y = self.b_norm1(y)
        y = F.relu(y)
        y = self.linear2(y)
        # print('y:', y)
        return y[2]

def train_gnn():
    model = RecurrentGCN(node_features=8, hidden_channels=30)
    try:
        model.load_state_dict(torch.load('./data/training_gnn_model_temp.pt'))
        print('MODEL LOADED')
    except:
        print('MODEL NOT LOADED')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    model.train()
    losses = []
    best_loss = float('inf')

    batch_size = 10000
    for epoch in tqdm.tqdm(range(10000)):
        count = 0
        for time, snapshot in enumerate(train_dataset):
            count += 1
            if count >= batch_size:
                break
            y_hat = model(snapshot.x, snapshot.edge_index)
            loss = loss_fn(y_hat, snapshot.y)
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1 == 0 and epoch != 0:
            model.eval()
            loss = 0
            test_count = 0
            for time, snapshot in enumerate(test_dataset):
                test_count += 1
                if test_count >= 10000:
                    break
                y_hat = model(snapshot.x, snapshot.edge_index)
                loss_temp = loss_fn(y_hat, snapshot.y)
                loss += torch.sqrt(loss_temp)
            loss = loss / (test_count + 1)
            loss = loss.item()

            with open('./gnn_log.txt', 'a') as log_f:
                log_f.write(str(loss))
                log_f.write('\n')

            print('losses:', losses)
            if loss < best_loss:
                best_loss = loss
                torch.save(model, './data/training_gnn_model_temp.pt')
            print('LOSS EPOCH{}'.format(epoch) + ' ' + str(loss))

# def train_gnn():
#     batch_size = 1000
#     batch_start = torch.arange(0, len(train_dataset.y), batch_size)
#     model = RecurrentGCN(node_features=7, hidden_channels=20)
#     optimizer = torch.optim.Adam(model.parameters, lr=0.0005)
#     for epoch in range(10000):
#         model.train()
#         with tqdm.tqdm(batch_start) as bar:
#             for start in bar:
#                 x_train =

# def train_gnn():
#     loss_fn = nn.MSELoss()
#     model = RecurrentGCN(node_feature=7)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     model.train()
#     for epoch in tdqm(range(100000)):
#         loss = 0
#         step = 0
#         for snapshot in train_dataset:
#             y_hat = model(snapshot.x, snapshot.edge_index)
#             loss = loss_fn(y_hat, snapshot.y) # calculuate MSE
#             loss = torch.sqrt(loss) # calculuate RMSE

if __name__ == "__main__":
    # for time, snapshot in enumerate(train_dataset):
    #     print('time:', time)
    #     print('snapshot:', snapshot, type(snapshot))
    #     print('snapshot data:', snapshot.y)
    #     ttt.sleep(1.5)

    # GNN_data_process()
    train_gnn()
    print('done')
