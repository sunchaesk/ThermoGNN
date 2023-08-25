
import tqdm

import os
import sys
import gc

import pickle
import joblib

import torch
import numpy as np

import matplotlib.pyplot as plt
from buffer import Buffer

from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from sklearn.preprocessing import StandardScaler

pf = open('./data/gnn_scaled_data.pt', 'rb')
raw_data = pickle.load(pf)
#raw_data = raw_data.buffer
print('raw_data len:', len(raw_data))
# 111 episodes
# for each timestep in each episode
# [[obs], y]
pf.close()

overlap_chunk_data = []
sequence_length = 20
def overlap_sequences(data, sequence_length):
    overlapping_sequences = [data[i:i + sequence_length] for i in range(len(data) - sequence_length + 1)]
    #print('overlapping seq:', overlapping_sequences)
    return overlapping_sequences
for episode in raw_data:
    overlap_chunk_data.extend(overlap_sequences(episode, sequence_length))

# clear data for reduce ram usage
del raw_data
gc.collect()

# convert the overlap_chunk_data
# for each of the overlap_chunk_data
# get each index to the PyG Data format
x_t_list = []
y_t_list = []

overlap_chunk_data = overlap_chunk_data[:1000000]
gc.collect()

for chunk_data in tqdm.tqdm(overlap_chunk_data):
    #chunk_data = overlap_chunk_data[i]
    curr_data_obj = None
    x_t = [] # x with extension to the time direction
    y_t = []
    for timestep_data in chunk_data:
        x = []

        curr_timestep = timestep_obs = timestep_data[0]
        timestep_target = timestep_data[1]
        #sys.exit(1)

        # generate feature matrix
        outdoor_feature = []
        outdoor_feature.append(curr_timestep[0])
        outdoor_feature.append(curr_timestep[4])
        outdoor_feature.append(curr_timestep[27])
        outdoor_feature.append(curr_timestep[28])
        outdoor_feature.append(0) # outdoor has 0 direct solar?
        outdoor_feature.append(curr_timestep[30])
        outdoor_feature.append(curr_timestep[31])
        outdoor_feature.append(0) # action
        x.append(np.array(outdoor_feature))

        ground_feature = []
        ground_feature.append(curr_timestep[3])
        ground_feature.append(0) # ground humidity is set to 0
        ground_feature.append(curr_timestep[27])
        ground_feature.append(curr_timestep[28])
        ground_feature.append(0) # ground doesn't have diffuse solar
        ground_feature.append(curr_timestep[30])
        ground_feature.append(curr_timestep[31])
        ground_feature.append(0) # action
        x.append(np.array(ground_feature))

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
        x.append(np.array(living_feature))

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
        x.append(np.array(attic_feature))

        #print(x, len(x), len(x[0]))
        x = np.array(x).reshape(4,8)
        x_t.append(x)
        y_t.append(np.array([timestep_target]))


    #x_t = np.array(x_t)


    x_t = np.array(x_t)
    x_t = np.transpose(x_t, (1,2,0))
    y_t = np.array(y_t)
    x_t_list.append(x_t)
    y_t_list.append(y_t)
    #y_t = np.reshape(1,20)
    # print(x_t)
    # print(x_t.shape)
    # print(y_t)
    # print(y_t.shape)

del overlap_chunk_data
gc.collect()


print(len(x_t_list), x_t_list[0].shape)
print('done')
pf = open('./data/gnn_training_data_chunk.pt', 'wb')
pickle.dump([x_t_list, y_t_list], pf)
pf.close()
print('joblib saved')

# goal data format:
# - Data()
