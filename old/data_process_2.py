
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

pf = open('./data/gnn_scaled_data.pt', 'rb')
raw_data = pickle.load(pf)
pf.close()
print('raw_data len:', len(raw_data))

x_vals = []
y_vals = []
for episode_trajectory in raw_data:
    for timestep in episode_trajectory:
        x_vals.append(timestep[0])
        y_vals.append(timestep[1])

print(x_vals[0])
print(y_vals[0])
print('len:', len(x_vals))

x_vals_new = []
for curr_timestep in x_vals:
    x = [] # feature matrix -> shape [num_nodes, num_node_features]

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
    living_diffuse_solar_list = []
    for index in [7,8,9,10,11,12,13,14,19,20,21,22,23,24,25,26]:
        living_diffuse_solar_list.append(curr_timestep[index])
    living_diffuse_solar = sum(living_diffuse_solar_list)
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
    attic_diffuse_solar_list = []
    for index in [15,16,17,18]:
        attic_diffuse_solar_list.append(curr_timestep[index])
    attic_diffuse_solar = sum(attic_diffuse_solar_list)
    attic_feature.append(attic_diffuse_solar)
    attic_feature.append(curr_timestep[30])
    attic_feature.append(curr_timestep[31])
    attic_feature.append(0) # action
    x.append(attic_feature)

    x_vals_new.append(x)

ret = [x_vals_new, y_vals]

pf = open('./data/gnn_training_data_no_chunk.pt', 'wb')
pickle.dump(ret, pf)
pf.close()
print('DONE')
