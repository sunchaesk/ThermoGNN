
import os
import sys
import pickle
import gc

from buffer import Buffer

from sklearn.preprocessing import StandardScaler

pf = open('./data/save_training_data.pt', 'rb')
raw_data = pickle.load(pf).buffer
pf.close()

x_vals = []
y_vals = []
for episode_trajectory in raw_data:
    for timestep in episode_trajectory:
        x_vals.append(timestep[0])
        y_vals.append(timestep[1])
x_vals = x_vals[:800000]
del raw_data
gc.collect()


x_scaler = StandardScaler()
x_vals = x_scaler.fit_transform(x_vals)

ret_vals = []
for i in range(len(x_vals)):
    temp = [x_vals[i], y_vals[i]]
    ret_vals.append(temp)

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

episode_trajectories = split_list(ret_vals, 15408)
print(len(episode_trajectories))
print(len(episode_trajectories[0]))

pf = open('./data/gnn_scaled_data.pt', 'wb')
pickle.dump(episode_trajectories, pf)
pf.close()

print('SCALED DATA SAVED')
