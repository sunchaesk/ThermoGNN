'''
Attempt to create a full toolchain to auto collect relevant data
given the inputted idf file (using lp.py)

This code is for creating data for PyG Temporal
collected data points:
- each zones temperature (including ground, outdoor)
- day of week, hour of day
- site solar radiation
- site horizontal infrared
- sky diffuse


- the data generated is  formatted for PyG Data format
'''

import base_data_collect as base
from base_data_collect import default_args
import lp

import os
import sys
import json
import pickle

import numpy as np
import gymnasium as gym

from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

class Buffer:
    'Fixed sized buffer for collecting model data, for Eplus building model generation'
    def __init__(self,
                 buffer_size: int = 5000000000,
                 data_collection_period_start: Tuple[int, int] = [1,1],
                 data_collection_period_end: Tuple[int, int] = [12,31],
                 data_collection_method: str = 'random',
                 weather_region: str = 'rochester'):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.weather_region = weather_region
        self.data_collection_period_start = data_collection_period_start
        self.data_collection_period_end = data_collection_period_end
        self.episode = 0
        # prob 'Random'
        self.data_collection_method = data_collection_method
        #self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])

    def add(self, elem):
        #e = tuple(state)
        e = elem
        self.buffer.append(elem)

    def b_full(self):
        if len(self.buffer) == self.buffer_size:
            return True
        else:
            return False

    def percentage_full(self):
        return round(len(self.buffer) / self.buffer_size, 2)

IDF_PATH = './in.idf'
IDF_FILE = open(IDF_PATH, 'r')
F = IDF_FILE.read()
PARSED_IDF = lp.parse(F)

def generate_eplus_variables(parsed_idf):
    ret_variables = dict()
    solar_surfaces_list = lp.get_solar_surface_list(parsed_idf)
    zone_list = lp.get_zone_list(parsed_idf)

    # get zone temperatures
    ret_variables['outdoor_temp'] = tuple(["Site Outdoor Air Drybulb Temperature", "Environment"])
    for zone in zone_list:
        if zone == "Outdoors" or "ground" in zone.lower():
            continue
        key = (zone.replace('-', '_') + '_temp').lower()
        ret_variables[key] = tuple(["Zone Air Temperature", zone])
    ret_variables['ground_temp'] = tuple(["Site Ground Temperature", "Environment"])

    # get zone humidity
    ret_variables['outdoor_humidity'] = tuple(["Site Outdoor Air Relative Humidity", "Environment"])
    for zone in zone_list:
        if zone == "Outdoors" or "ground" in zone.lower():
            continue
        key = (zone.replace('-', '_') + '_humidity').lower()
        ret_variables[key] = tuple(["Zone Air Relative Humidity", zone])

    # get solar
    for solar_surface in solar_surfaces_list:
        key = (solar_surface.replace('-', '_') + '_sky_diffuse').lower()
        ret_variables[key] = tuple(["Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", solar_surface])

    # time stuff is taken care of in self.next_obs

    # site solar
    ret_variables['site_direct_solar'] = tuple(["Site Direct Solar Radiation Rate per Area", "Environment"])
    ret_variables['site_horizontal_infrared'] = tuple(["Site Horizontal Infrared Radiation Rate per Area", "Environment"])

    return ret_variables

def main_data_collector(num_episodes, load=True):
    buf = Buffer(50000000000,
                 tuple([5,1]),
                 tuple([8,31]),
                 'zero-heating-coolig',
                 'Rochester International Arpt,MN,USA')

    start_episode = 0

    b_load = load
    if b_load:
        try:
            p_f = open('./data/training_data.pt', 'rb')
            buf = pickle.load(p_f)
            start_episode = buf.episode
            p_f.close()
            print('#######################')
            print('LOADING FROM training_data.pt...')
            print('BUF SIZE:', len(buf.buffer))
            print('#######################')
            time.sleep(2)
        except:
            print('ERROR: ./data/training_data.pt not found... data collection from scratch')

    variables_dict = generate_eplus_variables(PARSED_IDF)
    buf.indices = {key: index for index, key in enumerate(variables_dict)}
    default_args['variables'] = variables_dict
    env = base.EnergyPlusEnv(default_args)
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        done = False
        episode_datas = []
        print('------------------COMPLETION: {}% / Iteration {}-----------------'.format(buf.percentage_full() * 100, episode))
        while not done:
            action = env.action_space.sample()
            ret = n_state, reward, done, truncated, info = env.step(action)
            # episode_datas.append(n_state.append(action).tolist())
            add = state.tolist()
            add.append(action)
            episode_datas.append([add, n_state[1]])
            #buf.add(add, n_state[1])
            state = n_state
        #
        buf.episode += 1
        buf.add(episode_datas)
        if episode % 5 == 0 and episode != 0:
            pickle_file = open('./data/training_data.pt', 'wb')
            pickle.dump(buf, pickle_file)
            pickle_file.close()
    #
    return buf
    print('COMPLETED')

def generate_graph_data(next_obs: dict) -> Data:
    pass

if __name__ == "__main__":
    main_data_collector(10000)
