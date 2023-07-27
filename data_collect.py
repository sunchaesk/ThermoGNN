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

import base as base
import lp

import os
import sys

import numpy as np

import json

import torch
import torch.nn as nn
from torch_geometric.data import Data

idf_path = './in.idf'
idf_file = open(idf_path, 'r')
f = idf_file.read()
parsed_idf = lp.parse(f)

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

    # get solar
    for solar_surface in solar_surfaces_list:
        key = (solar_surface.replace('-', '_') + '_sky_diffuse').lower()
        ret_variables[key] = tuple(["Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", solar_surface])

    # time stuff is taken care of in self.next_obs

    # site solar
    ret_variables['site_direct_solar'] = tuple(["Site Direct Solar Radiation Rate per Area", "Environment"])
    ret_variables['site_horizontal_infrared'] = tuple(["Site Horizontal Infrared Radiation Rate per Area", "Environment"])

    return ret_variables

def generate_graph_data(next_obs: dict) -> Data:
    pass

if __name__ == "__main__":
    print(generate_eplus_variables(parsed_idf))
