
import sys
import os

import base

import lp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import pprint
import json

if __name__ == "__main__":
    adjacency = lp.generate_adjacency('./in.idf')
    pretty_adjacency_list = json.dumps(adjacency, indent=4)

    print(pretty_adjacency_list)
    #print(adjacency)
