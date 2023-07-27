
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import json

class ThermalGCN(nn.Module):
    def __init__(self, num_zone_features, num_outdoor_features, num_ground_features,
                 num_hidden):
        super(ThermalGCN, self).__init__()
        # Separate GCNConv layers for each node type
        self.conv_zone = GCNConv(num_zone_features, num_hidden)
        self.conv_outdoor = GCNConv(num_outdoor_features, num_hidden)
        self.conv_ground = GCNConv(num_ground_features, num_hidden)

        # Define edge attributes (thermal resistance) as learnable parameters for each edge type
        self.edge_attributes_zone = nn.Parameter(torch.randn(num_edges_zone, 1))
        self.edge_attributes_outdoor = nn.Parameter(torch.randn(num_edges_outdoor, 1))
        self.edge_attributes_ground = nn.Parameter(torch.randn(num_edges_ground, 1))

    def forward(self, x_zone, x_outdoor, x_ground, edge_index_zone, edge_index_outdoor, edge_index_ground):
        # Node message passing using GCNConv for each node type
        x_zone = self.conv_zone(x_zone, edge_index_zone, edge_weight=self.edge_attributes_zone)
        x_outdoor = self.conv_outdoor(x_outdoor, edge_index_outdoor, edge_weight=self.edge_attributes_outdoor)
        x_ground = self.conv_ground(x_ground, edge_index_ground, edge_weight=self.edge_attributes_ground)

        # Apply ReLU activation after aggregation for each node type
        x_zone = F.relu(x_zone)
        x_outdoor = F.relu(x_outdoor)
        x_ground = F.relu(x_ground)

        return x_zone, x_outdoor, x_ground

# Example usage:
num_zone_features = 2       # Number of thermal features per zone
num_outdoor_features = 3    # Number of features for outdoor nodes
num_ground_features = 1     # Number of features for ground nodes
num_hidden = 64             # Number of hidden units in GCNConv layers

# Sample adjacency list representation of the graph with string labels for zones
idf_path = './in.idf'
edges_list = lp.generate_connections(idf_path)

# Create a set of all unique node labels to assign numeric indices
all_nodes = set(node for edge in edges_list for node in edge)
node_to_index = {node: i for i, node in enumerate(all_nodes)}
print('all_nodes:', all_nodes)

# Convert the edge list into the edge index format (2D tensor of size (2, num_edges))
edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges_list], dtype=torch.long).t()

model = ThermalGCN(num_zone_features, num_outdoor_features, num_ground_features, num_hidden)

# Sample input data for zones, outdoor nodes, and ground nodes (replace this with your own data)
num_nodes = len(all_nodes)  # Number of total nodes in the graph
x_zone = torch.randn(num_nodes, num_zone_features)       # Node features for all nodes with 5 features each
x_outdoor = torch.randn(3, num_outdoor_features)  # 3 outdoor nodes with 3 features each
x_ground = torch.randn(2, num_ground_features)    # 2 ground nodes with 4 features each

# Forward pass through the GCN
zones_features, outdoor_features, ground_features = model(x_zone, x_outdoor, x_ground, edge_index)

# Perform regression tasks or other operations using the node features and edge attributes

# Create the adjacency list dictionary
adjacency_list = {node: [all_nodes[i] for i in edge_index[1][edge_index[0] == node_to_index[node]].tolist()] for node in all_nodes}

# Pretty print the adjacency list dictionary
pretty_adjacency_list = json.dumps(adjacency_list, indent=4)

print(pretty_adjacency_list)
