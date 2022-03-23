# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 08:18:51 2021

@author: dongwei
"""

import argparse
import numpy as np
import scipy.sparse as sp

import tools as tl

parser = argparse.ArgumentParser()

parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'Cora')

parser.add_argument(
        '-s',
        '--sampling-size',
        type = int,
        default = 1)

args = parser.parse_args()

K = args.sampling_size
raw_dir = './' + args.dataset + '/'

labels = np.load(raw_dir + 'labels.npy')
N = labels.shape[0]

A = sp.load_npz(raw_dir + 'adj.npz')
E = int(np.sum(A))

sub_graph_list = tl.mutual_info(A, K)
sub_graph_queue = []

for sub_graph in sub_graph_list:
    
    if(1 < sub_graph.number_of_nodes()):
        
        sub_graph_queue.append(sub_graph)

node_stat = np.zeros(N + 1)
edge_stat = np.zeros(E + 1)

for sub_graph in sub_graph_queue:
    
    node_stat[sub_graph.number_of_nodes()] += 1
    edge_stat[sub_graph.number_of_edges()] += 1
    
node_tick_label = np.flatnonzero(node_stat)
node_stat = node_stat.ravel()[node_tick_label]

edge_tick_label = np.flatnonzero(edge_stat)
edge_stat = edge_stat.ravel()[edge_tick_label]

np.savetxt(raw_dir + 'subgraphs/node_tick_label_' + str(K) + '.txt', node_tick_label, fmt='%d')
np.savetxt(raw_dir + 'subgraphs/node_stat_' + str(K) + '.txt', node_stat, fmt='%d')

np.savetxt(raw_dir + 'subgraphs/edge_tick_label_' + str(K) + '.txt', edge_tick_label, fmt='%d')
np.savetxt(raw_dir + 'subgraphs/edge_stat_' + str(K) + '.txt', edge_stat, fmt='%d')