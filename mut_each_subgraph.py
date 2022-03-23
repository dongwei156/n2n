# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:08:48 2021

@author: dongwei
"""

import argparse
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

import tools as tl


def mkdir(path):
    
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
    
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
labels = np.argmax(labels, 1)

A = sp.load_npz(raw_dir + 'adj.npz')

sub_graph_list = tl.mutual_info(A, K)
sub_graph_queue = []

for sub_graph in sub_graph_list:
    
    if(1 < sub_graph.number_of_nodes()):
        
        sub_graph_queue.append(sub_graph)

color_dict = {0:'red', 
              1: 'orange', 
              2:'deepskyblue', 
              3:'green', 
              4:'cyan', 
              5:'blue', 
              6:'purple',
              7:'firebrick',
              8:'darkgoldenrod',
              9:'olive',
              10:'lightgreen',
              11:'teal',
              12:'yellow',
              13:'violet',
              14:'hotpink'
              }

path = raw_dir + 'subgraphs/Sampling size ' + str(K) + '/'
mkdir(path)

count = 0
for sub_graph in sub_graph_queue:
    
    plt.figure(count, figsize = (2, 2))
    
    count += 1
    
    color_list = []
    
    node_list = list(sub_graph.nodes())
    
    for l in labels[node_list]:
        
        color_list.append(color_dict[l])
    
    nx.draw(sub_graph, 
            pos=nx.spring_layout(sub_graph),
            node_color = color_list,
            node_size = 10,
            edge_color = 'gray')
    
    plt.savefig(path + str(count) + '.pdf')
    plt.close()