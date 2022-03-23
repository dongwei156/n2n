# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 07:10:57 2021

@author: dongwei
"""

import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import math

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

fig_cols = 10

labels = np.load(raw_dir + 'labels.npy')
labels = np.argmax(labels, 1)

A = sp.load_npz(raw_dir + 'adj.npz')

sub_graph_list = tl.mutual_info(A, K)
sub_graph_queue = []

for sub_graph in sub_graph_list:
    
    if(1 < sub_graph.number_of_nodes()):
        
        sub_graph_queue.append(sub_graph)

sub_graph_number = len(sub_graph_queue)

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

if(100 <= sub_graph_number):


    index_list = [math.floor(sub_graph_number * 0.0),
                  math.floor(sub_graph_number * 0.1),
                  math.floor(sub_graph_number * 0.2),
                  math.floor(sub_graph_number * 0.3),
                  math.floor(sub_graph_number * 0.4),
                  math.floor(sub_graph_number * 0.5),
                  math.floor(sub_graph_number * 0.6),
                  math.floor(sub_graph_number * 0.7),
                  math.floor(sub_graph_number * 0.8),
                  math.floor(sub_graph_number * 0.9),
                  ]
    
    fig_rows = len(index_list)
    
    object_list = []
    for i in index_list:
        
        for j in range(fig_cols):
            
            object_list.append(i + j)
    
    plt.figure('Figure2', figsize=(20, 20))
    
    count = 1
    
    for obj in object_list:
        
        sub_graph = sub_graph_queue[obj]
        
        color_list = []
        
        node_list = list(sub_graph.nodes())
        
        for l in labels[node_list]:
            
            color_list.append(color_dict[l])
        
        
        plt.subplot(fig_rows, fig_cols, count)
        
        node_size = 5
        if(len(node_list) <= 10):
            
            node_size = 50
            
        nx.draw(sub_graph, 
                pos=nx.spring_layout(sub_graph),
                node_color = color_list,
                node_size = node_size,
                edge_color = 'gray')
        
        '''
        nx.draw_spring(sub_graph,
                       node_color = color_list,
                       node_size = node_size,
                       edge_color = 'gray')
        '''
        
        
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        
        ax = plt.gca()
        ax.patch.set_facecolor('#FFF5EE') 
        
        count += 1
        
else:
    
    plt.figure('Figure2', figsize=(20, 20))
    
    count = 1
    
    for sub_graph in sub_graph_queue:
        
        color_list = []
        
        node_list = list(sub_graph.nodes())
        
        for l in labels[node_list]:
            
            color_list.append(color_dict[l])
        
        
        plt.subplot(10, 10, count)
        
        node_size = 5
        if(len(node_list) <= 10):
            
            node_size = 50
        
        nx.draw(sub_graph, 
                pos=nx.spring_layout(sub_graph),
                node_color = color_list,
                node_size = node_size,
                edge_color = 'gray')
        
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        
        ax = plt.gca()
        ax.patch.set_facecolor('#FFF5EE') 
        
        count += 1


plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
plt.savefig(raw_dir + 'subgraphs/mut_' + str(K) + '.pdf')
plt.close('all')
