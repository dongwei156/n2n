# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 08:11:02 2021

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
        default = None)

args = parser.parse_args()

raw_dir = './' + args.dataset + '/'
labels = np.load(raw_dir + 'labels.npy')
labels = np.argmax(labels, 1)

if(None == args.sampling_size):
    K = labels.shape[0]
else:
    K = args.sampling_size

A = sp.load_npz(raw_dir + 'adj.npz')

I = tl.get_mutual_information(A)
true_sample_index = tl.get_mutli_sim(I, ratio = 1, K = K)

#node_list = []

smoothness_label = 0.

i = 0
for indexes in true_sample_index:
    
    smoothness_label += tl.smoothness_labels(i, indexes, labels, len(A.data))
        
    i += 1

print(smoothness_label)
#np.savetxt(raw_dir + 'pdf/label_smoothness_' + str(K) + '.txt', np.array(node_list), fmt='%f')
    
