# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:25:37 2021

@author: dongwei
"""

import argparse
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pickle

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
K = args.sampling_size

raw_dir = './' + args.dataset + '/'

A = sp.load_npz(raw_dir + 'adj.npz')

if(1 == K):
    
    I = tl.get_mutual_information(A)
    
    true_sample_index = tl.get_sim(I)
    np.savetxt(raw_dir + 'true_sample_index', true_sample_index, fmt = '%d')
    
else:
    
    if(None != K):
        
        I = tl.get_mutual_information(A)
    
        I = I.tocoo()
        I.data = np.exp(I.data)
        I = I.multiply(1 / I.sum(axis = 1))
        I = I.tocsr()
        true_sample_index = tl.get_mutli_sim(I, ratio = 1, K = K)
        
        row = []
        col = []
        i = 0
        for indexes in true_sample_index:
            
            for j in indexes:
                
                row.append(i)
                col.append(j)
                
            i += 1
        
        N = len(row)
           
        mask = sp.coo_matrix((np.ones(N), (row, col)), shape = A.shape).tocsr()
        A = A.multiply(mask)
    
    A = tl.get_L_DA(A)
    
    A = A.tocoo()
    
    indices = np.stack([A.row, A.col])
    indices = indices.T
    
    sparse_matrix = tf.SparseTensor(indices = indices,
                                    values = tf.convert_to_tensor(A.data, dtype = tf.float32),
                                    dense_shape = A.shape)
    
    with open(raw_dir + 'adj_tensor.pkl', 'wb') as fid:
    
        pickle.dump(tf.sparse.reorder(sparse_matrix), fid)