# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:07:51 2020

@author: yangm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random as rm
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import networkx as nx
import keras.backend as K


"""
Base Function
"""

def smoothness_features(adj, features):
    
    softmax_features = np.exp(features)
    
    features = softmax_features / np.sum(softmax_features, axis = 1, keepdims = True)
    
    E = len(adj.tocoo().data) / 2
    F = features.shape[-1]
    
    D = get_D(adj, 1.0, False)
    
    return np.linalg.norm(np.sum(np.power((D - adj).dot(features), 2), axis = 0), ord = 1) / (E * F)

def smoothness_labels(node, neighbors, labels, E):
    
    l = labels[node]
    n_l = labels[neighbors]
    
    return np.sum(n_l != l) / E

def get_D(adj, n, is_self_loop = True):
    
    d = adj.sum(axis = 1).A
    d = np.squeeze(d)
    if(True == is_self_loop):
        d = d + 1
    d = d.astype(np.float32)
    
    D_index = list(range(d.shape[0]))
    D = coo_matrix((np.power(d, n), (D_index, D_index)), shape=(d.shape[0], d.shape[0]), dtype=np.float32)
    D = D.tocsr()
    
    return D

def get_L_AD(adj):
    
    D = get_D(adj, -1.0)
    I = sp.identity(adj.shape[0])
    A = adj + I
    L_AD = A.dot(D)
    
    return L_AD

def get_L_DA(adj):
    
    D = get_D(adj, -1.0)
    I = sp.identity(adj.shape[0])
    A = adj + I
    L_DA = D.dot(A)
    
    return L_DA

def get_L_M(adj):
    
    D = get_D(adj, -1/2)
    I = sp.identity(adj.shape[0])
    A = adj + I
    L_M = D.dot(A)
    L_M = L_M.dot(D)
    
    return L_M

def loss_dot_product(y_pred, true_pred, size_splits = None, temperature = 1, bias = 1e-8):
    
    numerator = tf.exp(tf.reduce_sum(y_pred * true_pred, axis = 1, keepdims = True) / temperature)
    
    if(None != size_splits):
    
        y_pred_list = tf.split(y_pred, num_or_size_splits = size_splits)
        
        den_list = []
        
        for y_pred in y_pred_list:
            
            E_1 = tf.matmul(y_pred, tf.transpose(y_pred, perm = (1, 0)))
            den = tf.reduce_sum(tf.exp(E_1 / temperature), axis = 1, keepdims = True)
            den_list.append(den)
           
        denominator = tf.concat(den_list, axis = 0)
        
    else:
    
        E_1 = tf.matmul(y_pred, tf.transpose(y_pred, perm = (1, 0)))
    
        denominator = tf.reduce_sum(tf.exp(E_1 / temperature), axis = 1, keepdims = True)
    
    return -tf.reduce_mean(tf.math.log(numerator / (denominator + bias) + bias))

def loss_dot_product_v2(y_pred, true_pred, axis = 1, temperature = 1, bias = 1e-8):
    
    loss = 2 - 2 * tf.reduce_sum(y_pred * true_pred, axis = axis, keepdims = True) \
        / (temperature * tf.norm(y_pred, axis = axis, keepdims = True) * tf.norm(true_pred, axis = axis, keepdims = True) + bias)
    
    return tf.reduce_mean(loss)


def loss_dot_product_v3(y_pred, true_pred, axis = 1, temperature = 1, bias = 1e-8):
    
    numerator = tf.exp(tf.reduce_sum(y_pred * true_pred, axis = axis, keepdims = True) / temperature)
    
    if(0 == axis):
    
        E_1 = tf.matmul(tf.transpose(y_pred, perm = (1, 0)), y_pred)
        
    else:
        
        E_1 = tf.matmul(y_pred, tf.transpose(y_pred, perm = (1, 0)))
    
    denominator = tf.reduce_sum(tf.exp(E_1 / temperature), axis = axis, keepdims = True)
    
    return -tf.reduce_mean(tf.math.log(numerator / (denominator + bias) + bias))

def auto_correlation(y_pred, lam = 0.005, bias = 1e-8):
    
    D = y_pred.shape[-1]
    N = y_pred.shape[0]
    
    y_pred_mean = tf.reduce_mean(y_pred, axis = 0, keepdims = True)
    
    y_pred = y_pred - y_pred_mean
    
    C = tf.matmul(tf.transpose(y_pred, perm=[1, 0]), y_pred) / N
    
    I = tf.eye(D)
    
    C_diff = tf.pow(C - I, 2)
    
    part_diag = tf.linalg.diag_part(C_diff)
    
    return tf.reduce_sum(part_diag) + tf.reduce_sum(lam * (C_diff - tf.linalg.diag(part_diag)))


def cross_correlation(y_pred, true_pred, lam = 0.005, bias = 1e-8):
    
    D = y_pred.shape[-1]
    N = y_pred.shape[0]
    
    y_pred_mean = tf.reduce_mean(y_pred, axis = 0, keepdims = True)
    
    y_pred = y_pred - y_pred_mean
    
    true_pred_mean = tf.reduce_mean(true_pred, axis = 0, keepdims = True)
    
    true_pred = true_pred - true_pred_mean

    C = tf.matmul(tf.transpose(y_pred, perm=[1, 0]), true_pred) / N
    
    I = tf.eye(D)
    
    C_diff = tf.pow(C - I, 2)
    
    part_diag = tf.linalg.diag_part(C_diff)
    
    return tf.reduce_sum(part_diag) + tf.reduce_sum(lam * (C_diff - tf.linalg.diag(part_diag)))


def get_mutual_information(A, bias = 1e-10):
    
    A = A.tocsr()
    
    N = A.shape[0]
    
    d = A.sum(axis = 1).A
    p_x = d / N
    
    p_x = np.squeeze(p_x, 1)
    p_inv_x = 1.0 - p_x
    
    p_x_y_row = []
    p_x_y_col = []
    p_x_y_data = []
    
    p_x_inv_y_row = []
    p_x_inv_y_col = []
    p_x_inv_y_data = []
    
    p_inv_x_y_row = []
    p_inv_x_y_col = []
    p_inv_x_y_data = []
    
    p_inv_x_inv_y_row = []
    p_inv_x_inv_y_col = []
    p_inv_x_inv_y_data = []
    
    for i in range(N):
        
        A_i = np.squeeze(A[i].toarray(), 0)
        
        for j in np.squeeze(np.argwhere(A_i > 0), 1): 
        
            if(i == j):
                
                continue;
                
            A_j = np.squeeze(A[j].toarray(), 0)
                
            t_i = (0 < A_i)
            t_j = (0 < A_j)
            f_i = (0 == A_i)
            f_j = (0 == A_j)
            
            t_i_t_j = (t_i * t_j)
            e = (t_i_t_j > 0).sum() / float(N)
            p_x_y_row.append(i)
            p_x_y_col.append(j)
            p_x_y_data.append(e * np.log(e / (p_x[i] * p_x[j] + bias) + bias))
            
            t_i_f_j = (t_i * f_j)
            e = (t_i_f_j > 0).sum() / float(N)
            p_x_inv_y_row.append(i)
            p_x_inv_y_col.append(j)
            p_x_inv_y_data.append(e * np.log(e / (p_x[i] * p_inv_x[j] + bias) + bias))
            
            f_i_t_j = (f_i * t_j)
            e = (f_i_t_j > 0).sum() / float(N)
            p_inv_x_y_row.append(i)
            p_inv_x_y_col.append(j)
            p_inv_x_y_data.append(e * np.log(e / (p_inv_x[i] * p_x[j] + bias) + bias))
            
            f_i_f_j = (f_i * f_j)
            e = (f_i_f_j > 0).sum() / float(N)
            p_inv_x_inv_y_row.append(i)
            p_inv_x_inv_y_col.append(j)
            p_inv_x_inv_y_data.append(e * np.log(e / (p_inv_x[i] * p_inv_x[j] + bias) + bias))
    
    I_x_y = sp.coo_matrix((p_x_y_data, (p_x_y_row, p_x_y_col)), shape = (N, N)).tocsr()
    I_x_inv_y = sp.coo_matrix((p_x_inv_y_data, (p_x_inv_y_row, p_x_inv_y_col)), shape = (N, N)).tocsr()
    I_inv_x_y = sp.coo_matrix((p_inv_x_y_data, (p_inv_x_y_row, p_inv_x_y_col)), shape = (N, N)).tocsr()
    I_inv_x_inv_y = sp.coo_matrix((p_inv_x_inv_y_data, (p_inv_x_inv_y_row, p_inv_x_inv_y_col)), shape = (N, N)).tocsr()
    
    I = I_x_y + I_x_inv_y + I_inv_x_y + I_inv_x_inv_y
    
    return I
    
def get_sim(I):
    
    true_sample_index = []
    
    N = I.shape[0]
    
    for i in range(N):
        
        I_i = np.squeeze(I[i].toarray(), 0)
        j = rm.choice(get_max_indices(I_i))
        true_sample_index.append(j)
            
    return true_sample_index

def get_mutli_sim(I, ratio = 0.5, K = 3, is_padding = False):
    
    true_sample_index = []
    
    N = I.shape[0]
    
    for i in range(N):
        
        I_i = np.squeeze(I[i].toarray(), 0)
        index = get_reverse_sort(I_i, ratio = ratio, K = K, is_padding = is_padding)
        true_sample_index.append(list(map(int, index)))
            
    return true_sample_index

def get_random(A):
    
    A = A.tocsr()
    
    N = A.shape[0]
    
    true_sample_index = []
    
    for i in range(N):
        
        A_i = np.squeeze(A[i].toarray(), 0)
        j = rm.choice(np.squeeze(np.argwhere(A_i > 0), 1).tolist())
        true_sample_index.append(j)
        
    return true_sample_index


def get_max_indices(nums):

    max_of_nums = np.max(nums)
    tup = [(i, nums[i]) for i in range(len(nums))]
    return [i for i, n in tup if n == max_of_nums]

def get_reverse_sort(nums, ratio = 0.5, K = 5, is_padding = False):
    
    tup = [(i, nums[i]) for i in range(nums.size)]
    tup = [[item[0], item[1]] for item in sorted(tup, key = lambda x:x[1], reverse = True)]
    tup = np.array(tup)
    
    count = 0
    index = []
    
    for item in tup[0:K]:
        
        if(item[1] > 0):
            count = item[1] + count
        
            index.append(item[0])
        
        if(ratio <= count):
            break;
    
    if(True == is_padding):
        
        index = np.array(index, dtype=int)
        
        if(index.size < K):
        
            index = np.pad(index, [0, K - index.size], 'constant', constant_values = -1)
    
    return index


def get_mut_adj(I):
    
    mut_adj_row = []
    mut_adj_col = []
    mut_adj_data = []
    N = I.shape[0]
    
    for i in range(N):
        
        I_i = np.squeeze(I[i].toarray(), 0)
        
        for j in get_max_indices(I_i):
            
            mut_adj_row.append(i)
            mut_adj_col.append(j)
            mut_adj_data.append(1)
            
    return sp.coo_matrix((mut_adj_data, (mut_adj_row, mut_adj_col)), shape = (N, N))


def mutual_info(A, K):
    
    I = get_mutual_information(A)
    true_sample_index = get_mutli_sim(I, ratio = 1, K = K)

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
    #A = A.multiply(mask)
    
    sub_graph_list = []
    graph = nx.from_scipy_sparse_matrix(mask)
    
    for sub_nodes in sorted(nx.connected_components(graph),key=len,reverse=True):
        
        sub_graph = graph.subgraph(sub_nodes)
        sub_graph_list.append(sub_graph)

    return sub_graph_list
    
def Newton_ZCA_for_features(X, T = 5, temp = 10, epsilon = 1e-8):
    
    d = X.shape[-1]
    m = X.shape[0]
    
    mean = tf.reduce_mean(X, axis = 0, keepdims = True)
    
    X_mean = X - mean
    
    X_mean_T = tf.transpose(X_mean, perm = [1, 0])
    
    C = tf.matmul(X_mean_T, X_mean) / m + epsilon * tf.eye(d)
    
    part_diag = tf.linalg.diag_part(C)
    
    tr = tf.reduce_sum(part_diag)
    
    C = C / tr
    
    P = tf.eye(d)
    
    for _ in range(T):
        
        P = (3 * P - tf.matmul(P, tf.matmul(P, tf.matmul(P, C)))) / 2
        
    C = P / (temp * tf.sqrt(tr))
    
    return tf.matmul(X_mean, C), mean, C

def Schur_Newton_ZCA_for_features(X, T = 5, temp = 10, epsilon = 1e-8):
    
    d = X.shape[-1]
    m = X.shape[0]
    I = tf.eye(d)
    
    mean = tf.reduce_mean(X, axis = 0, keepdims = True)
    
    X_mean = X - mean
    
    X_mean_T = tf.transpose(X_mean, perm = [1, 0])
    
    C = tf.matmul(X_mean_T, X_mean) / m + epsilon * I
    
    part_diag = tf.linalg.diag_part(C)
    
    tr = tf.reduce_sum(part_diag)
    
    C = C / tr
    
    P = tf.eye(d)
    
    for _ in range(T):
        
        C_T = (3 * I - C) / 2
        
        P = tf.matmul(P, C_T)
        
        C = tf.matmul(C_T, tf.matmul(C_T, C))
        
    C = P / (temp * tf.sqrt(tr))
    
    return tf.matmul(X_mean, C), mean, C

def Newton_ZCA_for_samples(X, T = 5, temp = 10, epsilon = 1e-8):
    
    d = X.shape[-1]
    m = X.shape[0]
    
    mean = tf.reduce_mean(X, axis = 1, keepdims = True)
    
    X_mean = X - mean
    
    X_mean_T = tf.transpose(X_mean, perm = [1, 0])
    
    C = tf.matmul(X_mean, X_mean_T) / d + epsilon * tf.eye(m)
    
    part_diag = tf.linalg.diag_part(C)
    
    tr = tf.reduce_sum(part_diag)
    
    C = C / tr
    
    P = tf.eye(m)
    
    for _ in range(T):
        
        P = (3 * P - tf.matmul(P, tf.matmul(P, tf.matmul(P, C)))) / 2
        
    C = P / (temp * tf.sqrt(tr))
    
    return tf.matmul(C, X_mean), mean, C

def Schur_Newton_ZCA_for_samples(X, T = 5, temp = 10, epsilon = 1e-8):
    
    d = X.shape[-1]
    m = X.shape[0]
    I = tf.eye(m)
    
    mean = tf.reduce_mean(X, axis = 1, keepdims = True)
    
    X_mean = X - mean
    
    X_mean_T = tf.transpose(X_mean, perm = [1, 0])
    
    C = tf.matmul(X_mean, X_mean_T) / d + epsilon * I
    
    part_diag = tf.linalg.diag_part(C)
    
    tr = tf.reduce_sum(part_diag)
    
    C = C / tr
    
    P = tf.eye(m)
    
    for _ in range(T):
        
        C_T = (3 * I - C) / 2
        
        P = tf.matmul(P, C_T)
        
        C = tf.matmul(C_T, tf.matmul(C_T, C))
        
    C = P / (temp * tf.sqrt(tr))
    
    return tf.matmul(C, X_mean), mean, C

def centering(K):
    
    n = K.shape[0]
    unit = tf.ones([n, n])
    I = tf.eye(n)
    H = I - unit / n
    
    return tf.matmul(tf.matmul(H, K), H)

def sp_dot_sp_T(sp_Y):
    
    sp_Y = sp_Y.tocoo()
    
    sp_Y_T = coo_matrix((sp_Y.data, (sp_Y.col, sp_Y.row)), shape = (sp_Y.shape[1], sp_Y.shape[0]))
    
    sp_Y = sp_Y.tocsr()
    sp_Y_T = sp_Y_T.tocsr()
    
    L_Y = sp_Y.dot(sp_Y_T).todense()
    
    return L_Y

def linear_HSIC_d_s(X, sp_Y):
    
    L_X = tf.matmul(X, tf.transpose(X, perm=[1, 0]))
    
    L_Y = sp_dot_sp_T(sp_Y)
    
    L_Y = tf.convert_to_tensor(L_Y, dtype = tf.float32)
    
    return tf.reduce_sum(centering(L_X) * centering(L_Y))

def linear_HSIC_s_s(sp_X, sp_Y):
    
    L_X = sp_dot_sp_T(sp_X)
    
    L_X = tf.convert_to_tensor(L_X, dtype = tf.float32)
    
    L_Y = sp_dot_sp_T(sp_Y)
    
    L_Y = tf.convert_to_tensor(L_Y, dtype = tf.float32)
    
    return tf.reduce_sum(centering(L_X) * centering(L_Y))

def linear_HSIC(X, Y):
    
    L_X = tf.matmul(X, tf.transpose(X, perm=[1, 0]))
    L_Y = tf.matmul(Y, tf.transpose(Y, perm=[1, 0]))
    
    return tf.reduce_sum(centering(L_X) * centering(L_Y))
    #return tf.reduce_sum(L_X * L_Y)

def linear_CKA_d_s(X, sp_Y):
    
    hsic = linear_HSIC_d_s(X, sp_Y)
    var1 = tf.math.sqrt(linear_HSIC(X, X))
    var2 = tf.math.sqrt(linear_HSIC_s_s(sp_Y, sp_Y))
    
    return hsic / (var1 * var2)

def linear_CKA(y_pred, true_pred):
    
    hsic = linear_HSIC(y_pred, true_pred)
    var1 = tf.math.sqrt(linear_HSIC(y_pred, y_pred))
    var2 = tf.math.sqrt(linear_HSIC(true_pred, true_pred))
    
    return hsic / (var1 * var2)

def rbf(X, sigma = None):
    
    GX = tf.matmul(X, tf.transpose(X, perm=[1, 0]))
    GX = tf.linalg.diag_part(GX) - GX
    KX = GX + tf.transpose(GX, perm=[1, 0])
    
    if sigma is None:
        mean = tf.reduce_mean(KX[KX != 0])
        sigma = tf.math.sqrt(mean)
    
    KX *= - 0.5 / (sigma * sigma)
    KX = tf.math.exp(KX)
    
    return KX

def kernel_HSIC(X, Y, sigma):
    
    return tf.reduce_sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def kernel_CKA(X, Y, sigma = None):
    
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = tf.math.sqrt(kernel_HSIC(X, X, sigma))
    var2 = tf.math.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def KL_mean_var(z_mean, z_log_var, alpha = 0.5):
    
    kl_loss = - alpha * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var))
    
    return kl_loss
