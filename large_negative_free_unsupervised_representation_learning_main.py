# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:45:29 2022

@author: dongwei
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import sys
import argparse
import os
import heapq
import math

import pickle

from sklearn.metrics import f1_score

import tools as tl

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

    
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    
    
class SparseLinear(tf.keras.layers.Layer):
    def __init__(self, units = 32, use_bias = False, L2 = None, seed = 1):
        
        super(SparseLinear, self).__init__()
        
        self.units = units
        self.use_bias = use_bias
        self.L2 = L2
        self.seed = seed
            
    def build(self, input_shape):
        
        self.w = self.add_weight(
            shape = (input_shape[-1], self.units),
            initializer = tf.keras.initializers.GlorotUniform(seed = self.seed),
            trainable = True
        )
        
        if(True == self.use_bias):
        
            self.b = self.add_weight(
                shape=(self.units,), 
                initializer=tf.keras.initializers.GlorotUniform(seed = self.seed), 
                trainable=True
            )

    def call(self, sparse_inputs):
        
        result = tf.sparse.sparse_dense_matmul(sparse_inputs, self.w)
        
        if(None != self.L2):
        
            self.add_loss(self.L2 * tf.math.reduce_sum(tf.math.square(self.w)))
        
        if(True == self.use_bias):
            
            result = result + self.b
            
            if(None != self.L2):
            
                self.add_loss(self.L2 * tf.math.reduce_sum(tf.math.square(self.b)))
        
        return result
    
    
class MLPModel(tf.keras.Model):
    
    def __init__(self,
                 layer_units,
                 output_units,
                 dropout_func = tf.keras.layers.Dropout,
                 dropout = 0.6,
                 L2 = 0.01,
                 activation = lambda x:x,
                 residual = False, 
                 first_layer_is_dense = False,
                 batch_normalization = 'ZCA',
                 ZCA_iteration = 10,
                 ZCA_lam = 0.5,
                 ZCA_temp = 10,
                 seed = 1):
        
        super(MLPModel, self).__init__()
        
        self.output_units = output_units
        self.residual = residual
        self.ZCA_iteration = ZCA_iteration
        self.batch_normalization = batch_normalization
        self.lam = ZCA_lam
        self.ZCA_temp = ZCA_temp
        
        self.dense_layers = []
        self.dropout_layers = []
        self.residual_layers = []
        self.ZCA_mean_list = []
        self.ZCA_C_list = []
        
        #self.dropout_layer = dropout_func(dropout)
        self.activation = activation
        
        self.ZCA_mean_list.append(tf.zeros([1, layer_units[0]]))
        self.ZCA_C_list.append(tf.eye(layer_units[0]))
        
        if(False == first_layer_is_dense):
        
            dense_layer = SparseLinear(units = layer_units[0],
                                       use_bias = False,
                                       L2 = L2,
                                       seed = seed)
            
        else:
        
            dense_layer = tf.keras.layers.Dense(layer_units[0], 
                                                use_bias = False,
                                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                                kernel_regularizer = tf.keras.regularizers.l2(L2))
        
        self.dense_layers.append(dense_layer)
        self.dropout_layers.append(dropout_func(dropout))
        
        for l in range(1, len(layer_units)):
            
            self.ZCA_mean_list.append(tf.zeros([1, layer_units[l]]))
            self.ZCA_C_list.append(tf.eye(layer_units[l]))
            
            dense_layer = tf.keras.layers.Dense(layer_units[l], 
                                                use_bias = False,
                                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                                kernel_regularizer = tf.keras.regularizers.l2(L2))
            self.dense_layers.append(dense_layer)
            self.dropout_layers.append(dropout_func(dropout))
            
            if(True == residual):
                
                residual_layer = tf.keras.layers.Dense(layer_units[l], 
                                                       use_bias = False,
                                                       kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                                       kernel_regularizer = tf.keras.regularizers.l2(L2))
                self.residual_layers.append(residual_layer)
                
        self.ZCA_mean_list.append(tf.zeros([1, output_units]))
        self.ZCA_C_list.append(tf.eye(output_units))
                
        dense_layer = tf.keras.layers.Dense(output_units, 
                                            use_bias = False,
                                            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                            kernel_regularizer = tf.keras.regularizers.l2(L2))
        self.dense_layers.append(dense_layer)
        self.dropout_layers.append(dropout_func(dropout))
        
        
    def call(self, x, training = True):
        
        deep_copy_list = []
        
        x = self.dense_layers[0](x)
        
        if('ZCA' == self.batch_normalization):
            
            if(True == training):
            
                x, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                
                self.ZCA_mean_list[0] = (1 - self.lam) * self.ZCA_mean_list[0] + self.lam * ZCA_mean
                self.ZCA_C_list[0] = (1 - self.lam) * self.ZCA_C_list[0] + self.lam * ZCA_C
                
            else:
                
                x = tf.matmul(x - self.ZCA_mean_list[0], self.ZCA_C_list[0])
                
        elif('Schur_ZCA' == self.batch_normalization):
            
            if(True == training):
            
                x, ZCA_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                
                self.ZCA_mean_list[0] = (1 - self.lam) * self.ZCA_mean_list[0] + self.lam * ZCA_mean
                self.ZCA_C_list[0] = (1 - self.lam) * self.ZCA_C_list[0] + self.lam * ZCA_C
                
            else:
                
                x = tf.matmul(x - self.ZCA_mean_list[0], self.ZCA_C_list[0])
                
        elif('BN' == self.batch_normalization):
        
            x = tf.keras.layers.BatchNormalization()(x, training = training)
        
        x = self.activation(x)
        
        x = self.dropout_layers[0](x)
        
        deep_copy_list.append(tf.identity(x))
        
        for l in range(1, len(self.dense_layers) - 1):
            
            h = x
            
            h = self.dense_layers[l](h)
            
            if(True == self.residual):
                
                h = h + self.residual_layers[l-1](x)
                
            if('ZCA' == self.batch_normalization):
                
                if(True == training):
                
                    h, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(h, temp = self.ZCA_temp, T = self.ZCA_iteration)
                    
                    self.ZCA_mean_list[l] = (1 - self.lam) * self.ZCA_mean_list[l] + self.lam * ZCA_mean
                    self.ZCA_C_list[l] = (1 - self.lam) * self.ZCA_C_list[l] + self.lam * ZCA_C
                    
                else:
                    
                    h = tf.matmul(h - self.ZCA_mean_list[l], self.ZCA_C_list[l])
                    
            elif('Schur_ZCA' == self.batch_normalization):
                
                if(True == training):
                
                    h, ZCA_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(h, temp = self.ZCA_temp, T = self.ZCA_iteration)
                    
                    self.ZCA_mean_list[l] = (1 - self.lam) * self.ZCA_mean_list[l] + self.lam * ZCA_mean
                    self.ZCA_C_list[l] = (1 - self.lam) * self.ZCA_C_list[l] + self.lam * ZCA_C
                    
                else:
                    
                    h = tf.matmul(h - self.ZCA_mean_list[l], self.ZCA_C_list[l])
                    
            elif('BN' == self.batch_normalization):
            
                h = tf.keras.layers.BatchNormalization()(h, training = training)
                
            h = self.activation(h)
            
            x = self.dropout_layers[l](h)
            
            deep_copy_list.append(tf.identity(x))
        
        x = self.dense_layers[-1](x)
        
        if('ZCA' == self.batch_normalization):
            
            if(True == training):
            
                x, ZCA_mean, ZCA_C = tl.Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                
                self.ZCA_mean_list[-1] = (1 - self.lam) * self.ZCA_mean_list[-1] + self.lam * ZCA_mean
                self.ZCA_C_list[-1] = (1 - self.lam) * self.ZCA_C_list[-1] + self.lam * ZCA_C
                
            else:
                
                x = tf.matmul(x - self.ZCA_mean_list[-1], self.ZCA_C_list[-1])
        
        elif('Schur_ZCA' == self.batch_normalization):
            
            if(True == training):
            
                x, ZCA_mean, ZCA_C = tl.Schur_Newton_ZCA_for_features(x, temp = self.ZCA_temp, T = self.ZCA_iteration)
                
                self.ZCA_mean_list[-1] = (1 - self.lam) * self.ZCA_mean_list[-1] + self.lam * ZCA_mean
                self.ZCA_C_list[-1] = (1 - self.lam) * self.ZCA_C_list[-1] + self.lam * ZCA_C
                
            else:
                
                x = tf.matmul(x - self.ZCA_mean_list[-1], self.ZCA_C_list[-1])
                
        elif('BN' == self.batch_normalization):
        
            x = tf.keras.layers.BatchNormalization()(x, training = training)
            
        x = self.activation(x)
        
        x = self.dropout_layers[-1](x)
        
        deep_copy_list.append(x)
            
        return deep_copy_list
    
def unsupervised_learning(features,
                          anchor_indexes,
                          augmented_indexes,
                          model,
                          contrast_loss,
                          batch_size = 64,
                          lam = 0.005,
                          temperature = 1.0,
                          num_epochs = 5000,
                          learning_rate = 0.001,
                          alpha = 1.0,
                          beta = 1.0):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
    train_loss_results = []
    
    feat_ds = tf.data.Dataset.from_tensor_slices((anchor_indexes, augmented_indexes)).shuffle(10000).batch(batch_size)
    
    for epoch in range(num_epochs):
                
        loss_value_list = []
                
        for anc_ind, aug_ind in feat_ds:
        
            with tf.GradientTape() as tape:
                
                anc_feat = convert_sparse_matrix_to_sparse_tensor(features[anc_ind.numpy(), :])        
                
                y_pred_list = model(anc_feat, training = True)
                
                aug_feat = convert_sparse_matrix_to_sparse_tensor(features[aug_ind.numpy(), :])
                
                true_pred_list = model(aug_feat, training = True)
                
                loss_value = 0.0
                
                for y_pred, true_pred in list(zip(y_pred_list, true_pred_list)):
                    
                    y_pred = y_pred / (tf.norm(y_pred, ord = 2, axis = 1, keepdims = True) + 1e-8)
                    true_pred = true_pred / (tf.norm(true_pred, ord = 2, axis = 1, keepdims = True) + 1e-8)
                    
                    if('centered_kernel_alignment' == contrast_loss):
                        
                        loss_value += (1 / tl.linear_CKA(y_pred, true_pred))
                
                    elif('distance' == contrast_loss):
                            
                        loss_value += tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, temperature = temperature)
                        
                    elif('signal_distance' == contrast_loss):
                            
                        loss_value += tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature)
                        
                    elif('contrastive' == contrast_loss):
                        
                        loss_value += tl.loss_dot_product_v3(y_pred = y_pred, true_pred = true_pred, temperature = temperature)
                        
                    elif('signal_contrastive' == contrast_loss):
                        
                        loss_value += tl.loss_dot_product_v3(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature)
                        
                    elif('distance+auto-correlation' == contrast_loss):
                        
                        loss_value += (alpha * tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, temperature = temperature) + \
                                       beta * tl.auto_correlation(y_pred = y_pred, lam = lam) + tl.auto_correlation(y_pred = true_pred, lam = lam))
                            
                    elif('signal_distance+auto-correlation' == contrast_loss):
                        
                        loss_value += (alpha * tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature) + \
                                       beta * tl.auto_correlation(y_pred = y_pred, lam = lam) + tl.auto_correlation(y_pred = true_pred, lam = lam))
                            
                    elif('cross-correlation+auto-correlation' == contrast_loss):
                        
                        loss_value += (alpha * tl.cross_correlation(y_pred = y_pred, true_pred = true_pred, lam = lam) + \
                                       beta * tl.auto_correlation(y_pred = y_pred, lam = lam) + tl.auto_correlation(y_pred = true_pred, lam = lam))
                            
                    elif('cross-correlation' == contrast_loss):
                            
                        loss_value += tl.cross_correlation(y_pred = y_pred, true_pred = true_pred, lam = lam)
                
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
                loss_value_list.append(loss_value.numpy())
        
        loss = np.mean(np.array(loss_value_list))    
        train_loss_results.append(loss)
            
        print('Epoch {:03d}: Loss: {:.5f}'.format(epoch + 1, loss))
        
    #model.summary()
    
    '''
    fig, axes = plt.subplots(1, sharex = True, figsize = (12, 48))
    fig.suptitle('Metrics')
    
    axes.set_ylabel('Loss', fontsize=14)
    axes.plot(train_loss_results)
    
    plt.show()
    '''
    
    return model


def downstream_learning(train_data,
                        val_data,
                        test_data,
                        train_y_true,
                        val_y_true,
                        test_y_true,
                        model,
                        batch_size = 512,
                        num_epochs = 5000,
                        learning_rate = 0.0006,
                        early_stopping = 80):
    
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []
    test_accuracy_results = []
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    stop_count = early_stopping
    max_test_metric = 0.
    
    for epoch in range(num_epochs):
        
        loss_value = 0.
        
        y_pred_train_list = []
        
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_y_true)).shuffle(10000).batch(batch_size)
        
        for (train, train_label) in train_ds:
        
            with tf.GradientTape() as tape:
                
                y_pred_train = model(train, training = True)
            
                loss = loss_object(y_true = train_label, y_pred = y_pred_train)
                
                y_pred_train_list.append(tf.argmax(y_pred_train, axis = 1, output_type = tf.int32))
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                loss_value += loss
                
        loss_value /= len(train_ds)
            
        train_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        
        y_pred_train_list = []
        
        for train in train_ds:
            
            y_pred_train = model(train, training = False)
            
            y_pred_train_list.append(tf.argmax(y_pred_train, axis = 1, output_type = tf.int32))
            
        y_pred_train = tf.concat(y_pred_train_list, axis = 0)
        
        val_ds = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size)
        
        y_pred_val_list = []
        
        for val in val_ds:
            
            y_pred_val = model(val, training = False)
            
            y_pred_val_list.append(tf.argmax(y_pred_val, axis = 1, output_type = tf.int32))
            
        y_pred_val = tf.concat(y_pred_val_list, axis = 0)
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
        
        y_pred_test_list = []
        
        for test in test_ds:
            
            y_pred_test = model(test, training = False)
            
            y_pred_test_list.append(tf.argmax(y_pred_test, axis = 1, output_type = tf.int32))
            
        y_pred_test = tf.concat(y_pred_test_list, axis = 0)
            
        train_metric = f1_score(train_y_true.numpy(), y_pred_train.numpy(), average = 'micro')
        val_metric = f1_score(val_y_true.numpy(), y_pred_val.numpy(), average = 'micro')
        test_metric = f1_score(test_y_true.numpy(), y_pred_test.numpy(), average = 'micro')
            
        loss_value = loss_value.numpy()
        train_loss_results.append(loss_value)
        train_accuracy_results.append(train_metric)
        val_accuracy_results.append(val_metric)
        test_accuracy_results.append(test_metric)
            
        if(test_metric > max_test_metric):
            max_test_metric = test_metric
        
        print('Epoch {:03d}: Loss: {:.5f}, Train metric: {:.2%}, Val metric: {:.2%}, Test metric: {:.2%}'.format(
            epoch + 1,
            loss_value,
            train_metric,
            val_metric,
            test_metric
            ))
        
        if(len(train_loss_results) >= early_stopping):
            stop = np.array(heapq.nsmallest(early_stopping, train_loss_results))
            stop = stop >= loss_value
        
            if(stop.all()):
                stop_count = early_stopping
            else:
                stop_count -= 1
                
        if(0 > stop_count or True == math.isnan(loss_value)):
            break;
                
    model.summary()
        
    return max_test_metric


parser = argparse.ArgumentParser()

parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'feather-lastfm-social')

'''
    centered_kernel_alignment
    distance (ZCA)
    signal_distance (ZCA)
    contrastive (Info-NCE)
    signal_contrastive (Info-NCE)
    distance+auto-correlation
    signal_distance+auto-correlation
    cross-correlation+auto-correlation
    cross-correlation (Barlow Tiwns)
'''

parser.add_argument(
        '--alpha',
        type = float,
        default = 1.0)

parser.add_argument(
        '--beta',
        type = float,
        default = 1.0)

parser.add_argument(
        '-cl', 
        '--contrast-loss',
        type = str,
        default = 'distance')

parser.add_argument(
        '-la',
        '--lam',
        type = float,
        default = 0.0) #0.0

parser.add_argument(
        '-t',
        '--temperature',
        type = float,
        default = 1.0)

parser.add_argument(
        '-r',
        '--num-run',
        type = int,
        default = 1)

parser.add_argument(
        '-ubs',
        '--unsupervised-batch-size',
        type = int,
        default = 20480) #65536

parser.add_argument(
        '-dbs',
        '--downstream-batch-size',
        type = int,
        default = 20480) #65536

parser.add_argument(
        '-bn', 
        '--batch-normalization',
        type = str,
        default = 'Schur_ZCA') # None, BN, ZCA, Schur_ZCA

parser.add_argument(
        '-zi',
        '--ZCA-iteration',
        type = int,
        default = 14)

parser.add_argument(
        '-zt',
        '--ZCA-temp',
        type = float,
        default = 1.0) # 0.05

parser.add_argument(
        '-zl',
        '--ZCA-lam',
        type = float,
        default = 0.0)

parser.add_argument(
        '-pe',
        '--pretext-num-epochs',
        type = int,
        default = 30)

parser.add_argument(
        '-de',
        '--downstream-num-epochs',
        type = int,
        default = 1500)

parser.add_argument(
        '-pu',
        '--pretext-hidden-units',
        nargs='+', 
        type=int,
        default = [512])

parser.add_argument(
        '-pt',
        '--pretext-output-units',
        type = int,
        default = 512)

parser.add_argument(
        '-po',
        '--pretext-dropout',
        type = float,
        default = 0.)

parser.add_argument(
        '-do',
        '--downstream-dropout',
        type = float,
        default = 0.2)

parser.add_argument(
        '-pL',
        '--pretext-L2',
        type = float,
        default = 0.)

parser.add_argument(
        '-dL',
        '--downstream-L2',
        type = float,
        default = 0.001)

parser.add_argument(
        '-pl',
        '--pretext-learning-rate',
        type = float,
        default = 0.001)

parser.add_argument(
        '-dl',
        '--downstream-learning-rate',
        type = float,
        default = 0.002)

parser.add_argument(
        '-y',
        '--early-stopping',
        type = int,
        default = 200)



args = parser.parse_args()

data_dir = './' + args.dataset + '/'

features_list = []

with open(data_dir + 'adj.pkl', 'rb') as file_adj:
    
    features_list = pickle.load(file_adj)
    
anchor_indexes_list = []
augmented_indexes_list = []
    
with open(data_dir + 'anchor_indexes.pkl', 'rb') as file_anchor:
    
    anchor_indexes_list = pickle.load(file_anchor)
    
with open(data_dir + 'augmented_indexes.pkl', 'rb') as file_augmented:
    
    augmented_indexes_list = pickle.load(file_augmented)
    
num_classes = None

with open(data_dir + 'cls_count.txt', encoding='utf-8') as file_num_classes:

    num_classes = int(file_num_classes.readline())
    

train_set_list = []
train_y_true_list = []

with open(data_dir + 'dic_train.pkl', 'rb') as file_train:
    
    dic_train_list = pickle.load(file_train)
    
    for dic_train in dic_train_list:
    
        train_set_list.append(list(dic_train.keys()))
        train_y_true_list.append(np.array(list(dic_train.values())))
        
train_y_true = np.concatenate(train_y_true_list, axis = 0)
        
    
val_set_list = []
val_y_true_list = []

with open(data_dir + 'dic_val.pkl', 'rb') as file_val:
    
    dic_val_list = pickle.load(file_val)
    
    for dic_val in dic_val_list:
        
        val_set_list.append(list(dic_val.keys()))
        val_y_true_list.append(np.array(list(dic_val.values())))
        
val_y_true = np.concatenate(val_y_true_list, axis = 0)
    
test_set_list = []
test_y_true_list = []

with open(data_dir + 'dic_test.pkl', 'rb') as file_test:
    
    dic_test_list = pickle.load(file_test)
    
    for dic_test in dic_test_list:
        
        test_set_list.append(list(dic_test.keys()))
        test_y_true_list.append(np.array(list(dic_test.values())))
        
test_y_true = np.concatenate(test_y_true_list, axis = 0)

train_y_true = tf.convert_to_tensor(train_y_true, dtype = tf.int32)
val_y_true = tf.convert_to_tensor(val_y_true, dtype = tf.int32)
test_y_true = tf.convert_to_tensor(test_y_true, dtype = tf.int32)
    
    
train_representation_list = []
val_representation_list = []
test_representation_list = []

for i in range(len(features_list)):
    
    features = features_list[i].tocsr()
    
    print(features.shape)

    model = MLPModel(layer_units = args.pretext_hidden_units,
                     output_units = args.pretext_output_units,
                     dropout_func = tf.keras.layers.Dropout,
                     dropout = args.pretext_dropout,
                     L2 = args.pretext_L2,
                     activation = tf.nn.relu,
                     residual = False,
                     batch_normalization = args.batch_normalization,
                     ZCA_iteration = args.ZCA_iteration,
                     ZCA_lam = args.ZCA_lam,
                     ZCA_temp = args.ZCA_temp)
    
    unsupervised_model = unsupervised_learning(features = features,
                                               anchor_indexes = tf.convert_to_tensor(anchor_indexes_list[i], dtype=tf.int32),
                                               augmented_indexes = tf.convert_to_tensor(augmented_indexes_list[i], dtype=tf.int32),
                                               model = model,
                                               contrast_loss = args.contrast_loss,
                                               batch_size = args.unsupervised_batch_size,
                                               lam = args.lam,
                                               temperature = args.temperature,
                                               num_epochs = args.pretext_num_epochs,
                                               learning_rate = args.pretext_learning_rate,
                                               alpha = args.alpha,
                                               beta = args.beta)
    
    train_representation = unsupervised_model(convert_sparse_matrix_to_sparse_tensor(features[train_set_list[i], :]), training = False)[0]
    val_representation = unsupervised_model(convert_sparse_matrix_to_sparse_tensor(features[val_set_list[i], :]), training = False)[0]
    test_representation = unsupervised_model(convert_sparse_matrix_to_sparse_tensor(features[test_set_list[i], :]), training = False)[0]
    
    train_representation_list.append(train_representation.numpy())
    val_representation_list.append(val_representation.numpy())
    test_representation_list.append(test_representation.numpy())
    
train_data = np.concatenate(train_representation_list, axis = 0)
val_data = np.concatenate(val_representation_list, axis = 0)
test_data = np.concatenate(test_representation_list, axis = 0)


while(True):
    is_go_on = input("is continue to train? (yes(y) or no(n)):")
    
    if('y' == is_go_on or 'yes' == is_go_on):
        
        break;
        
    if('n' == is_go_on or 'no' == is_go_on):
        
        sys.exit(0)



for _ in range(args.num_run):
    
    downstream_model = tf.keras.Sequential(
        [
             tf.keras.layers.Dropout(args.downstream_dropout),
             tf.keras.layers.Dense(num_classes,
                                   use_bias = False,
                                   kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 1),
                                   kernel_regularizer = tf.keras.regularizers.l2(args.downstream_L2)),
        ]
    )
    
    downstream_learning(train_data = tf.convert_to_tensor(train_data, dtype = tf.float32),
                        val_data = tf.convert_to_tensor(val_data, dtype = tf.float32),
                        test_data = tf.convert_to_tensor(test_data, dtype = tf.float32),
                        train_y_true = train_y_true,
                        val_y_true = val_y_true,
                        test_y_true = test_y_true,
                        model = downstream_model,
                        batch_size = args.downstream_batch_size,
                        num_epochs = args.downstream_num_epochs,
                        learning_rate = args.downstream_learning_rate,
                        early_stopping = args.early_stopping)
