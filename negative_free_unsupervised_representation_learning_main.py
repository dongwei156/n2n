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

from sklearn.metrics import f1_score
from sklearn import preprocessing

import tools_v1 as tl

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
class MLPModel(tf.keras.Model):
    
    def __init__(self,
                 layer_units,
                 output_units,
                 dropout_func = tf.keras.layers.Dropout,
                 dropout = 0.6,
                 L2 = 0.01,
                 activation = lambda x:x,
                 residual = False, 
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
        self.residual_layers = []
        self.ZCA_mean_list = []
        self.ZCA_C_list = []
        
        self.dropout_layer = dropout_func(dropout)
        self.activation = activation
        
        self.ZCA_mean_list.append(tf.zeros([1, layer_units[0]]))
        self.ZCA_C_list.append(tf.eye(layer_units[0]))
        
        dense_layer = tf.keras.layers.Dense(layer_units[0], 
                                            use_bias = False,
                                            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                            kernel_regularizer = tf.keras.regularizers.l2(L2))
        self.dense_layers.append(dense_layer)
        
        for l in range(1, len(layer_units)):
            
            self.ZCA_mean_list.append(tf.zeros([1, layer_units[l]]))
            self.ZCA_C_list.append(tf.eye(layer_units[l]))
            
            dense_layer = tf.keras.layers.Dense(layer_units[l], 
                                                use_bias = False,
                                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
                                                kernel_regularizer = tf.keras.regularizers.l2(L2))
            self.dense_layers.append(dense_layer)
            
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
        
        x = self.dropout_layer(x)
        
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
            
            x = self.dropout_layer(h)
            
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
        
        x = self.dropout_layer(x)
        
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
                
                y_pred_list = model(tf.gather(features, anc_ind), training = True)
                true_pred_list = model(tf.gather(features, aug_ind), training = True)
                
                loss_value = 0.0
                
                for y_pred, true_pred in list(zip(y_pred_list, true_pred_list)):
                    
                    y_pred = y_pred / (tf.norm(y_pred, ord = 2, axis = 1, keepdims = True) + 1e-8)
                    true_pred = true_pred / (tf.norm(true_pred, ord = 2, axis = 1, keepdims = True) + 1e-8)
                
                    if('distance' == contrast_loss):
                            
                        loss_value += tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, temperature = temperature)
                        
                    elif('signal distance' == contrast_loss):
                            
                        loss_value += tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature)
                        
                    elif('contrastive' == contrast_loss):
                        
                        loss_value += tl.loss_dot_product_v3(y_pred = y_pred, true_pred = true_pred, temperature = temperature)
                        
                    elif('signal contrastive' == contrast_loss):
                        
                        loss_value += tl.loss_dot_product_v3(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature)
                        
                    elif('distance + auto-correlation' == contrast_loss):
                        
                        loss_value += (alpha * tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, temperature = temperature) + \
                                       beta * tl.auto_correlation(y_pred = y_pred, lam = lam) + tl.auto_correlation(y_pred = true_pred, lam = lam))
                            
                    elif('signal distance + auto-correlation' == contrast_loss):
                        
                        loss_value += (alpha * tl.loss_dot_product_v2(y_pred = y_pred, true_pred = true_pred, axis = 0, temperature = temperature) + \
                                       beta * tl.auto_correlation(y_pred = y_pred, lam = lam) + tl.auto_correlation(y_pred = true_pred, lam = lam))
                            
                    elif('cross-correlation + auto-correlation' == contrast_loss):
                        
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
        

    model.summary()
    
    '''
    fig, axes = plt.subplots(1, sharex = True, figsize = (12, 48))
    fig.suptitle('Metrics')
    
    axes.set_ylabel('Loss', fontsize=14)
    axes.plot(train_loss_results)
    
    plt.show()
    '''
    
    logits = model(features, training = False)
    
    return logits[0]


def downstream_learning(features,
                        labels,
                        train_set,
                        val_set,
                        test_set,
                        model,
                        num_epochs = 5000,
                        learning_rate = 0.0006,
                        early_stopping = 80):
    
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []
    test_accuracy_results = []
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    train_y_true = tf.gather(labels, train_set)
    val_y_true = tf.gather(labels, val_set)
    test_y_true = tf.gather(labels, test_set)
    
    stop_count = early_stopping
    max_test_metric = 0.
    
    for epoch in range(num_epochs):
        
        with tf.GradientTape() as tape:
            
            y_pred_training = model(features, training = True)
            y_pred_training = tf.gather(y_pred_training, train_set)
            
            loss_value = loss_object(y_true = train_y_true, y_pred = y_pred_training)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            logits = model(features, training = False)
            prediction = tf.argmax(logits, axis = 1, output_type=tf.int32)
            
            
            train_metric = f1_score(train_y_true, tf.gather(prediction, train_set).numpy(), average='micro')
            val_metric = f1_score(val_y_true, tf.gather(prediction, val_set).numpy(), average='micro')
            test_metric = f1_score(test_y_true, tf.gather(prediction, test_set).numpy(), average='micro')
            
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
        default = 'Cora')

'''
    distance (ZCA)
    signal distance (ZCA)
    contrastive (Info-NCE)
    signal contrastive (Info-NCE)
    distance + auto-correlation
    signal distance + auto-correlation
    cross-correlation + auto-correlation
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
        default = 0.0)

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
        '-bs',
        '--batch-size',
        type = int,
        default = 2048) #65536

parser.add_argument(
        '-zi',
        '--ZCA-iteration',
        type = int,
        default = 4)

parser.add_argument(
        '-zt',
        '--ZCA-temp',
        type = float,
        default = 0.05)

parser.add_argument(
        '-zl',
        '--ZCA-lam',
        type = float,
        default = 0.0)

parser.add_argument(
        '-bn', 
        '--batch-normalization',
        type = str,
        default = 'Schur_ZCA') # None, BN, ZCA, Schur_ZCA

parser.add_argument(
        '-pe',
        '--pretext-num-epochs',
        type = int,
        default = 70)

parser.add_argument(
        '-de',
        '--downstream-num-epochs',
        type = int,
        default = 3000)

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
        default = 256)

parser.add_argument(
        '-po',
        '--pretext-dropout',
        type = float,
        default = 0.)

parser.add_argument(
        '-do',
        '--downstream-dropout',
        type = float,
        default = 0.5)

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
        default = 0.001)

parser.add_argument(
        '-y',
        '--early-stopping',
        type = int,
        default = 200)



args = parser.parse_args()

data_dir = './' + args.dataset + '/'

features = np.load(data_dir + 'feats.npy')

anchor_indexes = np.load(data_dir + 'anchor_indexes.npy')
augmented_indexes = np.load(data_dir + 'augmented_indexes.npy')
    
labels = np.load(data_dir + 'labels.npy')
    
num_classes = labels.shape[-1]
labels = np.argmax(labels, axis = 1)

train_set = np.loadtxt(data_dir + 'train_set', dtype = int)
val_set = np.loadtxt(data_dir + 'val_set', dtype = int)
test_set = np.loadtxt(data_dir + 'test_set', dtype = int)

anchor_indexes = tf.convert_to_tensor(anchor_indexes, dtype=tf.int32)
augmented_indexes = tf.convert_to_tensor(augmented_indexes, dtype=tf.int32)

features = tf.convert_to_tensor(features, dtype=tf.float32)

labels = tf.convert_to_tensor(labels, dtype = tf.float32)

model = MLPModel(layer_units = args.pretext_hidden_units,
                 output_units = args.pretext_output_units,
                 dropout_func = tf.keras.layers.Dropout,
                 dropout = args.pretext_dropout,
                 L2 = args.pretext_L2,
                 activation = tf.nn.relu,
                 residual = True,
                 batch_normalization = args.batch_normalization,
                 ZCA_iteration = args.ZCA_iteration,
                 ZCA_lam = args.ZCA_lam,
                 ZCA_temp = args.ZCA_temp)

representations = unsupervised_learning(features = features,
                                        anchor_indexes = anchor_indexes,
                                        augmented_indexes = augmented_indexes,
                                        model = model,
                                        contrast_loss = args.contrast_loss,
                                        batch_size = args.batch_size,
                                        lam = args.lam,
                                        temperature = args.temperature,
                                        num_epochs = args.pretext_num_epochs,
                                        learning_rate = args.pretext_learning_rate,
                                        alpha = args.alpha,
                                        beta = args.beta)

while(True):
    is_go_on = input("is continue to train? (yes(y) or no(n)):")
    
    if('y' == is_go_on or 'yes' == is_go_on):
        
        break;
        
    if('n' == is_go_on or 'no' == is_go_on):
        
        sys.exit(0)

max_test_f1_list = []
for _ in range(args.num_run):

    downstream_model = tf.keras.Sequential(
        [
             tf.keras.layers.Dropout(args.downstream_dropout),
             tf.keras.layers.Dense(num_classes,
                                   kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 1),
                                   kernel_regularizer = tf.keras.regularizers.l2(args.downstream_L2)),
        ]
    )
    
    max_test_f1 = downstream_learning(features = representations, 
                                      labels = labels,
                                      train_set = train_set,
                                      val_set = val_set,
                                      test_set = test_set,
                                      model = downstream_model,
                                      num_epochs = args.downstream_num_epochs,
                                      learning_rate = args.downstream_learning_rate,
                                      early_stopping = args.early_stopping)
    
    max_test_f1_list.append(max_test_f1)
   
print("result: %.2f%% ± %.2f%%" % (np.mean(max_test_f1_list) * 100, np.std(max_test_f1_list) * np.power(args.num_run, -1/2) * 100))