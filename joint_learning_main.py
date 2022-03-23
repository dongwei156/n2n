# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:21:01 2021

@author: dongwei
"""
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import argparse
import os
import numpy as np
import heapq
import math
from sklearn.metrics import f1_score
import pickle
import scipy.sparse as sp

import tools as tl

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class JointLearningModel(tf.keras.Model):
    
    def __init__(self,
                 layer_units,
                 num_classes,
                 dropout_func = tf.keras.layers.Dropout,
                 dropout = 0.6,
                 L2 = 0.01,
                 activation = lambda x:x,
                 residual = False):
        
        super(JointLearningModel, self).__init__()
        
        self.num_classes = num_classes
        self.residual = residual
        
        self.dense_layers = []
        self.residual_layers = []
        self.dropout_layer = dropout_func(dropout)
        self.activation = activation
        
        dense_layer = tf.keras.layers.Dense(layer_units[0], 
                                            use_bias = False,
                                            kernel_regularizer = tf.keras.regularizers.l2(L2))
        self.dense_layers.append(dense_layer)
        
        for l in range(1, len(layer_units)):
            
            dense_layer = tf.keras.layers.Dense(layer_units[l], 
                                                kernel_regularizer = tf.keras.regularizers.l2(L2))
            self.dense_layers.append(dense_layer)
            
            if(True == residual):
                
                residual_layer = tf.keras.layers.Dense(layer_units[l], 
                                                       kernel_regularizer = tf.keras.regularizers.l2(L2))
                self.residual_layers.append(residual_layer)
                
        dense_layer = tf.keras.layers.Dense(num_classes, 
                                            kernel_regularizer = tf.keras.regularizers.l2(L2))
        self.dense_layers.append(dense_layer)
        
        
    def call(self, x, training = True):
        
        x = self.dropout_layer(x)
        
        x = self.dense_layers[0](x)
        
        x = self.activation(x)
        
        for l in range(1, len(self.dense_layers) - 1):
            
            h = x
            
            h = self.dropout_layer(h)
            
            h = self.dense_layers[l](h)
            
            if(True == self.residual):
                
                h = h + self.residual_layers[l-1](x)
                
            x = self.activation(h)
            
        deep_copy = tf.identity(x)
            
        x = self.dropout_layer(x)
        
        x = self.dense_layers[-1](x)
            
        return x, deep_copy
    
def downstream_learning(features,
                        adj_tensor,
                        true_sample_index,
                        labels,
                        train_set,
                        val_set,
                        test_set,
                        model,
                        contrast_loss,
                        is_sampling_one = True,
                        size_splits = None,
                        temperature = 5,
                        alpha = 0.6,
                        num_epochs = 1000,
                        learning_rate = 0.001,
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
    
    for epoch in range(num_epochs):
        
        with tf.GradientTape() as tape:
            
            y_pred_training, contrast_training = model(features, training = True)
            
            if(True == is_sampling_one):
                
                true_pred = tf.gather(contrast_training, true_sample_index)
            
            else:
                
                true_pred = tf.sparse.sparse_dense_matmul(adj_tensor, contrast_training)
            
            loss_value = (1 - alpha) * loss_object(y_true = train_y_true, y_pred = tf.gather(y_pred_training, train_set)) \
                + alpha * contrast_loss(y_pred = contrast_training, 
                                        true_pred = true_pred,
                                        size_splits = size_splits,
                                        temperature = temperature)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            logits, _ = model(features, training = False)
            
            prediction = tf.argmax(logits, axis = 1, output_type=tf.int32)
            
            train_f1 = f1_score(train_y_true, tf.gather(prediction, train_set).numpy(), average='micro')
            val_f1 = f1_score(val_y_true, tf.gather(prediction, val_set).numpy(), average='micro')
            test_f1 = f1_score(test_y_true, tf.gather(prediction, test_set).numpy(), average='micro')
    
            loss_value = loss_value.numpy()
            
            if(True == np.isnan(loss_value)):
                
                break;
            
            train_loss_results.append(loss_value)
            train_accuracy_results.append(train_f1)
            val_accuracy_results.append(val_f1)
            test_accuracy_results.append(test_f1)
            
            print('Epoch {:03d}: Loss: {:.5f}, Train Micro-F1: {:.2%}, Val Micro-F1: {:.2%}, Test Micro-F1: {:.2%}'.format(
                epoch,
                loss_value,
                train_f1,
                val_f1,
                test_f1
                ))
            
            if(None != early_stopping):
            
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

parser = argparse.ArgumentParser()

parser.add_argument(
        '-d', 
        '--dataset',
        type = str,
        default = 'Cora')

parser.add_argument(
        '-m', 
        '--is-mean-norm',
        type = bool,
        default = False)

parser.add_argument(
        '-s',
        '--is-sampling-one',
        type = bool,
        default = True)

parser.add_argument(
        '-p',
        '--part-nodes',
        type = int,
        default = None)

parser.add_argument(
        '-r',
        '--num-run',
        type = int,
        default = 1)

parser.add_argument(
        '-e',
        '--num-epochs',
        type = int,
        default = 1000)

parser.add_argument(
        '-u',
        '--hidden-units',
        type = int,
        default = 512)

parser.add_argument(
        '-o',
        '--dropout',
        type = float,
        default = 0.6)

parser.add_argument(
        '-L',
        '--L2',
        type = float,
        default = 0.01)

parser.add_argument(
        '-t',
        '--temperature',
        type = float,
        default = 5)

parser.add_argument(
        '-a',
        '--alpha',
        type = float,
        default = 0.99)

parser.add_argument(
        '-l',
        '--learning-rate',
        type = float,
        default = 0.001)

parser.add_argument(
        '-y',
        '--early-stopping',
        type = int,
        default = 200)


args = parser.parse_args()


is_mean_norm = args.is_mean_norm
is_sampling_one = args.is_sampling_one
part_nodes = args.part_nodes

num_run = args.num_run

adj_tensor = None
true_sample_index = None

data_dir = './' + args.dataset + '/'
features = np.load(data_dir + 'feats.npy')
adj = sp.load_npz(data_dir + 'adj.npz')

if(True == is_mean_norm):
    
    features = features - np.mean(features, 0)
    
if(True == is_sampling_one):

    true_sample_index = np.loadtxt(data_dir + 'true_sample_index', dtype = int)
    
else:
    
    with open(data_dir + 'adj_tensor.pkl', 'rb') as f:
    
        adj_tensor = pickle.load(f)
    
labels = np.load(data_dir + 'labels.npy')

size_splits = None
if(None != part_nodes):
    
    num_nodes = labels.shape[0]
    size_splits = []
    
    for _ in range(int(np.floor(num_nodes / part_nodes))):
        size_splits.append(part_nodes)
        
    size_splits.append(num_nodes % part_nodes)

num_classes = labels.shape[-1]
labels = np.argmax(labels, 1)

train_set = np.loadtxt(data_dir + 'train_set', dtype = int)
val_set = np.loadtxt(data_dir + 'val_set', dtype = int)
test_set = np.loadtxt(data_dir + 'test_set', dtype = int)

features = tf.convert_to_tensor(features, dtype=tf.float32)
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

for _ in range(num_run):
    
    model = JointLearningModel(layer_units = [args.hidden_units],
                               num_classes = num_classes,
                               dropout_func = tf.keras.layers.Dropout,
                               dropout = args.dropout,
                               L2 = args.L2,
                               activation = tf.nn.relu,
                               residual = True)
    
    downstream_learning(features = features, 
                        adj_tensor = adj_tensor,
                        true_sample_index = true_sample_index,
                        labels = labels,
                        train_set = train_set,
                        val_set = val_set,
                        test_set = test_set,
                        model = model,
                        contrast_loss = tl.loss_dot_product,
                        is_sampling_one = is_sampling_one,
                        size_splits = size_splits,
                        temperature = args.temperature,
                        alpha = args.alpha,
                        num_epochs = args.num_epochs,
                        learning_rate = args.learning_rate,
                        early_stopping = args.early_stopping)
