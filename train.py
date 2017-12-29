# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:07:38 2017

@author: Thomas Schatz

ABnet training and feature extraction with trained ABnet
"""


import numpy as np
import h5features
import abnet2.abnet2 as abnet
from dtw import DTW # install it from https://github.com/syhw/DTW_Cython
from ABXpy.distances.metrics.cosine import cosine_distance
import pickle
import lasagne.layers as layers
import os
import lasagne.nonlinearities as nl
#init_path = '/fhgfs/bootphon/scratch/rriad/projects/ABnet_corpus/log_train_sampling_pairs/'
#features_file = '/fhgfs/bootphon/scratch/rriad/projects/ABnet_corpus/mfcc.h5f'

def Parse_Dataset(path):
    """
    Parse folder for batches
    """
    batches = []
    batches += ([os.path.join(path, add) for add in os.listdir(path) if add.endswith(('.batch'))])
    return batches

class Features_Accessor(object):
    
    def __init__(self, times, features):
        self.times = times
        self.features = features


    def get(self, f, on, off):
        t = np.where(np.logical_and(self.times[f] >= on,
                                    self.times[f] <= off))[0]
        return self.features[f][t, :]


def read_pairs(pair_file):
    with open(pair_file, 'r') as fh:
        lines = fh.readlines()
    pairs = {'same' : [], 'diff' : []}
    for line in lines:
        tokens = line.strip().split(" ")
        assert len(tokens) == 7
        f1, s1, e1, f2, s2, e2, pair_type = tokens
        s1, e1, s2, e2 = float(s1), float(e1), float(s2), float(e2)
        assert pair_type in pairs, \
               'Unsupported pair type {0}'.format(pair_type)
        pairs[pair_type].append((f1, s1, e1, f2, s2, e2))
    return pairs


def read_feats(features_file, align_features_file=None):
    with h5features.Reader(features_file, 'features') as fh:
        features = fh.read()  # load all at once here...
    times = features.dict_labels()
    feats = features.dict_features()
    feat_dim = feats[list(feats.keys())[0]].shape[1]
    features = Features_Accessor(times, feats)
    if align_features_file is None:
        align_features = None
    else:
        with h5features.Reader(features_file, 'features') as fh:
            align_features = fh.read()  # load all at once here...
        times = align_features.dict_labels()
        feats = align_features.dict_features()
        align_features = Features_Accessor(times, feats)
    return features, align_features, feat_dim


def get_dtw_alignment(feat1, feat2):
    distance_array = cosine_distance(feat1, feat2)
    _, _, paths = DTW(feat1, feat2, return_alignment=True,
                             dist_array=distance_array)
    path1, path2 = paths[1:]
    assert len(path1) == len(path2)
    return path1, path2   


def pair_features(pairs, features, align_features=None, seed=0):
    pairs = read_pairs(pairs)
    token_feats, token_align_feats = {}, {}
    # 1. find unique fragments for which we want features
    for f1, s1, e1, f2, s2, e2 in pairs['same']:
        token_feats[f1, s1, e1] = True
        token_feats[f2, s2, e2] = True
    for f1, s1, e1, f2, s2, e2 in pairs['diff']:
        token_feats[f1, s1, e1] = True
        token_feats[f2, s2, e2] = True
    # 2. fill in features
    for f, s, e in token_feats:
        token_feats[f, s, e] = features.get(f, s, e)
        if not(align_features is None):
            token_align_feats[f, s, e] = align_features.get(f, s, e)
    # 3. align features for each pair
    X1, X2, y = [], [], []
    ## get features for each same pair based on DTW alignment paths
    for f1, s1, e1, f2, s2, e2 in pairs['same']:
        if (s1>e1) or (s2>e2):
            continue
        if align_features is None:
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
        else:
            feat1 = token_align_feats[f1, s1, e1]
            feat2 = token_align_feats[f2, s2, e2]
        try:
            path1, path2 = get_dtw_alignment(feat1, feat2)
        except:
            continue
        if not(align_features is None):
            feat1 = token_feats[f1, s1, e1]
            feat2 = token_feats[f2, s2, e2]
        X1.append(feat1[path1,:])
        X2.append(feat2[path2,:])
        # 1 for same pairs, 0 for diff pairs
        # np.int32 to match what is required by theano
        y.append(np.ones(len(path1), dtype=np.int32))
    ## get features for each diff pair based on diagonal alignment
    for f1, s1, e1, f2, s2, e2 in pairs['diff']:
        if (s1>e1) or (s2>e2):
            continue
        feat1 = token_feats[f1, s1, e1]
        feat2 = token_feats[f2, s2, e2]
        n1 = feat1.shape[0]
        n2 = feat2.shape[0]
        X1.append(feat1[:min(n1, n2),:])
        X2.append(feat2[:min(n1, n2),:])
        y.append(np.zeros(min(n1, n2), dtype=np.int32))
    ## concatenate all features
    X1, X2, y = np.vstack(X1), np.vstack(X2), np.concatenate(y)
    # shuffle pairs
    n_pairs = len(y)
    np.random.seed(seed)
    ind = np.random.permutation(n_pairs)
    return X1[ind,:], X2[ind,:], y[ind]


def get_pair_batch(features,file_name):
    """
    Generate batch from file
    """
    #features, align_features, feat_dim = read_feats(features) 
    X_batch1, X_batch2, y_batch = pair_features(file_name,features)
    return X_batch1, X_batch2, y_batch 
    
def get_next_batch(features,folder,num_max_batches=10000):
    """
    Get next batch from folder
    """
    batches = Parse_Dataset(folder)
    num_batches = len(batches)
    if num_max_batches<num_batches:
        selected_batches = np.random.choice(range(num_batches), num_max_batches, replace=False)
    else:
        print "Number of batches not sufficient, iterating over all the batches"
        selected_batches = np.random.permutation(range(num_batches))
    for idx in selected_batches:
        X_batch1, X_batch2, y_batch = get_pair_batch(features,batches[idx])
        yield X_batch1, X_batch2, y_batch

#g = get_next_batch(features_file,init_path)  
#x1, x2, y = g.next()
#pdb.set_trace()        

def read_params(params_neuralnet):
    """
    Read Neural Network parameters file for ABnet2, example file in neural_net_example.params
    """
    params_file = open(params_neuralnet, "r")
    results = {}
    for line in params_file:
        values = line.split(" ")
        if values[0] == "margin":
            results[values[0]] = float(values[1])
        if values[0] == "architecture":
            layers = map(int, values[1].split(","))
            results[values[0]] = layers
        else:
            results[values[0]] = values[1]
    if results == {}:
        return None 
    return results

def train(train_pairs, test_pairs, features_file, out_file,
          align_features=None, seed=0):
    """
    Train an abnet based on lists of train and
    test pairs with associated features in h5features format
    
    After training, the network is automatically set up to best params
    encountered.
    Validation and train loss at each epoch are printed on stdout
    """
    # load and format training and test data
    #features, align_features, feat_dim = read_feats(features, align_features)
    #X_train1, X_train2, y_train = pair_features(train_pairs, features,
    #                                 align_features, seed=seed)
    #X_val1, X_val2, y_val = pair_features(test_pairs, features,
    #                                 align_features, seed=seed)   
    # Workaround to get the dim of inputs
    #mean = np.load(mean_file)
    #std = np.load(std_file)
    features, align_features, feat_dim = read_feats(features_file, align_features)
    #params = read_params(params_neuralnet)
    #assert params != None, print "Parameters for abnet2 is empty"
    # instantiate neural net
    #Original abnet (without batch_norm)
    #nnet = abnet.ABnet([feat_dim, 500, 500, 100], batch_norm=True,loss_type='coscos2')
    #Abnet training with cosmargin loss function and batch normalization
    #nnet = abnet.ABnet([feat_dim, 500, 500, 100], batch_norm=True, margin=0.85)    
    #Abnet training with lstm layers, cosmargin and batch normalization
    #nnet = abnet.ABnet([feat_dim,500,100],lstm_layers=True, batch_norm=True)
    nonlinearities = [nl.sigmoid, nl.sigmoid, nl.rectify ]
    nonlinearities_deep = [nl.sigmoid, nl.sigmoid, nl.sigmoid ,nl.sigmoid ]
    #nnet = abnet.ABnet([feat_dim, 500,500,500], batch_norm=True, margin=0.9)  
    #nnet = abnet.ABnet([feat_dim, 500,500,500, 100 ],[nl.sigmoid] * 3 + [nl.sigmoid], margin=0.1)
    nnet = abnet.ABnet([feat_dim, 500,500, 100 ],[nl.sigmoid] * 2 + [nl.sigmoid], loss_type='coscos2')
    #nnet = abnet.ABnet([feat_dim]+params["architecture"], params["nonlinearities"], margin=params["margin"],
    #                  batch_norm=params["normalization"], loss_type=params["loss"])
    #nnet = abnet.ABnet([feat_dim, 1000,1000, 100], batch_norm=True, margin=0.85)  
    # train
    # patience=-1 means we do the whole 500 epochs even if test set loss
    # increases for many consecutive epochs
    # I put patience=50 in the end, meaning after 50 consecutive epochs
    # without increase in the test set loss, training stops
    # Output of abnet.train is the sequence of NN params at the various epochs
    # we ignore it, instead only retaining the best params found (epoch with
    # minimum validation set loss)
    #train_batches = get_next_batch(features,train_pairs)
    #validation_batches = get_next_batch(features,test_pairs)
    #_, best_epoch = nnet.train_by_batch(train_pairs,test_pairs,features,get_next_batch=get_next_batch, max_epochs=500,patience=50)
    _, best_epoch = nnet.train_by_batch(train_pairs,test_pairs,features,get_next_batch=get_next_batch, max_epochs=0,patience=50)
    
    #_, best_epoch = nnet.train(X_train1, X_train2, y_train,
    #                           X_val1, X_val2, y_val,
    #                           max_epochs=500, patience=50)
    # save trained net
    # there should be a method for this in abnet2
    # to avoid explicit call to lasagne layers
    # also the instantiation parameters of the network should be saved
    # ([feat_dim, 500, 500, 100], loss_type='coscos2')
    nnet_params = layers.get_all_param_values(nnet.network)
    with open(out_file, 'w') as fh:
        pickle.dump((best_epoch, nnet_params), fh)



if __name__=='__main__':    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_pairs', help = "file with list of train pairs")
    parser.add_argument('test_pairs', help = "file with list of test pairs")
    parser.add_argument('features', help = "h5features file with " + \
                                           "input features")
    #parser.add_argument('mean_file', help = "file with the mean of the data")
    #parser.add_argument('std_file', help = "file with the std of data")
    parser.add_argument('out_file', help = "file in which to store the " + \
                                           "trained neural net")
    parser.add_argument('--align_features', default="",
                        help = "h5features file with features used for DTW" + \
                               "alignment of same pairs, if different from" + \
                               "input features")
    parser.add_argument('--seed', type=int, default=0,
                        help = "Random seed for shuffling data")           
    args = parser.parse_args()
    if args.align_features == "":
        align_features = None
    else:
        align_features = args.align_features
    import time
    t1 = time.time()
    print "Start training Neural net"
    train(args.train_pairs, args.test_pairs, args.features, args.out_file,
          align_features, args.seed)
    print "Traing Neural net took {} s ".format(time.time() - t1)


#/pylon2/ci560op/odette/data/mscoco/val2014/wav
#/pylon2/ci560op/rachine/data/mscoco/val2014/mfcc




