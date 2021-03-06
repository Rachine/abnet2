# -*- coding: utf-8 -*-
"""

@author:  Rachid Riad

Analyzes KNN spoken term discovery and prepares input
data to train an ABnet model from it.

TODO: Modify description of the sampling
Extension of Thomas' work on sampling, here the sampling 
method is in two phases based on conditional prob:
        1) Sample among types of tokens based on frequencies 
        2) Sample among uniformly among speakers to find tokens
        
        Ex: For same type i/ same speaker j pair: p_(i,j) = p_(j|i) * p_i 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import os
import psutil
import time
import random

import pdb

def parse_KNN_results(KNN_file):
    """
    INPUT:
        Text file of pairs discovered from KNN discovery system
        Format:
            Speaker_id1 Speaker_id2
            t1 t2 t3 t4 dtw_similarity density1 density2
            t5 t6 t7 t8 dtw_similarity density1 density2
    OUTPUT:
        Dictionnary of pairs
    """
    
    with open(KNN_file, 'r') as fh:
        lines = fh.readlines()
    pairs = []
    i = 0
    num_lines = len(lines)
    while i < num_lines:
        cluster = []
        tokens = lines[i].strip().split(" ")
        assert len(tokens) == 2, str(tokens)
        spk1, spk2 = tokens
        i = i+1
        try:
            tokens = lines[i].strip().split(" ")
        except:
            return pairs
        if len(tokens) == 2:
            continue
        t0, t1, t2, t3, dtw, density1, density2 = [float(el) for el in tokens]
        pairs.append([spk1, spk2, t0, t1, t2, t3, dtw, density1, density2])
        new_speakers = False
        while not(new_speakers) and i < num_lines:
            i = i+1
            tokens = lines[i].strip().split(" ")
            if len(tokens) == 7:
                t0, t1, t2, t3, dtw, density1, density2 = [float(el) for el in tokens]
                pairs.append([spk1, spk2, t0, t1, t2, t3, dtw, density1, density2])
            else:
                assert len(tokens) == 2, str(tokens)
                new_speakers = True
                #i = i+1        
    return pairs


KNN_file = '/home/rriad/abnet2/abnet2/pairs_to_debug.txt'
pairs = parse_KNN_results(KNN_file)

pdb.set_trace()

def split_clusters(clusters, train_spk, test_spk, get_spkid_from_fid):    
    # train/test split based on talkers
    train_clusters, test_clusters = [], []
    for cluster in clusters:
        train_cluster = [tok for tok in cluster \
                             if get_spkid_from_fid(tok[0]) in train_spk]
        test_cluster = [tok for tok in cluster \
                             if get_spkid_from_fid(tok[0]) in test_spk]
        if train_cluster:
            train_clusters.append(train_cluster)
        if test_cluster:
            test_clusters.append(test_cluster)
    return train_clusters, test_clusters


def analyze_clusters(clusters, get_spkid_from_fid=None):
    """
    OUTPUT :
        tokens : flat list of tokens
        tokens_type: list of the associated type for each token         
        tokens_speaker : list of the associated speaker for each token
        types: list of the number of tokens for each type
        speakers: dict of the number of tokens for each speaker
        speakers_types : dict of the number of types per speaker
        types_speakers : list of the number of speakers per type
    """
    if get_spkid_from_fid is None:
        get_spkid_from_fid = lambda x: x
    tokens = [f for c in clusters for f in c]
    # check that no token is present twice in the list
    nb_unique_tokens = \
        len(np.unique([a+"--"+str(b)+"--"+str(c) for a, b, c in tokens]))
    assert len(tokens) == nb_unique_tokens
    tokens_type = [i for i, c in enumerate(clusters) for f in c]
    tokens_speaker = [get_spkid_from_fid(f[0]) for f in tokens]
    types = [len(c) for c in clusters]
    speakers = {}
    for spk in np.unique(tokens_speaker):
        speakers[spk] = len(np.where(np.array(tokens_speaker) == spk)[0])
    speakers_types = {spk : 0 for spk in speakers}
    types_speakers = []
    for c in clusters:
        cluster_speakers = np.unique([get_spkid_from_fid(f[0]) for f in c])
        for spk in cluster_speakers:
            speakers_types[spk] = speakers_types[spk]+1
        types_speakers.append(len(cluster_speakers))   
    std_descr = {'tokens' : tokens,
                 'tokens_type' : tokens_type,
                 'tokens_speaker' : tokens_speaker,
                 'types' : types,
                 'speakers' : speakers,
                 'speakers_types' : speakers_types,
                 'types_speakers' : types_speakers}
    return std_descr

    
#description = analyze_clusters(clusters)
def type_sample_p(std_descr,  type_samp = 'log'):
    """
    Sampling proba modes for the types:
        - 1 : equiprobable
        - f2 : proportional to type probabilities
        - f : proportional to square root of type probabilities
        - fcube : proportional to cube root of type probabilities
        - log : proportional to log of type probabilities
    TODO: Enable to inject directly custom function?
    """ 
    nb_tok = len(std_descr['tokens'])
    speakers = std_descr['speakers']
    tokens_type = std_descr['tokens_type']
    W_types = {}
    nb_types = len(std_descr['types'])
    types = std_descr['types']
    assert type_samp in ['1', 'f', 'f2','log', 'fcube']
    if type_samp == '1':        
        type_samp_func = lambda x : 1
    if type_samp == 'f2':
        type_samp_func = lambda x : x
    if type_samp == 'f':
        type_samp_func = lambda x : np.sqrt(x)
    if type_samp == 'fcube':
        type_samp_func = lambda x : np.cbrt(x)
    if type_samp == 'log':
        type_samp_func = lambda x : np.log(1+x)

    for tok in range(nb_tok):
        try:
            W_types[tokens_type[tok]] += 1.0
        except:
            W_types[tokens_type[tok]] = 1.0
    p_types = {"Stype" : {}, "Dtype":{}}
    
    for type_idx in range(nb_types):
        p_types["Stype"][type_idx] = type_samp_func(W_types[types[type_idx]])
        for type_jdx in range(type_idx+1,nb_types):
            p_types["Dtype"][(type_idx,type_jdx)] = type_samp_func(W_types[types[type_idx]])*type_samp_func(W_types[types[type_jdx]])
    
    return p_types        
#t0 = time.time()
#p_types = type_sample_p(description)
#print "Generate p_types took : {} s \n ".format(time.time() - t0)
#import pdb
#pdb.set_trace()
def sample_spk_p(std_descr, spk_samp = 'f2'):
    """
    Sampling proba modes for the speakers conditionned by the drawn type(s)
        - 1 : equiprobable
        - f2 : proportional to type probabilities
        - f : proportional to square root of type probabilities
        - fcube : proportional to cube root of type probabilites
        - log : proportional to log of type probabilities
    """
    nb_tok = len(std_descr['tokens'])
    tokens_type = std_descr['tokens_type']
    p_spk_types = {'Stype_Sspk' : {}, 'Stype_Dspk' : {}, 'Dtype_Sspk' : {}, 'Dtype_Dspk' : {}}
    speakers = std_descr['tokens_speaker'] 
    list_speakers = std_descr['speakers_types'].keys()
    W_spk_types = {} 
    for tok in range(nb_tok):
        try:
            W_spk_types[(speakers[tok],tokens_type[tok])] += 1.0
        except:
            W_spk_types[(speakers[tok],tokens_type[tok])] = 1.0
    
    if spk_samp == '1':
        spk_samp_func = lambda x : 1
    if spk_samp == 'f2':
        spk_samp_func = lambda x : x
    if spk_samp == 'f':
        spk_samp_func = lambda x : np.sqrt(x)
    if spk_samp == 'fcube':
        spk_samp_func = lambda x : np.cbrt(x)
    if spk_samp == 'log':
        spk_samp_func = lambda x : np.log(1+x)
        
    nb_types = len(std_descr['types'])
    for (spk,type_idx) in W_spk_types.keys(): 
        for (spk2,type_jdx) in W_spk_types.keys():
            if spk == spk2:
                if type_idx == type_jdx: 
                    p_spk_types['Stype_Sspk'][(spk,type_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)]*(W_spk_types[(spk,type_idx)] - 1) )  
                else:
                    min_idx,max_idx = np.min([type_idx,type_jdx]),np.max([type_idx,type_jdx])
                    p_spk_types['Dtype_Sspk'][(spk,min_idx,max_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk,type_jdx)]) 
            else:
                if type_idx == type_jdx:
                    p_spk_types['Stype_Dspk'][(spk,spk2,type_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk2,type_idx)]) 
                else:
                    min_idx,max_idx = np.min([type_idx,type_jdx]),np.max([type_idx,type_jdx])
                    p_spk_types['Dtype_Dspk'][(spk,spk2,min_idx,max_idx)] = spk_samp_func(W_spk_types[(spk,type_idx)])*spk_samp_func(W_spk_types[(spk2,type_jdx)])
    return p_spk_types

def normalize_distribution(p):
    sum_norm = 0.0
    keys = p.keys()
    for key in keys:
        sum_norm += p[key]
    
    for key in keys:
        p[key] = p[key]/sum_norm

    return p

def generate_possibilities(std_descr):
    """
    Generate possibilities between (types,speakers) and tokens/realisations
    """
    pairs = {'Stype_Sspk' : {},
             'Stype_Dspk' : {},
             'Dtype_Sspk' : {},
             'Dtype_Dspk' : {}}
    nb_tok = len(std_descr['tokens'])
    speakers = std_descr['tokens_speaker']
    types = std_descr['tokens_type']
    for tok1 in range(nb_tok):
        for tok2 in range(tok1+1, nb_tok):
            spk_type = 'S' if speakers[tok1] == speakers[tok2] else 'D'
            type_type = 'S' if types[tok1] == types[tok2] else 'D'
            pair_type = type_type + 'type_' + spk_type + 'spk'
            try:
                if pair_type == 'Stype_Sspk':
                    pairs[pair_type][(speakers[tok1],types[tok1])].append((tok1, tok2))
                if pair_type == 'Stype_Dspk':
                    pairs[pair_type][(speakers[tok1],speakers[tok2],types[tok1])].append((tok1, tok2))
                if pair_type == 'Dtype_Sspk':
                    min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                    pairs[pair_type][(speakers[tok1],min_idx,max_idx)].append((tok1, tok2))
                    #pairs[pair_type][(speakers[tok1],types[tok1],types[tok2])].append((tok1, tok2))
                if pair_type == 'Dtype_Dspk':
                    min_idx,max_idx = np.min(types[tok1],types[tok2]), np.max([types[tok1],types[tok2]])
                    pairs[pair_type][(speakers[tok1],speakers[tok2],min_idx,max_idx)].append((tok1, tok2))
                    #pairs[pair_type][(speakers[tok1],speakers[tok2],types[tok1],types[tok2])].append((tok1, tok2))
            except:
                if pair_type == 'Stype_Sspk':
                    pairs[pair_type][(speakers[tok1],types[tok1])] = [(tok1, tok2)]
                if pair_type == 'Stype_Dspk':
                    pairs[pair_type][(speakers[tok1],speakers[tok2],types[tok1])] = [(tok1, tok2)]
                if pair_type == 'Dtype_Sspk':
                    min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                    pairs[pair_type][(speakers[tok1],min_idx,max_idx)] = [(tok1, tok2)]
                if pair_type == 'Dtype_Dspk':
                    min_idx,max_idx = np.min([types[tok1],types[tok2]]), np.max([types[tok1],types[tok2]])
                    pairs[pair_type][(speakers[tok1],speakers[tok2],min_idx,max_idx)]= [(tok1, tok2)]
    return pairs

#pairs = generate_possibilities(description)

def type_speaker_sampling_p(std_descr, type_samp = 'f', speaker_samp='f'):
    """
    Sampling proba modes for p_i1,i2,j1,j2 based on conditionnal Bayes proba:
        - log : proporitonal to log of speaker or type probabilities
        - f : proportional to square roots of speaker
            or type probabilities (in order to obtain sampling
            probas for pairs proportional to geometric mean of 
            the members of the pair probabilities)
        - f2 : proportional to speaker
            or type probabilities
        - fcube: proportionnal to log probabilities
        - 1 : equiprobable
    """
    assert type_samp in ['1', 'f', 'f2','log', 'fcube']
    assert speaker_samp in ['1', 'f', 'f2','log', 'fcube'] 
    W_types = std_descr['types']
    speakers = [e for e in std_descr['speakers']]
    W_speakers = [std_descr['speakers'][e] for e in speakers]
    p_types = type_sample_p(std_descr,  type_samp = type_samp)
    p_spk_types = sample_spk_p(std_descr, spk_samp = speaker_samp)
    
    for config in p_types.keys():
        p_types[config] = normalize_distribution(p_types[config])
    
    for config in p_spk_types.keys():
        p_spk_types[config] = normalize_distribution(p_spk_types[config])
    
    for config in p_spk_types.keys():
        if config == 'Stype_Sspk':
            for el in p_spk_types[config].keys():
                p_spk_types[config][el] = p_types['Stype'][el[1]]*p_spk_types[config][el]
        if config == 'Stype_Dspk':
            for el in p_spk_types[config].keys():
                p_spk_types[config][el] = p_types['Stype'][el[2]]*p_spk_types[config][el]
        if config == 'Dtype_Sspk':
            for el in p_spk_types[config].keys():
                p_spk_types[config][el] = p_types['Dtype'][(el[1],el[2])]*p_spk_types[config][el]
        if config == 'Dtype_Dspk':
            for el in p_spk_types[config].keys():
                p_spk_types[config][el] = p_types['Dtype'][(el[2],el[3])]*p_spk_types[config][el]

    for config in p_spk_types.keys():
        p_spk_types[config] = normalize_distribution(p_spk_types[config])

    return p_spk_types


#proba = type_speaker_sampling_p(description)

def pair_sampling_p(pairs, p_tok):
    # get sampling probabilities for the pairs in the 
    # four different kinds of pairs
    p = {}
    for e in pairs:
        p[e] = p_tok[pairs[e][:,0]]*p_tok[pairs[e][:,1]]
        p[e] = p[e]/np.float(np.sum(p[e]))
    return p


def sample_batch(p_spk_types,
                 pairs,
                 seed=0, prefix='',
                 num_per_config=15):
    #Sample batch of size 4 * num_per_config
        
    np.random.seed(seed)
    sampled_tokens = {'Stype_Sspk' : [],
             'Stype_Dspk' : [],
             'Dtype_Sspk' : [],
             'Dtype_Dspk' : []}
    for config in p_spk_types.keys():
        proba_config = np.array(p_spk_types[config].values())
        sizes = len(p_spk_types[config].keys())
        keys = np.array(p_spk_types[config].keys())
        sample_idx = np.random.choice(sizes,num_per_config,p=proba_config, replace=False)
        sample = keys[sample_idx]
        if config == 'Stype_Sspk':
            for key in sample:
                spk, type_idx = key
                pot_tok = pairs[config][spk,int(type_idx)]
                num_tok = len(pot_tok)
                sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
        if config == 'Stype_Dspk':
            for key in sample:
                spk1,spk2, type_idx = key
                try:
                    pot_tok = pairs[config][spk1,spk2,int(type_idx)]
                except:
                    pot_tok = pairs[config][spk2,spk1,int(type_idx)]
                num_tok = len(pot_tok)
                sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
        if config == 'Dtype_Sspk':
            for key in sample:
                spk, type_idx, type_jdx = key
                pot_tok = pairs[config][spk,int(type_idx),int(type_jdx)]
                num_tok = len(pot_tok)
                sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
        if config == 'Dtype_Dspk':
            for key in sample:
                spk1,spk2 ,type_idx, type_jdx = key
                try:
                    pot_tok = pairs[config][spk1,spk2,int(type_idx),int(type_jdx)]
                except:
                    pot_tok = pairs[config][spk2,spk1,int(type_idx),int(type_jdx)]
                num_tok = len(pot_tok)
                sampled_tokens[config].append(pot_tok[np.random.choice(num_tok)])
    return sampled_tokens    


#sampled = sample_batch(proba,pairs)

def print_token(tok):
    return "{0} {1:.2f} {2:.2f}".format(tok[0], tok[1], tok[2])


def export_pairs(out_dir, descr,type_sampling_mode='f2',spk_sampling_mode='f2', seed=0, size_batch=16):
    # all the different types of pairs are randomly mixed in the output file
    # with an added 'same' or 'different' label added for types
    # same and different speakers pairs are not distinguishable in the output
    # TODO generate pairs for speaker labels
    np.random.seed(seed)
    same_pairs = ['Stype_Sspk', 'Stype_Dspk']
    diff_pairs = ['Dtype_Sspk', 'Dtype_Dspk']
    pairs =  generate_possibilities(descr)
    proba = type_speaker_sampling_p(descr, type_samp = type_sampling_mode, speaker_samp = spk_sampling_mode)
    num = np.min(descr['speakers'].values())
    num_batches = num*(num-1) / 2
    num_batches = num_batches // size_batch
    for idx_batch in range(num_batches):
        lines = []
        sampled_batch = sample_batch(proba,pairs,num_per_config=size_batch/4)
        for config in sampled_batch.keys():
            if config == 'Stype_Sspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Stype_Dspk':
                pair_type = 'same'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Dtype_Sspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
            if config == 'Dtype_Dspk':
                pair_type = 'diff'
                for pair in sampled_batch[config]:
                    tok1 = print_token(descr['tokens'][pair[0]])
                    tok2 = print_token(descr['tokens'][pair[1]])
                    lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
        
        random.shuffle(lines)
        with open(os.path.join(out_dir,'batch_pair_'+str(idx_batch)), 'w') as fh:
            fh.writelines(lines)


#out_dir = '/home/rriad/abnet2/test_generate_pairs'
#export_pairs(out_dir, description,'log','f2', seed=10, size_batch=16)


def read_spkid_file(spkid_file):
    with open(spkid_file, 'r') as fh:
        lines = fh.readlines()
    spk = {}
    for line in lines:
        fid, spkid = line.strip().split(" ")
        assert not(fid in spk)
        spk[fid] = spkid
    return lambda x: spk[x]


def read_spk_list(spk_file):
    with open(spk_file, 'r') as fh:
        lines = fh.readlines()
    return [line.strip() for line in lines]


def std2abnet(std_file, spkid_file, train_spk_file, dev_spk_file,
              out_dir, stats=False, seed=0,
              type_sampling_mode='f', spk_sampling_mode='f',
              size_batch = 16):
    """
    Main function : takes Term Discovery results and sample pairs
        for training and testing an ABnet from it.
        Input : 
            std_file : Term Discovery results (.class file)
            spkid_file : mapping between wav_id and spk_id
            train_spk_file : list of speakers constituting the train set
            dev_spk_file : list of speakers constituting the dev set
            out_dir : target directory for output files 
            stats : if True the function outputs 'figures.pdf' containing 
                    some stats on the Term Discovery results;
                    otherwise the function outputs files train.pairs and
                    dev.pairs containing the sampled pairs
            seed : random seed
            type_sampling_mode : sampling mode for token types
                                ('1', 'f','f2','fcube' or 'log')
            spk_sampling_mode : sampling mode for token speakers
                                ('1', 'f','f2','fcube' or 'log')
            size_batch : number of pairs per batch,
        Output : 
            train.pairs and dev.pairs files in out_dir (or figures.pdf if 
            stat=True)
    """
    get_spkid_from_fid = read_spkid_file(spkid_file)
    # parsing STD results
    clusters = parse_STD_results(std_file)
    std_descr = analyze_clusters(clusters, get_spkid_from_fid)
    # parsing train/dev split
    train, dev = read_spk_list(train_spk_file), read_spk_list(dev_spk_file)
    # check that no speaker is present twice
    assert len(np.array(train+dev)) == len(np.unique(np.array(train+dev)))
    # check that all speakers match speakers in the STD results
    for spk in train + dev:
        assert spk in std_descr['speakers'].keys(), spk
    # train, dev split
    train_clusters, dev_clusters = split_clusters(clusters, train, dev,
                                                   get_spkid_from_fid)
    train_descr = analyze_clusters(train_clusters, get_spkid_from_fid)
    dev_descr = analyze_clusters(dev_clusters, get_spkid_from_fid)
    # train and dev stats
    if stats:
        try:
            #TODO remake a plot_stats function with new functions.
            pdf_file = os.path.join(out_dir, 'figures.pdf')
            pdf_pages = PdfPages(pdf_file)
            plot_stats(std_descr, 'Whole data', pdf_pages)
            plot_stats(train_descr, 'Train set', pdf_pages)
            plot_stats(test_descr, 'Test set', pdf_pages)
        finally:
            pdf_pages.close()
    else:
        # generate and write pairs to disk
        seed = seed+1
        export_pairs(os.path.join(out_dir, 'train.pairs'),
                     train_descr, type_sampling_mode = type_sampling_mode, 
                     spk_sampling_mode = spk_sampling_mode,
                     seed=seed, size_batch = size_batch)
        seed = seed+1
        export_pairs(os.path.join(out_dir,'dev.pairs'),
                     dev_descr, type_sampling_mode = type_sampling_mode, 
                     spk_sampling_mode = spk_sampling_mode,
                     seed = seed,size_batch = size_batch)

########
# MAIN #
########

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('std_file', help = "Term Discovery results")
    parser.add_argument('spkid_file', help = "wav_id to spk_id matching")
    parser.add_argument('train_spk_file', help = "train speakers")      
    parser.add_argument('dev_spk_file', help = "dev speakers")
    parser.add_argument('out_dir', help = "output directory")
    parser.add_argument('--stats', action='store_true',
                        help="If specified, only plot some stats " + \
                             "in figures.pdf")
    parser.add_argument('seed', type=int, help = "random seed")
    parser.add_argument('--type_sampling_mode', default='f',
                        help = "1, f, f2, fcube, or log, default is f")
    parser.add_argument('--spk_sampling_mode', default='f',
                        help = "1, f,f2, fcube, or log, default is f")
    parser.add_argument('--size_batch', type=int, default=16,
                        help = "number of pairs per batch")
    args = parser.parse_args()
    assert args.type_sampling_mode in ['1', 'f', 'f2','fcube','log' ]
    assert args.spk_sampling_mode in ['1', 'f', 'f2','fcube','log']
    print("Start sampling")
    t1 = time.time()
    std2abnet(args.std_file, args.spkid_file, args.train_spk_file,
              args.dev_spk_file, args.out_dir, args.stats, args.seed,
              args.type_sampling_mode, args.spk_sampling_mode,args.size_batch)
    print("Sample and output the pairs took {} s".format(time.time()-t1))

