# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:47:04 2017

@authors: Thomas Schatz & Rachid Riad

Analyzes Spoken Term Discovery (STD) results and prepares input
data to train an ABnet model from it.

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
import pdb
"""
TODO: Refactor parse function for STD results/Golds word labels
"""

def parse_STD_results(STD_file):
    """
    INPUT:
        Text file of Clusters, can come from STD results or ground truth Gold words
        Format:
            Class_index1 description1
            Speaker_id1 t1 t2
            Speaker_id2 t3 t4

            Class_index2 description2
            Speaker_id2 t5 t6
    OUTPUT:
        List of List
    """
    
    with open(STD_file, 'r') as fh:
        lines = fh.readlines()
    clusters = []
    i = 0
    while i < len(lines):
        cluster = []
        tokens = lines[i].strip().split(" ")
        assert len(tokens) == 2, str(tokens)
        i = i+1
        tokens = lines[i].strip().split(" ")
        assert len(tokens) == 3, "Empty class!"
        fid, t0, t1 = tokens
        t0, t1 = float(t0), float(t1)
        cluster.append([fid, t0, t1])
        new_class = False
        while not(new_class):
            i = i+1
            tokens = lines[i].strip().split(" ")
            if len(tokens) == 3:
                fid, t0, t1 = tokens
                t0, t1 = float(t0), float(t1)
                cluster.append([fid, t0, t1])
            else:
                assert tokens == ['']
                new_class = True
                clusters.append(cluster)
                i = i+1        
    return clusters
STD_file = '/fhgfs/bootphon/scratch/rriad/projects/ABnet_corpus/gold_words.txt'
clusters = parse_STD_results(STD_file)

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

    
#t0 = time.time()
description = analyze_clusters(clusters)
#fichier.write( "Analysis of clusters took : {} s \n ".format(time.time() - t0))
def type_sample_p(std_descr,  type_samp = 'log'):
    """
    Sampling proba modes for the types:
        - 1 : equiprobable
        - f2 : proportional to type probabilities
        - f : proportional to square root of type probabilities
        - log : proportional to log of type probabilities
    """ 
    nb_tok = len(std_descr['tokens'])
    speakers = std_descr['speakers']
    tokens_type = std_descr['tokens_type']
    W_types = {}
    nb_types = len(std_descr['types'])
    types = std_descr['types']
    assert type_samp in ['1', 'f', 'f2','log']
    for tok in range(nb_tok):
        try:
            W_types[tokens_type[tok]] += 1.0
        except:
            W_types[tokens_type[tok]] = 1.0
    p_types = {"Stype" : {}, "Dtype":{}}
    for type_idx in range(nb_types):
        if type_samp == '1':
            p_types["Stype"][type_idx] = 1.0
        if type_samp == 'f2':
            p_types["Stype"][type_idx] = W_types[types[type_idx]]   
        if type_samp == 'f':
            p_types["Stype"][type_idx] = np.sqrt(W_types[types[type_idx]])   
        if type_samp == 'log':
            p_types["Stype"][type_idx] = np.log(1.0 + W_types[types[type_idx]])   
        for type_jdx in range(type_idx+1,nb_types):
            if type_samp == '1':
                p_types["Dtype"][(type_idx,type_jdx)] = 1.0 
            if type_samp == 'f2':
                p_types["Dtype"][(type_idx,type_jdx)]  = W_types[types[type_idx]]*W_types[types[type_jdx]] 
            if type_samp == 'f':
                p_types["Dtype"][(type_idx,type_jdx)]  = np.sqrt(W_types[types[type_idx]]*W_types[types[type_jdx]]) 
            if type_samp == 'log':
                p_types["Dtype"][(type_idx,type_jdx)]  = np.max(np.log(1.0 + np.sqrt(W_types[types[type_idx]]*W_types[types[type_jdx]])),0) 
    return p_types        
#t0 = time.time()
#p_types = type_sample_p(description)
#print "Generate p_types took : {} s \n ".format(time.time() - t0)

def sample_p(std_descr, type_samp = 'log', spk_samp = 'log'):
    """
    Sampling proba modes for the types and for the speakers
        - 1 : equiprobable
        - f2 : proportional to type probabilities
        - f : proportional to square root of type probabilities
        - log : proportional to log of type probabilities
    """
    nb_tok = len(std_descr['tokens'])
    tokens_type = std_descr['tokens_type']
    p_spk_types = {'StypeSspk' : {}, 'StypeDspk' : {}, 'DtypeSspk' : {}, 'DtypeDspk' : {}}
    speakers = std_descr['tokens_speaker'] 
    list_speakers = std_descr['speakers_types'].keys()
    W_spk_types = {} 
    for tok in range(nb_tok):
        try:
            W_spk_types[(speakers[tok],tokens_type[tok])] += 1.0
        except:
            W_spk_types[(speakers[tok],tokens_type[tok])] = 1.0
    
    nb_types = len(std_descr['types'])
    for (spk,type_idx) in W_spk_types.keys(): 
        if type_samp == '1':
            p_spk_types["StypeSspk"][(spk,type_idx)] = 1.0
        if type_samp == 'f2':
            p_spk_types["StypeSspk"][(spk,type_idx)] = W_spk_types[(spk,type_idx)]   
        if type_samp == 'f':
            p_spk_types["StypeSspk"][(spk,type_idx)] = np.sqrt(W_spk_types[(spk,type_idx)])   
        if type_samp == 'log':
            p_spk_types["StypeSspk"][(spk,type_idx)]  = np.log(1.0 + W_spk_types[(spk,type_idx)])   
        #for type_jdx in range(type_idx+1,nb_types):
        #    if type_samp == '1':
        #        p_types["Dtype"][(type_idx,type_jdx)] = 1.0 
        #    if type_samp == 'f2':
        #        p_types["Dtype"][(type_idx,type_jdx)]  = W_types[types[type_idx]]*W_types[types[type_jdx]] 
        #    if type_samp == 'f':
        #        p_types["Dtype"][(type_idx,type_jdx)]  = np.sqrt(W_types[types[type_idx]]*W_types[types[type_jdx]]) 
        #    if type_samp == 'log':
        #        p_types["Dtype"][(type_idx,type_jdx)]  = np.max(np.log(1.0 + np.sqrt(W_types[types[type_idx]]*W_types[types[type_jdx]])),0) 
    import pdb 
    pdb.set_trace()

sample_p(description)


def generate_all_pairs(std_descr):
    # could be optimized
    # probably possible to do the sampling without
    # generating all the pairs explicitly, but not obvious
    # if this become necessary, I should look at it formally
    # before programming anything
    """
    TODO: Test if generate_all_pairs can scale to the zerospeech2017
    Answer: Difficult to fit everything in memory
    """
    pairs = {'Stype_Sspk' : [],
             'Stype_Dspk' : [],
             'Dtype_Sspk' : [],
             'Dtype_Dspk' : []}
    nb_tok = len(std_descr['tokens'])
    speakers = std_descr['tokens_speaker']
    types = std_descr['tokens_type']
    for tok1 in range(nb_tok):
        for tok2 in range(tok1+1, nb_tok):
            spk_type = 'S' if speakers[tok1] == speakers[tok2] else 'D'
            type_type = 'S' if types[tok1] == types[tok2] else 'D'
            pair_type = type_type + 'type_' + spk_type + 'spk'
            pairs[pair_type].append((tok1, tok2))
    for e in pairs:
        pairs[e] = np.vstack(pairs[e])
    return pairs

#t0 = time.time()
#pairs = generate_all_pairs(description)
#fichier.write( "Generate all pairs took : {} s \n".format(time.time() - t0))
#process = psutil.Process(os.getpid())
#fichier.write( "Memory occupied with pairs {} kiloBytes \n".format(process.memory_info().rss/1024.0))
#fichier.write( "Generate all pairs took : {} s \n".format(time.time() - t0))

#fichier.close()
#pdb.set_trace()


def token_sampling_p(std_descr, type_samp = 'f', speaker_samp='f', 
                     threshold=False):
    """
    Sampling proba modes:
        - log : proporitonal to log of speaker or type probabilities
        - f : proportional to square roots of speaker
            or type probabilities (in order to obtain sampling
            probas for pairs proportional to geometric mean of 
            the members of the pair probabilities)
        - f2 : proportional to speaker
            or type probabilities
        - 1 : equiprobable
    If threshold is True, the speaker and type probabilities
        are capped at mean+3std
    """
    assert type_samp in ['1', 'f', 'f2','log']
    assert speaker_samp in ['1', 'f', 'f2','log'] 
    W_types = std_descr['types']
    speakers = [e for e in std_descr['speakers']]
    W_speakers = [std_descr['speakers'][e] for e in speakers]
    if threshold:
        spk_thr = np.mean(W_speakers) + 3*np.std(W_speakers)
        W_speakers = np.minimum(W_speakers, spk_thr)
        type_thr = np.mean(W_types) + 3*np.std(W_types)
        W_types = np.minimum(W_types, type_thr)
    if type_samp == 'f':
        W_types = np.sqrt(W_types)
    elif type_samp == '1':
        W_types = list(np.ones(len(W_types)))
    elif type_samp == 'log':
        W_types = np.log(W_types)
        W_types[W_types == - float('Inf')] = 0
    if speaker_samp == 'f':
        W_speakers = np.sqrt(W_speakers)
    elif speaker_samp == '1':
        W_speakers = list(np.ones(len(W_speakers)))
    elif speaker_samp == 'log':
        W_speakers = np.log(W_speakers)
        W_speakers[W_speakers == - float('Inf')] = 0
    N = len(std_descr['tokens'])  # nb tokens
    p = np.zeros((N,), dtype=np.float)
    for i in range(N):
        tok_type = std_descr['tokens_type'][i] 
        tok_spk = speakers.index(std_descr['tokens_speaker'][i])      
        p[i] = W_types[tok_type]*W_speakers[tok_spk]
    p = p/np.float(np.sum(p))
    return p


def pair_sampling_p(pairs, p_tok):
    # get sampling probabilities for the pairs in the 
    # four different kinds of pairs
    p = {}
    for e in pairs:
        p[e] = p_tok[pairs[e][:,0]]*p_tok[pairs[e][:,1]]
        p[e] = p[e]/np.float(np.sum(p[e]))
    return p


def sample_pairs(descr, seed=0, prefix='',
                 spk_sampling_mode='f',
                 type_sampling_mode='f'):
    # we do not balance sampling between same speaker and different speaker
    # pairs
    # not clear why we want to sample the number of pairs below, given
    # that we reweight sampling probabilities (with a uniform sampling
    # probability on pairs, we would sample on average each pair from the 
    # smallest samples exactly once with these numbers)
    # The right way would be to ask given the resampling scheme instantiated
    # how many sample do we need from the smallest sample to be probably
    # much closer to whatever the scheme conditioned on the data converges to
    # than it itself is close to what it converges to as more data becomes
    # available
    pairs = generate_all_pairs(descr)
    p_tok = token_sampling_p(descr,
                             speaker_samp=spk_sampling_mode,
                             type_samp=type_sampling_mode)
    p = pair_sampling_p(pairs, p_tok)
    Nsamespk = min(len(pairs['Dtype_Sspk']), len(pairs['Stype_Sspk']))
    Ndiffspk = min(len(pairs['Dtype_Dspk']),len(pairs['Stype_Dspk']))
    print(prefix + \
        'Same type: {0} same speaker pairs sampled'.format(Nsamespk))
    print(prefix + \
        'Same type: {0} diff speaker pairs sampled'.format(Ndiffspk))
    print(prefix + \
        'Diff type: {0} same speaker pairs sampled'.format(Nsamespk))
    print(prefix + \
        'Diff type: {0} diff speaker pairs sampled'.format(Ndiffspk))
    n = {'Dtype_Sspk' : Nsamespk, 'Stype_Sspk' : Nsamespk,
         'Dtype_Dspk': Ndiffspk, 'Stype_Dspk': Ndiffspk}
    np.random.seed(seed)
    sampled_pairs = {}
    for pair_type in pairs:
        sample = np.random.multinomial(n[pair_type], p[pair_type])
        pairs_index = np.where(sample)[0]
        samp_pairs = [sample[ind]*[pairs[pair_type][ind]] \
                      for ind in pairs_index]
        samp_pairs = np.vstack([f for e in samp_pairs for f in e])
        np.random.shuffle(samp_pairs)
        sampled_pairs[pair_type] = samp_pairs
    return sampled_pairs



def analysis_sample_pair_performance(clusters, max_clusters=2000):
        """ 
        Plot performance 
        """
        num_clusters = range(100,max_clusters,300) 
        timings, memories = [], []
        for num_cluster in num_clusters:
            t1 = time.time()
            description =  analyze_clusters(clusters[np.random.choice(range(len(clusters)), num_cluster, replace=False)])
            pairs = sample_pairs(description)
            process = psutil.Process(os.getpid())
            memories.append(process.memory_info().rss/1024.0)
            timings.append(time.time() - t1)
        pdf_pages = PdfPages('usage_results.pdf')
        fig = plt.figure()
        plt.plot(num_clusters,timings,'r',label='Timing')
        plt.xlabel('Number of clusters/type of words')
        plt.ylabel('Timing in s')
        plt.title('Timing to sample pairs from clusters')
        plt.legend()
        pdf_pages.savefig(fig)
        fig = plt.figure()
        plt.plot(num_clusters,memories,'b',label='Memory')
        plt.ylabel('Memory occupied in kiloBytes')
        plt.xlabel('Number of clusters/type of words')
        plt.legend('Memory used to generate sample')
        pdf_pages.savefig(fig)
        pdf_pages.close()

        
#analysis_sample_pair_performance(clusters,3000)
"""
Not used as it would risk overfitting by using some
tokens and speakers in both sets
def split_pairs(pairs, p, ratio=.5, seed=0):
    np.random.seed(seed)
    train, p_train, test, p_test = {}, {}, {}, {}
    for e in pairs:
        perm = np.random.permutation(len(p[e]))
        perm_pairs = pairs[e][perm,:]
        perm_p = p[e][perm]
        ind = np.searchsorted(np.cumsum(perm_p), ratio)
        p_tr, p_te = perm_p[:ind+1], perm_p[ind+1:]
        p_train[e] = p_tr/np.float(np.sum(p_tr))
        p_test[e] = p_te/np.float(np.sum(p_te))
        train[e], test[e] = perm_pairs[:ind+1], perm_pairs[ind+1:]
    return train, p_train, test, p_test
"""


def plot_stats(stats, t, pdf_pages):
    fig = plt.figure()
    plt.text(0.1, 0.8, t)
    plt.text(0.1, 0.6, 'Nb Tokens : {0}'.format(len(stats['tokens'])))
    plt.text(0.1, 0.4, 'Nb Types : {0}'.format(len(stats['types'])))
    plt.text(0.1, 0.2, 'Nb Speakers : {0}'.format(len(stats['speakers'])))
    plt.axis([0, 1, 0, 1])
    plt.axis('off')
    pdf_pages.savefig(fig)
    #
    fig = plt.figure()
    ticks = np.arange(1,max(stats['types_speakers'])+2)
    bins = ticks-.5
    h = plt.hist(stats['types_speakers'], log=True, 
                 bins=bins)
    plt.title('Nb Speakers per Type distribution (' + t + ')')
    plt.ylim([.9, 10*max(h[0])])
    plt.xlabel('Nb Speakers')
    plt.ylabel('Nb Types')
    pdf_pages.savefig(fig)
    #
    fig = plt.figure()
    h = plt.hist(stats['types'], bins=100, log=True)
    plt.title('Nb Tokens per Type distribution (' + t + ')')
    plt.ylim([.9, 10*max(h[0])])
    plt.xlabel('Nb Tokens')
    plt.ylabel('Nb Types')
    pdf_pages.savefig(fig)
    #
    data = [stats['speakers_types'][e] for e in stats['speakers_types']]
    fig = plt.figure()
    h = plt.hist(data, bins=30)
    plt.title('Nb Types per Speaker distribution (' + t + ')')
    plt.ylim([.9, 1+max(h[0])])
    plt.xlabel('Nb Types')
    plt.ylabel('Nb Speakers')
    pdf_pages.savefig(fig)
    #
    data = [stats['speakers'][e] for e in stats['speakers']]
    fig = plt.figure()
    h = plt.hist(data, bins=30)
    plt.title('Nb Tokens per Speaker distribution (' + t + ')')
    plt.ylim([.9, 1+max(h[0])])
    plt.xlabel('Nb Tokens')
    plt.ylabel('Nb Speakers')
    pdf_pages.savefig(fig)
    #TODO pair stats
    """
    For diff types, diff speakers pairs :
        could plot the distribution of pair of diff types
        as a function of the number of pair of diff speakers available to
        represent it 
        Same with distribution of pair of speakers
        as a function of pairs of types
    Then for other types of pairs :
        do the same 2 graphs just replacing pairs of types or pairs of 
        speakers with types or speakers where appropriate
    """


def print_token(tok):
    return "{0} {1:.2f} {2:.2f}".format(tok[0], tok[1], tok[2])


def export_pairs(out_file, descr, pairs, seed=0):
    # all the different types of pairs are randomly mixed in the output file
    # with an added 'same' or 'different' label added for types
    # same and different speakers pairs are not distinguishable in the output
    np.random.seed(seed)
    same_pairs = ['Stype_Sspk', 'Stype_Dspk']
    diff_pairs = ['Dtype_Sspk', 'Dtype_Dspk']
    same_pairs = np.vstack([pairs[e] for e in same_pairs])
    n_same = same_pairs.shape[0]
    diff_pairs = np.vstack([pairs[e] for e in diff_pairs])
    n_diff = diff_pairs.shape[0]
    pairs = np.vstack([same_pairs, diff_pairs])
    pair_types = np.concatenate([np.ones(n_same), np.zeros(n_diff)])
    n = pairs.shape[0]
    ind = np.arange(n)
    np.random.shuffle(ind)
    pairs = pairs[ind,:]
    pair_types = pair_types[ind]
    lines = []
    for i in range(n):
        pair_type = 'same' if pair_types[i]==1 else 'diff'
        pair = pairs[i,:]
        tok1 = print_token(descr['tokens'][pair[0]])
        tok2 = print_token(descr['tokens'][pair[1]])
        lines.append(tok1 + " " + tok2 + " " +  pair_type + "\n")
    with open(out_file, 'w') as fh:
        fh.writelines(lines)


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


def std2abnet(std_file, spkid_file, train_spk_file, test_spk_file,
              out_dir, stats=False, seed=0,
              type_sampling_mode='f', spk_sampling_mode='f'):
    """
    Main function : takes Term Discovery results and sample pairs
        for training and testing an ABnet from it.
        Input : 
            std_file : Term Discovery results (.class file)
            spkid_file : mapping between wav_id and spk_id
            train_spk_file : list of speakers constituting the train set
            test_spk_file : list of speakers constituting the test set
            out_dir : target directory for output files 
            stats : if True the function outputs 'figures.pdf' containing 
                    some stats on the Term Discovery results;
                    otherwise the function outputs files train.pairs and
                    test.pairs containing the sampled pairs
            seed : random seed
            type_sampling_mode : sampling mode for token types
                                ('1', 'f','f2' or 'log')
            spk_sampling_mode : sampling mode for token speakers
                                ('1', 'f','f2' or 'log')
        Output : 
            train.pairs and test.pairs files in out_dir (or figures.pdf if 
            stat=True)
    """
    get_spkid_from_fid = read_spkid_file(spkid_file)
    # parsing STD results
    clusters = parse_STD_results(std_file)
    std_descr = analyze_clusters(clusters, get_spkid_from_fid)
    # parsing train/test split
    train, test = read_spk_list(train_spk_file), read_spk_list(test_spk_file)
    # check that no speaker is present twice
    assert len(np.array(train+test)) == len(np.unique(np.array(train+test)))
    # check that all speakers match speakers in the STD results
    for spk in train + test:
        assert spk in std_descr['speakers'].keys(), spk
    # train, test split
    # TODO read clusters from the two gold files directly
    train_clusters, test_clusters = split_clusters(clusters, train, test,
                                                   get_spkid_from_fid)
    train_descr = analyze_clusters(train_clusters, get_spkid_from_fid)
    test_descr = analyze_clusters(test_clusters, get_spkid_from_fid)
    # train and test stats
    if stats:
        try:
            pdf_file = os.path.join(out_dir, 'figures.pdf')
            pdf_pages = PdfPages(pdf_file)
            plot_stats(std_descr, 'Whole data', pdf_pages)
            plot_stats(train_descr, 'Train set', pdf_pages)
            plot_stats(test_descr, 'Test set', pdf_pages)
        finally:
            pdf_pages.close()
    else:
        # generating train and test pairs
        train_pairs = sample_pairs(train_descr, seed=seed, prefix='TRAIN ',
                                   spk_sampling_mode=spk_sampling_mode,
                                   type_sampling_mode=type_sampling_mode)
        seed = seed+1
        test_pairs = sample_pairs(test_descr, seed=seed, prefix='TEST ',
                                  spk_sampling_mode=spk_sampling_mode,
                                  type_sampling_mode=type_sampling_mode)
        # write pairs to disk
        seed = seed+1
        export_pairs(os.path.join(out_dir,type_sampling_mode+ 'train.pairs'),
                     train_descr, train_pairs,
                     seed=seed)
        seed = seed+1
        export_pairs(os.path.join(out_dir, type_sampling_mode+ 'test.pairs'),
                     test_descr, test_pairs,
                     seed=seed)

########
# MAIN #
########

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('std_file', help = "Term Discovery results")
    parser.add_argument('spkid_file', help = "wav_id to spk_id matching")
    parser.add_argument('train_spk_file', help = "train speakers")      
    parser.add_argument('test_spk_file', help = "test speakers")
    parser.add_argument('out_dir', help = "output directory")
    parser.add_argument('--stats', action='store_true',
                        help="If specified, only plot some stats " + \
                             "in figures.pdf")
    parser.add_argument('seed', type=int, help = "random seed")
    parser.add_argument('--type_sampling_mode', default='f',
                        help = "1, f, f2 or log, default is f")
    parser.add_argument('--spk_sampling_mode', default='f',
                        help = "1, f,f2 or log, default is f")
    args = parser.parse_args()
    assert args.type_sampling_mode in ['1', 'f', 'f2','log' ]
    assert args.spk_sampling_mode in ['1', 'f', 'f2','log']
    print("Start sampling")
    t1 = time.time()
    std2abnet(args.std_file, args.spkid_file, args.train_spk_file,
              args.test_spk_file, args.out_dir, args.stats, args.seed,
              args.type_sampling_mode, args.spk_sampling_mode)
    print("Sample and output the pairs took {} s".format(time.time()-t1))

