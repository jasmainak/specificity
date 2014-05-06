import scipy.io
from scipy.stats import nanmean
import numpy as np

import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

from itertools import product

sentences = scipy.io.loadmat('../data/sentences_all.mat')
sentences = sentences['sentences']

# this data is written by the script calculate importance.m
object_mat = scipy.io.loadmat('../data/object_presence.mat')
objectnames = object_mat['objectnames'][0, :].tolist()
object_pres = object_mat['object_pres']

f = open('../qualitative/importance_output.txt', 'w')

S, D, importance = list(), list(), list()
analyze = CountVectorizer().build_analyzer()

# seed the numpy selector
np.random.seed(42)

def sentence_similarity(idx, ob):

    S_list = list()
    for im_idx, sentence_group in enumerate(sentences[idx, :].squeeze()):
                
        print >> f,''
        for sent in sentence_group:     

            words = analyze(sent[0])

            sim = list()
            for w in words:

                syn1 = wn.synsets(w)
                syn2 = wn.synsets(ob[0])

                if syn1 and syn2:
                    sim.append(max(s1.path_similarity(s2) for (s1, s2) in product(syn1, syn2)))
                else: 
                    sim.append(None) # ignore word if no synset combination was found on wordnet
            
            if max(sim):                    
                S_list.append(max(sim))
                print >> f, (("\t%s: %s (%0.2f : %s)" % 
                             (ob[0], sent[0], S_list[-1], 
                              words[sim.index(max(sim))])).encode('utf-8'))
            else: 
                S_list.append(float('nan')) # ignore sentence if no word was similar enough

    return S_list

for ob_idx, ob in enumerate(objectnames):
    
    print >> f, "\nCalculating importance for object category: %s" % ob[0]
     
    present_idx = np.where(object_pres[ob_idx, :]==1)
    absent_idx = np.random.choice(np.asarray(np.where(object_pres[ob_idx, :]==0)).squeeze(), 
                                             size=len(present_idx[0]), replace=False)

    print >> f,'\nSIMILARITY'  
    S_list = sentence_similarity(idx=present_idx, ob=ob)
    S.append(nanmean(S_list) if S_list else 0)

    print >> f,'\nDISIMILARITY'
    D_list = sentence_similarity(idx=absent_idx, ob=ob)
    D.append(nanmean(D_list) if D_list else 0)

    importance.append(S[-1] - D[-1])
    print >> f, ("\nImportance calculated for object category %s = %0.4f (%0.4f - %0.4f)"
                 % (ob[0], importance[-1], S[-1], D[-1]))

scipy.io.savemat('../data/importance_scores.mat', 
                 {'importance':importance, 'S': S, 'D': D, 'object_pres':object_pres, 
                  'objectnames':object_mat['objectnames']})