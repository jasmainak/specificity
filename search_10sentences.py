import scipy.io
import numpy as np

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from similarity import find_sentence_similarity


mat1 = scipy.io.loadmat('../data/sentence_descriptions.mat')
mat2 = scipy.io.loadmat('../data/sentence_descriptions_140315.mat')

sentences = np.concatenate((mat1['sentences'], mat2['sentences']), axis=1)

n_sentences = sentences.shape[0]

def image_similarity(sent_group, im_idx):    
    return [find_sentence_similarity(sent1, sent2) for (sent1, sent2) in combinations(sent_group,2)]

def ref_query(ref):    
    s_ref = [find_sentence_similarity(ref,sent_group[4]) for sent_group in sentences]    
    return s_ref

jobs = 22
print "Number of jobs = %d" % jobs

scores_w = Parallel(n_jobs=jobs, verbose=100)(delayed(image_similarity)(sent_group, im_idx) for im_idx, sent_group in enumerate(sentences))
scores_w = np.array(scores_w)

s = Parallel(n_jobs=jobs, verbose=100)(delayed(ref_query)(sent_group[0]) for sent_group in sentences)
s = np.array(s)

sent_pairs = list()
for sent_group in sentences:         
    sent_pairs.append([(sent1[0], sent2[0]) for (sent1, sent2) in combinations(sent_group,2)])       
    
scipy.io.savemat('../data/image_search_10sentences_reference_1stsent.mat',
                 {'sentences': sentences, 'sent_pairs': np.array(sent_pairs, dtype=object), 
                  'scores_w': scores_w, 's': s})
