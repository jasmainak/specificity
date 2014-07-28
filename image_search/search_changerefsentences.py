import sys
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from utils.similarity import find_sentence_similarity # remember to reload ipython if any changes are made to similarity.py

dataset_name = 'pascal' # {'memorability', 'pascal'}

if dataset_name=='memorability':
    mat1 = scipy.io.loadmat('../data/sentence_descriptions.mat')
    mat2 = scipy.io.loadmat('../data/sentence_descriptions_140315.mat')

    sentences = np.concatenate((mat1['sentences'], mat2['sentences']), axis=1)
    ref_idx, query_idx, x, n_comb = 0, 4, range(1,11), 45

elif dataset_name=='pascal':
    mat = scipy.io.loadmat('../data/pascal_1000_img_50_sent.mat')
    sentences = mat['pascal_sentences']
    ref_idx, query_idx, x, n_comb = range(1,8), 49, range(1,51), 1225

n_sentences = sentences.shape[0]

def ref_query(ref, im_idx):    
    s_ref = list()
    for idx, sent_group in enumerate(sentences):
        s_ref.append(find_sentence_similarity(ref,sent_group[query_idx], sentences))
        sys.stdout.write('\rImage %3d:[%3d]' % (im_idx, idx))
        sys.stdout.flush() 
    return s_ref

jobs = 12
print "Number of jobs = %d" % jobs

s = np.zeros((len(ref_idx), n_sentences, n_sentences))
print "\nSearching for images ... "
for i, r_idx in enumerate(ref_idx):
    s_temp = Parallel(n_jobs=jobs, verbose=0)(delayed(ref_query)(sent_group[r_idx], im_idx) for im_idx, sent_group in enumerate(sentences))
    s[i,:,:] = np.array(s_temp)
    print "\n"

    # Save the data
    if dataset_name=='memorability':
        scipy.io.savemat('../data/image_search_10sentences_reference_1stsent.mat',
                        {'sentences': sentences, 's': s, 'curr_idx': i})
    elif dataset_name=='pascal':   
        scipy.io.savemat('../data/image_search_50sentences_query_refs.mat', 
                        {'sentences': sentences, 's': s, 'curr_idx': i})