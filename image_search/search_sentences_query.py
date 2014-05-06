import sys
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from similarity import find_sentence_similarity # remember to reload ipython if any changes are made to similarity.py

dataset_name = 'pascal' # {'memorability', 'pascal'}

if dataset_name=='memorability':
    mat1 = scipy.io.loadmat('../data/sentence_descriptions.mat')
    mat2 = scipy.io.loadmat('../data/sentence_descriptions_140315.mat')

    sentences = np.concatenate((mat1['sentences'], mat2['sentences']), axis=1)
    ref_idx, query_idx, x, n_comb = 0, 4, range(1,11), 45

elif dataset_name=='pascal':
    mat = scipy.io.loadmat('../data/pascal_1000_img_50_sent.mat')
    sentences = mat['pascal_sentences']
    ref_idx, query_idx, x, n_comb = 0, 49, range(1,51), 1225

n_sentences = sentences.shape[0]

def image_similarity(sent_group, im_idx):    
    sim = list()
    for idx, (sent1, sent2) in enumerate(combinations(sent_group,2)):
        sim.append(find_sentence_similarity(sent1, sent2, sentences))        
        sys.stdout.write('\rImage %3d:[%4d]' % (im_idx, idx))
        sys.stdout.flush()
    return sim
    
def ref_query(ref, im_idx):    
    s_ref = list()
    for idx, sent_group in enumerate(sentences):
        s_ref.append(find_sentence_similarity(ref,sent_group[query_idx], sentences))
        sys.stdout.write('\rImage %3d:[%3d]' % (im_idx, idx))
        sys.stdout.flush() 
    return s_ref

jobs = 12
print "Number of jobs = %d" % jobs

#print "Calculating search parameters ..."
#scores_w = Parallel(n_jobs=jobs, verbose=0)(delayed(image_similarity)(sent_group, im_idx) for im_idx, sent_group in enumerate(sentences))
#scores_w = np.array(scores_w)

print "\nSearching for images ..."
s = Parallel(n_jobs=jobs, verbose=0)(delayed(ref_query)(sent_group[ref_idx], im_idx) for im_idx, sent_group in enumerate(sentences))
s = np.array(s)

# List ways of combining the sentences
comb = np.zeros((n_comb, 2))
for idx, (x1,x2) in enumerate(combinations(x,2)):
    comb[idx, 0], comb[idx,1] = x1, x2

# Save the data
if dataset_name=='memorability':
    scipy.io.savemat('../data/image_search_10sentences_reference_1stsent.mat',
                    {'sentences': sentences, 's': s,
                     'comb': comb})
elif dataset_name=='pascal':   
    scipy.io.savemat('../data/image_search_50sentences_query.mat', 
                    {'sentences': sentences, 's': s,
                     'comb': comb})
