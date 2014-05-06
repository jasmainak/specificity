import sys
import os.path
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from similarity import find_sentence_similarity # remember to reload ipython if any changes are made to similarity.py

dataset_name = raw_input("Please enter name of data set (pascal/memorability): ")
jobs = int(raw_input("Please enter number of parallel jobs: "))

np.random.seed(42) # set random seed

# Choose data set
if dataset_name == 'pascal':
    input_filename = '../../data/sentences/pascal_1000_img_50_sent.mat'
    output = '../../data/search_parameters/pascal/'
    
    mat1 = scipy.io.loadmat(input_filename)
    sentences = mat1['pascal_sentences']

    n_images, n_sentences = sentences.shape
    m_sentences = 24 # 24 sentences outside image for every sentence within image    
    n_scores = n_sentences*m_sentences # 24*50 = 1200 pairs for estimating mu_d


output_status = output + 'image_search_50sentences_mud.mat'
output_mud = output + 'mu_d/'

# Create directories/files that don't exist
if not os.path.exists(output):
    os.makedirs(output)
    os.makedirs(output_mud)

# Load status file if it exists
if os.path.isfile(output_status):
    mat2 = scipy.io.loadmat(output_status)    
    curr_idx = mat2['curr_idx']    
else:
    curr_idx = -1    
    sent_pairs = []

# List of all sentences
sent_list = list()
for sent_group in sentences:
    sent_list = sent_list + [sent for sent in sent_group]

print "Calculating search parameters ..."

pick_within, pick_others = np.zeros((n_images, n_scores), dtype=int), np.zeros((n_images, n_scores), dtype=int)

for im_idx, sentence_group in enumerate(sentences):        
        
        if im_idx > curr_idx:

            im_sentences = [x for x in sentence_group]
            other_sentences = [x for x in sent_list if x not in im_sentences]
                        
            pick_within[im_idx, :] = [i for i in range(0, n_sentences) for j in range(0, m_sentences)]
            pick_others[im_idx, :] = np.random.randint(0, len(other_sentences), n_scores)

            print "Image index = [%d/%d]" % (im_idx, n_images)
            similarity_b = Parallel(n_jobs=jobs, verbose=10)(delayed(find_sentence_similarity)(im_sentences[i], other_sentences[j], dataset_name=dataset_name, verbose=False) 
                                                            for i,j in zip(pick_within[im_idx, :], pick_others[im_idx, :]))
            
            sent_pairs = [(im_sentences[i], other_sentences[j]) for i, j in zip(pick_within[im_idx, :], pick_others[im_idx, :])]
           
            # Save the data
            print "Saving intermediate data ..."
            
            scipy.io.savemat(output_mud + 'image_search_50sentences_mud_' + str(im_idx) + '.mat', 
                            {'scores_b': similarity_b, 'sent_pairs': np.asarray(sent_pairs, dtype='object')})

            scipy.io.savemat(output_status, {'curr_idx': im_idx})