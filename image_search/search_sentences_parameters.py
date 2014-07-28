import os
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from itertools import combinations

# remember to reload ipython if any changes are made to similarity.py
from utils.similarity import find_sentence_similarity

dataset_name = raw_input("Please enter name of data set (pascal/memorability/clipart): ")
task = raw_input("Please enter the task (mus/mud/s): ")
jobs = int(raw_input("Please enter number of parallel jobs: "))

if dataset_name == 'memorability':
    mat1 = scipy.io.loadmat('../data/sentence_descriptions.mat')
    mat2 = scipy.io.loadmat('../data/sentence_descriptions_140315.mat')

    sentences = np.concatenate((mat1['sentences'], mat2['sentences']), axis=1)
    ref_idx, query_idx, x, n_comb = 0, 4, range(1, 11), 45

elif dataset_name == 'pascal':
    mat = scipy.io.loadmat('../data/pascal_1000_img_50_sent.mat')
    sentences = mat['pascal_sentences']
    ref_idx, query_idx, x, n_comb = 0, 49, range(1, 51), 1225

elif dataset_name == 'clipart':
    input_filename = '../../data/sentences/clipart_500_img_48_sent.mat'
    output = '../../data/search_parameters/clipart/'

    mat = scipy.io.loadmat(input_filename)
    sentences = mat['clipart_sentences']

    ref_idx, query_idx, x, n_comb = 0, 47, range(1, 49), 1128

n_images, n_sentences = sentences.shape

output_status = output + 'image_search_' + task + '.mat'
output_task = output + task + '/'

# Create directories/files that don't exist
if not os.path.exists(output):
    os.makedirs(output)

if not os.path.exists(output_task):
    os.makedirs(output_task)

# Load status file if it exists
if os.path.isfile(output_status):
    mat2 = scipy.io.loadmat(output_status)
    curr_idx = mat2['curr_idx']
else:
    curr_idx = -1
    sent_pairs = []

print "Calculating search parameters ..."

if task == 'mus':
    
    # List ways of combining the sentences
    comb = np.zeros((n_comb, 2))
    for idx, (x1, x2) in enumerate(combinations(x, 2)):
        comb[idx, 0], comb[idx, 1] = x1, x2

    for im_idx, sent_group in enumerate(sentences):

            if im_idx > curr_idx:

                print "Image index = [%d/%d]" % (im_idx + 1, n_images)

                similarity_w = (Parallel(n_jobs=jobs, verbose=10)(delayed(find_sentence_similarity)
                               (sent1, sent2, dataset_name=dataset_name, verbose=False) for (sent1, sent2) in
                                combinations(sent_group, 2)))

                # Save the data
                print "Saving intermediate data ..."

                sent_pairs = [(sent1, sent2) for sent1, sent2
                              in combinations(sent_group, 2)]

                scipy.io.savemat(output_task + 'image_search_' + task + '_' + str(im_idx) + '.mat',
                                 {'scores_w': similarity_w, 'comb': comb,
                                  'sent_pairs': np.asarray(sent_pairs, dtype='object')})

                scipy.io.savemat(output_status, {'curr_idx': im_idx})

elif task == 's':

    for im_idx, sent_group in enumerate(sentences):

        if im_idx > curr_idx:

            print "Image index = [%d/%d]" % (im_idx + 1, n_images)

            refs = [ref_group[0] for ref_group in sentences]

            s = (Parallel(n_jobs=jobs, verbose=10)(delayed(find_sentence_similarity)
                         (ref, sent_group[query_idx], dataset_name=dataset_name, verbose=False)
                          for ref in refs))

            scipy.io.savemat(output_task + 'image_search_' + task + '_' + str(im_idx) + '.mat',
                             {'s': s, 'query_sentence': sent_group[query_idx],
                              'ref_sentences': np.asarray(refs, dtype='object')})

            scipy.io.savemat(output_status, {'curr_idx': im_idx})
