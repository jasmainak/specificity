import os
import os.path as op
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from itertools import combinations

# remember to reload ipython if any changes are made to similarity.py
from utils.similarity import find_sentence_similarity


def _load_dataset(dataset_name, method=''):
    if dataset_name == 'memorability':
        input_filename = '../../data/sentences/memorability_888_img_5_sent.mat'
        output_dir = '../../data/search_parameters/memorability/' + method

    elif dataset_name == 'pascal':
        input_filename = '../../data/sentences/pascal_1000_img_50_sent.mat'
        output_dir = '../../data/search_parameters/pascal/' + method

    elif dataset_name == 'clipart':
        input_filename = '../../data/sentences/clipart_500_img_48_sent.mat'
        output_dir = '../../data/search_parameters/clipart/' + method

        mat = scipy.io.loadmat(input_filename)
        sentences = mat['clipart_sentences']

    mat = scipy.io.loadmat(input_filename)
    sentences = mat[dataset_name + '_sentences']
    urls = mat[dataset_name + '_urls']

    return sentences, urls, output_dir


if __name__ == '__main__':

    dataset_name = raw_input("Please enter name of data set "
                             "(pascal/memorability/clipart): ")
    task = raw_input("Please enter the task (mus/s): ")
    jobs = int(raw_input("Please enter number of parallel jobs: "))
    method = 'cosine'

    sentences, urls, output_dir = _load_dataset(dataset_name, method)
    n_images, n_sentences = sentences.shape

    if task == 's':
        msg = 'Please enter ref_idx (0, %d): ' % (n_sentences - 1)
        ref_idx = int(raw_input(msg))
        msg = 'Please enter query_idx (0, %d): ' % (n_sentences - 1)
        query_idx = int(raw_input(msg))

        output_dir = op.join(output_dir, 's', ('ref%d_query%d'
                                               % (ref_idx, query_idx)))
    elif task == 'mus':
        output_dir = op.join(output_dir, 'mus')

    # setup for parallel computation
    parallel = Parallel(n_jobs=jobs, verbose=30)
    my_fun = delayed(find_sentence_similarity)

    # Create output directory if it doesn't exist
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    print('Calculating search parameters ...')

    if task == 'mus':
        for im_idx, (url, sent_group) in enumerate(zip(urls, sentences)):

            # output filename
            img_fname = op.basename(urls[im_idx][0][0])
            fname = op.join(output_dir, ('img_%s.mat' % img_fname))
            print('Image index = [%d/%d]' % (im_idx + 1, n_images))

            if op.isfile(fname):
                continue

            # compute similarity
            similarity_w = parallel(my_fun(sent1, sent2,
                                           dataset_name=dataset_name,
                                           verbose=False, method=method) for
                                    (sent1, sent2) in
                                    combinations(sent_group, 2))

            # Save the data
            sent_pairs = [(sent1, sent2) for sent1, sent2
                          in combinations(sent_group, 2)]

            # List ways of combining the sentences
            comb = np.zeros((len(sent_pairs), 2))
            iter_sent = range(1, n_sentences + 1)
            for idx, (x1, x2) in enumerate(combinations(iter_sent, 2)):
                comb[idx, 0], comb[idx, 1] = x1, x2

            # save results
            savedict = {'scores_w': similarity_w, 'comb': comb,
                        'sent_pairs': np.asarray(sent_pairs, dtype='object')}
            print('Saving result to %s' % fname)
            scipy.io.savemat(fname, savedict)

    elif task == 's':
        for im_idx, (url, sent_group) in enumerate(zip(urls, sentences)):

            # output filename
            img_fname = op.basename(urls[im_idx][0][0])
            fname = op.join(output_dir, ('target_%s.mat' % img_fname))
            print('Image index = [%d/%d]' % (im_idx + 1, n_images))

            if op.isfile(fname):
                continue

            # compute similarity
            refs = [ref_group[ref_idx] for ref_group in sentences]
            s = parallel(my_fun(ref, sent_group[query_idx],
                                dataset_name=dataset_name, verbose=False,
                                method=method)
                         for ref in refs)

            # save results
            savedict = {'s': s, 'query_sentence': sent_group[query_idx],
                        'ref_sentences': np.asarray(refs, dtype='object')}
            print('Saving result to %s' % fname)
            scipy.io.savemat(fname, savedict)
