import os
import os.path as op
import scipy.io
import numpy as np

from joblib import Parallel, delayed

# remember to reload ipython if any changes are made to similarity.py
from utils.similarity import find_sentence_similarity


def _load_dataset(dataset_name):
    if dataset_name == 'memorability':
        input_filename = '../../data/sentences/memorability_888_img_5_sent.mat'
        output_dir = '../../data/search_parameters/memorability/'

    elif dataset_name == 'pascal':
        input_filename = '../../data/sentences/pascal_1000_img_50_sent.mat'
        output_dir = '../../data/search_parameters/pascal/'
        m_sentences = 40  # no of sentences per sentence in image

    elif dataset_name == 'clipart':
        input_filename = '../../data/sentences/clipart_500_img_48_sent.mat'
        output_dir = '../../data/search_parameters/clipart/'

        mat = scipy.io.loadmat(input_filename)
        sentences = mat['clipart_sentences']
        m_sentences = 30

    mat = scipy.io.loadmat(input_filename)
    sentences = mat[dataset_name + '_sentences']
    urls = mat[dataset_name + '_urls']
    output_dir = op.join(output_dir, 'mud')

    return sentences, m_sentences, urls, output_dir


if __name__ == '__main__':

    dataset_name = raw_input('Please enter name of dataset (pascal/'
                             'memorability/clipart): ')
    n_jobs = int(raw_input("Please enter number of parallel jobs: "))

    sentences, m_sentences, urls, output_dir = _load_dataset(dataset_name)
    n_images, n_sentences = sentences.shape
    n_scores = n_sentences * m_sentences  # number of pairs for estimating mu_d

    # Create directories/files that don't exist
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # List of all sentences as list of tuples (im_idx, sentence)
    sent_list = list()
    for im_idx, sent_group in enumerate(sentences):
        sent_list = sent_list + [(im_idx, sent) for sent in sent_group]

    print "Calculating search parameters ..."

    pick_within = np.zeros((n_images, n_scores), dtype=int)
    pick_others = np.zeros((n_images, n_scores), dtype=int)

    for im_idx, (url, sent_group) in enumerate(zip(urls, sentences)):

        img_fname = op.basename(urls[im_idx][0][0])
        fname = op.join(output_dir, ('img_%s.mat' % img_fname))

        if op.isfile(fname):
            continue

        im_sentences = [(idx, sent) for (idx, sent) in sent_list
                        if idx == im_idx]
        other_sentences = [(idx, sent) for (idx, sent) in sent_list
                           if idx != im_idx]

        pick_within[im_idx, :] = [i for i in range(0, n_sentences)
                                  for j in range(0, m_sentences)]
        pick_others[im_idx, :] = np.random.randint(0, len(other_sentences),
                                                   n_scores)

        print("Image index = [%d/%d]" % (im_idx, n_images))
        parallel = Parallel(n_jobs=n_jobs, verbose=30)
        my_fun = delayed(find_sentence_similarity)

        similarity_b = parallel(my_fun(im_sentences[i][1],
                                       other_sentences[j][1],
                                       dataset_name=dataset_name,
                                       verbose=False)
                                for i, j in
                                zip(pick_within[im_idx, :],
                                    pick_others[im_idx, :]))

        sent_pairs = [(im_sentences[i][0], im_sentences[i][1],
                       other_sentences[j][0], other_sentences[j][1])
                      for i, j in
                      zip(pick_within[im_idx, :], pick_others[im_idx, :])]

        # Save the data
        print('Saving result to %s' % fname)
        savedict = {'scores_b': similarity_b,
                    'sent_pairs': np.asarray(sent_pairs, dtype='object')}
        scipy.io.savemat(fname, savedict)
