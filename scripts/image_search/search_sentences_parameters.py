import os
import os.path as op
import scipy.io
import numpy as np

from joblib import Parallel, delayed
from itertools import combinations

from optparse import OptionParser

# remember to reload ipython if any changes are made to similarity.py
from utils.similarity import find_sentence_similarity


def _load_dataset(dataset_name, method=''):
    """Auxiliary function to load the dataset."""
    if dataset_name == 'pascal':
        input_filename = '../../data/sentences/pascal_1000_img_50_sent.mat'
    elif dataset_name == 'clipart':
        input_filename = '../../data/sentences/clipart_500_img_48_sent.mat'
    else:
        raise RuntimeError('Dataset %s does not exist' % dataset_name)

    output_dir = op.join('../../data/image_search', dataset_name,
                         'similarity_scores')

    mat = scipy.io.loadmat(input_filename)
    sentences = mat[dataset_name + '_sentences']
    urls = mat[dataset_name + '_urls']
    m_sentences = 40  # no of sentences per sentence in image

    return sentences, m_sentences, urls, output_dir


if __name__ == '__main__':

    # #### Options for the main function ####
    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset_name",
                      help="dataset to evaluate (pascal/clipart)")
    parser.add_option("-j", "--njobs", dest="n_jobs",
                      help="number of parallel processes to run")
    parser.add_option("-t", "--task", dest="task",
                      help="task to do. Compute similarity score of"
                           "(train_pos_class/train_neg_class/test)")

    options, args = parser.parse_args()
    dataset_name = options.dataset_name
    n_jobs = options.n_jobs
    task = options.task

    # #### Set up things #####
    sentences, m_sentences, urls, output_dir = _load_dataset(dataset_name)
    n_images, n_sentences = sentences.shape

    output_dir = op.join(output_dir, task)

    if task == 'train_pos_class':
        ref_idx, query_idx = 0, (n_sentences - 1)
    elif task == 'train_neg_class':
        # number of pairs for negative class
        n_scores = n_sentences * m_sentences
        pick_within = np.zeros((n_images, n_scores), dtype=int)
        pick_others = np.zeros((n_images, n_scores), dtype=int)

        # List of all sentences as list of tuples (im_idx, sentence)
        sent_list = list()
        for im_idx, sent_group in enumerate(sentences):
            sent_list = sent_list + [(im_idx, sent) for sent in sent_group]

    # #### setup for parallel computation #####
    parallel = Parallel(n_jobs=n_jobs, verbose=30)
    my_fun = delayed(find_sentence_similarity)

    # #### Create output directory if it doesn't exist #####
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    print('Calculating similarity scores for image search (%s) ...' % task)

    for im_idx, (url, sent_group) in enumerate(zip(urls, sentences)):
        # output filename
        img_fname = op.basename(urls[im_idx][0][0])
        fname = op.join(output_dir, ('%s.mat' % img_fname))
        print('[%s] Image index = [%d/%d]' % (task, im_idx + 1, n_images))

        if op.isfile(fname):
            continue

        if task == 'train_pos_class':
            # compute similarity
            similarity_w = parallel(my_fun(sent1, sent2,
                                           dataset_name=dataset_name,
                                           verbose=False) for
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

        elif task == 'train_neg_class':
            pass
            im_sentences = [(idx, sent) for (idx, sent) in sent_list
                            if idx == im_idx]
            other_sentences = [(idx, sent) for (idx, sent) in sent_list
                               if idx != im_idx]

            pick_within[im_idx, :] = [i for i in range(0, n_sentences)
                                      for j in range(0, m_sentences)]
            pick_others[im_idx, :] = np.random.randint(0, len(other_sentences),
                                                       n_scores)
            parallel = Parallel(n_jobs=n_jobs, verbose=30)
            my_fun = delayed(find_sentence_similarity)

            similarity_b = parallel(my_fun(im_sentences[i][1],
                                           other_sentences[j][1],
                                           dataset_name=dataset_name,
                                           verbose=False)
                                    for i, j in zip(pick_within[im_idx, :],
                                    pick_others[im_idx, :]))

            sent_pairs = [(im_sentences[i][0], im_sentences[i][1],
                           other_sentences[j][0], other_sentences[j][1])
                          for i, j in zip(pick_within[im_idx, :],
                          pick_others[im_idx, :])]

            # Save the data
            savedict = {'scores_b': similarity_b,
                        'sent_pairs': np.asarray(sent_pairs, dtype='object')}

        elif task == 'test':
            # compute similarity
            refs = [ref_group[ref_idx] for ref_group in sentences]
            s = parallel(my_fun(ref, sent_group[query_idx],
                                dataset_name=dataset_name, verbose=False)
                         for ref in refs)

            # save results
            savedict = {'s': s, 'query_sentence': sent_group[query_idx],
                        'ref_sentences': np.asarray(refs, dtype='object')}

        print('Saving result to %s' % fname)
        scipy.io.savemat(fname, savedict)
