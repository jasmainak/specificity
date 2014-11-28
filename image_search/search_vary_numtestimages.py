from operator import itemgetter
import numpy as np
import scipy.io as io
from joblib import Parallel, delayed

from utils.search import (search_specificity, search_baseline)


def vary_test_size(s, y, z, test_size):
    n_images = len(y)
    # note: this will give different result from matlab b/c of
    # different random number generators
    test_idx = np.random.choice(n_images, test_size, replace=False)

    #s_test = np.asarray([s[i][test_idx] for i in test_idx])
    s_test = np.asarray([itemgetter(test_idx)(s_t) for
                         s_t in itemgetter(test_idx)(s)])
    rank_b = search_baseline(s_test)
    rank_s = search_specificity(s_test, y[test_idx], z[test_idx])

    return rank_b, rank_s, test_idx


if __name__ == '__main__':

    dataset = 'pascal'

    mat1 = io.loadmat('search_parameters_pascal.mat')
    s = mat1['s']

    mat2 = io.loadmat('../../data/predict_search/%s/'
                      'predicted_specificity_1000fold.mat'
                      % dataset)
    y, z = mat2['y_pred'][0, :], mat2['z_pred'][0, :]
    n_images = len(y)

    parallel, p_func = Parallel(n_jobs=6, verbose=30), delayed(vary_test_size)
    r = parallel(p_func(s, y, z, test_size) for test_size in
                 range(100, n_images + 1, 10))
    rank_b, rank_s, test_idx = zip(*r)
    savedict = {'rank_baseline': rank_b, 'rank_specificity': rank_s,
                'test_idx': test_idx, 'test_size': range(100, n_images + 1, 10)}
    io.savemat('../../data/predict_search/%s/'
               'predicted_logistic_1000fold_python.mat' % dataset, savedict)
