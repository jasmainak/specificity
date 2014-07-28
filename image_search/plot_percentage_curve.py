import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from utils.search import search_specificity, search_baseline


if __name__ == '__main__':

    #### LOAD DATA ####

    dataset = 'pascal'

    mat1 = io.loadmat('search_parameters_pascal.mat')
    mat2 = io.loadmat('../../data/predict_search/%s/'
                      'predicted_specificity_1000fold.mat' % dataset)
    mat3 = io.loadmat('../../data/predict_search/%s/'
                      'groundtruth_specificity.mat' % dataset)

    s, y, z = mat1['s'], mat2['y_pred'].ravel(), mat2['z_pred'].ravel()
    y_pred, z_pred = mat3['y'].ravel(), mat3['z'].ravel()

    #### FIND RANKS ####

    rank_b = search_baseline(s, verbose=True)
    rank_s = search_specificity(s, y, z, verbose=True)
    rank_g = search_specificity(s, y_pred, z_pred, verbose=True)
    rank_p1 = search_specificity(s, y, z_pred, verbose=True)
    rank_p2 = search_specificity(s, y_pred, z, verbose=True)

    #### CALCULATE RETRIEVAL CURVE ####
    per_b, per_s, per_g, per_p1, per_p2 = [], [], [], [], []
    for i in xrange(len(y)):
        per_b.append(len(rank_b[rank_b <= i]) / float(len(y)) * 100)
        per_s.append(len(rank_s[rank_s <= i]) / float(len(y)) * 100)
        per_g.append(len(rank_g[rank_g <= i]) / float(len(y)) * 100)
        per_p1.append(len(rank_p1[rank_p1 <= i]) / float(len(y)) * 100)
        per_p2.append(len(rank_p2[rank_p2 <= i]) / float(len(y)) * 100)

    #### PLOT RETRIEVAL CURVE ####

    plt.figure(figsize=(10, 8))
    plt.plot(xrange(len(y)), per_b, label='baseline')
    plt.plot(xrange(len(y)), per_s, label='predicted specificity')
    plt.plot(xrange(len(y)), per_g, label='ground truth specificity')
    plt.plot(xrange(len(y)), per_p1, label='param2-pred-param2-gt specificity')
    plt.plot(xrange(len(y)), per_p2, label='param1-gt-param2-pred specificity')
    plt.plot(xrange(len(y)), np.array(range(len(y))) / float(len(y)) * 100,
             label='random')
    plt.legend(loc=0)
    plt.xlabel('Rank <= x', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)

    plt.show()
