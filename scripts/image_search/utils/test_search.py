from scipy import io
from numpy.testing import assert_array_equal

from search import search_specificity, search_baseline


if __name__ == '__main__':
    data = io.loadmat('test_search.mat')
    s, y, z = data['s'], data['y'], data['z']

    rank_s = search_specificity(s, y, z, verbose=True)
    rank_b = search_baseline(s, verbose=True)

    assert_array_equal(rank_b, data['rank_b'].ravel())
    assert_array_equal(rank_s, data['rank_s'].ravel())
