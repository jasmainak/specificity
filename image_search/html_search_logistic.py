import sys
import numpy as np
import scipy.io as io
from tempita import Template

from utils.search import search_specificity, search_baseline, _matlab_sort

QueryTemplate = Template("""
<br/><b>Query sentence:</b> {{query_sentence}}<br/>
<table border="1">

    <tr><th>Baseline</th>
    {{for img_src in baseline_srcs}}
        <td><img src="{{img_src}}" width="150"></img></td>
    {{endfor}}
    </tr>

    <tr><th>Specificity (gt)</th>
    {{for img_src in spec_gt_srcs}}
        <td><img src="{{img_src}}" width="150"></img></td>
    {{endfor}}
    </tr>

    <tr><th>Specificity (pred)</th>
    {{for img_src in spec_pred_srcs}}
        <td><img src="{{img_src}}" width="150"></img></td>
    {{endfor}}
    </tr>

</table>
<br/>
""")


def _load_data(dataset):

    if dataset == 'pascal':
        mat0 = io.loadmat('../../data/sentences/pascal_1000_img_50_sent.mat')
    elif dataset == 'clipart':
        mat0 = io.loadmat('../../data/sentences/clipart_500_img_48_sent.mat')

    sentences = mat0['%s_sentences' % dataset]
    urls = [url[0][0] for url in mat0['%s_urls' % dataset]]

    mat1 = io.loadmat('search_parameters_%s.mat' % dataset)
    mat2 = io.loadmat('../../data/predict_search/%s/'
                      'predicted_specificity_1000fold.mat' % dataset)
    mat3 = io.loadmat('../../data/predict_search/%s/'
                      'groundtruth_specificity.mat' % dataset)

    s, y, z = mat1['s'], mat2['y_pred'].ravel(), mat2['z_pred'].ravel()
    y_pred, z_pred = mat3['y'].ravel(), mat3['z'].ravel()

    return sentences, urls, s, y, z, y_pred, z_pred

if __name__ == '__main__':

    #### LOAD DATA ####

    dataset = sys.argv[1]
    print('Loading dataset %s' % dataset)
    sentences, urls, s, y, z, y_pred, z_pred = _load_data(dataset)

    rank_b = search_baseline(s, verbose=True)
    rank_s, r_s = search_specificity(s, y, z, return_score=True, verbose=True)
    rank_p, r_p = search_specificity(s, y_pred, z_pred, return_score=True,
                                     verbose=True)

    html = []
    for query_idx in range(len(s)):
        _idx_b = _matlab_sort(s[query_idx, :])
        _idx_s = _matlab_sort(r_s[query_idx, :])
        _idx_p = _matlab_sort(r_p[query_idx, :])

        srcs_b = np.array(urls)[_idx_b[0:10]].tolist()
        srcs_s = np.array(urls)[_idx_s[0:10]].tolist()
        srcs_p = np.array(urls)[_idx_p[0:10]].tolist()

        query_sentence = sentences[query_idx, 49][0]
        html.append(QueryTemplate.substitute(
                    query_sentence=query_sentence,
                    baseline_srcs=srcs_b,
                    spec_gt_srcs=srcs_s,
                    spec_pred_srcs=srcs_p))

    # write out the html
    f = open('../../qualitative/search_specificity_logistic_%s.html'
             % (dataset), 'w')
    f.write(''.join(html))
    f.close()
