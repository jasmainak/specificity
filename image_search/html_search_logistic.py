import sys
import numpy as np
import scipy.io as io
from tempita import Template

from utils.search import search_specificity, search_baseline, _matlab_sort

QueryTemplate = Template("""
<br/><b>Query sentence:</b> {{query_sentence}} (s={{s}}) ({{t_b + 1}}, {{t_s + 1}}, {{t_p + 1}})<br/>
<table border="1">

    <tr><th>Baseline</th>
    {{for im_idx, img_src in enumerate(baseline_srcs)}}
        {{if t_b == im_idx}}
        <td><img src="{{img_src}}" width="150" style="border:3px solid red;"></img></td>
        {{else}}
        <td><img src="{{img_src}}" width="150"></img></td>
        {{endif}}
    {{endfor}}
    </tr>

    <tr><th>Specificity (gt)</th>
    {{for im_idx, img_src in enumerate(spec_gt_srcs)}}
        {{if t_s == im_idx}}
        <td><img src="{{img_src}}" width="150" style="border:3px solid red;"></img></td>
        {{else}}
        <td><img src="{{img_src}}" width="150"></img></td>
        {{endif}}
    {{endfor}}
    </tr>

    <tr><th>Specificity (pred)</th>
    {{for im_idx, img_src in enumerate(spec_pred_srcs)}}
        {{if t_p == im_idx}}
        <td><img src="{{img_src}}" width="150" style="border:3px solid red;"></img></td>
        {{else}}
        <td><img src="{{img_src}}" width="150"></img></td>
        {{endif}}
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

    mat1 = io.loadmat('../../data/search_parameters/search_parameters_%s.mat' % dataset)
    #mat2 = io.loadmat('../../data/predict_search/%s/'
    #                  'predicted_specificity_1000fold.mat' % dataset)
    mat2 = io.loadmat('../../data/search_parameters/%s/predicted_LR.mat' % dataset)
    #mat3 = io.loadmat('../../data/search_parameters/%s/'
    #                  'groundtruth_specificity.mat' % dataset)
    mat3 = io.loadmat('../../data/specificity_alldatasets.mat', struct_as_record=False)

    s, y_pred, z_pred = mat1['s'], mat2['y_pred'].ravel(), mat2['z_pred'].ravel()
    
    if dataset == 'pascal':
        y = mat3['specificity'][0, 0].pascal[0, 0].B0.ravel()
        z = mat3['specificity'][0, 0].pascal[0, 0].B1.ravel()
    elif dataset == 'clipart':
        y = mat3['specificity'][0, 0].clipart[0, 0].B0.ravel()
        z = mat3['specificity'][0, 0].clipart[0, 0].B1.ravel()

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

    query_sentence_idx = 49 if dataset == 'pascal' else 47

    html = []
    for query_idx in range(len(s)):
        _idx_b = _matlab_sort(s[query_idx, :])
        _idx_s = _matlab_sort(r_s[query_idx, :])
        _idx_p = _matlab_sort(r_p[query_idx, :])

        t_b = np.where(_idx_b == query_idx)[0][0]
        t_s = np.where(_idx_s == query_idx)[0][0]
        t_p = np.where(_idx_p == query_idx)[0][0]

        srcs_b = np.array(urls)[_idx_b[0:100]].tolist()
        srcs_s = np.array(urls)[_idx_s[0:100]].tolist()
        srcs_p = np.array(urls)[_idx_p[0:100]].tolist()

        query_sentence = '[%d] %s' % (query_idx, sentences[query_idx, query_sentence_idx][0])
        html.append(QueryTemplate.substitute(
                    query_sentence=query_sentence,
                    baseline_srcs=srcs_b,
                    spec_gt_srcs=srcs_s,
                    t_b=t_b,
                    t_s=t_s,
                    t_p=t_p,
                    spec_pred_srcs=srcs_p,
                    s='%0.2f' % s[query_idx, _idx_s[t_s]]))

    # write out the html
    f = open('../../qualitative/search_specificity_logistic_%s.html'
             % (dataset), 'w')
    f.write(''.join(html))
    f.close()
