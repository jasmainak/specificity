"""Calculate importance scores for correlation with specificity."""

# Author: Mainak Jas

from scipy import io
from scipy.stats import nanmean
import numpy as np

from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

from itertools import product
from progressbar import ProgressBar, SimpleProgress

mat = io.loadmat('../../data/sentences/memorability_888_img_5_sent.mat')
sentences = mat['memorability_sentences']

mat = io.loadmat('../../data/target_features.mat')
object_names = mat['objectnames'][0, :].tolist()

mat = io.loadmat('../../data/target_features.mat', matlab_compatible=True)
areas = mat['Areas']

mat = io.loadmat('../../data/memorability_mapping.mat', matlab_compatible=True)
mapping = mat['mapping'].ravel()

# this data is written by the script calculate importance.m
min_area = 4000  # Same value as in Isola et al. NIPS memorability paper
object_pres = areas > min_area

# only objects which occur in at least 10 images are included
include = []
for ii in range(object_pres.shape[0]):
    if object_pres[ii, mapping - 1].sum() >= 10:
        include.append(ii)

# Truncate to only images for which specificity measurements are available
object_names = [object_name[0] for (ii, object_name)
                in enumerate(object_names) if ii in include]
object_pres = object_pres[np.array(include), :][:, mapping - 1].todense()

# save so that the matlab file can use it
io.savemat('../../data/object_presence.mat',
           {'object_names': object_names,
            'object_pres': object_pres})

sentence_list = []
# convert sentences to list
for sent_group in sentences:
    sentence_list.append([sent[0] for sent in sent_group])
sentences = sentence_list
del sentence_list

S, D, importance = list(), list(), list()
analyze = CountVectorizer().build_analyzer()

# seed the numpy selector
np.random.seed(42)


def sentence_similarity(idx, ob, mode):

    s_list = list()
    pbar = ProgressBar(widgets=['%s: image ' % mode, SimpleProgress()],
                       maxval=len(sentences)).start()

    for im_idx, sentence_group in enumerate(np.array(sentences)[idx, :]):

        pbar.update(im_idx + 1)
        for sent in sentence_group:

            words = analyze(sent)

            sim = list()
            for w in words:

                syn1 = wn.synsets(w)
                syn2 = wn.synsets(ob)

                if syn1 and syn2:
                    sim.append(max(s1.path_similarity(s2) for (s1, s2)
                                   in product(syn1, syn2)))
                else:
                    # ignore word if no synset combination was found on wordnet
                    sim.append(None)

            if max(sim):
                s_list.append(max(sim))
            else:
                # ignore sentence if no word was similar enough
                s_list.append(float('nan'))

    pbar.finish()
    return s_list

for ob_idx, ob in enumerate(object_names):

    print("\nCalculating importance for object category: %s" % ob)

    present_idx = np.where(object_pres[ob_idx, :] == True)
    present_idx = np.array(present_idx[1])[0]
    absent_idx = np.random.choice(np.asarray(np.where(object_pres[ob_idx, :] == False)).squeeze()[1],
                                  size=present_idx.shape, replace=False)

    # Mentions
    s_list = sentence_similarity(idx=present_idx, ob=ob, mode='mentions')
    S.append(nanmean(s_list) if s_list else 0)

    # A priori
    D_list = sentence_similarity(idx=absent_idx, ob=ob, mode='a priori')
    D.append(nanmean(D_list) if D_list else 0)

    importance.append(S[-1] - D[-1])
    print('Importance calculated: %0.2f (%0.2f - %0.2f)'
          % (importance[-1], S[-1], D[-1]))

io.savemat('../../data/importance_scores.mat',
           {'importance': importance, 'S': S, 'D': D,
            'object_pres': object_pres, 'object_names': object_names})
