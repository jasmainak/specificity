"""Calculate specificity automatically for the MEM-5S dataset."""

# Author: Mainak Jas

from nltk.corpus import wordnet as wn

import numpy as np
import scipy.io
from scipy.stats import nanmean

from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import combinations, product

print(__doc__)


class PrettyFloat(float):
    def __repr__(self):
        return "%0.2f" % self

# Load sentences
sentences = scipy.io.loadmat('../../data/sentences/memorability_888_img_5_sent.mat')
sentences = sentences['memorability_sentences']

f = open('../../automated_specificity.txt', 'w')

sent_pairs, scores_w = list(), list()

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w\\w+\\b')
corpus = list()

# Build corpus
for sent_group in sentences:
    corpus.append(' '.join([sent[0] for sent in sent_group]))

vectorizer.fit(corpus)
analyze = vectorizer.build_analyzer()

specificity_max, specificity_w = list(), list()
for im_idx, sentence_group in enumerate(sentences):

    similarity_max, similarity_w = list(), list()
    for (sent1, sent2) in combinations(sentence_group, 2):

        words1, words2 = analyze(sent1[0]), analyze(sent2[0])

        sent1_weights = [vectorizer.transform(sent1).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words1]
        sent2_weights = [vectorizer.transform(sent2).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words2]

        print >> f, [w.encode('utf-8') for w in words1]
        print >> f, [PrettyFloat(w) for w in sent1_weights]
        print >> f, [w.encode('utf-8') for w in words2]
        print >> f, [PrettyFloat(w) for w in sent2_weights]

        sim_max = list()

        print >>f, ''
        # first find best-matches for words in sentence 1
        for w1 in words1:
            best_match, best_score = list(), list()
            for w2 in words2:

                syn1 = wn.synsets(w1)
                syn2 = wn.synsets(w2)

                if syn1 and syn2:
                    best_score.append(max(s1.path_similarity(s2) for (s1, s2)
                                      in product(syn1, syn2)))
                else:
                    best_score.append(None)

            if max(best_score):
                sim_max.append(max(best_score))
                print >>f, "%s: %s (%0.2f)" % (w1, words2[best_score.index(max(best_score))], sim_max[-1])
            else:
                sim_max.append(None)

        print >>f, ''

        # then find best-matches for words in sentence 2
        for w1 in words2:
            best_score = list()
            for w2 in words1:

                syn1 = wn.synsets(w1)
                syn2 = wn.synsets(w2)

                if syn1 and syn2:
                    best_score.append(max(s1.path_similarity(s2) for (s1, s2)
                                      in product(syn1, syn2)))
                else:
                    best_score.append(None)

            if max(best_score):
                sim_max.append(max(best_score))
                print >>f, "%s: %s (%0.2f)" % (w1, words1[best_score.index(max(best_score))], sim_max[-1])
            else:
                sim_max.append(None)

        # sentence similarity

        if all(x is None for x in sim_max):
            similarity_max.append(float('nan'))
            similarity_w.append(float('nan'))
        else:
            (sim_cleaned, a) = zip(*[(x, w) for (x, w) in zip(sim_max, sent1_weights + sent2_weights) if x!=None])
            similarity_max.append(np.mean(sim_cleaned))
            similarity_w.append(np.average(sim_cleaned, weights=a))

        print >>f, "\nSimilarity score = (%0.2f, %0.2f)\n" % (similarity_max[-1], similarity_w[-1])

    scores_w.append(similarity_w)
    sent_pairs.append([(sent1[0], sent2[0]) for (sent1, sent2) in combinations(sentence_group,2)])

    specificity_max.append(nanmean(similarity_max))
    specificity_w.append(nanmean(similarity_w))

    print "Specificity score for image_%d = (%0.4f, %0.4f)" % (im_idx, specificity_max[-1], specificity_w[-1])

scipy.io.savemat('../../data/specificity_automated.mat',
                 {'specificity_max': specificity_max,
                  'specificity_automated': specificity_w, 'scores_w': scores_w,
                  'sent_pairs': np.array(sent_pairs, dtype=object)})

f.close()
