import string
import nltk
from nltk.corpus import wordnet as wn

import numpy as np
import scipy.io
from scipy.stats import nanmean

from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import combinations, product

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

# Load sentences
sentences = scipy.io.loadmat('../data/sentences_all.mat')
sentences = sentences['sentences']

f = open('../qualitative/automated_specificity.txt', 'w')

all_pairs1, all_pairs2 = list(), list()
all_scores = list()

vectorizer = TfidfVectorizer()
corpus = list()

"""Build Corpus"""
for sent_group in sentences:
    corpus.append(' '.join([sent[0] for sent in sent_group]))

X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()

specificity_max = list()
for im_idx, sentence_group in enumerate(sentences):
    #X = vectorizer.fit_transform
    similarity_max = list()
    for (sent1, sent2) in combinations(sentence_group,2):
                
        # separate words
        #words1 = nltk.word_tokenize(sent1[0])
        #words2 = nltk.word_tokenize(sent2[0])

        # remove punctuation
        #words1 = filter(lambda x: x not in string.punctuation, words1)
        #words2 = filter(lambda x: x not in string.punctuation, words2) 

        words1, words2 = analyze(sent1[0]), analyze(sent2[0])
        
        sent1_weights = [vectorizer.transform(sent1).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words1]
        sent2_weights = [vectorizer.transform(sent2).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words2]
        
        print >> f, [w.encode('utf-8') for w in words1]        
        print >> f, [prettyfloat(w) for w in sent1_weights]
        print >> f, [w.encode('utf-8') for w in words2]
        print >> f, [prettyfloat(w) for w in sent2_weights]
        
        sim_max = list()
        
        print >>f, ''
        # first find best-matches for words in sentences 1
        for w1 in words1:
            best_match, best_score = list(), list()
            for w2 in words2:

                syn1 = wn.synsets(w1)
                syn2 = wn.synsets(w2)

                if syn1 and syn2:
                    best_score.append(max(s1.path_similarity(s2) for (s1, s2) in product(syn1, syn2)))
                else:
                    best_score.append(None)

            if max(best_score):
                sim_max.append(max(best_score))
                print >>f, "%s: %s" % (w1, words2[best_score.index(max(best_score))])
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
                    best_score.append(max(s1.path_similarity(s2) for (s1, s2) in product(syn1, syn2)))
                else:
                    best_score.append(None)

            if max(best_score):
                sim_max.append(max(best_score))
                print >>f, "%s: %s" % (w1, words1[best_score.index(max(best_score))])
            else:
                sim_max.append(None)

        print >>f, '\n'
        
        # sentence similarity
        similarity_max.append(np.mean([x for x in sim_max if x!=None]))
        
        all_scores.append(similarity_max[-1])
        all_pairs1.append(sent1)
        all_pairs2.append(sent2)
        
    specificity_max.append(np.mean(similarity_max))

    print "Specificity score for image_%d = %0.4f" % (im_idx, specificity_max[-1])

scipy.io.savemat('../data/specificity_automated_modified.mat', 
                 {'specificity_max' : specificity_max, 'all_scores': all_scores,
                 'all_pairs1': all_pairs1, 'all_pairs2': all_pairs2 })

f.close()