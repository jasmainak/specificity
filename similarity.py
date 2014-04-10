import scipy.io
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from nltk.corpus import wordnet as wn

from itertools import product

mat1 = scipy.io.loadmat('../data/sentence_descriptions.mat')
mat2 = scipy.io.loadmat('../data/sentence_descriptions_140315.mat')

sentences = np.concatenate((mat1['sentences'], mat2['sentences']), axis=1)

# Build corpus
corpus = list()
for sent_group in sentences:
    corpus.append(' '.join([sent[0] for sent in sent_group]))

# Build tf-idf vectorizer
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w\\w+\\b') # at-least three letters in word
vectorizer.fit(corpus)
analyze = vectorizer.build_analyzer()

def find_best_match(words1, words2):
    """
    Parameters
    ----------
    words1 : list of strings
        Words of the first sentence
    words2 : list of strings
        Words of the second sentence

    Returns
    -------
    sim : list of floats
        Similarity scores for each word in the first sentence
    """

    sim = list()
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
                sim.append(max(best_score))                
            else:
                sim.append(None)
    return sim

def find_sentence_similarity(sent1, sent2):
    """
    Parameters
    ----------
    sent1 : str
        The first sentence
    sent2 : str
        Words of the second sentence
    
    Returns
    -------
    similarity : float
        Similarity scores for two sentences
    """

    words1, words2 = analyze(sent1[0]), analyze(sent2[0])
        
    sent1_weights = [vectorizer.transform(sent1).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words1]
    sent2_weights = [vectorizer.transform(sent2).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words2]
    
    sim_max = find_best_match(words1, words2) + find_best_match(words2, words1)

    if all(x is None for x in sim_max):            
        similarity = float('nan')
    else:
        (sim_cleaned, a) = zip(*[(x, w) for (x, w) in zip(sim_max, sent1_weights + sent2_weights) if x!=None])
        similarity = np.average(sim_cleaned, weights=a)

    return similarity
