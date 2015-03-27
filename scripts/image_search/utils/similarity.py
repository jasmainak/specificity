# Authors : Mainak Jas <mainak@neuro.hut.fi>

from itertools import product

import numpy as np
import scipy.io

# script runs correctly with scikit-learn v0.14
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

from sklearn.metrics.pairwise import cosine_similarity


def sentence_tokenizer(dataset_name='pascal'):
    """
    Parameters
    ----------
    dataset_name : string
        'memorability' or 'pascal' or 'clipart'

    Returns
    -------
    analyze : object
        breaks sentences into words using scikit-learn tokenizer
    vectorizer : object of class TfidfVectorizer
        see scikit-learn documentation
    """

    if dataset_name == 'memorability':
        mat = scipy.io.loadmat('../../data/sentences/memorability_888_img_5_sent.mat')
        sentences = mat['memorability_sentences']

    elif dataset_name == 'pascal':
        mat = scipy.io.loadmat('../../data/sentences/pascal_1000_img_50_sent.mat')
        sentences = mat['pascal_sentences']

    elif dataset_name == 'clipart':
        mat = scipy.io.loadmat('../../data/sentences/clipart_500_img_48_sent.mat')
        sentences = mat['clipart_sentences']

    # Build corpus
    corpus = list()
    for sent_group in sentences:
        corpus.append(' '.join([sent[0] for sent in sent_group]))

    ### Build tf-idf vectorizer ###

    # at-least three letters in word
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w\\w+\\b')
    vectorizer.fit(corpus)
    analyze = vectorizer.build_analyzer()

    return analyze, vectorizer


analyze_pascal, vectorizer_pascal = sentence_tokenizer('pascal')
analyze_memorability, vectorizer_memorability = sentence_tokenizer('memorability')
analyze_clipart, vectorizer_clipart = sentence_tokenizer('clipart')


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
                best_score.append(_find_best_score(w1, w2))

            if max(best_score):
                sim.append(max(best_score))
            else:
                sim.append(None)
    return sim


def _find_best_score(w1, w2):

    syn1 = wn.synsets(w1)
    syn2 = wn.synsets(w2)

    if syn1 and syn2:
        return max(s1.path_similarity(s2) for (s1, s2)
                   in product(syn1, syn2))
    else:
        return None


def find_sentence_similarity(sent1, sent2, dataset_name='pascal',
                             verbose=False, method='custom'):
    """
    Parameters
    ----------
    sent1 : str
        The first sentence
    sent2 : str
        Words of the second sentence
    dataset_name : string
        'memorability' or 'pascal' or 'clipart'
    verbose : bool
        Print messages if true

    Returns
    -------
    similarity : float
        Similarity scores for two sentences
    """

    if dataset_name == 'pascal':
        analyze, vectorizer = analyze_pascal, vectorizer_pascal
    elif dataset_name == 'memorability':
        analyze, vectorizer = analyze_memorability, vectorizer_memorability
    elif dataset_name == 'clipart':
        analyze, vectorizer = analyze_clipart, vectorizer_clipart

    # Break sentences into words
    words1, words2 = analyze(sent1[0]), analyze(sent2[0])

    if method == 'cosine':
        similarity = cosine_similarity(vectorizer.transform(sent1),
                                       vectorizer.transform(sent2))[0, 0]
        if verbose:
            print sent1, sent2, similarity

        return similarity

    # Get Tfidf weights
    sent1_weights = [vectorizer.transform(sent1).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words1]
    sent2_weights = [vectorizer.transform(sent2).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words2]

    # Find similarity scores for best matching words
    sim_max = find_best_match(words1, words2) + find_best_match(words2, words1)

    # Take weighted average of similarity scores
    if all(x is None for x in sim_max):
        similarity = float('nan')
    else:
        (sim_cleaned, a) = zip(*[(x, w) for (x, w) in zip(sim_max, sent1_weights + sent2_weights) if x != None])
        similarity = np.average(sim_cleaned, weights=a)

    if verbose:
        print sent1, sent2, similarity

    return similarity
