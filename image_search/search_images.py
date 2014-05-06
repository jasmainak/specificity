# Author: Mainak Jas <mainak@neuro.hut.fi>

import pdb
import nltk
from nltk.corpus import wordnet as wn

import numpy as np
import scipy.io
from scipy.stats import nanmean, norm

from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import combinations, product

# Load sentences
sentences = scipy.io.loadmat('../data/sentences_all.mat')
sentences = sentences['sentences']

f = open('../qualitative/search_images.txt', 'w')

sent_pairs, scores_w, corpus = list(), list(), list()

sent_list = list()
for sent_group in sentences:
    sent_list = sent_list + [sent[0] for sent in sent_group]

np.random.seed(42) # set random seed
query_idx = np.random.randint(0, len(sent_list))
reference_idx = np.random.randint(0, 5, len(sentences))

# Build corpus
for sent_group in sentences:
    corpus.append(' '.join([sent[0] for sent in sent_group]))

# Build tf-idf vectorizer
vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w\\w+\\b') # at-least three letters in word
vectorizer.fit(corpus)
analyze = vectorizer.build_analyzer()

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

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

def estimate_parameters(sentences):
    specificity_w, mu_s, sigma_s, mu_d, sigma_d = list(), list(), list(), list(), list()

    n_sentences, n_images = len(sentences[0]), len(sentences)
    scores_b = np.zeros((n_images, n_sentences, 10)) # between image similarity

    # iterate over images
    for im_idx, sentence_group in enumerate(sentences):
        
        other_sentences = [x for x in sent_list if x not in sent_list[im_idx*5:im_idx*5 + 5]]

        for i, pick_s1 in enumerate(sentence_group):
            
            # for each sentence in the image
            # compare similarity to 10 sentences randomly picked from other images
            for j in range(0, 10):
                pick_s2 = other_sentences[np.random.randint(0, len(other_sentences))]                
                scores_b[im_idx][i][j] = find_sentence_similarity(pick_s1, np.asarray([pick_s2]))

                # pick another sentence if it is a 'nan'. Try up to 5 times
                tries = 0;
                while np.isnan(scores_b[im_idx][i][j]) and tries<=5:
                    pick_s2 = other_sentences[np.random.randint(0, len(other_sentences))]
                    scores_b[im_idx][i][j] = find_sentence_similarity(pick_s1, np.asarray([pick_s2]))
                    tries+=1

        # within-image similarity
        similarity_w = [find_sentence_similarity(sent1,sent2) for (sent1, sent2) in combinations(sentence_group,2)]
        
        # Fit normal distribution
        mu, sigma = norm.fit(similarity_w)
        mu_s.append(mu); sigma_s.append(sigma)

        mu, sigma = norm.fit(scores_b[im_idx, :, :])
        mu_d.append(mu); sigma_d.append(sigma)

        # Save scores and sentence pairs
        scores_w.append(similarity_w)
        sent_pairs.append([(sent1[0], sent2[0]) for (sent1, sent2) in combinations(sentence_group,2)])       

        specificity_w.append(nanmean(similarity_w))

        print ("(specificity, mu_s, sigma_s, mu_d, sigma_d) for image_%d = (%0.4f, %0.4f, %0.4f, %0.4f, %0.4f)" 
               % (im_idx, specificity_w[-1], mu_s[-1], sigma_s[-1], mu_d[-1], sigma_d[-1]))
        print >> f, ("(specificity, mu_s, sigma_s, mu_d, sigma_d) for image_%d = (%0.4f, %0.4f, %0.4f, %0.4f, %0.4f)" 
               % (im_idx, specificity_w[-1], mu_s[-1], sigma_s[-1], mu_d[-1], sigma_d[-1]))

        # commented out below to avoid accidental overwriting
        # scipy.io.savemat('../data/image_search_parameters.mat', 
        #                  {'specificity_w': specificity_w, 'scores_w': scores_w, 'scores_b': scores_b,
        #                  'sent_pairs': np.array(sent_pairs, dtype=object), 'mu_s': mu_s,
        #                  'sigma_s': sigma_s, 'mu_d': mu_d, 'sigma_d': sigma_d})

    return specificity_w, mu_s, sigma_s, mu_d, sigma_d 


estimate_parameters(sentences)
# specificity_w, mu_s, sigma_s, mu_d, sigma_d = estimate_parameters(sentences)

load_parameters = scipy.io.loadmat('../data/image_search_parameters.mat')
specificity_w, mu_s, mu_d = load_parameters['specificity_w'], load_parameters['mu_s'], load_parameters['mu_d']
sigma_s, sigma_d = load_parameters['sigma_s'], load_parameters['sigma_d']

# Baseline approach
reference_sentences = [sent_group[r][0] for (r, sent_group) in zip(list(reference_idx), sentences)]
ranking = list()
for ref in reference_sentences:
    ranking.append(find_sentence_similarity(np.asarray([sent_list[query_idx]]), np.asarray([ref])))
    print ref

print "\nQuery sentence is : (%s)" % sent_list[query_idx]
print "\nMatch found with sentence (baseline) : (%s)" % reference_sentences[ranking.index(max(ranking))]

# Our approach
r_s, r_d = list(), list()
for (ref, p, mus, sigmas, mud, sigmad) in zip(reference_sentences, ranking, mu_s, sigma_s, mu_d, sigma_d):
    
    p_s, p_d = norm.pdf(p, mus, sigmas)[0], norm.pdf(p, mud, sigmad)[0]
    r_s.append(p_s/(p_s + p_d))
    r_d.append(p_d/(p_s + p_d))

print "Match found with sentence (our method) : (%s)" % reference_sentences[r_s.index(max(r_s))]
f.close()