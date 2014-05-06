import sys
import numpy as np
import scipy.io
from scipy.stats import norm

from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import product

sentences = scipy.io.loadmat('../data/sentences_all.mat')
sentences = sentences['sentences']

cfg = scipy.io.loadmat('../data/image_search_parameters.mat')
scores_w, scores_b = cfg['scores_w'], cfg['scores_b']

f = open('../qualitative/search_images.txt', 'w')

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

sent_list = list()
for sent_group in sentences:
    sent_list = sent_list + [sent[0] for sent in sent_group]

np.random.seed(42) # set random seed

n_sentences = len(sentences)
reference_idx = np.random.randint(0, 4, n_sentences)
train_idx_b, train_idx_w = range(0, 4), np.array([0, 1, 2, 4, 5, 7])
query_list = range(4, len(sent_list), 5)

r_s, r_d = np.zeros((n_sentences, n_sentences)), np.zeros((n_sentences, n_sentences))
s = np.zeros((n_sentences, n_sentences))

searchmat = scipy.io.loadmat('../data/image_search_results.mat') 
r_s, r_d, s, curr_idx = searchmat['r_s'], searchmat['r_d'], searchmat['s'], searchmat['curr_idx']

for im_idx, query_idx in enumerate(query_list):

    if im_idx > curr_idx:
        
        print "\n[%d] Query sentence is : (%s)" % (im_idx, sent_list[query_idx])
        print >> f, "\n[%d] Query sentence is : (%s)" % (im_idx, sent_list[query_idx])

        # Baseline approach
        reference_sentences = [sent_group[r][0] for (r, sent_group) in zip(list(reference_idx), sentences)]
        for idx, ref in enumerate(reference_sentences):
            s[im_idx, idx] = find_sentence_similarity(np.asarray([sent_list[query_idx]]), np.asarray([ref]))

            if np.remainder(idx,10)==0:
                sys.stdout.write('.')
                sys.stdout.flush()
           
        print ''
        print >> f, ''
        for i in range(1,10):
            print "Match found with sentence (baseline) : (%s)" % reference_sentences[np.argsort(s[im_idx, :])[-i]]  
            print >> f, "Match found with sentence (baseline) : (%s)" % reference_sentences[np.argsort(s[im_idx, :])[-i]]

        # Our approach  
        for idx, ref in enumerate(reference_sentences):
            
            p = s[im_idx, idx]
            mu_s, sigma_s = norm.fit(scores_w[idx, train_idx_w])
            mu_d, sigma_d = norm.fit(scores_b[idx, train_idx_b, :].ravel())

            p_s, p_d = norm.pdf(p, mu_s, sigma_s), norm.pdf(p, mu_d, sigma_d)
            # r_s.append(p_s/(p_s + p_d))
            # r_d.append(p_d/(p_s + p_d))

            r_s[im_idx, idx] = p_s/(p_s + p_d)
            r_d[im_idx, idx] = p_d/(p_s + p_d)

        scipy.io.savemat('../data/image_search_results.mat', 
                        {'reference_idx': reference_idx, 'r_s': r_s, 'r_d': r_d,
                         's': s, 'curr_idx': im_idx}) 
        print ''
        print >> f, ''
        for i in range(1, 10):
            print "Match found with sentence (specificity) : (%s)" % reference_sentences[np.argsort(r_s[im_idx, :])[-i]]
            print >> f, "Match found with sentence (specificity) : (%s)" % reference_sentences[np.argsort(r_s[im_idx, :])[-i]]

        f.flush() # output to file

f.close()