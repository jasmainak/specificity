"""Calculates length of sentence descriptions in the MEM-5S dataset."""

# Author: Mainak Jas

import numpy as np
from scipy import io
from sklearn.feature_extraction.text import TfidfVectorizer

print(__doc__)

mat = io.loadmat('../../data/sentences/memorability_888_img_5_sent.mat')
sentences = mat['memorability_sentences']

# use all words
vectorizer = TfidfVectorizer(min_df=0, token_pattern=r"\b\w+\b")
analyze = vectorizer.build_analyzer()

# Build corpus and find sentence lengths
sent_lengths = np.zeros_like(sentences)
for i, sent_group in enumerate(sentences):
    for j, sent in enumerate(sent_group):
        sent_lengths[i][j] = len(analyze(sent[0]))

io.savemat('../../data/memorability_sent_lengths.mat',
           dict(sent_lengths=sent_lengths))
