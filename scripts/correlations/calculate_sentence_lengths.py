import numpy as np
from scipy import io
from sklearn.feature_extraction.text import TfidfVectorizer

mat = io.loadmat('../../data/sentences/memorability_888_img_5_sent.mat')
sentences = mat['memorability_sentences']

# at-least three letters in word
vectorizer = TfidfVectorizer(min_df=0, token_pattern=r"\b\w+\b")
analyze = vectorizer.build_analyzer()

# Build corpus
sent_lengths = np.zeros_like(sentences)
for i, sent_group in enumerate(sentences):
    for j, sent in enumerate(sent_group):
        sent_lengths[i][j] = len(analyze(sent[0]))

io.savemat('../../data/sentences/memorability_sent_lengths.mat',
           dict(sent_lengths=sent_lengths))
