import sys
import string
import nltk
from nltk.corpus import wordnet as wn
from optparse import OptionParser
from itertools import product

if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("-s", "--sentence", dest="sentence")
    parser.add_option("-o", "--object", dest="object_category")

    options, _ = parser.parse_args()

    # separate words
    words = nltk.word_tokenize(options.sentence)
    
    # remove punctuation
    words = filter(lambda x: x not in string.punctuation, words)
    
    sim = list()
    for w in words:

        syn1 = wn.synsets(w)
        syn2 = wn.synsets(options.object_category)

        if syn1 and syn2:
            sim.append(max(s1.path_similarity(s2) for (s1, s2) in product(syn1, syn2)))

    sys.stdout.write(str(max(sim)))
