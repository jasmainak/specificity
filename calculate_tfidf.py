from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io

sentences = scipy.io.loadmat('../data/sentences_all.mat')
sentences = sentences['sentences']

f = open('../qualitative/tfidf.txt', 'w')

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

vectorizer = TfidfVectorizer()
corpus = list()

"""Build Corpus"""
for sent_group in sentences:
	for sent in sent_group:
		corpus.append(sent[0])

"""Calculate tf-idf scores"""
X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()

for sent in corpus:
	all_weights = vectorizer.transform([sent]).toarray()
	words = analyze(sent)

	weights = list()
	for w in words:
		weights.append(all_weights[0][vectorizer.vocabulary_.get(w)])
	
	print >> f, [w.encode('utf-8') for w in words]
	print >> f, [prettyfloat(w) for w in weights]
	print >>f, ''

f.close()