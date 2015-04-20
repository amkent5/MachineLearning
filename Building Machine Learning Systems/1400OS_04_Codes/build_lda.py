import nltk.corpus
import milk
import numpy as np
import string
from gensim import corpora, models, similarities
import sklearn.datasets
import nltk.stem
from collections import defaultdict

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['from:', 'subject:', 'writes:', 'writes'])
class DirectText(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input
    def __len__(self):
        return len(self.input)
dataset = sklearn.datasets.load_mlcomp("20news-18828", "train", 
                                       mlcomp_root='../data')
otexts = dataset.data
texts = dataset.data

texts = [t.decode('utf-8', 'ignore') for t in texts]
texts = [t.split() for t in texts]
texts = [map(lambda w: w.lower(), t) for t in texts]
texts = [filter(lambda s : not len(set("+-.?!()>@012345689") & set(s)), t) for t in texts]
texts = [filter(lambda s : (len(s) > 3) and (s not in stopwords), t) for t in texts]
texts = [map(english_stemmer.stem,t) for t in texts]
usage = defaultdict(int)
for t in texts:
    for w in set(t):
        usage[w] += 1
limit = len(texts)/10
too_common = [w for w in usage if usage[w] > limit]
too_common = set(too_common)
texts = [filter(lambda s : s not in too_common, t) for t in texts]

corpus = DirectText(texts)
dictionary = corpus.dictionary
try:
    dictionary['computer']
except:
    pass

model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary.id2token)

thetas = np.zeros((len(texts),100))
for i,c in enumerate(corpus):
    for ti,v in model[c]:
        thetas[i,ti] += v
distances = milk.unsupervised.pdist(thetas)
large = distances.max() + 1
for i in xrange(len(distances)): distances[i,i] = large

print otexts[1]
print
print
print
print otexts[distances[1].argmin()]

