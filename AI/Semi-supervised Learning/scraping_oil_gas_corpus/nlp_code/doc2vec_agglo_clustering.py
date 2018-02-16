##### Gensim doc2vec and agglomerative clustering of resulting dense vector representations (rather than bow's)

### Load scraped data
import pickle

filename = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/scraping_code/d_oil_and_gas_terms.pickle'
with open(filename, 'rb') as handle:
	d_data = pickle.load(handle)

# sample
for k, v in d_data.items(): print k, '\n', v, '\n'*2
"""

forward multiple-contact test
A laboratory test to determine the phase envelope between lean gas and oil by equilibrating a gas sample 
several times with fresh samples of oil. In a forward-contact test, light and intermediate components are 
stripped from the oil by multiple contacts with the gas. The test also indicates how many contacts are required 
before the gas with added components becomes miscible with the oil. The molar ratios at each contact step are 
typically designed using PVT simulation software that incorporates the fluid composition at each contact.


standing valve
A downhole valve assembly that is designed to hold pressure from above while allowing fluids to flow from 
below. Standing valves generally are run and retrieved on slickline with the valve assembly located in an 
appropriate nipple. Applications for standing valves include testing the tubing string, setting packers, or 
other applications in which it is desirable to maintain fluid in the tubing string.


wellbore orientation
Wellbore direction. Wellbore orientation may be described in terms of inclination and azimuth. Inclination 
refers to the vertical angle measured from the down direction-the down, horizontal and up directions have 
inclinations of 0, 90 and 180, respectively. Azimuth refers to the horizontal angle measured clockwise 
from true north-the north, east, south and west directions have azimuths of 0, 90, 180 and 270, respectively.

"""


### Process text
import random
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

# randomly sample 400 elements of the data dictionary
print len(d_data)	# 4931
d_data = dict( (k, d_data[k]) for k in random.sample(d_data, 400) )
print len(d_data)	# 400

# form keywords list and descriptions list
keywords, descriptions = [], []
for i in range(len(d_data)):
	keywords.append( d_data.keys()[i] )
	descriptions.append( d_data.values()[i] )

# perform natural language cleaning
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
def nlp_clean(doc):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as its own token
    tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
    
    # remove stop words
    tokens = [ token for token in tokens if token not in stopwords ]

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # stem
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

docs = []
for doc in descriptions: docs.append( nlp_clean(doc) )

# sample
print descriptions[-1]
print docs[-1]
"""

A phenomenon of relative seismic velocities of strata whereby a shallow layer or feature with a high seismic 
velocity (e.g., a salt layer or salt dome, or a carbonate reef) surrounded by rock with a lower seismic velocity 
causes what appears to be a structural high beneath it. After such features are correctly converted from time to 
depth, the apparent structural high is generally reduced in magnitude.

[u'a', u'phenomenon', u'relat', u'seismic', u'veloc', u'strata', u'wherebi', u'shallow', u'layer', u'featur', u'high', 
u'seismic', u'veloc', u'e.g.', u'salt', u'layer', u'salt', u'dome', u'carbon', u'reef', u'surround', u'rock', u'lower', 
u'seismic', u'veloc', u'caus', u'appear', u'structur', u'high', u'beneath', u'after', u'featur', u'correct', u'convert', 
u'time', u'depth', u'appar', u'structur', u'high', u'general', u'reduc', u'magnitud']

"""



### Build, train and save the doc2vec model
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# create tagged docs (https://github.com/RaRe-Technologies/gensim/issues/1542)
# (tagging the docs allows us to easily extract the labels when doing doc similarity)
docs_tagged = [ TaggedDocument(docs[i], [keywords[i]]) for i in range(len(docs)) ]
print docs_tagged[0]

# initialise a model
model = Doc2Vec(
	size=100, 		# dimensionality of document vectors
	window=4, 
	min_count=5, 
	workers=4
	)

# build vocab
model.build_vocab(docs_tagged)

# train
model.train(docs_tagged, total_examples=len(docs_tagged), epochs=100)



### Inspect some results (https://stackoverflow.com/questions/41709318/what-is-gensims-docvecs)

# first keyword and its 100-D document vector representation
keyword1 = keywords[0]
docvec = model.docvecs[0]

print '\n'*2
print keyword1
print docvec
print len(docvec)

# most similar document vectors (by their keyword)
docsim = model.docvecs.most_similar( keyword1 )
for i in range(10): print docsim[i]

""" Output:

mud-aging cell
[-0.46930075  0.24111511 -0.65452486  0.07960624  0.27933225 -0.57205486
  0.01857634 -0.43373516  0.5950049   0.09591938  0.8515341  -0.23467036
  0.61899596  0.20965149  0.9030415  -0.38237852 -0.4520214  -0.5371389
 -0.02965951 -0.7457512   0.21630289  0.18072322 -0.01356592  0.16475712
  0.3085342   0.76956207  0.42369726  0.3957191   0.49336338  0.6514096
  0.7015149  -0.56747806  0.6603277   0.00761382  0.85111606 -0.29173177
  0.46942535 -0.5020117  -0.17106616 -1.1938404   0.6599053   0.55130905
 -0.80835307  0.40446362  0.03073083 -0.25957283  0.12954792 -0.26075384
  0.19322304  0.00296548  0.05748585 -0.3986431  -0.05385951 -0.15913898
 -0.95375824  0.50589114 -0.74666363  0.8480258   0.6071295  -0.14590833
 -0.05624911  0.05853309 -0.3353707   1.2660801   0.6817378  -0.27144068
 -0.14211453  0.81194127  0.14978758 -0.4786596   0.3583522  -0.53172165
 -1.2283134  -0.15744266  0.0312759  -1.4410212  -0.49220684  0.28382796
 -0.7101878   0.21427606  0.48749664 -0.10307872  0.3946607   0.78112704
  0.38293564 -0.28952685  0.68147844 -0.25098896  0.65532887  0.10993876
 -0.29568055 -0.16812487  0.11657126  0.38111183  0.2104465  -0.22793952
  0.07921502 -0.26019612  0.2111703  -0.42160577]
100
(u'mud cell', 0.9888162016868591)
(u'flowstream sample', 0.6759142875671387)
(u'mud in sample', 0.5847841501235962)
(u'wireline-retrievable safety valve (WRSV)', 0.583116888999939)
(u'stock tank barrel', 0.5754137635231018)
(u'wireline retrievable safety valve (WRSV)', 0.5743108987808228)
(u'surge tank', 0.5686211585998535)
(u'buoyancy method', 0.5684531331062317)
(u'bentonite equivalent', 0.5675902962684631)
(u'standard temperature and pressure', 0.5541241765022278)

"""










