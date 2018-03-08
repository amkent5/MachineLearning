##### Gensim doc2vec and agglomerative clustering of resulting dense vector representations (rather than bow's)

### Load scraped data
import pickle

#filename = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/scraping_code/d_oil_and_gas_terms.pickle'
filename = '/Users/Ash/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/scraping_code/d_oil_and_gas_terms.pickle'
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
import os
import random
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer

# randomly sample 400 elements of the data dictionary
# need the same result each time so pickle the cut down dict and load it in each time
print len(d_data)	# 4931
#random_terms = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/nlp_code/random_terms.pickle'
random_terms = '/Users/Ash/Projects/MachineLearning/AI/Semi-supervised Learning/scraping_oil_gas_corpus/nlp_code/random_terms.pickle'
if not os.path.isfile(random_terms):
    d_data = dict( (k, d_data[k]) for k in random.sample(d_data, 400) )
    with open(random_terms, 'wb') as handle:
        pickle.dump(d_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(random_terms, 'rb') as handle:
        d_data = pickle.load(handle)

print len(d_data) # 400
print d_data.keys()[0]

# form keywords list and descriptions list
keywords, descriptions = [], []
for i in range(len(d_data)):
    keywords.append( d_data.keys()[i] )
    descriptions.append( d_data.values()[i] )

# perform natural language cleaning
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
def nlp_clean(doc):

    # create list of tokens from the document (a token being a individual component of the vocabulary
    # i.e. a single word, or single punctuation)
    tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]

    # make all words lower case
    lowers = [ token.lower() for token in tokens ]

    # remove stop words
    stopped = [ lower for lower in lowers if lower not in self.stopwords ]

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in stopped:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # reduce tokens to base / stemmed form
    stems = [ self.stemmer.stem(token) for token in filtered_tokens ]
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



### Agglomerative clustering of resultant first-order tensor space

# create distance matrix
from sklearn.metrics.pairwise import cosine_similarity
dist = cosine_similarity(model.docvecs)	# note previously I have had dist = 1 - cs...


print 'Zero''th doc vector:', '\n', model.docvecs[0], '\n'*3
print 'Zero''th distance vector:', '\n',  dist[0], '\n'*3
print 'N''th doc vector:', '\n', model.docvecs[-1], '\n'*3
print 'N''th distance vector:', '\n', dist[-1], '\n'*3

print len(model.docvecs)
print len(dist)

# although the number of instance in model.docvecs and dist is the same (length 400), 
# the length of each element is different. The length of each element of model.docvecs 
# is 100 (for each dimension), whereas the length for each element in dist is of length 
# 400. This is because each element represents the cosine similarity between the instance
# and each of the other 400 instances (http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/).
# Each element of dist, is an array representing the Cosine Similarity between the element (document) with all 
# other elements (documents) in the set

# Notice that that is why the first element of the zero'th dist vector is 1 (as this represents the cosine-similarity
# between itself), and the last element of the last (n'th) dist vector is 1 (as this represents the cosine-similarity
# between itself).

print len(model.docvecs[0])
print len(dist[0])



### Run clustering algorithm to understand hidden structure within the keywords / descriptions
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

# define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
    linkage_matrix,
    labels=keywords,
    leaf_font_size=12.,
    leaf_rotation=45.
    )

plt.tight_layout() # show plot with tight layout
plt.ylabel('Ward distance')
plt.show()



### Extract document classes from dendrogram

# create a truncated dendrogram showing only the last 12 merges
# this will cut away the noisey micro-clusters and enable us to consider macro-clusters
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
    linkage_matrix,
    truncate_mode='lastp',   # show only the last p merged clusters
    p=12,                    # last 12 merges
    leaf_font_size=12.,
    show_contracted=True,    # to get a distribution impression in truncated branches
    )

plt.xlabel('Number of merges in cluster')
plt.ylabel('Ward distance')
plt.show()

# a large jump in distance is typically what we're interested in if we want to argue for
# a certain number of clusters (https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)

# by inspecting the cut down dendrogram we can see that a distance cut-off at around 
# ward distance 20 maximises jumps in distance for each cluster tree, generating 6 document classes
max_dist = 20
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(
    linkage_matrix,
    truncate_mode='lastp',   # show only the last p merged clusters
    p=12,                    # last 12 merges
    leaf_font_size=12.,
    show_contracted=True,    # to get a distribution impression in truncated branches
    )

plt.axhline(y=max_dist, color='r', linestyle='--')
plt.xlabel('Number of merges in cluster')
plt.ylabel('Ward distance')
plt.show()

# knowing max_dist (our number of clusters) we can use the fcluster class to map each observation to a cluster id
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(linkage_matrix, max_dist, criterion='distance')
print clusters
"""
[2 5 4 2 6 3 1 4 1 2 5 1 1 1 4 5 4 1 2 4 3 2 5 5 3 1 1 5 4 2 1 1 1 2 3 3 1
 6 3 3 3 2 3 3 2 5 3 6 5 6 2 2 3 3 4 4 1 3 5 1 1 1 4 1 4 1 1 1 6 4 4 3 2 6
 6 5 6 1 2 6 2 4 4 3 1 5 2 2 1 2 6 5 2 1 6 1 1 2 6 1 1 4 3 3 5 1 5 2 4 2 1
 4 4 3 2 6 6 5 5 1 6 2 5 4 1 2 1 3 3 2 3 2 4 2 5 4 4 1 6 4 5 5 4 1 3 3 6 6
 2 4 4 2 6 1 1 2 4 5 2 4 2 4 2 3 2 1 6 2 4 6 1 3 6 5 6 2 3 3 3 1 3 3 3 1 1
 3 2 1 2 2 6 2 4 2 6 5 1 2 2 2 4 4 6 1 1 1 2 5 1 1 1 1 1 2 1 4 4 2 6 4 4 1
 1 1 4 5 1 6 2 4 3 3 1 6 1 1 1 2 3 2 5 2 4 1 6 4 6 6 3 3 2 6 6 5 2 4 6 5 2
 5 1 2 3 4 1 3 6 5 6 3 4 2 1 4 4 4 1 5 6 2 5 3 1 3 1 3 6 2 2 6 1 4 3 1 6 1
 5 6 6 2 6 3 5 6 1 3 1 4 1 3 1 1 3 6 6 6 1 1 3 6 6 5 2 3 6 4 2 1 2 4 6 6 1
 5 1 2 1 1 3 4 6 2 1 3 1 1 1 5 1 2 2 1 3 5 3 4 4 2 5 1 6 2 5 3 2 2 1 1 3 2
 5 1 3 2 3 1 6 4 4 4 1 5 6 2 1 2 2 3 6 2 6 1 6 3 5 3 1 5 2 3]
"""
# So, using unsupervised learning we have managed to derive class labels for a dataset that is completely beyond the understanding
# of the programmer. We have managed to craft a semi-supervised dataset, where 400 of the labels are known. Now let's use keras to
# create a classifier to predict labels for more unseen data.

# First though, let's visualise the clusters in 2D, with the keyword labels attached and try some vector arithmetic in our embedding-space!



### Use t-SNE to cluster the doc vectors too with their keywords displayed, and the class clustering colour
from sklearn.manifold import TSNE

# as per this SO article (https://stackoverflow.com/questions/36545434/cosine-similarity-tsne-in-sklearn-manifold)
# we need to change our distance metric slightly (https://en.wikipedia.org/wiki/Cosine_similarity):
dist_metric = 1.0 - dist

tsne_model = TSNE(metric="precomputed")
X_reduced = tsne_model.fit_transform( abs(dist_metric) )    # https://github.com/scikit-learn/scikit-learn/issues/5772
print X_reduced[:, 0], X_reduced[:, 1]

#plt.scatter( X_reduced[:, 0], X_reduced[:, 1], s=20*2**4 )
#plt.show()

colours = ['#F18F01', '#048BA8', '#2E4057', '#99C24D', '#FF2216', '#D4ADCF']     # have 6 colours here but you can't guarantee the number of clusters we'll generate as there is some randomness in the system
for i in range(len(keywords)):
    plt.scatter( X_reduced[i, 0], X_reduced[i, 1], s=20*2**4, color=colours[ clusters[i] - 1 ])
    plt.annotate(
        keywords[i],
        xy=( X_reduced[i, 0], X_reduced[i, 1] ),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom'
    )

plt.show()



### Can we do some vector arithmetic for this example?
"""
The classic example is:
King - Man + Woman = Queen

This works well, Man and Woman are clearly semantically correlated and are also a subset of Royalty (King and Queen).
We should try and find an example that follows a similar principle..

In word2vec we can do things like:
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)   # king - man + woman = queen
[('queen', 0.50882536)]
model.doesnt_match("breakfast cereal dinner lunch";.split())
'cereal'
model.similarity('woman', 'man')
0.73723527
"""

# As we have limited knowledge of the meaning behind the dataset, try some random combinations of vector
# additions and see if the results are a) interpretable and b) meaningful
for i in range(20):
    rix_1, rix_2 = random.randint(1, 399), random.randint(1, 399)
    vec_add = model.docvecs.most_similar( positive=[keywords[rix_1], keywords[rix_2]], topn=1 )

    print keywords[rix_1]
    print descriptions[rix_1], '\n'

    print keywords[rix_2]
    print descriptions[rix_2], '\n'

    print 'Vector addition result:'
    print vec_add[0][0]
    print [ descriptions[i] for i in range(len(descriptions)) if keywords[i] == vec_add[0][0] ], '\n'*3

"""
Good results from the addition:

kilogram per cubic meter
The SI unit of measurement for density. Mud weights are typically expressed in kg/m3. The conversion factor from lbm/gal 
to kg/m3 is 120. For example, 12 lbm/gal = 1440 kg/m3.

+

zinc carbonate
A neutral zinc salt, ZnCO3, which can be used as a sulfide scavenger in water-base muds. Zinc carbonate is less soluble 
than zinc basic carbonate and perhaps slower to react with sulfide ions. Treatment level is about 0.1 lbm/bbl per 50 mg/L sulfide 
ion (determined by Garrett Gas Train sulfide analysis of the filtrate).

=

equivalent weight
[u'The molecular weight of an element, molecule or ion divided by its valence (or valence change for a redox reaction). For example, 
the molecular weight of calcium hydroxide, or "slaked lime," [Ca(OH)2] is 72.10. Because the valency of calcium in this case is 2, 
the equivalent weight of lime is 36.05. Mud analyses give concentrations in various units: ppm, mg/L, wt.% and epm. Mud engineers should 
recognize the meaning of epm and equivalent weight of a mud chemical.']

We can see that by adding a chemical aspect to the kilogram, we have a resultant vector of equivalent weight - which is all about molecular
weight. This makes a lot of sense.

"""



### Creating a semi-supervised classifier












