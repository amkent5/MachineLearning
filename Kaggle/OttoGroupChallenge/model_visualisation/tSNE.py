# Run t-SNE on the Otto group Kaggle training dataset

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/OttoGroupChallenge/model_data/training_data_otto.csv'
data = pd.read_csv(datafile, delimiter=',')

# make the target numerical
target = data['target']
import sklearn.preprocessing
le = sklearn.preprocessing.LabelEncoder()
le.fit(target)
t1 = le.transform(target)

data.drop('target', axis=1, inplace=True)

# stick it back into the main DataFrame
t1 = np.expand_dims(t1, axis=1)
data = np.concatenate((data, t1), axis=1)
print data

X = data[:, 0:-1]
y = data[:, -1]
print X.shape
print y.shape

# take a random sample of 1000 rows the array as it's too big for t-SNE
ixs = np.random.choice(X.shape[0], 1000, replace=False)
X = X[ixs, :]
y = y[ixs,]
print X.shape
print y.shape

# use t-SNE clustering/ dimred
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_results = tsne.fit_transform(X)
dim1_tsne = tsne_results[:, 0]
dim2_tsne = tsne_results[:, 1]

# define 10 colours for output class
colors = [
	'red',
	'blue',
	'green',
	'springgreen',
	'coral',
	'yellow',
	'lightslategrey',
	'magenta',
	'firebrick',
	'wheat'
]

# https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels/12487355
plt.scatter(dim1_tsne, dim2_tsne, c=y, cmap=mpl.colors.ListedColormap(colors))
plt.title('t-SNE: Otto Group Kaggle Challenge')
plt.show()