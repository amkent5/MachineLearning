"""
Another way in which we can explore the dataset is to use dimensionality reduction techniques. 
t-SNE is a method which reduces high-dimensional datasets to a dimensionality of your choosing (i.e. 2 or 3 dimensional). 
If the resulting 2 or 3 dimensional distribution exhibits clustering, the implication is that there is structure in the 
dataset (i.e. things can be predicted). 

If no clustering is observed, then the data set is messy, and it could be hard to find a predictive signal in the underlying 
higher dimensional data.
"""

import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# base-2 encoded data file (features and classes)
encodedfile = '/model_data/encoded_data_base2.csv'

pandas_df = pd.read_csv(encodedfile, header=None)
numpy_dataset = pandas_df.as_matrix()
print numpy_dataset.shape
X = numpy_dataset[:, 0:-1]
Y = numpy_dataset[:, -1]

cut_down_dataset = numpy_dataset[:3000, :]
print cut_down_dataset.shape

X_cut = cut_down_dataset[:, 0:-1]
Y_cut = cut_down_dataset[:, -1]

# use t-SNE clustering/ dimred
# 2d
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_results = tsne.fit_transform(X_cut)
X_tsne = tsne_results[:, 0]
Y_tsne = tsne_results[:, 1]

# 2d plotting
colors = ['red','blue']
plt.scatter(X_tsne, Y_tsne, Y_tsne, c=Y_cut, cmap=mpl.colors.ListedColormap(colors))
plt.title('t-SNE Base-2 Encoded First 3,000 Records')
plt.show()
