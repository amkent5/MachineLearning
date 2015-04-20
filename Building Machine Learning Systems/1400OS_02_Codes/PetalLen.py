'''
Find a threshold of petal length so we can identify setosa
flower species over the other 2
'''
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# load datset from sklearn
data = load_iris()

# just isolate the data from the dataset
features = data['data']

feature_names = data['feature_names']
target = data['target']

target_names = data['target_names']
labels = target_names[target]

petal_len = features[:, 2]

is_setosa = (labels == 'setosa')

max_setosa_len = petal_len[is_setosa].max()
min_non_setosa_len = petal_len[~is_setosa].min()

print 'Max setosa petal len %s' % max_setosa_len
print 'Min other petals %s' % min_non_setosa_len