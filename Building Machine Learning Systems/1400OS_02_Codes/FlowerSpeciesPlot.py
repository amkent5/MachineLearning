from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# load datset from sklearn
data = load_iris()

# just isolate the data from the dataset
features = data['data']

feature_names = data['feature_names']

target = data['target']

for t, marker, c in zip(xrange(3), ">ox", "rgb"):
	plt.scatter(features[target == t, 0], features[target == t, 1], marker = marker, c = c)


plt.show()