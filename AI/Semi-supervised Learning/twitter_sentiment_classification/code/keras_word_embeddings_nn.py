### Keras word embeddings classifier for tweet sentiment
### Benchmark classifier for SSL experiments
"""
The two smileys :) and :( have been removed from the tweets in the data set.
The task is to classify where a :) has been removed (class 1), and where a :(
has been removed (class -1)

"""

### load data
import pandas as pd

# there are 'full' datasets in the same location
neg_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_neg.txt'
pos_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_pos.txt'

df_neg = pd.read_csv(neg_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False)
print df_neg
print df_neg.shape

df_pos = pd.read_csv(pos_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False)
print df_pos
print df_pos.shape
print df_pos.iloc[97692]	# creepy..

