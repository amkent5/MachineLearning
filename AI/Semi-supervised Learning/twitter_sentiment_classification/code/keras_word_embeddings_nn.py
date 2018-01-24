### Keras word embeddings classifier for tweet sentiment
### Benchmark classifier for SSL experiments
"""
The two smileys :) and :( have been removed from the tweets in the data set.
The task is to classify where a :) has been removed (class 1), and where a :(
has been removed (class -1)

"""

### Load data
import pandas as pd

# there are 'full' datasets in the same location
neg_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_neg.txt'
pos_datafile = '/Users/admin/Documents/Projects/MachineLearning/AI/Semi-supervised Learning/twitter_sentiment_classification/data/train_pos.txt'

df_neg = pd.read_csv(neg_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False, encoding='latin-1')
df_neg['target'] = -1
print df_neg.shape	# (98954, 2)

df_pos = pd.read_csv(pos_datafile, names=['tweet_str'], skiprows=0, delimiter='\n', error_bad_lines=False, encoding='latin-1')
df_pos['target'] = 1
print df_pos.shape	# (97718, 2)
print df_pos.iloc[97692]	# creepy..

df = pd.concat([df_neg, df_pos], ignore_index=True)
print df.shape	# (196672, 2)



### Preliminary cleaning of vocab
