# Exploratory data analysis 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""import matplotlib
matplotlib.use
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# we can style matplotlib using the inbuilt css
plt.style.use('bmh')
#print plt.style.available	"""

# load data into pandas
datafile = '/Users/admin/Documents/Projects/MachineLearning/Kaggle/MercariPriceSuggestionChallenge/data/train.tsv'
df = pd.read_csv(datafile, delimiter='\t')

# some stats
print df.head(5)
print '\n'*3
print df.size
print df.shape
print df.columns


### look at price distribution (and log price)
#
#
#
###


"""
From data, should be easy to include:
- item_condition	(plot and find num distinct)
- shipping			(easy to include)
- brand_name		(plot and find num distinct)

But what are we going to do with:
- name
- category_name

Both are key fields...

"""

# plot item_condition, shipping, brand_name (all versus target variable)
s_itcond = df['item_condition_id']
s_itcond = s_itcond.value_counts()
ax = s_itcond.plot(kind='bar')
ax.set_title('Item Condition Distinct Counts')
plt.show()


s_shipping = df['shipping'].value_counts()
ax = s_shipping.plot(kind='bar')
ax.set_title('Shipping Distinct Counts')
plt.show()


s_brname = df['brand_name'].value_counts()
print s_brname

# now do value counts of the counts...
s_brname.hist(bins=200)
plt.show()

# shows us very long tail at 1 item per brand.
# need to know when to cut the tail.
# Could be we don't include a brand
# that has less than 50 items in the dataset...
df = pd.DataFrame({'brand': s_brname.index, 'count': s_brname.values})
print df
print df.shape				# 4809 length with no filter

df = df[df['count'] > 50]
print df.shape				# 806 length with >50 occurence filter

# So we have a length 806 categorical array for brand.
# Add an additional column to the input data where a brand occurs that
# isn't within the >50 set of brands.



"""
item_condition:		5 categories all with decent representation

shipping:			2 categories almost equal representation

brand_name:			Very long tail, with most of the dataset being
					represented by the first couple of hundred brands.
					Limiting to brands with >50 representation in the dataset
					leaves us with a categorical vector of length 800 as input
					for brand
"""


### category name

"""
Notice the field is a tree structure:
	- Women/Tops & Blouses/Blouse
	- Women/Jewelry/Necklaces

So we can instantly break into sub categories and sub-sub categories etc.
Let's visualise these categories and sub categories...
"""






