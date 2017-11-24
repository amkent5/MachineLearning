"""
*** Kaggle Dog Breed Identification Challenge *** 
(https://www.kaggle.com/c/dog-breed-identification)




Benchmark the model using the VGG16 deep ConvNet (which is available as a 
saved model in Keras).


The VGG16 was developed in Oxford Universities Visual Geometry Group.
It was designed to solve the ImageNet challenge and is a 16 layered convolutional neural network.

However, when we hear the term 'ImageNet' in the context of deep learning and Convolutional Neural Networks,
we are likely referring to the ImageNet Large Scale Visual Recognition Challenge, or ILSVRC for short.

The goal of this image classification challenge is to train a model that can correctly classify an input image 
into 1,000 separate object categories.

Models are trained on ~1.2 million training images with another 50,000 images for validation and 100,000 images
for testing.

These 1,000 image categories represent object classes that we encounter in our day-to-day lives, such as
species of dogs, cats, various household objects, vehicle types, and much more. You can find the full list of object
categories in the ILSVRC challenge here:
http://image-net.org/challenges/LSVRC/2014/browse-synsets

When it comes to image classification, the ImageNet challenge is the de facto benchmark for computer
vision classification algorithms.


We could benchmark with all of the industry-standard algorithms:
https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

"""

# load and explore data

import numpy as np
import pandas as pd
import glob
import random

from PIL import Image


labels = pd.read_csv('labels.csv')
print labels
print labels.head()
print labels.describe()

# add file names to a list
path = '/Users/admin/Documents/Projects/CNNs/Dog_Breed_Identification_Kaggle/train/'
l_images = glob.glob(path + '*.jpg')
print l_images[0]

# take a look at some random images and add to a small validation list
l_subset_of_imgs = []
for i in range(5):
	rand = random.randint(0, len(l_images))
	img = Image.open(l_images[rand])
	#img.show()
	width, height = img.size
	print width, height
	l_subset_of_imgs.append(l_images[rand])

# Note images are different dimensions:
# 400 362
# 500 377
# 500 359
# 500 375
# 500 347

# import the necessary libraries
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2

# define a dict that maps model names to their classes inside Keras
models = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

# initialize the input image shape (224x224 pixels) for VGG16
# initialize the pre-processing function
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# load VGG16
Network = models['vgg16']
model = Network(weights='imagenet')	# pre-trained imagenet weights

# load image into PIL format and re-size its dimen
image = load_img(l_subset_of_imgs[0], target_size=inputShape)
print image

# convert image to multi-dimensional numpy array
image = img_to_array(image)
print image
print image.shape
print '\n'*2

# add a dimension to the input (otherwise will get an error when we use the network)
image = np.expand_dims(image, axis=0)

# pre-process the image using the helper function for VGG16
image = preprocess(image)

# classify the image
preds = model.predict(image)
#print preds

# use the ImageNet utility function to make sense of the predictions
preds = imagenet_utils.decode_predictions(preds)

print 'Class Label:'
# class label code
label = l_subset_of_imgs[0].split('/')[-1].split('.')[0]
print labels.loc[labels['id'] == label]

print '\n'

print 'VGG16 results:'
for tup in preds[0]:
	print tup[0], '\t', tup[1], '\t', tup[-1] * 100

# and load image to validate results
img = Image.open(l_subset_of_imgs[0])
img.show()


