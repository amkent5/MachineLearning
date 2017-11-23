# Classifying the CIFAR-10 dataset with convolutional neural networks

# Requires latest version of keras and Tensorflow
# Recommend running in a virtualenv and doing:
# sudo pip install tensorflow
# sudo pip install keras
# sudo pip install matplotlib

# manipulation imports
import random
import numpy as np

# machine learning imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# visualisation imports
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# import dataset
from keras.datasets import cifar10




# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# explore data
print X_train[0]
print X_train[0].shape

# plot some random images from the training data
# (imread converts image to multi-dimensional numpy array representation)
# (imshow then renders the numpy array)
max_ix = X_train.shape[0]
for i in range(5):
	plt.imshow(X_train[random.randint(0, max_ix)])
	plt.show()


### Main
seed = 7
np.random.seed(seed)

# normalise inputs from 0-255 to 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/ 255.0
X_test = X_test/ 255.0

# one hot encode outputs (we have 10 output classes)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print y_train
print num_classes

# create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print model.summary()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)



### Example using trained network to classify 'new' data



