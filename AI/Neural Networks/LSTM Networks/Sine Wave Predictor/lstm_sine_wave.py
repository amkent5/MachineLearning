import numpy as np
import matplotlib.pyplot as plt

from numpy import newaxis
from math import sin, pi
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#Main Run Thread
epochs = 3
seq_len = 50

# Generate sine wave data over 5000 data points with time delta of 0.1
def generate_sine():
	with open('sine_wave.csv', 'w') as f:
		l_sine = []
		for i in range(0, 5001):
			# sin function takes values [0, 500] with increment 0.1
			l_sine.append(sin(float(i)/ 10))
			f.write(str(sin(float(i)/ 10)) + '\n')

	l_sine_first50 = l_sine[:50]
	plt.plot(l_sine)
	plt.show()
	plt.plot(l_sine_first50)
	plt.show()

	f.close()

generate_sine()
filename = "sine_wave.csv"

f = open(filename, 'rb').read()
data = f.decode().split('\n')

# Get data into shape
"""
Keras LSTM layers work by taking in a numpy array of 3 
dimensions (N, W, F) where N is the number of training 
sequences, W is the sequence length and F is the number
of features of each sequence.

The x_train and x_test sets will contain the first 49 
elements of each 50-element sine sequence.
The y_train and y_test sets will contain the corresponding
50th element for these sequences.
"""
sequence_length = seq_len + 1
result = []

# generate an array of arrays where the inner arrays
# are sequences of 50 elements of sine data.
# The sequences shift by 1 element each time
for index in range(len(data) - sequence_length):
	result.append(data[index: index + sequence_length])

result = np.array(result)

row = round(0.9 * result.shape[0])	# generate training set (90% of the data)
train = result[:int(row), :]

np.random.shuffle(train)
x_train = train[:, :-1]	# all 50 sequences with the last elt of the 50 removed
y_train = train[:, -1]	# all 50 sequences, but only the last elt of each

# same for the test sets, but on the 10% we left behind
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1]

# now reshape the train and test sets with dimensions (N, W, F)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()

model.add(LSTM( 50, input_shape=(50, 1), return_sequences=True))
# Dropout can be interpreted as sampling a Neural Network within the full Neural Network, and only updating the parameters of the sampled network based on the input data. 
model.add(Dropout(0.2))

model.add(LSTM( 100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense( output_dim=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")
model.summary()
model.fit( X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)

prediction_seqs = []
for i in range(len(X_test)):
	curr_frame = X_test[i]
	predicted = model.predict(curr_frame[newaxis,:,:])[0,0]
	prediction_seqs.append(predicted)

"""
We are just predicting the 50th element of each sequence (based on the previous
49 elements in each sequence.
Because we have shifted the sequences by 1 each time, the predictions form
a continuous curve where there is a prediction for each time interval.
"""
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True Data')
plt.plot(prediction_seqs, label='Prediction')
plt.legend()
plt.show()

