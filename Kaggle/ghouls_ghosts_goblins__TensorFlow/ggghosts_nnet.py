import operator
import numpy

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold

# encoding categorical features
d_color, color_val = {}, 0
d_output, output_val = {}, 0
for i, line in enumerate(open('train.csv', 'r')):
	if i == 0:
		row = line.rstrip().split(',')
		colorIX = [i for i, elt in enumerate(row) if elt == 'color'][0]
		outputIX = [i for i, elt in enumerate(row) if elt == 'type'][0]
		continue
	row = line.rstrip().split(',')
	if not row[colorIX] in d_color:
		d_color[row[colorIX]] = color_val
		color_val += 1
	if not row[outputIX] in d_output:
		d_output[row[outputIX]] = output_val
		output_val += 1

# test
sorted_d = sorted(d_output.items(), key=operator.itemgetter(1))
for elt in sorted_d: print elt

# func to pad inputs
def padInput(dictionary, key):
	dict_index = dictionary[key]
	padded = [0 for i in range(len(dictionary))]
	padded[dict_index] = 1
	padded = [str(elt) for elt in padded]
	return ','.join(padded)

# create transformed dataset
with open('transformed_train.csv', 'w') as f:
	for i, line in enumerate(open('train.csv', 'r')):
		if i == 0:
			continue

		row = line.rstrip().split(',')
		inp_color = padInput(d_color, row[colorIX])
		inp_output = padInput(d_output, row[outputIX])
		csv_row = ','.join([row[1][:8], row[2][:8], row[3][:8], row[4][:8], inp_color, inp_output])
		print csv_row
		f.write(csv_row + '\n')

f.close()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load transformed dataset into numpy arrays
dataset = numpy.loadtxt("transformed_train.csv", delimiter=",")
X = dataset[:, 0:-3]	# X correctly sliced
Y = dataset[:, -3:]		# Y correctly sliced

model_input_dims = numpy.shape(X)[1]
model_output_dims = numpy.shape(Y)[1]

""" check slicing
print model_input_dims
print model_output_dims
print X[-2, :]
print X[-1, :]
print Y[-2, :]
print Y[-1, :]	"""

print model_input_dims
print model_output_dims


# create network topology
# 80.59% on training dataset
model = Sequential()
model.add(Dense(model_input_dims + 1, input_dim=model_input_dims, init='uniform', activation='relu'))
#model.add(Dense(6, init='uniform', activation='sigmoid'))
model.add(Dense(3, init='uniform', activation='sigmoid'))

# compile model
adam = Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# fit model
model.fit(X, Y, nb_epoch=500, batch_size=4)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

""" evaluate some training predictions
preds = model.predict(X)
for i in range(0, len(preds)):
	print Y[i, :]
	print preds[i, :]
	print '\n'	"""

# load the test kaggle submission dataset and apply our trained model to generate our submission outputs
l_ids = []
with open('transformed_test.csv', 'w') as f:
	for i, line in enumerate(open('test.csv', 'r')):
		if i == 0:
			continue

		row = line.rstrip().split(',')
		l_ids.append(row[0])
		inp_color = padInput(d_color, row[colorIX])
		csv_row = ','.join([row[1][:8], row[2][:8], row[3][:8], row[4][:8], inp_color])
		f.write(csv_row + '\n')

f.close()
testX = numpy.loadtxt("transformed_test.csv", delimiter=",")
submission_preds = model.predict(testX)

with open('submission.csv', 'w') as f:
	f.write('id,type' + '\n')
	for i in range(0, len(submission_preds)):

		pred_list = []
		for elt in submission_preds[i, :]:
			pred_list.append(elt)
		
		for k, v in d_output.iteritems():
			if v == pred_list.index(max(pred_list)):
				print k

				f.write(str(l_ids[i]) + ',' + k + '\n')

f.close()



