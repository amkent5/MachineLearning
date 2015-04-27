'''
Titanic: machine learning from disaster

Simple gender model - training set
We see 74% of women survived
Only 19% of men survivied

Therefore we create our submission file (GenderModel.csv) based
very simply on women = survived, men = died. 
'''

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next()

print header
'''	0				1			2		3	...
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
'''
data = []                          
for row in csv_file_object:      
    data.append(row)             

data = np.array(data)

print data[1]
''' What proportion survivied the disaster? '''
# 0:: means all (from start to end)
number_passengers = np.size(data[0::, 1].astype(np.float))	# size counts elements in array
number_survived = np.sum(data[0::, 1].astype(np.float))		# sum sums the elements up
proportion_survivors = number_survived/ number_passengers

print proportion_survivors


''' What proportion of men and of women survivied the disaster? '''
num_fem, num_male = 0, 0
num_fem_surv, num_male_surv = 0, 0
for row in data:
	print row[1], row[4]
	if row[4] == 'female':
		num_fem += 1
	else:
		num_male += 1
	if row[1] == '1':
		if row[4] == 'female':
			num_fem_surv += 1
		else:
			num_male_surv += 1

print float(num_fem_surv)/ num_fem
print float(num_male_surv)/ num_male


''' Lets create our submission file based on the knowledge we've gained '''
# Open up the test csv file in to a Python object
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# create our submission file to submit
prediction_file = open("GenderModel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0],'1'])
    else:       
        prediction_file_object.writerow([row[0],'0'])
test_file.close()
prediction_file.close()


















