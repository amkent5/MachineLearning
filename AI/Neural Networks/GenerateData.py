# Generate data for neural net

import random

def genData(fileName):
	f = open(fileName, "w")

	for i in range(0, 100):	# 100 boilers
		data = []
		for j in range(0, 24):	# 24 months
			# do we need a repair?
			r = random.random()
			if r < 0.2:	# need a repair 1/5 of the time
				cost = random.randint(1, 500)
				data.append(cost/ 500.0)	# normalise
			else:
				data.append(0)
		data.append(random.randint(0, 1))	# output

		# Write to file
		sdata = ""
		for d in data:
			sdata += str(d)
			sdata += " "
		sdata += "\n"
		f.write(sdata)

	f.close()

if __name__ == "__main__":
	genData("test.txt")