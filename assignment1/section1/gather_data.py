import numpy as np
import sys
import os

argc = len(sys.argv)
if argc == 2:
	n_max = int(sys.argv[1])
	n_min = 2
elif argc == 3:
	n_max = int(sys.argv[2])
	n_min = int(sys.argv[1])

data = np.zeros((n_max-n_min+1, 4))

for i in range(n_min,n_max+1):
	stream = os.popen('cat ring{}.txt'.format(i))
	output = stream.readlines()
        
	samples = []
	for line in output:
		samples.append(float(line[3:]))

	data[i-n_min,0] = np.max(samples)
	data[i-n_min,1] = np.min(samples)
	data[i-n_min,2] = np.mean(samples)
	data[i-n_min,3] = np.std(samples)

np.save('results.npy', data)
