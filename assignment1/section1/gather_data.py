import numpy as np
import sys
import os

n_max = int(sys.argv[1])

data = np.zeros((n_max-1, 3))

for i in range(2,n_max+1):
	stream = os.popen('cat ring{}.txt | grep T#'.format(i))
	output = stream.readlines()
        
	sample_idx = 0
	samples = np.zeros((1000-1,i))

	values = np.zeros(i)
	value_idx = 0
	for line in output:
		if 'STOP' in line:
			samples[sample_idx] = values
			sample_idx += 1
			
			# reset
			values = np.zeros(i)
			value_idx = 0
		else:
			values[value_idx] = float(line[3:])
			value_idx += 1

	data[i-2,0] = np.max(samples)
	data[i-2,1] = np.min(samples)
	data[i-2,2] = np.mean(np.mean(samples, axis=1))

np.save('results.npy', data)
