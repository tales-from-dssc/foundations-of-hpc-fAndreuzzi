import numpy as np
import sys
import os

data = np.zeros((24, 2, 3, 4))

for i in range(1,25):
	print('Processing {} processors'.format(i))
	stream = os.popen('cat results/matrix{}.txt'.format(i))
	output = list(stream.readlines())

	idx = 0        
	for mb_idx in range(data.shape[1]):
		samples = []

		if idx < len(output):
			line = output[idx].strip()
		else:
			print('\tnothing here')
			break
		while line != 'DONE':
			samples.append(list(map(float, line.split(','))))

			# get next line
			idx += 1
			if idx < len(output):
				line = output[idx].strip()
			else:
				break
		# me move on after encountering stop
		print('\tDONE found')
		idx += 1

		samples = np.array(samples)

		data[i-1,mb_idx,:,0] = np.max(samples, axis=0)
		data[i-1,mb_idx,:,1] = np.min(samples, axis=0)
		data[i-1,mb_idx,:,2] = np.mean(samples, axis=0)
		data[i-1,mb_idx,:,3] = np.std(samples, axis=0)

np.save('results.npy', data)
