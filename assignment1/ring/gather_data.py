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

# 0->P
# 1->mapby
# 2->max,min,mean,std
data = np.zeros((n_max-n_min+1, 3, 4))

for i in range(n_min,n_max+1):
	print('Processing {} processors'.format(i))
	stream = os.popen('cat results/ring{}.txt'.format(i))
	output = list(stream.readlines())

	idx = 0        
	for mb_idx in range(data.shape[1]):
		samples = []

		if idx < len(output):
			line = output[idx].strip()
		else:
			print('\tnothing here')
			break
		while line != 'DONE' and idx < len(output):
			samples.append(float(line))
			# get next line
			idx += 1
			line = output[idx].strip()
		# me move on after encountering stop
		print('\tDONE found')
		idx += 1

		data[i-n_min,mb_idx,0] = np.max(samples)
		data[i-n_min,mb_idx,1] = np.min(samples)
		data[i-n_min,mb_idx,2] = np.mean(samples)
		data[i-n_min,mb_idx,3] = np.std(samples)

np.save('results.npy', data)
