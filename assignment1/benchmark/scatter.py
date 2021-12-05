from subprocess import Popen, getstatusoutput, PIPE
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def process_filename(filename):
	arr = filename.split('_')
	arr[-1] = arr[-1][:-4]
	return arr

def latband(filename):
	label = ' '.join(process_filename(filename))
	print('python3 fit.py ' + label)

	for word in sys.argv[1:]:
		if not word in label:
			print('\tDoes not contain ' + word)
			return None

	result = Popen('python3 fit.py ' + label, shell=True, stdout=PIPE, stderr=PIPE)
	output = iter(result.stdout.readline, b'')

	buff = []
	for line in output:
		buff.append(float(line))
	if len(buff) == 0:
		print('\tSomething went wrong..')
		return None

	latency = buff[0]
	bandwidth = buff[1]

	return label, latency, bandwidth

labels, lat, band = tuple(zip(*list(filter(lambda x: x is not None, map(latband, os.listdir('results'))))))

plt.scatter(lat, band)

plt.xlabel('$\mu$s')
plt.ylabel('Mbytes/s')

for i, txt in enumerate(labels):
	plt.annotate(txt, (lat[i], band[i]))

plt.grid()
plt.show()
