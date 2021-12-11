from math import log2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams.update({'font.size': 25})

def flt(filename):
	flag = True

	last_idx = len(sys.argv) if 'bandw' not in sys.argv[-1] else -1

	for w in sys.argv[1:last_idx]:
		if w in filename:
			if w == 'intel':
				flag = flag and 'nintel' not in filename
			elif w == 'gpu':
				flag = flag and 'ngpu' not in filename
		else:
			return False
	return flag

def file_and_label(filename):
	n1 = np.load('results/' + filename)
	# we group by packet size
	n1 = n1[np.argsort(n1[:,0])]
	n1 = np.array(np.split(n1, 30, axis=0))
	# we take the mean on each sub-matrix
	n1 = np.mean(n1, axis=1)

	lb = ' '.join(filename[:-4].split('_'))
	lb = lb.replace('nintel', 'OpenMPI')
	lb = lb.replace('intel', 'IntelMPI')
	lb = lb.replace('infiniband', 'InfiniBand')
	lb = lb.replace('gigabit', 'Gigabit')
	lb = lb.replace('ngpu', 'THIN')
	lb = lb.replace('gpu', 'GPU')
	lb = lb.replace('core', '--map-by core')
	lb = lb.replace('socket', '--map-by socket')
	lb = lb.replace('node', '--map-by node')

	return n1, lb

files = map(file_and_label, filter(flt, os.listdir('results/')))

plt.figure(figsize=(15,12))

for file, label in files:
	if 'bandw' in sys.argv[-1]:
		plt.plot(file[:,0]*pow(2,-20), file[:,0]*pow(2,-20) / (file[:,2]*1.e-6), marker='o', label=label)
	else:
		plt.plot(file[:,0]*pow(2,-20), file[:,2], marker='o', label=label)

plt.legend()
plt.grid()

plt.xscale('log')
if 'bandw' not in sys.argv[-1]:
	plt.yscale('log')

plt.xlabel('MBytes')
if 'bandw' in sys.argv[-1]:
	plt.ylabel('MBytes/s')
else:
	plt.ylabel('$\mu$s')

# just for ticks
n1 = np.load('results/intel_gigabit_node_ngpu.npy')
plt.xticks(n1[:30,0][1::3]*pow(2,-20), ['$2^{{{}}}$'.format(int(log2(n))) for n in n1[:30,0][1::3]*pow(2,-20)])

plt.show()
