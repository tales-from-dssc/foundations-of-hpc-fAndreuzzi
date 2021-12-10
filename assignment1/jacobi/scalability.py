import numpy as np
import matplotlib.pyplot as plt
import sys
from pln import compute_pln
from latband import find_latency_bandwidth

L = 1200

nprocs_map = {'serial': [1,], 'onenode': [4,8,12], 'twonodes': [12,24,48]}
nprocs = nprocs_map[sys.argv[1]]

def load_files():
	file_names = ['{}{}_{}'.format(sys.argv[1], i, sys.argv[2]) for i in nprocs]
	data = [np.load('results/{}.npy'.format(label)) for label in file_names]

	if sys.argv[1] == 'serial':
		raise ValueError('scalability is not defined in this case')
	elif sys.argv[1] == 'onenode':
		third = sys.argv[2]
		fourth = 'ngpu'
	elif sys.argv[1] == 'twonodes':
		fourth = sys.argv[2]
		third = 'socket'

	return data, third, fourth

# time and LUP
def load_serial_quantities():
	serial_thin_jacobi = np.mean(np.load('results/serial_thin.npy')[2:-2, 2:4])
	serial_gpu_jacobi = np.mean(np.load('results/serial_gpu.npy')[2:-2, 2:4])

	serial_thin_lup = np.mean(np.load('results/serial_thin.npy')[2:-2, -1])
	serial_gpu_lup = np.mean(np.load('results/serial_gpu.npy')[2:-2, -1])

	if sys.argv[1] == 'serial':
		raise ValueError('scalability is not defined in this case')
	elif sys.argv[1] == 'onenode':
		return serial_thin_jacobi, serial_thin_lup
	elif sys.argv[1] == 'twonodes':
		if sys.argv[2] == 'gpu':
			return serial_thin_jacobi, serial_thin_lup
		else:
			return serial_gpu_jacobi, serial_gpu_lup

data, third, fourth = load_files()

serial_time, serial_lup = load_serial_quantities()
print('found serial time {} and LUP {}'.format(serial_time, serial_lup))

latency, bandwidth = find_latency_bandwidth(third, fourth)
print('lat: {}, band: {}'.format(latency, bandwidth))

#

PLN = compute_pln(data, serial_time, latency=latency, bandwidth=bandwidth, L=L)

if len(sys.argv) == 4:
	if sys.argv[3] == '0':
		measured_mlup = np.array(data)[:,:,-1]
		mlup = np.mean(measured_mlup[:,2:-2], axis=1)

		plt.plot(nprocs, PLN*1.e-6, 'ro-', label='$P(L,N)$')
		plt.plot(nprocs, mlup, 'bo-', label='measured')
		plt.plot(nprocs, np.array(nprocs) * serial_lup, label='N$P_1(L)$')

		plt.yscale('log')

		plt.xlabel('P')
		plt.ylabel('MLUP/sec')

		plt.title('LUP')

		plt.grid()
		plt.legend()
		plt.show()
	elif sys.argv[3] == '1':
		measured_mlup = np.array(data)[:,:,-1]
		mlup = np.mean(measured_mlup[:,2:-2], axis=1)

		plt.plot(nprocs, np.array(nprocs) * serial_lup / (PLN*1.e-6), 'ro-')

		plt.xlabel('P')

		plt.title('scalability')

		plt.grid()
		plt.show()
