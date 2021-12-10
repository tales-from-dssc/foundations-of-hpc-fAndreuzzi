import numpy as np
import matplotlib.pyplot as plt
import sys
from subprocess import Popen, getstatusoutput, PIPE
from pln import compute_pln

L = 1200

nprocs_map = {'serial': [1,], 'onenode': [4,8,12], 'twonodes': [12,24,48]}
nprocs = nprocs_map[sys.argv[1]]

file_names = ['{}{}_{}'.format(sys.argv[1], i, sys.argv[2]) for i in nprocs]
data = [np.load('results/{}.npy'.format(label)) for label in file_names]

serial_thin_jacobi = np.mean(np.load('results/serial_thin.npy')[2:-2, 2:4])
serial_gpu_jacobi = np.mean(np.load('results/serial_gpu.npy')[2:-2, 2:4])
# you need to set this somewhere
serial_time = None

serial_thin_lup = np.mean(np.load('results/serial_thin.npy')[2:-2, -1])
serial_gpu_lup = np.mean(np.load('results/serial_gpu.npy')[2:-2, -1])
# you need to set this somewhere
serial_lup = None

if sys.argv[1] == 'serial':
	raise ValueError('scalability is not defined in this case')
elif sys.argv[1] == 'onenode':
	third = sys.argv[2]
	fourth = 'ngpu'

	serial_time = serial_thin_jacobi
	serial_lup = serial_thin_lup
elif sys.argv[1] == 'twonodes':
	fourth = sys.argv[2]
	third = 'socket'

	if fourth == 'gpu':
		serial_time = serial_thin_jacobi
		serial_lup = serial_thin_lup
	else:
		serial_time = serial_gpu_jacobi
		serial_lup = serial_gpu_lup

print('found serial time {} and LUP {}'.format(serial_time, serial_lup))

# get latency and bandiwdth
cmd = 'python3 ../benchmark/fit.py nintel infiniband {} {}'.format(third, fourth)
print(cmd)
result = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
output = iter(result.stdout.readline, b'')

buff = []
for line in output:
	buff.append(float(line))
latency = buff[0] * 1.e-6
bandwidth = buff[1]
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
