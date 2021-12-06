import numpy as np
import matplotlib.pyplot as plt
import sys
from subprocess import Popen, getstatusoutput, PIPE

L = 1200

nprocs_map = {'serial': [1,], 'onenode': [4,8,12], 'twonodes': [12,24,48]}
nprocs = nprocs_map[sys.argv[1]]

file_names = ['{}{}_{}'.format(sys.argv[1], i, sys.argv[2]) for i in nprocs]
data = [np.load('results/{}.npy'.format(label)) for label in file_names]

# get latency and bandiwdth

if sys.argv[1] == 'serial':
	raise ValueError('scalability is not defined in this case')
elif sys.argv[1] == 'onenode':
	third = sys.argv[2]
	fourth = 'ngpu'
elif sys.argv[1] == 'twonodes':
	fourth = sys.argv[2]
	third = 'socket'

cmd = 'python3 ../benchmark/fit.py nintel infiniband {} {}'.format(third, fourth)
print(cmd)
result = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
output = iter(result.stdout.readline, b'')

buff = []
for line in output:
	buff.append(float(line))
print('latency, bandwidth ', buff)
latency = buff[0] * 1.e-6
bandwidth = buff[1]

#

def estimate_pln(jacobi_time, Tc, N):
	return L*L*L * N / (jacobi_time + Tc)

PLN = []
for dt in data:
	k = np.count_nonzero(dt[0,:3] - 1) * 2
	c = (L*L * k * 2 * 8) * pow(2,-20)
	print(c / bandwidth, k*latency)
	Tc = c / bandwidth + k * latency
	N = np.product(dt[0,:3])

	print('k: {}, c: {}, Tc: {}, N: {}'.format(k, c, Tc, N))
	PLN.append([estimate_pln(dt[1:,2], Tc, N), estimate_pln(dt[1:,3], Tc, N)])
PLN = np.array(PLN)

for idx,i in enumerate(nprocs):
	print('PLN for {} processors'.format(i))
	print('using min jacobi time:')
	print(PLN[idx,0])
	print('using max jacobi time:')
	print(PLN[idx,1])

if len(sys.argv) == 4 and sys.argv[3] == '1':
	measured_mlup = np.array(data)[:,:,-1]
	mlup = np.mean(measured_mlup[:,2:-2], axis=1)

	plt.plot(nprocs, np.mean(PLN[:,0],axis=1)*1.e-6, 'ro-', label='min jacobi time')
	plt.plot(nprocs, np.mean(PLN[:,1],axis=1)*1.e-6, 'go-', label='max jacobi time')
	plt.plot(nprocs, mlup, 'bo-', label='measured')

	plt.yscale('log')

	plt.xlabel('P')
	plt.ylabel('MLUP/sec')

	plt.grid()
	plt.legend()
	plt.show()
