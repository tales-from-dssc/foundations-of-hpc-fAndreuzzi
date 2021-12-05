import numpy as np
from math import log2
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

def func(x, lat, band):
    return lat*1.e-6 + x / band

n1 = np.load('../benchmark/results/{}_{}_{}_{}.npy'.format(*sys.argv[1:5]))
# we group by packet size
n1 = n1[np.argsort(n1[:,0])]
n1 = np.array(np.split(n1, 30, axis=0))
# we take the mean on each sub-matrix
n1 = np.mean(n1, axis=1)

n1[:,0] *= pow(2,-20)
n1[:,2] *= 1.e-6

cut_idx = 8

# FITTING

fit1, _ = curve_fit(func, n1[:cut_idx,0], n1[:cut_idx,2])
fit_lat = fit1[0]

fit2, _ = curve_fit(func, n1[-cut_idx:, 0], n1[-cut_idx:, 2])
fit_band = fit2[1]

#

print(fit_lat)
print(fit_band)

if len(sys.argv) == 6 and sys.argv[5] == '1':
	l1, = plt.plot(n1[:,0], func(n1[:,0], fit_lat, fit_band), 'ro-')

	plt.ylabel('$\mu$s')
	plt.xlabel('MBytes')
	plt.xscale('log')
	plt.yscale('log')

	l2, = plt.plot(n1[:,0], n1[:,2], 'go-')

	plt.legend([l1, l2], ['Fit', 'Measured'])
	plt.title(' '.join(sys.argv[1:5]))
	plt.grid()

	plt.xticks(n1[:,0][1::3], ['$2^{{{}}}$'.format(int(log2(n))) for n in n1[:,0][1::3]])

	plt.show()
