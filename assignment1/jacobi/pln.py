import numpy as np

def estimate_pln(L, jacobi_serial_time, Tc, N):
	return L*L*L * N / (jacobi_serial_time + Tc)

def estimate_Tc(c, bandwidth, k, latency):
	return c / bandwidth + k * latency

def compute_k(dt):
	return np.count_nonzero(dt[0,:3] - 1) * 2

def compute_N(dt):
	return np.product(dt[0,:3])

# MBytes
def compute_c(L,k):
	return (L*L * k * 2 * 8) * pow(2,-20)

# data is the .npy extracted from the output of jacobi
def compute_pln(data, serial_time, latency, bandwidth, L):
	PLN = []
	for dt in data:
		k = compute_k(dt)
		c = compute_c(L,k)
		Tc = estimate_Tc(c, bandwidth, k, latency)
		N = compute_N(dt)

		PLN.append(estimate_pln(L, serial_time, Tc, N))

		print('{} processors ({},{},{})'.format(N, *dt[0,:3]))
		print('\tc = {} MBytes'.format(c))
		print('\tk = {}'.format(k))
		print('\tTc = {} s'.format(Tc))
		print('\tP(L,N) = {} MLUP/s'.format(PLN[-1]*1.e-6))
	return np.array(PLN)
