from latband import find_latency_bandwidth
from pytablewriter import LatexTableWriter
from pytablewriter.style import Style
import numpy as np
from scalability import load_serial_quantities, load_files, L
from pln import *

data, third, fourth = load_files()

serial_time, serial_lup = load_serial_quantities()
print('found serial time {} and LUP {}'.format(serial_time, serial_lup))

latency, bandwidth = find_latency_bandwidth(third, fourth)
print('lat: {}, band: {}'.format(latency, bandwidth))

PLN = compute_pln(data, serial_time, latency=latency, bandwidth=bandwidth, L=L)

def row(i):
	dt = data[i]

	k = compute_k(dt)
	N = compute_N(dt)
	c = compute_c(L,k)

	Tc = estimate_Tc(c, bandwidth, k, latency)

	pln = estimate_pln(L, serial_time, Tc, N) * 1.e-6

	return [N, *dt[0,:3], k, np.round(c,3), np.round(Tc,3), np.round(pln,3), np.round(np.mean(dt[:,-1]), 3), np.round(N*serial_lup / pln,3)]

writer = LatexTableWriter()
writer.table_name = "example_table"
writer.headers = ["N", "Nx", "Ny", "Nz", "k", "C(L,N)", "T_c(L,N)", "\\tilde{P}(L,N)", "P(L,N)", "\\frac{P(1)*N}{\\tilde{P}(L,N)}"]
writer.value_matrix = list(map(row, range(len(PLN))))
writer.value_matrix.insert(0, [1, 1, 1, 1, 'n/a', 'n/a', 'n/a', 'n/a', np.round(serial_lup,3), 1])
writer.column_styles=[
            Style(align='center') for _ in range(len(writer.headers))
]

writer.write_table()
