import numpy as np
from string import Template
import sys
from scatter import latband
from cmds import cmds_dict
from nodes import nodes_list

filename = sys.argv.pop(1)

file_template = """
#header_line 1: $command
#header_line 2: $nodes
#header_line 3: $fitparams
#header: #bytes #repetitions      t[usec]   Mbytes/sec    t[usec] computed   Mbytes/sec (computed )
$rows
"""

row_template = """    {}      {}             {}   {}         {}          {}"""

# read the results file and save means
r = np.load('results/' + filename)
s = np.array(np.split(r, r.shape[0]//30))
s = np.mean(s, axis=0)

t = Template(file_template)

# fit latency and bandwidth
_, lat, band = latband(filename)
lat = np.round(lat, 3)
band = np.round(band, 3)

rows = []
for row_values in s:
	bytes = str(int(row_values[0])).rjust(9)
	reps = str(int(row_values[1])).ljust(4)
	time = str(np.round(row_values[2], 2)).ljust(8)
	bd = str(np.round(row_values[3], 2)).ljust(8)

	t_computed = lat + (pow(2,-20)*row_values[0] / band)*1.e6
	t_computed = np.round(t_computed, 2)

	b_computed = pow(2,-20)*row_values[0] / (t_computed*1.e-6)
	b_computed = np.round(b_computed, 2)

	t_computed = str(t_computed).ljust(8)
	b_computed = str(b_computed).ljust(8)

	rows.append(row_template.format(bytes, reps, time, bd, t_computed, b_computed))

command = cmds_dict[tuple(filename[:-4].split('_'))]
nodes = ', '.join(nodes_list(filename))

text = t.substitute(command=command, nodes=nodes, fitparams = 'latency={} us, bandwidth={} MByte/s'.format(lat, band), rows='\n'.join(rows))
print(text)
