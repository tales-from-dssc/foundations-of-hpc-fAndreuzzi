import numpy as np
import re
import glob

pttrn = 'JacobiMa\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*Residual\s*(\d+\.\d+)\s*MLUPs\s*(\d+\.\d+)'
process_grid_pattern = 'grid\s+(\d+)\s+(\d+)\s+(\d+)'

file_list = ['serial_thin_gpu.txt']

def tuplify(match,n,func=None):
	if func is None:
		return tuple(match.group(i) for i in range(1,n+1))
	else:
		return tuple(func(match.group(i)) for i in range(1,n+1))

for filename in glob.glob("*.txt"):
	vls = []
	with open(filename) as file:
		lines = '\n'.join(list(file.readlines()))
		mtch = re.search(process_grid_pattern, lines)
		vls.append([*tuplify(mtch, 3, float), 0,0,0])
		for match in re.findall(pttrn, lines):
			vls.append(list(map(float, match)))
	vls = np.array(vls)
	np.save(filename[:-4] + ".npy", vls)
