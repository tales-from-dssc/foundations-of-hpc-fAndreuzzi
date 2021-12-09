import sys
import re

def nodes_list(filename):
	if 'nintel' in filename:
		filename = 'errors/' + filename[:-4] + ".err"
	else:
		filename = 'results_txt/' + filename[:-4] + ".txt"

	with open(filename) as file:
		lines = file.readlines()
		lines = [line.rstrip() for line in lines]
		lines = '\n'.join(lines)

		p = re.compile(r'ct1p(?:g|t)-(?:g|t)node\d{3}')
		matches = set(p.findall(lines))
		return matches

if __name__ == "__main__":
	print(nodes_list(sys.argv[1]))
