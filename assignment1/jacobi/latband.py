from subprocess import Popen, getstatusoutput, PIPE

def find_latency_bandwidth(third, fourth):
	cmd = 'python3 ../benchmark/fit.py nintel infiniband {} {}'.format(third, fourth)
	print(cmd)
	result = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
	output = iter(result.stdout.readline, b'')

	buff = []
	for line in output:
        	buff.append(float(line))
	latency = buff[0] * 1.e-6
	bandwidth = buff[1]
	return latency, bandwidth
