from subprocess import Popen, getstatusoutput, PIPE

result = Popen('python3 combos.py', shell=True, stdout=PIPE, stderr=PIPE)
output = iter(result.stdout.readline, b'')

carr = []
for line in output:
	carr.append(line.decode("utf-8").split(':'))

cmds_dict = {
	tuple(key.strip().split(',')): value.strip()
	for key,value in carr
}

