from string import Template
import sys

n_max = int(sys.argv[1])

async = int(sys.argv[2])
async_flag = ''
if async == 1:
	async_flag = '-D async'

template = """
#!/bin/bash
### Job Name
#PBS -N ring$P
### Project code
#PBS -l walltime=$walltime
#PBS -q dssc
### Merge output and error files
#PBS -o /u/dssc/fandreuz/HPC/assignment1/ring$P.txt
#PBS -l select=1:ncpus=$P:mpiprocs=$P
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M andreuzzi.francesco@gmail.com

cd /u/dssc/fandreuz/HPC/assignment1

module load openmpi/4.0.3/gnu/9.3.0

mpic++ $async ring.cpp

### Run the executable
for n in {{2..1000}};
do mpirun -np $P a.out && echo "T# STOP";
done;
"""

for i in range(2,n_max+1):
	t = Template(template)
	text = t.substitute({'P': i, 'walltime': '00:30:00', 'async': async_flag})

	text_file = open("ring{}.pbs".format(i), "w")
	text_file.write(text)
	text_file.close()
