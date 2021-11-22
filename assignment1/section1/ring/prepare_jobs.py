from string import Template
import sys

if len(sys.argv) >= 3:
	n_min = int(sys.argv[1])
	n_max = int(sys.argv[2])
else:
	n_min = 2
	n_max = int(sys.argv[1])

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

declare -a opt_flags=("-O0" "-O1" "-O2" "-O3" "-O3 -march=native" )

for opt in $${opt_flags[@]}; do
	mpic++ $$opt -D TIME_ONLY ring.cpp
	mpirun -np $P a.out
	echo "STOP"
done
"""

for i in range(n_min,n_max+1):
	t = Template(template)
	text = t.substitute({'P': i, 'walltime': '00:30:00'})

	text_file = open("ring{}.pbs".format(i), "w")
	text_file.write(text)
	text_file.close()
