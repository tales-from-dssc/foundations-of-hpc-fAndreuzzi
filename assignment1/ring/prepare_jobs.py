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
#PBS -o /u/dssc/fandreuz/HPC/assignment1/ring/results/ring$P.txt
#PBS -l select=1:ncpus=$P:mpiprocs=$P

cd /u/dssc/fandreuz/HPC/assignment1/ring

module load openmpi/4.0.3/gnu/9.3.0
mpic++ -D TIME_ONLY ring.cpp

mpirun -np $P --map-by core a.out

echo DONE

mpirun -np $P --map-by socket a.out

echo DONE
"""

for i in range(n_min,n_max+1):
	t = Template(template)
	text = t.substitute({'P': i, 'walltime': '00:30:00'})

	text_file = open("ring{}.pbs".format(i), "w")
	text_file.write(text)
	text_file.close()
