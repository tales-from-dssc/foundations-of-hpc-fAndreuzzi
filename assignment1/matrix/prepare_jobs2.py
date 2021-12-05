from string import Template

template = """
#!/bin/bash
### Job Name
#PBS -N matrix${nproc}
### Project code
#PBS -l walltime=00:30:00
#PBS -q dssc
#PBS -o /u/dssc/fandreuz/HPC/assignment1/matrix/results/matrix${nproc}.txt
#PBS -e /u/dssc/fandreuz/HPC/assignment1/matrix/errors/matrix${nproc}.txt
#PBS -l nodes=1:ppn=$nproc

cd /u/dssc/fandreuz/HPC/assignment1/matrix

module load openmpi/4.0.3/gnu/9.3.0

mpic++ -O3 matrix_sum.cpp
mpirun -np $nproc --map-by core --report-bindings a.out 2400 100 100
echo DONE
mpirun -np $nproc --map-by socket --report-bindings a.out 2400 100 100
"""

for i in range(1,25):
	t = Template(template)
	text = t.substitute(nproc=i)
	
	text_file = open("matrix{}.pbs".format(i), "w")
	text_file.write(text)
	text_file.close()
