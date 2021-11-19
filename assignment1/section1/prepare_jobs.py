import sys

n_max = int(sys.argv[1])

template = """
#!/bin/bash
### Job Name
#PBS -N ring
### Project code
#PBS -l walltime=01:00:00
#PBS -q dssc
### Merge output and error files
#PBS -o /u/dssc/fandreuz/HPC/assignment1/ring{}.txt
#PBS -l select=1:ncpus={}:mpiprocs={}
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M andreuzzi.francesco@gmail.com

cd /u/dssc/fandreuz/HPC/assignment1

module load openmpi/4.0.3/gnu/9.3.0

### Run the executable
for n in {{2..1000}};
do mpirun -np {} a.out && echo "T# STOP";
done;
"""

for i in range(2,n_max+1):
	text_file = open("ring{}.pbs".format(i), "w")
	text_file.write(template.format(i,i,i,i))
	text_file.close()
