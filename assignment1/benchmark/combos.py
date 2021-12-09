import itertools
import sys

# does not produce any .pbs file if called with argument "0"

pbs_script = """
#!/bin/bash
### Job Name
#PBS -N {}
### Project code
#PBS -l walltime=01:00:00
#PBS -q dssc{}
#PBS -o /u/dssc/fandreuz/HPC/assignment1/benchmark/results/{}.txt
#PBS -e /u/dssc/fandreuz/HPC/assignment1/benchmark/errors/{}.err
#PBS -l nodes={}:ppn={}

cd /u/dssc/fandreuz/mpi-benchmarks/src_c
module load {}

make clean
make

for i in {{1..100}}; do {}; done
"""

network = ["infiniband", "gigabit"]
map_bys = ["core", "node", "socket"]
intels = ["intel", "nintel"]
gpu = ["gpu", "ngpu"]

N = 28


def cmd(net, mb, intel, gp):
    if intel == "nintel":
        network_params = (
            "--mca pml ucx"
            if net == "infiniband"
            else "--mca pml ob1 --mca btl tcp,self"
        )
        return "mpirun -np 2 {} --report-bindings {} IMB-MPI1 PingPong -msglog {}".format(
            network_params,
            map_by(mb, intel),
            N,
        )
    else:
        network_params = "mlx" if net == "infiniband" else "tcp"
        return "mpirun -np 2 -print-rank-map -genv I_MPI_FABRICS {} -env I_MPI_DEBUG 5 {} IMB-MPI1 PingPong -msglog {}".format(
            network_params,
            map_by(mb, intel),
            N,
        )


def load_module(intel):
    if intel == 'intel':
        return "intel"
    else:
        return "openmpi/4.0.3/gnu/9.3.0"


def map_by(mb, intel):
    if intel == 'intel':
        if mb == "core":
            return "-genv I_MPI_PIN_PROCESSOR_LIST 0,2"
        elif mb == "socket":
            return "-genv I_MPI_PIN_PROCESSOR_LIST 0,1"
        else:
            return ""
    else:
        return "--map-by " + mb


def nnodes(mb):
    if mb == "node":
        return 2
    else:
        return 1


def nprocs(mb):
    if mb == "node":
        return 1
    else:
        return 2


def summary(pml, btl, mb, intel):
    if intel == 'intel':
        return "IntelMPI, map-by {}, "


for intel in intels:
    for net in network:
        for mb in map_bys:
            for gp in gpu:
                jobname = "{}_{}_{}_{}".format(intel, net, mb, gp)

                content = pbs_script.format(
                    jobname,
                    "_gpu" if gp == 'gpu' else "",
                    jobname, jobname,
                    nnodes(mb),
                    nprocs(mb),
                    load_module(intel),
                    cmd(net, mb, intel, gp),
                )

                print("{},{},{},{}:{}".format(intel, net, mb, gp, cmd(net, mb, intel, gp)))

                if len(sys.argv) < 2 or sys.argv[1] != '0':
                    text_file = open(
                        "{}_{}_{}_{}.pbs".format(intel, net, mb, gp), "w"
                    )
                    text_file.write(content)
                    text_file.close()
