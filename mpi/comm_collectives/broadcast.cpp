#include <iostream>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int value;
	if (rank == 0) {
		value = rand() % 100;
		std::cout << "I'm master, extracted " << value << std::endl;
	}
	MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "I'm " << rank << ", master extracted " << value << std::endl;

	MPI_Finalize();
}
