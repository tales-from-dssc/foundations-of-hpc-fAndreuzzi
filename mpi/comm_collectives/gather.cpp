#include <iostream>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//call rand rank times to simulate a random seed
	for (int i = 0; i < rank; i++) rand();
	int value = rand() % 100;
	std::cout << rank << " extracted: " << value << std::endl;

	int random_values[size]{0};
	MPI_Gather(&value, 1, MPI_INT, random_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::cout << "I'm rank " << rank << std::endl;
	for (int i = 0; i < size; i++) std::cout << random_values[i] << " ";
	std::cout << std::endl;

	MPI_Finalize();
}
