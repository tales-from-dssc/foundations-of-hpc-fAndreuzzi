#include <iostream>
#include <cstdlib>
#include <mpi.h>

#define N 10

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int packet = N/size;

	int values[N]{0};
	if (rank == 0) {
		std::cout << "Master extracted: ";
		for (int i = 0; i < N; i++) {
			int value = rand() % 100;
			values[i] = value;
			std::cout << values[i] << " ";
		}
		std::cout << std::endl;
	}

	int random_values[packet]{0};
	MPI_Scatter(values, packet, MPI_INT, random_values, packet, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "I'm " << rank << ", master extracted: ";

	for (int i = 0; i < packet; i++) std::cout << random_values[i] << " ";
	std::cout << std::endl;

	MPI_Finalize();
}
