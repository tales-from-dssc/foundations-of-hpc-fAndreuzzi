#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
	int rank,size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int buffer[2];	
	if (rank < size-1) {
		MPI_Status status;
		MPI_Recv(buffer, 2, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		std::cout << "Received from " << rank+1 << std::endl;
	} else {
		buffer[0] = std::stoi(argv[1]);
		buffer[1] = std::stoi(argv[2]);
		std::cout << "Received arguments " << buffer[0] << ", " << buffer[1] << std::endl;
	}

	buffer[0] = buffer[0] * buffer[0];
	buffer[1] = buffer[1] * buffer[1];

	std::cout << "After the transformation occured in process " << rank << " data is: " << buffer[0] << ", " << buffer[1] << std::endl;

	if (rank != 0) MPI_Send(buffer, 2, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
	else std::cout << "Result: " << buffer[0] << ", " << buffer[1] << std::endl;

	MPI_Finalize();
}	
