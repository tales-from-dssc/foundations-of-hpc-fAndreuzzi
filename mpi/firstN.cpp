#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
        int rank,size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

	int root = 0;

	int N = atoi(argv[1]);
	int nums_this_node = N / size; int remainder = N%size;
	if (remainder != 0 && rank < remainder) nums_this_node++;

	std::cout << "Processing " << nums_this_node << " numbers on process " << rank << std::endl;

	// fill send buffer
	int send_buffer = 0;
	for (int i = 0; i < nums_this_node; i++) send_buffer += rank + size*i;

	std::cout << "Filled array at processor " << rank << " with value " << send_buffer << std::endl;

	int receive_buffer = 0;

	MPI_Reduce(&send_buffer, &receive_buffer, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

	if (rank == root) std::cout << receive_buffer << std::endl;
	MPI_Finalize();
}
