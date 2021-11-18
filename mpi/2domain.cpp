#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
        int rank,size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

	int periodic[] = {atoi(argv[1]), atoi(argv[2])};
	int first_dim = atoi(argv[3]);
	int cart_size[] = {first_dim, size/first_dim};

	std::cout << "Topology: " << cart_size[0] << "x" << cart_size[1] << std::endl;

	MPI_Comm cartesian_communicator;
	MPI_Cart_create(MPI_COMM_WORLD, 2, cart_size, periodic, 0, &cartesian_communicator);

	int rank_after;
	MPI_Comm_rank(cartesian_communicator, &rank_after);

	int cart_coords[2];
	MPI_Cart_coords(cartesian_communicator, rank_after, 2, cart_coords);

	std::cout << "Rank before: " << rank << ", rank after: " << rank_after << " --> (" << cart_coords[0] << "," << cart_coords[1] << ")" << std::endl;

	MPI_Finalize();
}
