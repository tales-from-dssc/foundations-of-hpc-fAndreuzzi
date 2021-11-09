
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
        int rank,size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

	int periodic = 1;
	MPI_Comm cartesian_communicator;
	MPI_Cart_create(MPI_COMM_WORLD, 1, &size, &periodic, 0, &cartesian_communicator);

	int rank_after;
	MPI_Comm_rank(cartesian_communicator, &rank_after);

	std::cout << "Rank before: " << rank << ", rank after: " << rank_after << std::endl;

	int left, right;
	MPI_Cart_shift(cartesian_communicator, 0, 1, &left, &right);
	std::cout << left << " - " << rank_after << " - " << right << std::endl;

	MPI_Finalize();
}
