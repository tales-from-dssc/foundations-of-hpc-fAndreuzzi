#include "matrix3d.h"
#include <iostream>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

std::vector<Matrix3D<double>>
scatter(std::vector<BlockifiedMatrix3D<double>> &matrices,
        const int *block_size, const MPI_Comm &comm) {
#ifdef DEBUG
  std::cout << "scatter called" << std::endl;
#endif
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int rank;
  MPI_Comm_rank(comm, &rank);

  int top_left_corner[3];
  int process_coords[3];

  // broadcast the expected number of blocks to be received
  int n_matrices = 0;
  if (rank == 0)
    n_matrices = matrices.size();
  MPI_Bcast(&n_matrices, 1, MPI_INT, 0, comm);

  double *receive_buffer = new double[n_matrices * block_n_cells];
  for (int i = 0; i < n_matrices; i++) {
    double *buffer = nullptr;
    if (rank == 0)
      buffer = matrices.at(i).data();
    MPI_Scatter(buffer, block_n_cells, MPI_DOUBLE,
                receive_buffer + i * block_n_cells, block_n_cells, MPI_DOUBLE,
                0, comm);
  }

  // do NOT delete this, this is used inside Matrix3D as elem
  // delete[] receive_buffer;

  std::vector<Matrix3D<double>> vec;
  for (int i = 0; i < n_matrices; ++i) {
    Matrix3D<double> blk(block_size, receive_buffer + i * block_n_cells);
    vec.push_back(std::move(blk));
  }
  return vec;
}

/*
        Receives summed blocks from the other processes via MPI_Scatter, and
        then composes the complete result using also the cartesian topology.

        This is a blocking function since it waits for the reception of all the
        pieces, and then composes the matrix.

        Non-root processes can pass anything in dest, for instance the same
   matrix passed in send.
*/
void receive_compose_matrix(BlockifiedMatrix3D<double> &dest,
                            Matrix3D<double> &send, const int *block_size,
                            const MPI_Comm &comm) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int matrix_size[]{dest.dim(0), dest.dim(1), dest.dim(2)};

#ifdef DEBUG
  std::cout << "Starting waiting for responses" << std::endl;
#endif

  MPI_Gather(send.data(), block_n_cells, MPI_DOUBLE, dest.data(), block_n_cells,
             MPI_DOUBLE, 0, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

#ifdef DEBUG
  if (rank == 0)
    std::cout << "Everyone sent the sum!" << std::endl;
#endif
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int matrix_size[3];
  int blocks_size[3];
  // number of blocks along the axis
  int n_blocks[3];
  int residual = 0;
  for (int i = 0; i < 3; i++) {
    matrix_size[i] = atoi(argv[i + 1]);
    blocks_size[i] = atoi(argv[3 + i + 1]);
    // if the dimension of the blocks is not covering the matrix exactly on
    // the axis we augment the dimension along that axis
    n_blocks[i] = matrix_size[i] / blocks_size[i];

    residual = matrix_size[i] % blocks_size[i];

    matrix_size[i] += (residual != 0) * (blocks_size[i] - residual);
    n_blocks[i] += (residual != 0) * 1;
#ifdef DEBUG
    if (rank == 0 && residual != 0)
      std::cout << "Matrix augmented by " << blocks_size[i] - residual
                << " cells along the " << i << "-th axis ("
                << matrix_size[i] - (blocks_size[i] - residual) << "->"
                << matrix_size[i] << ")" << std::endl;
#endif
  }

  MPI_Comm comm = MPI_COMM_WORLD;
  int expected_n_processes = n_blocks[0] * n_blocks[1] * n_blocks[2];
  if (expected_n_processes > size) {
    if (rank == 0)
      std::cout << "Invalid number of MPI processes, expected "
                << expected_n_processes << ", found " << size << std::endl;
    MPI_Finalize();
    return 1;
  } else if (expected_n_processes < size) {
    // we drop unneeded processes
    MPI_Comm_split(MPI_COMM_WORLD, rank < expected_n_processes, 0, &comm);

    if (rank == 0) {
      std::cout << "Dropped " << size - expected_n_processes
                << " process due to the dimension of the blocks" << std::endl;
    }

    if (rank >= expected_n_processes) {
      std::cout << "Process " << rank << " is going to sleep" << std::endl;
      MPI_Finalize();
      return 0;
    }
  }

#ifdef MPI_DEBUG
  if (rank == 0) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "rank " << rank << " -> PID " << getpid() << " on " << hostname
              << " ready for attach" << std::endl;
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
#endif

  int processors_distribution[3];
  for (int i = 0; i < 3; i++)
    processors_distribution[i] = matrix_size[i] / blocks_size[i];
#ifdef DEBUG
  if (rank == 0)
    std::cout << "Processor distribution: " << processors_distribution[0]
              << ", " << processors_distribution[1] << ", "
              << processors_distribution[2] << std::endl;
#endif

  int periodic[]{0, 0, 0};
  int reorder = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(comm, 3, processors_distribution, periodic, reorder,
                  &cartesian_communicator);

  MPI_Comm_rank(cartesian_communicator, &rank);

  std::vector<BlockifiedMatrix3D<double>> matrices;

  if (rank == 0) {
#ifdef DEBUG
    for (int i = 0; i < 3; i++)
      std::cout << "Matrix.shape[" << i << "] = " << matrix_size[i]
                << std::endl;
#endif

    std::default_random_engine ran{};

    BlockifiedMatrix3D<double> matrix1 = std::move(
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran));
    matrix1.set_block_size(blocks_size);
    BlockifiedMatrix3D<double> matrix2 = std::move(
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran));
    matrix2.set_block_size(blocks_size);
#ifdef DEBUG
    std::cout << matrix1 << std::endl;
    std::cout << matrix2 << std::endl;
#endif

    matrices.push_back(std::move(matrix1));
    matrices.push_back(std::move(matrix2));
  }

  // after generating the matrices (and before communicating them to the
  // other processes) we take the starting time
  double start_time = MPI_Wtime();

  // store the block obtained via scatter
  std::vector<Matrix3D<double>> blocks =
      std::move(scatter(matrices, blocks_size, cartesian_communicator));

#ifdef DEBUG
  std::cout << "I'm process " << rank << std::endl;
  std::cout << rank << " received "
            << "(" << blocks.at(0).dim(0) << "," << blocks.at(0).dim(1) << ","
            << blocks.at(0).dim(2) << ")" << std::endl;

  if (rank == 1) {
    std::cout << std::endl << "rank " << rank << " received: " << std::endl;
    std::cout << "Block 1 ------------------------" << std::endl;
    std::cout << blocks.at(0) << std::endl;
    std::cout << "Block 2 ------------------------" << std::endl;
    std::cout << blocks.at(1) << std::endl;
  }
#endif

  Matrix3D<double> m1(std::move(blocks.at(0)));
  Matrix3D<double> m2(std::move(blocks.at(1)));
  Matrix3D<double> summed_block = m1 + m2;

  // m2.elem and m1.elem are a single array, or two contiguous pointers. calling
  // delete m1 deletes also the part of the array dedicated to m2, therefore we
  // do not need (and cannot) call the destructor of m2 on the memory address
  m2.unbind_data();

  BlockifiedMatrix3D<double> result;

  if (rank == 0) {
    result = std::move(BlockifiedMatrix3D<double>(
        matrix_size[0], matrix_size[1], matrix_size[2]));
    result.set_block_size(blocks_size);
  }

  receive_compose_matrix(result, summed_block, blocks_size,
                         cartesian_communicator);

  if (rank == 0) {
#ifdef DEBUG
    std::cout << "Result ------------------------" << std::endl << result;
#endif

    std::cout << MPI_Wtime() - start_time << std::endl;
  }

  MPI_Finalize();
}
