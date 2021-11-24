#include "Matrix.h"
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>
using namespace Numeric_lib;

Matrix<double, 3> random_3d_matrix(int dim1, int dim2, int dim3) {
  Matrix<double, 3> m(dim1, dim2, dim3);

  std::default_random_engine ran{};
  std::uniform_real_distribution<> ureal{-10, 10};

  for (Index i = 0; i < dim1; ++i)
    for (Index j = 0; j < dim2; ++j)
      for (Index k = 0; k < dim3; ++k)
        m(i, j, k) = ureal(ran);

  return m;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T, 3> &p) {
  os << "Shape=(" << p.dim1() << ", " << p.dim2() << ", " << p.dim3() << ")"
     << std::endl;
  for (Index i = 0; i < p.dim1(); ++i) {
    std::cout << "Slice N. " << i << std::endl;
    for (Index j = 0; j < p.dim2(); ++j) {
      for (Index k = 0; k < p.dim3(); ++k)
        std::cout << p(i, j, k) << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  return os;
}

template <typename T>
Matrix<T, 3> block(const Matrix<T, 3> matrix, const int *block_size,
                   const int *top_left_corner) {
  Matrix<T, 3> data(block_size[0], block_size[1], block_size[2]);

  int slice, row, column;
  for (int i = 0; i < block_size[0]; i++) {
    slice = top_left_corner[0] + i;
    for (int j = 0; j < block_size[1]; j++) {
      row = top_left_corner[1] + j;
      for (int k = 0; k < block_size[2]; k++) {
        column = top_left_corner[2] + k;
        data(i, j, k) = matrix(slice, row, column);
      }
    }
  }

  return data;
}

void blockify_and_msg(const Matrix<double, 3> matrix, const int *block_size,
                      const MPI_Comm comm) {
#ifdef DEBUG
  std::cout << "blockify_and_msg called" << std::endl;
#endif
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Request *requests = new MPI_Request[size];

  int top_left_corner[3];
  int process_coords[3];
  int request_idx = 0;

  int send_to;
  for (process_coords[0] = 0; process_coords[0] < matrix.dim1() / block_size[0];
       ++process_coords[0]) {
    top_left_corner[0] = process_coords[0] * block_size[0];
    for (process_coords[1] = 0;
         process_coords[1] < matrix.dim2() / block_size[1];
         ++process_coords[1]) {
      top_left_corner[1] = process_coords[1] * block_size[1];
      for (process_coords[2] = 0;
           process_coords[2] < matrix.dim3() / block_size[2];
           ++process_coords[2]) {
        top_left_corner[2] = process_coords[2] * block_size[2];

#ifdef DEBUG
        std::cout << "(" << process_coords[0] << "," << process_coords[1] << ","
                  << process_coords[2] << ")" << std::endl;
#endif
        MPI_Cart_rank(comm, process_coords, &send_to);
#ifdef DEBUG
        std::cout << "sending to " << send_to << std::endl;
#endif

        Matrix<double, 3> blk = block(matrix, block_size, top_left_corner);
        MPI_Isend(blk.data(), block_n_cells, MPI_DOUBLE, send_to, 0, comm,
                  &requests[request_idx++]);
      }
    }
  }

#ifdef DEBUG
  std::cout << "Freeing MPI_Request(s)" << std::endl;
#endif
  delete[] requests;
}

Matrix<double, 3> receive_block(const int *block_size, const MPI_Comm comm,
                                int root_process) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];
  Matrix<double, 3> blk(block_size[0], block_size[1], block_size[2]);
  MPI_Status status;
  MPI_Recv(blk.data(), block_n_cells, MPI_DOUBLE, root_process, MPI_ANY_TAG,
           comm, &status);
  return blk;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int matrix_size[3];
  int blocks_size[3];
  for (int i = 0; i < 3; i++) {
    matrix_size[i] = atoi(argv[i + 1]);
    blocks_size[i] = atoi(argv[3 + i + 1]);
    // if the dimension of the blocks is not covering the matrix exactly on
    // the axis we augment the dimension along that axis
    int residual = matrix_size[i] % blocks_size[i];
    matrix_size[i] += (residual != 0) * (blocks_size[i] - residual);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // verify that the number of MPI processes is enough
  int product = 1;
  for (int i = 0; i < 3; i++)
    product *= matrix_size[i] / blocks_size[i];
  if (product != size) {
    if (rank == 0)
      std::cout << "Invalid number of MPI processes" << std::endl;
    MPI_Finalize();
    return 1;
  }

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
  MPI_Cart_create(MPI_COMM_WORLD, 3, processors_distribution, periodic, reorder,
                  &cartesian_communicator);

  MPI_Comm_rank(cartesian_communicator, &rank);

  if (rank == 0) {
    for (int i = 0; i < 3; i++)
      std::cout << "Matrix.shape[" << i << "] = " << matrix_size[i]
                << std::endl;

    Matrix<double, 3> matrix =
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2]);
    std::cout << matrix << std::endl;

    blockify_and_msg(matrix, blocks_size, cartesian_communicator);
  }

#ifdef DEBUG
  std::cout << "I'm process " << rank << std::endl;
#endif
  Matrix<double, 3> blk = receive_block(blocks_size, cartesian_communicator, 0);

#ifdef DEBUG
  std::cout << rank << " received "
            << "(" << blk.dim1() << "," << blk.dim2() << "," << blk.dim3()
            << ")" << std::endl;

  if (rank == 0) {
    std::cout << blk << std::endl;
  }
#endif

  MPI_Finalize();
}
