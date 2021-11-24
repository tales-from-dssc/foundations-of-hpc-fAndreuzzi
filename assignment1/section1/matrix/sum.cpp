#include "Matrix.h"
#include <iostream>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>
using namespace Numeric_lib;

#define BLOCK_1_TAG 0
#define BLOCK_2_TAG 1

Matrix<double, 3> random_3d_matrix(int dim1, int dim2, int dim3,
                                   std::default_random_engine &ran) {
  Matrix<double, 3> m(dim1, dim2, dim3);

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

Matrix<double, 3> operator+(const Matrix<double, 3> matrix1,
                            const Matrix<double, 3> matrix2) {
  if (matrix1.dim1() != matrix2.dim2() || matrix1.dim2() != matrix2.dim2() ||
      matrix1.dim3() != matrix2.dim3())
    throw std::invalid_argument(
        "Summation requires that the matrix have the same shape");

  Index dims[]{matrix1.dim1(), matrix1.dim2(), matrix1.dim3()};

  Matrix<double, 3> sum(dims[0], dims[1], dims[2]);
  for (int i = 0; i < dims[0]; i++)
    for (int j = 0; j < dims[1]; j++)
      for (int k = 0; k < dims[2]; k++)
        sum(i, j, k) = matrix1(i, j, k) + matrix2(i, j, k);

  return sum;
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

void blockify_and_msg(std::vector<Matrix<double, 3>> matrices,
                      std::vector<int> tags, const int *block_size,
                      const MPI_Comm comm) {
#ifdef DEBUG
  std::cout << "blockify_and_msg called" << std::endl;
#endif
  if (matrices.size() != tags.size())
    throw std::invalid_argument("n. of matrices != n. of tags");

  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Request *requests = new MPI_Request[size * tags.size()];

  int top_left_corner[3];
  int process_coords[3];
  int request_idx = 0;

  Index matrix_size[]{matrices.at(0).dim1(), matrices.at(0).dim2(),
                      matrices.at(0).dim3()};

  int send_to;
  for (process_coords[0] = 0;
       process_coords[0] < matrix_size[0] / block_size[0];
       ++process_coords[0]) {
    top_left_corner[0] = process_coords[0] * block_size[0];
    for (process_coords[1] = 0;
         process_coords[1] < matrix_size[1] / block_size[1];
         ++process_coords[1]) {
      top_left_corner[1] = process_coords[1] * block_size[1];
      for (process_coords[2] = 0;
           process_coords[2] < matrix_size[2] / block_size[2];
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

        for (int w = 0; w < matrices.size(); w++) {
          Matrix<double, 3> blk =
              block(matrices.at(w), block_size, top_left_corner);
          MPI_Isend(blk.data(), block_n_cells, MPI_DOUBLE, send_to, tags.at(w),
                    comm, &requests[request_idx++]);
        }
      }
    }
  }

#ifdef DEBUG
  std::cout << "Freeing MPI_Request(s)" << std::endl;
#endif
  delete[] requests;
}

std::vector<Matrix<double, 3>> receive_block(const int *block_size,
                                             const MPI_Comm comm,
                                             int root_process,
                                             std::vector<int> tags) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  std::vector<Matrix<double, 3>> matrices;
  for (int i = 0; i < tags.size(); i++) {
    Matrix<double, 3> blk(block_size[0], block_size[1], block_size[2]);
    MPI_Status status;
    MPI_Recv(blk.data(), block_n_cells, MPI_DOUBLE, root_process, tags.at(i),
             comm, &status);
    matrices.push_back(blk);
  }
  return matrices;
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

  /*  if (rank == 0) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      std::cout << "rank " << rank << " -> PID " << getpid() << " on " <<
    hostname
                << " ready for attach" << std::endl;
      fflush(stdout);
      while (0 == i)
        sleep(5);
    }
  */

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

  std::vector<int> tags{BLOCK_1_TAG, BLOCK_2_TAG};
  if (rank == 0) {
    for (int i = 0; i < 3; i++)
      std::cout << "Matrix.shape[" << i << "] = " << matrix_size[i]
                << std::endl;

    std::default_random_engine ran{};

    Matrix<double, 3> matrix1 =
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran);
    std::cout << matrix1 << std::endl;

    Matrix<double, 3> matrix2 =
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran);
    std::cout << matrix2 << std::endl;

    std::vector<Matrix<double, 3>> matrices{matrix1, matrix2};
    blockify_and_msg(matrices, tags, blocks_size, cartesian_communicator);
  }

#ifdef DEBUG
  std::cout << "I'm process " << rank << std::endl;
#endif
  std::vector<Matrix<double, 3>> blks =
      receive_block(blocks_size, cartesian_communicator, 0, tags);

#ifdef DEBUG
  std::cout << rank << " received "
            << "(" << blks.at(0).dim1() << "," << blks.at(0).dim2() << ","
            << blks.at(0).dim3() << ")" << std::endl;

  if (rank == 0) {
    std::cout << "Block 1 ------------------------" << std::endl;
    std::cout << blks.at(0) << std::endl;
    std::cout << "Block 2 ------------------------" << std::endl;
    std::cout << blks.at(1) << std::endl;
  }
#endif

  MPI_Finalize();
}
