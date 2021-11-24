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
Matrix<double, 3> block(const Matrix<T, 3> matrix,
                        const std::tuple<int, int, int> block_size,
                        const std::tuple<int, int, int> top_left_corner) {
  Matrix<double, 3> data(std::get<0>(block_size), std::get<1>(block_size),
                         std::get<2>(block_size));

  int slice, row, column;
  for (int i = 0; i < std::get<0>(block_size); i++) {
    slice = std::get<0>(top_left_corner) + i;
    for (int j = 0; j < std::get<1>(block_size); j++) {
      row = std::get<1>(top_left_corner) + j;
      for (int k = 0; k < std::get<2>(block_size); k++) {
        column = std::get<2>(top_left_corner) + k;
        data(i, j, k) = matrix(slice, row, column);
      }
    }
  }

  return data;
}

template <typename T>
std::vector<Matrix<double, 3>>
blockify(const Matrix<T, 3> matrix,
         const std::tuple<int, int, int> block_size) {
  std::vector<Matrix<double, 3>> vec;
  if (matrix.dim1() % std::get<0>(block_size) != 0 ||
      matrix.dim2() % std::get<1>(block_size) != 0 ||
      matrix.dim3() % std::get<2>(block_size) != 0)
    return vec;

  for (int i = 0; i < matrix.dim1(); i += std::get<0>(block_size))
    for (int j = 0; j < matrix.dim2(); j += std::get<1>(block_size))
      for (int k = 0; k < matrix.dim3(); k += std::get<2>(block_size))
        vec.push_back(block(matrix, block_size, std::make_tuple(i, j, k)));
  return vec;
}

template <typename T>
void bcast_matrix(const MPI_Comm &communicator, const Matrix<T, 3> matrix,
                  const std::tuple<int, int, int> blocks_size) {}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int processors_distribution[] = {0,0,0};
  MPI_Dims_create(size, 3, processors_distribution);

  int periodic = 0;
  int reorder = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(MPI_COMM_WORLD, 3, processors_distribution, &periodic,
                  reorder, &cartesian_communicator);

  int rank;
  MPI_Comm_rank(cartesian_communicator, &rank);

  if (rank == 0) {
    int matrix_size[3];
    for (int i = 0; i < 3; i++)
      matrix_size[i] = atoi(argv[i + 1]);

    auto m = random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2]);
    std::cout << m << std::endl;

    int blocks_size[3];
    for (int i = 0; i < 3; i++)
      blocks_size[i] = atoi(argv[3 + i + 1]);

    std::vector<Matrix<double, 3>> ms = blockify(
        m, std::make_tuple(blocks_size[0], blocks_size[1], blocks_size[2]));
    for (int i = 0; i < ms.size(); i++)
      std::cout << ms[i] << std::endl;
  }

  MPI_Finalize();
}
