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
#define BLOCK_SUM_TAG 2

template <typename T> class Matrix3D {
  T *elem;
  int dim1, dim2, dim3;

public:
  Matrix3D(int dim1, int dim2, int dim3) {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;

    elem = new T[dim1 * dim2 * dim3];
  }

  Matrix3D(Matrix3D &&that) {
    dim1 = that.dim1;
    dim2 = that.dim2;
    dim3 = that.dim3;

    elem = that.elem;
    that.elem = nullptr;
  }

  Matrix3D &operator=(Matrix3D &&that) {
    delete[] elem;

    elem = that.elem;
    that.elem = nullptr;

    dim1 = that.dim1;
    dim2 = that.dim2;
    dim3 = that.dim3;

    return *this;
  }

  // disable copy semantics
  Matrix3D(const Matrix3D &) = delete;
  Matrix3D &operator=(const Matrix3D &) = delete;

  ~Matrix3D() { delete[] elem; }
};

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

void send_matrix(Matrix<double, 3> matrix, int send_to, int tag, MPI_Comm comm,
                 int n_cells, MPI_Request *req) {
  if (n_cells == -1)
    n_cells = matrix.dim1() * matrix.dim2() * matrix.dim3();
  MPI_Isend(matrix.data(), n_cells, MPI_DOUBLE, send_to, tag, comm, req);
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
          send_matrix(blk, send_to, tags.at(w), comm, block_n_cells,
                      &requests[request_idx++]);
        }
      }
    }
  }

#ifdef DEBUG
  std::cout << "Freeing MPI_Request(s)" << std::endl;
#endif
  delete[] requests;
}

/*
        Receive a vector of matrices from sending_process. The number
        of expected matrices is equal to the number of tags. This
        function is blocking.
*/
std::vector<Matrix<double, 3>> receive_matrix(const int *matrix_size,
                                              const MPI_Comm comm,
                                              int sending_process,
                                              std::vector<int> tags) {
  int matrix_n_cells = matrix_size[0] * matrix_size[1] * matrix_size[2];

  std::vector<Matrix<double, 3>> matrices;
  for (int i = 0; i < tags.size(); i++) {
    Matrix<double, 3> m(matrix_size[0], matrix_size[1], matrix_size[2]);
    MPI_Status status;
    MPI_Recv(m.data(), matrix_n_cells, MPI_DOUBLE, sending_process, tags.at(i),
             comm, &status);
    matrices.push_back(m);
  }
  return matrices;
}

/*
        Receives asynchronously blocks from the processes and composes
        the complete matrix. This is a blocking function since it waits
        for the reception of all the pieces.
*/
void receive_compose_matrix(Matrix<double, 3> &dest, const int *block_size,
                            const MPI_Comm comm) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Request *requests = new MPI_Request[size];

  std::vector<Matrix<double, 3>> blocks;

  int top_left_corner[3];
  int process_coords[3];
  int request_idx = 0;

  Index matrix_size[]{dest.dim1(), dest.dim2(), dest.dim3()};

  int receive_from;
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

        Matrix<double, 3> blk = block(dest, block_size, top_left_corner);
        blocks.push_back(blk);

        MPI_Cart_rank(comm, process_coords, &receive_from);
        MPI_Irecv(blk.data(), block_n_cells, MPI_DOUBLE, receive_from,
                  BLOCK_SUM_TAG, comm, &requests[request_idx++]);
      }
    }
  }

#ifdef DEBUG
  std::cout << "Starting waiting for responses" << std::endl;
#endif

  MPI_Waitall(size, requests, MPI_STATUSES_IGNORE);
  delete[] requests;

#ifdef DEBUG
  std::cout << "Everyone sent the sum!" << std::endl;
#endif
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
      receive_matrix(blocks_size, cartesian_communicator, 0, tags);

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

  MPI_Request send_req;
  send_matrix(blks.at(0) + blks.at(1), 0, BLOCK_SUM_TAG, cartesian_communicator,
              -1, &send_req);

  if (rank == 0) {
    Matrix<double, 3> result(matrix_size[0], matrix_size[1], matrix_size[2]);
    receive_compose_matrix(result, blocks_size, cartesian_communicator);
    std::cout << "Result ------------------------" << std::endl << result;
  }

  MPI_Finalize();
}
