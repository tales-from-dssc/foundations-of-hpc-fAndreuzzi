#include "matrix3d.h"
#include "processor_utils.h"

#include <iostream>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

/*
        Extract a block of the given size starting from the given top
        left corner from the given matrix, and place it in the given
        buffer starting at the given index (included).

        The matrix is inserted iterating on the row first, i.e. the
        dimension change in this order:
                1. column;
                2. row;
                3. slice

        We put a copy of the values of matrix into the buffer, therefore
        there is no link between the values in the buffer and the
        values in the matrix.

        Important: there must be at least 'starting_index+matrix.size()'
        slots in the buffer, otherwise a segmentation fault is extremely
        likely.
*/
template <typename T>
void block(const Matrix3D<T> &matrix, const int *block_size,
           const int *top_left_corner, T *buffer) {
  int slice, row, column;
  int idx = 0;
  for (int i = 0; i < block_size[0]; i++) {
    slice = top_left_corner[0] + i;
    for (int j = 0; j < block_size[1]; j++) {
      row = top_left_corner[1] + j;
      for (int k = 0; k < block_size[2]; k++) {
        column = top_left_corner[2] + k;
        buffer[idx++] = matrix(slice, row, column);
      }
    }
  }
}

/*
        Split the given matrix(es) in blocks (using the given dimension) along
        the 3 axes. Then send it to the appropriate processes using the given
        (cartesian) communicator. The position of the process which receives a
        certain block is determined by its position in the virtual topology.
        This is also useful in order to reconstruct the matrix afterwards.

        Each matrix is sent with the corresponding tag in 'tags'. Therefore the
        size of the vectors 'tags' and 'matrices' must coincide.
*/
std::vector<Matrix3D<double>>
blockify_and_scatter(const std::vector<Matrix3D<double>> &matrices,
                     const int *block_size, const MPI_Comm &comm) {
#ifdef DEBUG
  std::cout << "blockify_and_scatter called" << std::endl;
#endif
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int rank;
  MPI_Comm_rank(comm, &rank);

  int top_left_corner[3];
  int process_coords[3];

  // contain the values for all the matrices to be sent. note that we
  // put alongside in the buffer the blocks which are going to be sent
  // to the same block, even if they belong to different matrices.
  double *buffer = nullptr;

  int current_process_rank;
  if (rank == 0) {
    int matrix_size[]{matrices.at(0).dim(0), matrices.at(0).dim(1),
                      matrices.at(0).dim(2)};

    buffer = new double[matrices.at(0).get_size() * matrices.size()];

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

          MPI_Cart_rank(comm, process_coords, &current_process_rank);

          for (int w = 0; w < matrices.size(); w++)
            block(matrices.at(w), block_size, top_left_corner,
                  buffer +
                      current_process_rank * block_n_cells * matrices.size() +
                      w * block_n_cells);
        }
      }
    }
  }

  // broadcast the expected number of blocks to be received
  int n_matrices = 0;
  if (rank == 0)
    n_matrices = matrices.size();
  MPI_Bcast(&n_matrices, 1, MPI_INT, 0, comm);

  double *receive_buffer = new double[n_matrices * block_n_cells];
  MPI_Scatter(buffer, n_matrices * block_n_cells, MPI_DOUBLE, receive_buffer,
              n_matrices * block_n_cells, MPI_DOUBLE, 0, comm);

  delete[] buffer;
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
void receive_compose_matrix(Matrix3D<double> &dest, Matrix3D<double> &send,
                            const int *block_size, const MPI_Comm &comm) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  std::vector<Matrix3D<double>> blocks;
  // each block is sent by a particular rank, we hold the rank
  // corresponding to each block in this vector
  std::vector<int> blocks_associated_rank;

  int top_left_corner[3];
  int process_coords[3];

  int matrix_size[]{dest.dim(0), dest.dim(1), dest.dim(2)};

  double *receive_buffer = nullptr;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    receive_buffer = new double[dest.get_size()];
  }

#ifdef DEBUG
  std::cout << "Starting waiting for responses" << std::endl;
#endif

  MPI_Gather(send.data(), block_n_cells, MPI_DOUBLE, receive_buffer,
             block_n_cells, MPI_DOUBLE, 0, comm);

  if (rank == 0) {
#ifdef DEBUG
    std::cout << "Everyone sent the sum!" << std::endl;
#endif

    int size;
    MPI_Comm_size(comm, &size);

    int upper_left_coords[3];
    for (int block_idx = 0; block_idx < size; ++block_idx) {
      double *data = receive_buffer + block_idx * block_n_cells;
      // we wrap the data in a Matrix3D
      Matrix3D<double> block(block_size, data);

      // remark: in this case block_idx is also the rank of the process which
      // sent this block.
      MPI_Cart_coords(comm, block_idx, 3, process_coords);

      // compute the position of the upper left corner of the block in dest
      for (int w = 0; w < 3; w++)
        upper_left_coords[w] = process_coords[w] * block_size[w];

      for (int a = 0; a < block_size[0]; ++a) {
        for (int b = 0; b < block_size[1]; ++b) {
          for (int c = 0; c < block_size[2]; ++c) {
            dest(upper_left_coords[0] + a, upper_left_coords[1] + b,
                 upper_left_coords[2] + c) = block(a, b, c);
          }
        }
      }

      // we do not want to free the buffer since it is shared by all the blocks
      block.unbind_data();
    }

    // we free manually the buffer
    delete[] receive_buffer;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // size of the cartesian grid along the three dimensions
  int decomposition_processors_count[3];
  int product = 1;
  for (int i = 4; i < argc; i++) {
    decomposition_processors_count[i - 4] = atoi(argv[i]);
    product *= decomposition_processors_count[i - 4];
  }

#ifdef ENFORCE_24
  if (product != 24) {
    if (rank == 0)
      std::cout << "Invalid size of cartesian grid, the given grid contains "
                << product << " processes instead of 24" << std::endl;
    MPI_Finalize();
    return 1;
  }
#endif

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

  int matrix_size[3];
  int blocks_size[3];
  determine_block_size(rank, argv, decomposition_processors_count, matrix_size,
                       blocks_size);

  int periodic[]{0, 0, 0};
  int reorder = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(MPI_COMM_WORLD, 3, decomposition_processors_count, periodic,
                  reorder, &cartesian_communicator);

  MPI_Comm_rank(cartesian_communicator, &rank);

  std::vector<Matrix3D<double>> matrices;

  if (rank == 0) {
#ifdef DEBUG
    for (int i = 0; i < 3; i++)
      std::cout << "Matrix.shape[" << i << "] = " << matrix_size[i]
                << std::endl;
#endif

    std::default_random_engine ran{};

    Matrix3D<double> matrix1 =
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran);
    Matrix3D<double> matrix2 =
        random_3d_matrix(matrix_size[0], matrix_size[1], matrix_size[2], ran);
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
  std::vector<Matrix3D<double>> blocks = std::move(
      blockify_and_scatter(matrices, blocks_size, cartesian_communicator));

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

  Matrix3D<double> result;

  if (rank == 0)
    result = std::move(
        Matrix3D<double>(matrix_size[0], matrix_size[1], matrix_size[2]));

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
