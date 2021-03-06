#include "matrix3d.h"
#include "processor_utils.h"
#include <iostream>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#define BLOCK_1_TAG 0
#define BLOCK_2_TAG 1
#define BLOCK_SUM_TAG 2

/*
        Extract a block of the given size starting from the given top
        left corner from the given matrix, and return it as a new 3D
        matrix.

        Afterwards there is no link between the two matrices, the data
        is copied by value. In other wards, the returned matrix is not
        a "view" on the matrix passed in the argument.
*/
template <typename T>
Matrix3D<T> block(const Matrix3D<T> &matrix, const int *block_size,
                  const int *top_left_corner) {
  Matrix3D<T> data(block_size[0], block_size[1], block_size[2]);

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

/*
        Send the given matrix to the process whose rank (in the given
        communicator) is 'send_to'. The communication happens using the
        given tag (which must be already known to the receiver).
*/
void send_matrix(Matrix3D<double> &matrix, const int send_to, const int tag,
                 const MPI_Comm &comm, MPI_Request *req) {
#ifdef DEBUG
  std::cout << "Sending to " << send_to << " (address=" << matrix.data() << ")"
            << std::endl;
  std::cout << matrix << std::endl;
#endif

  MPI_Isend(matrix.data(), matrix.get_size(), MPI_DOUBLE, send_to, tag, comm,
            req);
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
void blockify_and_msg(const std::vector<Matrix3D<double>> &matrices,
                      const std::vector<int> &tags, const int *block_size,
                      const MPI_Comm &comm) {
#ifdef DEBUG
  std::cout << "blockify_and_msg called" << std::endl;
#endif
  if (matrices.size() != tags.size())
    throw std::invalid_argument("n. of matrices != n. of tags");

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Request *requests = new MPI_Request[size * tags.size()];

  int top_left_corner[3];
  int process_coords[3];
  int block_idx = 0;

  int matrix_size[]{matrices.at(0).dim(0), matrices.at(0).dim(1),
                    matrices.at(0).dim(2)};

  // we use a vector to prevent the matrix to be destroyed
  // after the end of its scope (i.e. the innermostblock inside
  // the for loop)
  std::vector<Matrix3D<double>> sent_blocks;

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

        MPI_Cart_rank(comm, process_coords, &send_to);

        for (int w = 0; w < matrices.size(); w++) {
          sent_blocks.push_back(
              block(matrices.at(w), block_size, top_left_corner));
          send_matrix(sent_blocks.at(block_idx), send_to, tags.at(w), comm,
                      &requests[block_idx]);
          block_idx++;
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
        of expected matrices is equal to the number of tags. Note that
        this function is blocking.
*/
std::vector<Matrix3D<double>> receive_matrix(const int *matrix_size,
                                             const MPI_Comm &comm,
                                             const int sending_process,
                                             const std::vector<int> tags) {
  std::vector<Matrix3D<double>> matrices;
  for (int i = 0; i < tags.size(); i++) {
    Matrix3D<double> m(matrix_size[0], matrix_size[1], matrix_size[2]);
    MPI_Status status;
    MPI_Recv(m.data(), m.get_size(), MPI_DOUBLE, sending_process, tags.at(i),
             comm, &status);
    matrices.push_back(std::move(m));
  }
  return matrices;
}

/*
        Receives asynchronously a set of blocks from the processes using the
        given cartesian communicator (the expected tag is BLOCK_SUM_TAG). Then
        use them to reconstruct the complete matrix, mapping the position of the
        block in the complete matrix as a function of the position of the
        sending processor in the virtual topology. This is a blocking function
        since it waits for the reception of all the pieces, and then composes
        the matrix.
*/
void receive_compose_matrix(Matrix3D<double> &dest, const int *block_size,
                            const MPI_Comm &comm) {
  int block_n_cells = block_size[0] * block_size[1] * block_size[2];

  int size;
  MPI_Comm_size(comm, &size);
  MPI_Request *requests = new MPI_Request[size];

  std::vector<Matrix3D<double>> blocks;
  // each block is sent by a particular rank, we hold the rank
  // corresponding to each block in this vector
  std::vector<int> blocks_associated_rank;

  int top_left_corner[3];
  int process_coords[3];
  int block_idx = 0;

  int matrix_size[]{dest.dim(0), dest.dim(1), dest.dim(2)};

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

        blocks.push_back(std::move(block(dest, block_size, top_left_corner)));

        MPI_Cart_rank(comm, process_coords, &receive_from);
        blocks_associated_rank.push_back(receive_from);
        MPI_Irecv(blocks.at(block_idx).data(), block_n_cells, MPI_DOUBLE,
                  receive_from, BLOCK_SUM_TAG, comm, &requests[block_idx]);

        block_idx++;
      }
    }
  }

#ifdef DEBUG
  std::cout << "Starting waiting for responses" << std::endl;
#endif

  MPI_Waitall(size, requests, MPI_STATUSES_IGNORE);
  delete[] requests;

#ifdef DEBUG
  for (int i = 0; i < block_idx; i++) {
    std::cout << "Printing block number " << i << std::endl;
    std::cout << blocks.at(i) << std::endl << std::endl;
  }

  std::cout << "Everyone sent the sum!" << std::endl;
#endif

  int upper_left_coords[3];
  for (int i = 0; i < block_idx; i++) {
    Matrix3D<double> block = std::move(blocks.at(i));
    MPI_Cart_coords(comm, blocks_associated_rank.at(i), 3, process_coords);
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
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  double start_time;

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

  if (product != size) {
    if (rank == 0)
      std::cout << "Invalid size of cartesian grid, the given grid contains "
                << product << " processes instead of " << size << std::endl;
    MPI_Finalize();
    return 1;
  }

  int matrix_size[3];
  int blocks_size[3];
  determine_block_size(rank, argv, decomposition_processors_count, matrix_size,
                       blocks_size);

  MPI_Comm comm = MPI_COMM_WORLD;

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

  int periodic[]{0, 0, 0};
  int reorder = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(comm, 3, decomposition_processors_count, periodic, reorder,
                  &cartesian_communicator);

  MPI_Comm_rank(cartesian_communicator, &rank);

  std::vector<int> tags{BLOCK_1_TAG, BLOCK_2_TAG};
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

    std::vector<Matrix3D<double>> matrices;
    matrices.push_back(std::move(matrix1));
    matrices.push_back(std::move(matrix2));

    // after generating the matrices (and before communicating them to the
    // other processes) we take the starting time
    start_time = MPI_Wtime();

    blockify_and_msg(matrices, tags, blocks_size, cartesian_communicator);
  }

#ifdef DEBUG
  std::cout << "I'm process " << rank << std::endl;
#endif

  std::vector<Matrix3D<double>> blks =
      receive_matrix(blocks_size, cartesian_communicator, 0, tags);

#ifdef DEBUG
  std::cout << rank << " received "
            << "(" << blks.at(0).dim(0) << "," << blks.at(0).dim(1) << ","
            << blks.at(0).dim(2) << ")" << std::endl;

  if (rank == 1) {
    std::cout << "rank " << rank << " received: " << std::endl;
    std::cout << "Block 1 ------------------------" << std::endl;
    std::cout << blks.at(0) << std::endl;
    std::cout << "Block 2 ------------------------" << std::endl;
    std::cout << blks.at(1) << std::endl;
  }
#endif

  MPI_Request send_req;

  Matrix3D<double> summed_block = blks.at(0) + blks.at(1);
  send_matrix(summed_block, 0, BLOCK_SUM_TAG, cartesian_communicator,
              &send_req);

  if (rank == 0) {
    Matrix3D<double> result(matrix_size[0], matrix_size[1], matrix_size[2]);
    receive_compose_matrix(result, blocks_size, cartesian_communicator);

#ifdef DEBUG
    std::cout << "Result ------------------------" << std::endl << result;
#endif

    std::cout << MPI_Wtime() - start_time << std::endl;
  }

  MPI_Finalize();
}
