
#include <iostream>
#include <mpi.h>

#define TAG_MULTIPLIER 10

#ifdef async
#define ASYNC 1
#else
#define ASYNC 0
#endif

#define DEBUG 1

// updates the content of buffer (which is expected to contain four integers)
// according to the statement of the assignment
void update_send_buffer(int *buffer, int rank) {
  // if async is true, the buffer is 4 bytes long
  // and we keep separate variables for sent and received variables
  buffer[1] = buffer[ASYNC * 2 + 1] + rank;
  buffer[0] = buffer[ASYNC * 2] - rank;
}

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // the tag of all the messages sent by this process
  const int tag = rank * TAG_MULTIPLIER;

  int periodic = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(MPI_COMM_WORLD, 1, &size, &periodic, 0,
                  &cartesian_communicator);

  int left, right;
  MPI_Cart_shift(cartesian_communicator, 0, 1, &left, &right);
  const int right_tag = TAG_MULTIPLIER * right;
  const int left_tag = TAG_MULTIPLIER * left;

  const int expected_send = size * (size - 1) / 2 - rank;

  int msg_count = 0;
  int last_msg_left, last_msg_right = 0;

#if DEBUG
  std::cout << "Process " << rank << " objective: " << expected_send
            << std::endl;
#endif

#if ASYNC
  // 0: send left
  // 1: send right
  // 2: received from right
  // 3: received from left
  int buffer[] = {rank, -rank, 0, 0};
  MPI_Request requests[4];

  do {
    MPI_Isend(&buffer[0], 1, MPI_INT, left, tag, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&buffer[2], 1, MPI_INT, right, right_tag, MPI_COMM_WORLD,
              &requests[2]);
    MPI_Isend(&buffer[1], 1, MPI_INT, right, tag, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(&buffer[3], 1, MPI_INT, left, left_tag, MPI_COMM_WORLD,
              &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    ++msg_count;

#if DEBUG
    std::cout << "Process " << rank << " RECEIVED: " << buffer[3] << ", "
              << buffer[2] << " --- SENT: " << buffer[0] << ", " << buffer[1]
              << std::endl;
#endif

    update_send_buffer(buffer, rank);
  } while (buffer[1] != expected_send || buffer[0] != -expected_send);

  last_msg_left = buffer[1];
  last_msg_right = buffer[0];
#else // SYNC
  int buffer[] = {rank, -rank};
  MPI_Status status;

  do {
    MPI_Send(&buffer[0], 1, MPI_INT, left, tag, MPI_COMM_WORLD);
    last_msg_left = buffer[0];
    MPI_Recv(&buffer[0], 1, MPI_INT, right, right_tag, MPI_COMM_WORLD, &status);
    MPI_Send(&buffer[1], 1, MPI_INT, right, tag, MPI_COMM_WORLD);
    last_msg_right = buffer[1];
    MPI_Recv(&buffer[1], 1, MPI_INT, left, left_tag, MPI_COMM_WORLD, &status);
    ++msg_count;

#if DEBUG
    std::cout << "Process " << rank << " RECEIVED: " << buffer[3] << ", "
              << buffer[2] << " --- SENT: " << buffer[0] << ", " << buffer[1]
              << std::endl;
#endif

    update_send_buffer(buffer, rank);
  } while (buffer[1] != expected_send || buffer[0] != -expected_send);
#endif

  std::cout << "I am process " << rank << " and i have received " << msg_count
            << " messages. My final messages have tag " << tag << " and value "
            << last_msg_left << ", " << last_msg_right << std::endl;

  MPI_Finalize();
}
