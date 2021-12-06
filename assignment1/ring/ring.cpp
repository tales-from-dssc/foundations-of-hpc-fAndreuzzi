
#include <iostream>
#include <mpi.h>

#define TAG_MULTIPLIER 10
// number of time we repeat the whole communication to obtain a decent
// statistic on the running time
#define ITERATIONS 1000000
// we skip (i.e. do not count) the first INITIAL_SKIP iterations since they
// usually contain a little bit of noise
#define INITIAL_SKIP 1000

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  double times[ITERATIONS - INITIAL_SKIP];

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int periodic = 1;
  MPI_Comm cartesian_communicator;
  MPI_Cart_create(MPI_COMM_WORLD, 1, &size, &periodic, 1,
                  &cartesian_communicator);

  int rank;
  MPI_Comm_rank(cartesian_communicator, &rank);

  // the tag of all the messages sent by this process
  const int tag = rank * TAG_MULTIPLIER;

  int left, right;
  MPI_Cart_shift(cartesian_communicator, 0, 1, &left, &right);
  const int right_tag = TAG_MULTIPLIER * right;
  const int left_tag = TAG_MULTIPLIER * left;

  double start_time, end_time = 0;

  int msg_count = 0;

#ifdef DEBUG
  std::cout << "Process " << rank << " objective: " << expected_send
            << std::endl;
#endif

  // 0: send left
  // 1: send right
  // 2: received from right
  // 3: received from left
  int buffer[] = {rank, -rank, 0, 0};
  MPI_Request requests[4];

  for (int it = 0; it < ITERATIONS; ++it) {
    // synchronize the processes before starting the communication
    MPI_Barrier(cartesian_communicator);

    // save the starting time of this iteration
    start_time = MPI_Wtime();

    for (int i = 0; i < size; ++i) {
      MPI_Isend(&buffer[0], 1, MPI_INT, left, tag, cartesian_communicator,
                &requests[0]);
      MPI_Irecv(&buffer[2], 1, MPI_INT, right, right_tag,
                cartesian_communicator, &requests[2]);
      MPI_Isend(&buffer[1], 1, MPI_INT, right, tag, cartesian_communicator,
                &requests[1]);
      MPI_Irecv(&buffer[3], 1, MPI_INT, left, left_tag, cartesian_communicator,
                &requests[3]);
      MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

      msg_count += 2;

#ifdef DEBUG
      std::cout << "Process " << rank << " RECEIVED: " << buffer[3] << ", "
                << buffer[2] << " --- SENT: " << buffer[0] << ", " << buffer[1]
                << std::endl;
#endif

      // update the content of the SEND part of the buffer
      buffer[1] = buffer[3] + rank;
      buffer[0] = buffer[2] - rank;
    }

    // before taking the time we wait that all the processes are done with their
    // own communication
    MPI_Barrier(cartesian_communicator);
    end_time = MPI_Wtime();

    if (it >= INITIAL_SKIP)
      times[it - INITIAL_SKIP] = end_time - start_time;

#ifndef TIME_ONLY
    std::cout << "I am process " << rank << " and i have received " << msg_count
              << " messages. My final messages have tag " << tag
              << " and value " << buffer[3] << ", " << buffer[2] << std::endl;
#endif

    // reset buffer and count
    msg_count = 0;
    buffer[0] = rank;
    buffer[1] = -rank;
    buffer[2] = buffer[3] = 0;
  }

  // output the times
  if (rank == 0)
    for (int i = 0; i < ITERATIONS - INITIAL_SKIP; ++i)
      std::cout << times[i] << std::endl;

  MPI_Finalize();
}
