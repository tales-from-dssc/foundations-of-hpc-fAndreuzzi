#include <iostream>
#include <mpi.h>
#include <random>
#include <unistd.h>

// used for abs()
#ifdef CHECK
#include <stdlib.h>
#endif

#define ITERATIONS 10000

double *generate_random_array(int n, std::default_random_engine &ran) {
  std::uniform_real_distribution<> ureal{-10, 10};

  double *data = new double[n];
  for (int i = 0; i < n; i++)
    data[i] = ureal(ran);

  return data;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SERIAL
  if (size > 1) {
    std::cout << "Called as serial even though size > 1" << std::endl;
    MPI_Finalize();
    return 1;
  }
#endif

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dim1 = atoi(argv[1]);
  int dim2 = atoi(argv[2]);
  int dim3 = atoi(argv[3]);

  int N = dim1 * dim2 * dim3;
  // we extend the array in order to make it divisible by size
  int residual = N % size;
  N += (residual != 0) * (size - residual);

  int blocks_size = N / size;

  double *matrix_1 = nullptr, *matrix_2 = nullptr;
  if (rank == 0) {
    std::default_random_engine ran{};

    // generate two 3D matrices using the given dimensions
    matrix_1 = generate_random_array(N, ran);
    matrix_2 = generate_random_array(N, ran);
  }

#ifdef MPI_DEBUG
  if (rank == 0) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
#endif

  double start_time = 0, end_time = 0, before_gather_time = 0,
         after_gather_time = 0, before_scatter_time = 0, after_scatter_time = 0;
  double tot_time[ITERATIONS];
  double gather_time[ITERATIONS];
  double scatter_time[ITERATIONS];

  // on each process, stores the summed block
  double *block_sum = new double[blocks_size];
  // on each process, stores the two blocks coming
  // from the first and second matrices
  double *block_1 = nullptr;
  double *block_2 = nullptr;
  // on main thread, stores the summed matrix
  double *result = nullptr;

  // if we are serial, then the size of the summed matrix is the same
  // of the size of one of the blocks sent to a (the only one) process
#ifndef SERIAL
  if (rank == 0)
    result = new double[N];
  block_1 = new double[blocks_size];
  block_2 = new double[blocks_size];
#else
  if (rank == 0)
    result = block_sum;
  block_1 = matrix_1;
  block_2 = matrix_2;
#endif

  for (int iter = 0; iter < ITERATIONS; iter++) {
    start_time = MPI_Wtime();

#ifndef SERIAL
    before_scatter_time = MPI_Wtime();
    // send/recv the first matrix
    MPI_Scatter(matrix_1, blocks_size, MPI_DOUBLE, block_1, blocks_size,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // send/recv the second matrix
    MPI_Scatter(matrix_2, blocks_size, MPI_DOUBLE, block_2, blocks_size,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    after_scatter_time = MPI_Wtime();
#endif

    // we sum the two blocks we received from scatter
    for (int i = 0; i < blocks_size; i++)
      block_sum[i] = block_1[i] + block_2[i];

#ifndef SERIAL
    before_gather_time = MPI_Wtime();
    // send back the summed block
    MPI_Gather(block_sum, blocks_size, MPI_DOUBLE, result, blocks_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    after_gather_time = MPI_Wtime();
#endif

    end_time = MPI_Wtime();

    tot_time[iter] = end_time - start_time;
    gather_time[iter] = after_gather_time - before_gather_time;
    scatter_time[iter] = after_scatter_time - before_scatter_time;

#ifdef CHECK
    if (rank == 0) {
      double max_error = -1;
      double error = 0;
      for (int i = 0; i < dim1 * dim2 * dim3; i++) {
        error = abs(matrix_1[i] + matrix_2[i] - result[i]);
        if (error > max_error)
          max_error = error;
      }
      std::cout << "Max error: " << max_error << " on iteration " << iter << std::endl;
    }
#endif
  }

  if (rank == 0) {
    for (int i = 0; i < ITERATIONS; i++)
      std::cout << tot_time[i] << "," << scatter_time[i] << ","
                << gather_time[i] << std::endl;
  }

  delete[] matrix_2;
  delete[] matrix_1;
  delete[] block_sum;

#ifndef SERIAL
  delete[] block_1;
  delete[] block_2;
  delete[] result;
#endif

  MPI_Finalize();
}
