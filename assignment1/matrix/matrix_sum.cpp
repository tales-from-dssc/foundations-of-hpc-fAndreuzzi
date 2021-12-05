#include <iostream>
#include <mpi.h>
#include <random>
#include <unistd.h>

#define ITERATIONS 1000

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

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dim1 = atoi(argv[1]);
  int dim2 = atoi(argv[2]);
  int dim3 = atoi(argv[3]);

  int N = dim1 * dim2 * dim3;
  // we extend the array in order to make it divisible by size
  N += size - N % size;

  int blocks_size = N / size;

  double *matrix_1, *matrix_2 = nullptr;
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

  double start_time, end_time, before_gather_time, after_gather_time,
      before_scatter_time, after_scatter_time = 0;
  double tot_time[ITERATIONS];
  double gather_time[ITERATIONS];
  double scatter_time[ITERATIONS];
  for (int iter = 0; iter < ITERATIONS; iter++) {
    double *block_1;
    double *block_2;

    if (size > 1) {
      block_1 = new double[blocks_size];
      block_2 = new double[blocks_size];
    }

    start_time = MPI_Wtime();

    if (size > 1) {
      before_scatter_time = MPI_Wtime();
      // send/recv the first matrix
      MPI_Scatter(matrix_1, blocks_size, MPI_DOUBLE, block_1, blocks_size,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // send/recv the second matrix
      MPI_Scatter(matrix_2, blocks_size, MPI_DOUBLE, block_2, blocks_size,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
      after_scatter_time = MPI_Wtime();
    } else {
      block_1 = matrix_1;
      block_2 = matrix_2;
    }

    double *block_sum = new double[blocks_size];
    for (int i = 0; i < blocks_size; i++)
      block_sum[i] = block_1[i] + block_2[i];

    // no need to release if size is 1
    if (size > 1) {
      delete[] block_1;
      delete[] block_2;
    }

    double *result;
    if (size > 1)
      result = new double[N];
    else
      result = block_sum;

    if (size > 1) {
      before_gather_time = MPI_Wtime();
      // send back the summed block
      MPI_Gather(block_sum, blocks_size, MPI_DOUBLE, result, blocks_size,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
      after_gather_time = MPI_Wtime();
    } else {
      result = block_sum;
    }

    end_time = MPI_Wtime();

    delete[] result;
    if (size > 1)
      delete[] block_sum;

    tot_time[iter] = end_time - start_time;
    gather_time[iter] = after_gather_time - before_gather_time;
    scatter_time[iter] = after_scatter_time - before_scatter_time;
  }

  if (rank == 0) {
    for (int i = 0; i < ITERATIONS; i++)
      std::cout << tot_time[i] << "," << scatter_time[i] << ","
                << gather_time[i] << std::endl;
  }

  // release the memory used to store the 3D matrices
  if (rank == 0) {
    delete[] matrix_2;
    delete[] matrix_1;
  }

  MPI_Finalize();
}
