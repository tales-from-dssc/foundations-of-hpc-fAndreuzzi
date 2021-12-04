#include <iostream>

/* decomposition_processors_count is an array containing 3 integers which
   represent the number of processors along the given axis. matrix_size and
   block_size are 3-arrays to be filled by the function determine_block_size.
*/
void determine_block_size(int rank, char **argv,
                          const int *decomposition_processors_count,
                          int *matrix_size, int *blocks_size) {
  // number of blocks along the axis
  int residual = 0;
  for (int i = 0; i < 3; i++) {
    matrix_size[i] = atoi(argv[i + 1]);
    blocks_size[i] = matrix_size[i] / decomposition_processors_count[i];

    residual = matrix_size[i] % decomposition_processors_count[i];
    blocks_size[i] += (residual != 0) * 1;
    matrix_size[i] +=
        (residual != 0) * (decomposition_processors_count[i] - residual);
#ifdef DEBUG2
    if (rank == 0 && residual != 0)
      std::cout << "Matrix augmented by "
                << decomposition_processors_count[i] - residual
                << " cells along the " << i << "-th axis ("
                << matrix_size[i] -
                       (decomposition_processors_count[i] - residual)
                << "->" << matrix_size[i] << ")" << std::endl;
#endif
  }
}
