#include <iostream>
#include <random>

/*
        A wrapper of an array of T (allocated on the free store)
        which behaves like a 3D matrix.

        Copy semantics is disabled to avoid multiple instances
        living at the same time. Move constructor/assignment is
        implemented and encourage (remember not to use the rvalue
        afterwards).
*/
template <typename T> class Matrix3D {
  int size;

protected:
  T *elem;
  int dim1, dim2, dim3;

  virtual inline int map_3D_to_1D(int i, int j, int k) const {
    return k + j * dim3 + i * dim2 * dim3;
  }

  void set_dim(int dim1, int dim2, int dim3) {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->size = dim1 * dim2 * dim3;
  }

public:
  Matrix3D() : dim1(0), dim2(0), dim3(0), size(0), elem(nullptr) {}

  Matrix3D(const int dim1, const int dim2, const int dim3, T *data) {
    this->set_dim(dim1, dim2, dim3);
    elem = data;
  }

  Matrix3D(const int dim1, const int dim2, const int dim3)
      : Matrix3D(dim1, dim2, dim3, new T[dim1 * dim2 * dim3]) {}

  Matrix3D(const int *ds) : Matrix3D(ds[0], ds[1], ds[2]) {}

  Matrix3D(const int *ds, T *data) : Matrix3D(ds[0], ds[1], ds[2], data) {}

  Matrix3D(Matrix3D &&that)
      : Matrix3D(that.dim1, that.dim2, that.dim3, that.elem) {
    that.elem = nullptr;
  }

  Matrix3D &operator=(Matrix3D &&that) {
    delete[] elem;

    elem = that.elem;
    that.elem = nullptr;

    set_dim(that.dim1, that.dim2, that.dim3);

    return *this;
  }

  // disable copy semantics
  Matrix3D(const Matrix3D &) = delete;
  Matrix3D &operator=(const Matrix3D &) = delete;

  ~Matrix3D() {
#ifdef DEBUG
    std::cout << "destructor called on address " << elem << std::endl;
#endif
    delete[] elem;
  }

  void unbind_data() { elem = nullptr; }

  void set_data(T *data) {
    delete[] elem;
    elem = data;
  }

  // access the shape of this matrix
  int dim(int i) const {
    switch (i) {
    case 0:
      return dim1;
    case 1:
      return dim2;
    case 2:
      return dim3;
    default:
      return -1;
    }
  }

  const int get_size() const { return size; }

  const T operator()(int i, int j, int k) const {
    return elem[map_3D_to_1D(i, j, k)];
  }

  T &operator()(int i, int j, int k) { return elem[map_3D_to_1D(i, j, k)]; }

  T *data() { return elem; }

  Matrix3D<double> operator+(const Matrix3D<double> &matrix2) const {
    if (dim(0) != matrix2.dim(0) || dim(1) != matrix2.dim(1) ||
        dim(2) != matrix2.dim(2))
      throw std::invalid_argument(
          "Summation requires that the two matrices have the same shape");

    Matrix3D<double> sum(dim(0), dim(1), dim(2));
    for (int i = 0; i < dim(0); i++)
      for (int j = 0; j < dim(1); j++)
        for (int k = 0; k < dim(2); k++)
          sum(i, j, k) = (*this)(i, j, k) + matrix2(i, j, k);

    return sum;
  }
};

template <typename T> class BlockifiedMatrix3D : public Matrix3D<T> {
  int block_dim1, block_dim2, block_dim3 = 0;
  int block_size = 0;
  int block_2d_size = 0;
  int n_blocks_in_slice = 0;
  int n_blocks_in_row = 0;

protected:
  inline int map_3D_to_1D(int i, int j, int k) const override {
    // the index in terms of blocks along the 3 dimensions
    // for instance if we have blocks of size 2x2x1, the
    // cell (1,4,3) has idx_along_dim={0,1,3}
    int idx_along_dim[3]{i / block_dim1, j / block_dim2, k / block_dim3};
    int idx_inside_block[3]{i % block_dim1, j % block_dim2, k % block_dim3};

    return n_blocks_in_slice * block_size * idx_along_dim[0] +
           n_blocks_in_row * block_size * idx_along_dim[1] +
           idx_along_dim[2] * block_size +
           block_dim2 * block_dim3 * idx_inside_block[0] +
           block_dim2 * idx_inside_block[1] + idx_inside_block[2];
  }

public:
  BlockifiedMatrix3D() : Matrix3D<T>() {}
  BlockifiedMatrix3D(const int dim1, const int dim2, const int dim3)
      : Matrix3D<T>(dim1, dim2, dim3) {}
  BlockifiedMatrix3D(const int dim1, const int dim2, const int dim3, T *data)
      : Matrix3D<T>(dim1, dim2, dim3, data) {}
  BlockifiedMatrix3D(const int *ds) : Matrix3D<T>(ds) {}
  BlockifiedMatrix3D(Matrix3D<T> &&that) : Matrix3D<T>(std::move(that)) {}

  BlockifiedMatrix3D(BlockifiedMatrix3D<T> &&that)
      : BlockifiedMatrix3D(that.dim1, that.dim2, that.dim3, that.elem) {
    set_block_size(that.block_dim1, that.block_dim2, that.block_dim3);
    that.elem = nullptr;
  }

  BlockifiedMatrix3D &operator=(BlockifiedMatrix3D &&that) {
    delete[] Matrix3D<T>::elem;

    Matrix3D<T>::elem = that.data();
    that.elem = nullptr;

    Matrix3D<T>::set_dim(that.dim(0), that.dim(1), that.dim(2));
    set_block_size(that.block_dim1, that.block_dim2, that.block_dim3);

    return *this;
  }

  // disable copy semantics
  BlockifiedMatrix3D(const BlockifiedMatrix3D &) = delete;
  BlockifiedMatrix3D &operator=(const BlockifiedMatrix3D &) = delete;

  void set_block_size(int d1, int d2, int d3) {
    block_dim1 = d1;
    block_dim2 = d2;
    block_dim3 = d3;

    block_size = d1 * d2 * d3;

    // if this is not true the user is expected to set the block size
    // via set_block_size
    if (d3 != 0 && d2 != 0) {
      n_blocks_in_row = Matrix3D<T>::dim3 / d3;
      n_blocks_in_slice = n_blocks_in_row * Matrix3D<T>::dim2 / d2;
    }
  }

  void set_block_size(int *bs) { set_block_size(bs[0], bs[1], bs[2]); }
};

/*
        Generate a random 3D matrix of doubles using the given random
        engine.
*/
Matrix3D<double> random_3d_matrix(int dim1, int dim2, int dim3,
                                  std::default_random_engine &ran) {
  Matrix3D<double> m(dim1, dim2, dim3);

  std::uniform_real_distribution<> ureal{-10, 10};
  for (int i = 0; i < dim1; ++i)
    for (int j = 0; j < dim2; ++j)
      for (int k = 0; k < dim3; ++k)
        m(i, j, k) = ureal(ran);

  return m;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix3D<T> &p) {
  os << "Shape=(" << p.dim(0) << ", " << p.dim(1) << ", " << p.dim(2) << ")"
     << std::endl;
  for (int i = 0; i < p.dim(0); ++i) {
    os << "Slice N. " << i << std::endl;
    for (int j = 0; j < p.dim(1); ++j) {
      for (int k = 0; k < p.dim(2); ++k)
        os << p(i, j, k) << " ";
      os << std::endl;
    }
    os << std::endl;
  }
  return os;
}
