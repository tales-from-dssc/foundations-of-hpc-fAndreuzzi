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
  T *elem;
  int dim1, dim2, dim3;
  int size;

  inline int map_3D_to_1D(int i, int j, int k) const {
    return k + j * dim3 + i * dim2 * dim3;
  }

public:
  Matrix3D(int dim1, int dim2, int dim3) {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->size = dim1 * dim2 * dim3;

    elem = new T[dim1 * dim2 * dim3];
  }

  Matrix3D(Matrix3D &&that) {
    dim1 = that.dim1;
    dim2 = that.dim2;
    dim3 = that.dim3;
    size = that.size;

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
    size = that.size;

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

  Matrix3D<double> operator+(const Matrix3D<double> &matrix2) {
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
