#include <iostream>

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

  inline int map_3D_to_1D(int i, int j, int k) const {
    return k + j * dim3 + i * dim2 * dim3;
  }

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
