#ifndef MATRIX_PARALLEL_HPP
#define MATRIX_PARALLEL_HPP

#include "Tensor.hpp"
#include "TensorParallel.hpp"
#include "Matrix.hpp"

namespace gema{

template<class T, TensorType<T> ITensor = TensorParallel<T>>
class MatrixParallel : Matrix<T>{

    private:

};

}

#include "MatrixParallel.tpp"

#endif