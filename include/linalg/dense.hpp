#ifndef LINALG_DENSE_HPP
#define LINALG_DENSE_HPP

/**
 *  Class includes all of the required headers to use the dense linear algebra component of the linalg library
 */

#include "tensor/dense/tensor.hpp"
#include "algebra/algebra.hpp"

namespace linalg
{
template <typename T, typename backend = blas_backend> using matrix = tensor<T, 2, backend>;
template <typename T, typename backend = blas_backend> using vector = tensor<T, 1, backend>;

template <typename T, size_t D> using host_tensor = tensor<T, D, blas_backend>;
template <typename T> using host_matrix = tensor<T, 2, blas_backend>;
template <typename T> using host_vector = tensor<T, 1, blas_backend>;
#ifdef __NVCC__
template <typename T, size_t D> using device_tensor = tensor<T, D, cuda_backend>;
template <typename T> using device_matrix = tensor<T, 2, cuda_backend>;
template <typename T> using device_vector = tensor<T, 1, cuda_backend>;
#endif


}

#endif

