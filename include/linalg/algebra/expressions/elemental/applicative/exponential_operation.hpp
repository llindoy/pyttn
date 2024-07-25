#ifndef LINALG_ALGEBRA_ELEMENTAL_EXPONENTIAL_OPERATION_IMPL_HPP
#define LINALG_ALGEBRA_ELEMENTAL_EXPONENTIAL_OPERATION_IMPL_HPP

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{
namespace expression_templates
{

template <>
class elemental_exp_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;

public:
    template <typename T> 
    static inline T apply(const T* a, size_type i) {return exp(a[i]);}

    template <typename T>
    static inline typename T::value_type apply(const T& a, size_type i){return exp(a[i]);}
};


#ifdef __NVCC__


template <>
class elemental_exp_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;

public:
    template <typename T> 
    static inline __device__ T apply(const T* a, size_type i) {return exp(a[i]);}

    template <typename T>
    static inline __device__ typename T::value_type apply(const T& a, size_type i){return exp(a[i]);}
};

#endif

}   //namespace expression_templates
}   //namespace linalg

#endif //LINALG_ALGEBRA_ELEMENTAL_EXPONENTIAL_OPERATION_IMPL_HPP

