#ifndef LINALG_ALGEBRA_SCALAR_MULTIPLICATION_OPERATION_IMPL_HPP
#define LINALG_ALGEBRA_SCALAR_MULTIPLICATION_OPERATION_IMPL_HPP

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{

namespace expression_templates
{

template <>
class multiplication_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline auto apply(T1 a, const T2* b, size_type i) -> decltype(a*b[i]) {return a*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline auto apply(T a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> decltype(a*b[i]) {return a*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline auto apply(T a, unary_expression<vtype, op, blas_backend> b, size_type i) -> decltype(a*b[i]) {return a*b[i];}
};




#ifdef __NVCC__
template <>
class multiplication_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline __device__ auto apply(T1 a, const T2* b, size_type i) -> decltype(a*b[i]) {return a*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline __device__ auto apply(T a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> decltype(a*b[i]) {return a*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline __device__ auto apply(T a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> decltype(a*b[i]) {return a*b[i];}
};
#endif

}   //namespace expression_templates
}   //namespace linalg

#endif //LINALG_ALGEBRA_SCALAR_MULTIPLICATION_OPERATION_IMPL_HPP

