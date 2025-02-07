#ifndef LINALG_ALGEBRA_HADAMARD_OPERATION_IMPL_HPP
#define LINALG_ALGEBRA_HADAMARD_OPERATION_IMPL_HPP

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{

namespace expression_templates
{

template <>
class hadamard_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline auto apply(const T1* a, const T2* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline auto apply(binary_expression<ltype, rtype, op, blas_backend> a, const T* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline auto apply(const T* a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename ltype1, typename rtype1, template <typename > class op1, typename ltype2, typename rtype2, template <typename > class op2  > 
    static inline auto apply(binary_expression<ltype1, rtype1, op1, blas_backend> a, binary_expression<ltype2, rtype2, op2, blas_backend> b, size_type i)-> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline auto apply(unary_expression<vtype, op, blas_backend> a, const T* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline auto apply(const T* a, unary_expression<vtype, op, blas_backend> b, size_type i) -> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename vtype1, template <typename > class op1, typename vtype2, template <typename > class op2  > 
    static inline auto apply(unary_expression<vtype1, op1, blas_backend> a, unary_expression<vtype2, op2, blas_backend> b, size_type i)-> decltype(a[i]*b[i]){return a[i]*b[i];}
};


#ifdef PYTTN_BUILD_CUDA


template <>
class hadamard_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline __device__ auto apply(const T1* a, const T2* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline __device__ auto apply(binary_expression<ltype, rtype, op, cuda_backend> a, const T* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline __device__ auto apply(const T* b, binary_expression<ltype, rtype, op, cuda_backend> a, size_type i) -> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename ltype1, typename rtype1, template <typename > class op1, typename ltype2, typename rtype2, template <typename > class op2  > 
    static inline __device__ auto apply(binary_expression<ltype1, rtype1, op1, cuda_backend> a, binary_expression<ltype2, rtype2, op2, cuda_backend> b, size_type i)-> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline __device__ auto apply(unary_expression<vtype, op, cuda_backend> a, const T* b, size_type i) -> decltype(a[i]*b[i]) {return a[i]*b[i];}

    template <typename T, typename vtype, template <typename > class op > 
    static inline __device__ auto apply(const T* a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> decltype(a[i]*b[i]){return a[i]*b[i];}

    template <typename vtype1, template <typename > class op1, typename vtype2, template <typename > class op2  > 
    static inline __device__ auto apply(unary_expression<vtype1, op1, cuda_backend> a, unary_expression<vtype2, op2, cuda_backend> b, size_type i)-> decltype(a[i]*b[i]){return a[i]*b[i];}
};

#endif

}   //namespace expression_templates
}   //namespace linalg

#endif  //LINALG_ALGEBRA_HADAMARD_OPERATION_IMPL_HPP

