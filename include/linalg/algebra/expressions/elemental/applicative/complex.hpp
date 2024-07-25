#ifndef LINALG_ALGEBRA_COMPLEX_OPERATION_IMPL_HPP
#define LINALG_ALGEBRA_COMPLEX_OPERATION_IMPL_HPP

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{

namespace expression_templates
{

template <>
class complex_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline auto apply(const T1* a, const T2* b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline auto apply(const T* a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}

    template <typename T, typename vtype, template <typename > class op > 
    static inline auto apply(const T* a, unary_expression<vtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}
};
#ifdef __NVCC__
template <>
class complex_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline __device__ auto apply(const T1* a, const T2* b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline __device__ auto apply(const T* a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}

    template <typename T, typename vtype, template <typename > class op > 
    static inline __device__ auto apply(const T* a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return complex<decltype(a[i]+b[i])>(a[i],b[i]);}
};
#endif

template <>
class polar_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline auto apply(const T1* a, const T2* b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline auto apply(const T* a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}

    template <typename T, typename vtype, template <typename > class op > 
    static inline auto apply(const T* a, unary_expression<vtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}
};
#ifdef __NVCC__
template <>
class polar_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;

public:
    template <typename T1, typename T2> 
    static inline __device__ auto apply(const T1* a, const T2* b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}

    template <typename T, typename ltype, typename rtype, template <typename > class op > 
    static inline __device__ auto apply(const T* a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}

    template <typename T, typename vtype, template <typename > class op > 
    static inline __device__ auto apply(const T* a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i]+b[i])> {return polar(a[i],b[i]);}
};
#endif

//unary expression for evaluating the real part of a complex array
template <>
class unit_polar_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;
public:
    template <typename T> static inline auto apply(const T* a, size_type i) -> complex<decltype(a[i])>{return complex<decltype(a[i])>(cos(a[i]), sin(a[i]));}
    template <typename T> static inline auto apply(const T& a, size_type i) -> complex<decltype(a[i])>{return complex<decltype(a[i])>(cos(a[i]), sin(a[i]));}
};
#ifdef __NVCC__
template <>
class unit_polar_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline __device__ auto apply(const T* a, size_type i) -> complex<decltype(a[i])>{return complex<decltype(a[i])>(cos(a[i]), sin(a[i]));}
    template <typename T> static inline __device__ auto apply(const T& a, size_type i) -> complex<decltype(a[i])>{return complex<decltype(a[i])>(cos(a[i]), sin(a[i]));}
};
#endif



//unary expression for evaluating the real part of a complex array
template <>
class real_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;
public:
    template <typename T> static inline typename get_real_type<T>::type apply(const T* a, size_type i) {return real(a[i]);}
    template <typename T> static inline typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return real(a[i]);}
};
#ifdef __NVCC__
template <>
class real_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline __device__ typename get_real_type<T>::type apply(const T* a, size_type i) {return real(a[i]);}
    template <typename T> static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return real(a[i]);}
};
#endif

template <>
class imag_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;
public:
    template <typename T> static inline typename get_real_type<T>::type apply(const T* a, size_type i) {return imag(a[i]);}
    template <typename T> static inline typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return imag(a[i]);}
};
#ifdef __NVCC__
template <>
class imag_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline __device__ typename get_real_type<T>::type apply(const T* a, size_type i) {return imag(a[i]);}
    template <typename T> static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return imag(a[i]);}
};
#endif

template <>
class norm_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;
public:
    template <typename T> static inline typename get_real_type<T>::type apply(const T* a, size_type i) {return norm(a[i]);}
    template <typename T> static inline typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return norm(a[i]);}
};
#ifdef __NVCC__
template <>
class norm_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline __device__ typename get_real_type<T>::type apply(const T* a, size_type i) {return norm(a[i]);}
    template <typename T> static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return norm(a[i]);}
};
#endif

template <>
class arg_op<blas_backend>
{
public:
    using size_type = blas_backend::size_type;
public:
    template <typename T> static inline typename get_real_type<T>::type apply(const T* a, size_type i) {return arg(a[i]);}
    template <typename T> static inline typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return arg(a[i]);}
};
#ifdef __NVCC__
template <>
class arg_op<cuda_backend>
{
public:
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline __device__ typename get_real_type<T>::type apply(const T* a, size_type i) {return arg(a[i]);}
    template <typename T> static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T& a, size_type i){return arg(a[i]);}
};
#endif

}   //namespace expression_templates
}   //namespace linalg

#endif //LINALG_ALGEBRA_COMPLEX_OPERATION_IMPL_HPP

