#ifndef LINALG_ALGEBRA_STORAGE_TRAITS_HPP
#define LINALG_ALGEBRA_STORAGE_TRAITS_HPP

//TODO: comment this file

#include "../../../linalg_forward_decl.hpp"

namespace linalg
{

namespace expression_templates
{
///////////////////////////////////////////////////////////////////////////////////////////////////
//Structures for storing the type to be stored for various different types found in expressions
///////////////////////////////////////////////////////////////////////////////////////////////////
//traits for an literal type
template <typename T, typename backend> 
struct storage_traits<literal_type<T, backend> > 
{
    static_assert(is_number<T>::value, "Failed to initialise storage_traits object.  The literal expects a number type as an input template parameter.");
    using type = literal_type<T, backend>;    using eval_type = T;

    static inline eval_type data(type a){return a;}
};

//traits for the tensor objects
template <typename T, size_t D, typename backend> 
struct storage_traits<tensor<T, D, backend> > 
{
    static_assert(is_number<T>::value, "Failed to initialise storage_traits object.  The tensor object expects a number type as an input template parameter.");
    using type = const tensor<T, D, backend>&;
    using eval_type = const T*;

    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

template <typename T, size_t D, typename backend> 
struct storage_traits<reinterpreted_tensor<T, D, backend> > 
{
    static_assert(is_number<T>::value, "Failed to initialise storage_traits object.  The reinterpreted tensor object expects a number type as an input template parameter.");
    using type = const reinterpreted_tensor<T, D, backend>&;
    using eval_type = const T*;

    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};


template <template <typename, size_t, typename> class arrtype, typename T1, size_t D1, typename backend, typename T, size_t D> 
struct storage_traits<tensor_slice<arrtype<T1, D1, backend>, T, D> > 
{
    static_assert(is_number<T>::value && (std::is_same<T1, T>::value || std::is_same<typename std::add_const<T1>::type, T>::value) , "Failed to initialise storage_traits object.  The reinterpreted tensor object expects a number type as an input template parameter.");
    using type = const tensor_slice<arrtype<T1, D1, backend>, T, D>&;    
    using eval_type = const T*;

    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};


template <typename T, typename backend> 
struct storage_traits<hermitian_matrix<T, backend> > 
{
    static_assert(is_number<T>::value, "Failed to initialise storage_traits object.  The reinterpreted tensor object expects a number type as an input template parameter.");
    using type = const hermitian_matrix<T, backend>&;
    using eval_type = const T*;

    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

template <typename T, typename backend> 
struct storage_traits<upper_hessenberg_matrix<T, backend> > 
{
    static_assert(is_number<T>::value, "Failed to initialise storage_traits object.  The reinterpreted tensor object expects a number type as an input template parameter.");
    using type = const upper_hessenberg_matrix<T, backend>&;
    using eval_type = const T*;

    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

//traits for the different expression objects
template <typename expr, size_t rank, typename backend> 
struct storage_traits<expression_tree<expr, rank, backend> > 
{
    using type = expression_tree<expr, rank, backend>;
    using eval_type = expr;

    static inline eval_type data(type a){return a.expression();}
};

//traits for the tensor objects
template <typename T, typename backend> 
struct storage_traits<csr_matrix<T, backend> > 
{
    using type = const csr_matrix<T, backend>&;
    using eval_type = const T*;
    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

namespace internal
{
template <typename T, typename backend> struct diagonal_matrix_view_storage_type;

template <typename T> 
struct diagonal_matrix_view_storage_type<T, blas_backend>
{
    using size_type = typename blas_backend::size_type;
    T* buffer;
    size_type incx;
    T operator[](size_type i)const{return buffer[i*incx];}
};

#ifdef __NVCC__
template <typename T> 
struct diagonal_matrix_view_storage_type<T, cuda_backend>
{
    using size_type = typename cuda_backend::size_type;
    T* buffer;  size_type incx; 
    __host__ __device__ T operator[](size_type i)const{return buffer[i*incx];}
};

#endif
}   //namespace intenral

//traits for the tensor objects
template <typename T, typename backend> 
struct storage_traits<diagonal_matrix<T, backend> > 
{
    using type = const diagonal_matrix<T, backend>&;
    using eval_type = const T*;
    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

//traits for the tensor objects
template <typename T, typename backend> 
struct storage_traits<symmetric_tridiagonal_matrix<T, backend> > 
{
    using type = const symmetric_tridiagonal_matrix<T, backend>&;
    using eval_type = const T*;
    static inline eval_type data(type a){return static_cast<eval_type>(a.buffer());}
};

}   //namespace expression_templates
}   //namespace linalg

#endif  //LINALG_ALGEBRA_STORAGE_TRAITS_HPP//

