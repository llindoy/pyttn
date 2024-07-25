#ifndef LINALG_ALGEBRA_OVERLOADS_TRACE_HPP
#define LINALG_ALGEBRA_OVERLOADS_TRACE_HPP

namespace linalg
{

namespace internal
{
template <typename T>
struct is_valid_trace : 
    public std::conditional
    < 
        is_dense_tensor<T>::value ,
        typename std::conditional
        <   
            traits<T>::rank == 2 && !is_expression<T>::value,
            std::true_type,
            std::false_type
        >::type,
        typename std::conditional
        <
            is_diagonal_matrix_type<T>::value && !is_expression<T>::value,
            std::true_type,
            std::false_type
        >::type
    >::type
{};
}   //namespace internal

//dot product of two arrays
template <typename T>
typename std::enable_if<internal::is_valid_trace<T>::value, typename traits<T>::value_type>::type trace(const T& a)
{
    using value_type = typename std::remove_cv<typename traits<T>::value_type>::type;
    using backend_type = typename traits<T>::backend_type;
    
    ASSERT(a.shape(0) == a.shape(1), "Failed to evaluate trace of matrix.  The matrix is not a square matrix.");
    value_type val; CALL_AND_HANDLE(val = backend_type::trace(a.shape(1), a.buffer(), a.diagonal_stride()), "Failed to evaluate trace of matrix.  Failed to call the backend::trace routine.");
    return val;
}

//dot product of conjugate expressions
template <typename T>
typename std::enable_if<internal::is_valid_trace<T>::value, typename traits<T>::value_type>::type trace(const conj_type<T>& a)
{
    using value_type = typename std::remove_cv<typename traits<T>::value_type>::type;
    using backend_type = typename traits<T>::backend_type;
    
    ASSERT(a.shape(0) == a.shape(1), "Failed to evaluate trace of matrix.  The matrix is not a square matrix.");
    value_type val; CALL_AND_HANDLE(val = backend_type::trace(a.shape(1), a.buffer(), a.diagonal_stride()), "Failed to evaluate trace of matrix.  Failed to call the backend::trace routine.");
    return conj(val);
}
}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_TRACE_HPP//


