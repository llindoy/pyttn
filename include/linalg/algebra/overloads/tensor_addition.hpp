#ifndef LINALG_ALGEBRA_OVERLOADS_ADDITION_DENSE_HPP
#define LINALG_ALGEBRA_OVERLOADS_ADDITION_DENSE_HPP

namespace linalg
{

//operator+ overloads for the various array additions
/*
template <typename T1, typename T2, typename = typename std::enable_if< is_tensor<T1>::value && is_tensor<T2>::value, void>::type>>
axpy_return_type<T1, T2> operator+(const T1& a, const T2& b)
{
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<T1, T2>(axpy_binary_type<T1, T2>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, size_t D, typename backend>
template <typename T1, typename T2, typename = typename std::enable_if< is_tensor<T1>::value && is_expression<T2>::value, void>::type>>
axpy_return_type<T1, T2> operator+(const T1& a, const T2& b)
{
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<T1, T2>(axpy_binary_type<T1, T2>(a, b.expression()), a.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, size_t D, typename backend>
template <typename T1, typename T2, typename = typename std::enable_if< is_expression<T1>::value && is_tensor<T2>::value, void>::type>>
axpy_return_type<T1, T2> operator+(const T1& a, const T2& b)
{
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<T1, T2>(axpy_binary_type<T1, T2>(a.expression(), b), a.shape());
}
*/

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T1, typename T2, size_t D, typename backend>
axpy_return_type<c1<T1, D, backend>, c2<T2, D, backend>> operator+(const c1<T1, D, backend>& a, const c2<T2, D, backend>& b)
{
    using t1type = c1<T1, D, backend>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, size_t D, typename backend>
axpy_return_type<c1<T1, D, backend>, c2<T2, backend>> operator+(const c1<T1, D, backend>& a, const c2<T2, backend>& b)
{
    using t1type = c1<T1, D, backend>;    using t2type = c2<T2, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, typename> class c1, template <typename, size_t, typename> class c2, typename T1, typename T2, size_t D, typename backend>
axpy_return_type<c1<T1, backend>, c2<T2, D, backend>> operator+(const c1<T1, backend>& a, const c2<T2, D, backend>& b)
{
    using t1type = c1<T1, backend>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, typename backend>
axpy_return_type<c1<T1, backend>, c2<T2, backend>> operator+(const c1<T1, backend>& a, const c2<T2, backend>& b)
{
    using t1type = c1<T1, backend>;    using t2type = c2<T2, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
axpy_return_type<tensor_slice<c1<T, D1, backend>, T1, D>, c2<T2, D, backend>>
operator+(const tensor_slice<c1<T, D1, backend>, T1, D>& a, const c2<T2, D, backend>& b)
{
    using t1type = tensor_slice<c1<T, D1, backend>, T1, D>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
axpy_return_type<c2<T2, D, backend>, tensor_slice<c1<T, D1, backend>, T1, D>> operator+(const c2<T2, D, backend>& a, const tensor_slice<c1<T, D1, backend>, T1, D>& b)
{
    using t1type = c2<T2, D, backend>;    using t2type = tensor_slice<c1<T, D1, backend>, T1, D>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename U, typename T1, typename T2, size_t D, size_t D1, size_t D2, typename backend>
axpy_return_type<tensor_slice<c1<T, D1, backend>, T1, D>, tensor_slice<c2<U, D2, backend>, T2, D>> operator+( const tensor_slice<c1<T, D1, backend>, T1, D>& a, const tensor_slice<c2<U, D2, backend>, T2, D>& b)
{
    using t1type = tensor_slice<c1<T, D1, backend>, T1, D>;    using t2type = tensor_slice<c2<T, D2, backend>, T2, D>;
    ASSERT(a.shape() == b.shape(), "Failed to add two objects.  The two objects do not have the same shape.");
    return axpy_type<t1type, t2type>(axpy_binary_type<t1type, t2type>(a, b), b.shape());
}

//operator -
template <typename T1, typename T2, typename = typename std::enable_if< is_linalg_object<T1>::value && is_linalg_object<T2>::value, void>::type>
auto operator-(const T1& a, const T2& b) -> decltype( a + static_cast<typename traits<T2>::value_type>(-1.0)*b){using cv = typename traits<T2>::value_type; CALL_AND_RETHROW(return a + static_cast<cv>(-1.0)*b);}


}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_ADDITION_OPERATOR_DENSE_HPP//

