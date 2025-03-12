#ifndef LINALG_ALGEBRA_OVERLOADS_HADAMARD_HPP
#define LINALG_ALGEBRA_OVERLOADS_HADAMARD_HPP

namespace linalg
{

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T1, typename T2, size_t D, typename backend>
hadamard_return_type<c1<T1, D, backend>, c2<T2, D, backend>> hadamard(const c1<T1, D, backend>& a, const c2<T2, D, backend>& b)
{
    using t1type = c1<T1, D, backend>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, size_t D, typename backend>
hadamard_return_type<c1<T1, D, backend>, c2<T2, backend>> hadamard(const c1<T1, D, backend>& a, const c2<T2, backend>& b)
{
    using t1type = c1<T1, D, backend>;    using t2type = c2<T2, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, typename> class c1, template <typename, size_t, typename> class c2, typename T1, typename T2, size_t D, typename backend>
hadamard_return_type<c1<T1, backend>, c2<T2, D, backend>> hadamard(const c1<T1, backend>& a, const c2<T2, D, backend>& b)
{
    using t1type = c1<T1, backend>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, typename> class c1, template <typename, typename> class c2, typename T1, typename T2, typename backend>
hadamard_return_type<c1<T1, backend>, c2<T2, backend>> hadamard(const c1<T1, backend>& a, const c2<T2, backend>& b)
{
    using t1type = c1<T1, backend>;    using t2type = c2<T2, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
hadamard_return_type<tensor_slice<c1<T, D1, backend>, T1, D>, c2<T2, D, backend>>
hadamard(const tensor_slice<c1<T, D1, backend>, T1, D>& a, const c2<T2, D, backend>& b)
{
    using t1type = tensor_slice<c1<T, D1, backend>, T1, D>;    using t2type = c2<T2, D, backend>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
hadamard_return_type<c2<T2, D, backend>, tensor_slice<c1<T, D1, backend>, T1, D>> hadamard(const c2<T2, D, backend>& a, const tensor_slice<c1<T, D1, backend>, T1, D>& b)
{
    using t1type = c2<T2, D, backend>;    using t2type = tensor_slice<c1<T, D1, backend>, T1, D>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

template <template <typename, size_t, typename> class c1, template <typename, size_t, typename> class c2, typename T, typename U, typename T1, typename T2, size_t D, size_t D1, size_t D2, typename backend>
hadamard_return_type<tensor_slice<c1<T, D1, backend>, T1, D>, tensor_slice<c2<U, D2, backend>, T2, D>> hadamard( const tensor_slice<c1<T, D1, backend>, T1, D>& a, const tensor_slice<c2<U, D2, backend>, T2, D>& b)
{
    using t1type = tensor_slice<c1<T, D1, backend>, T1, D>;    using t2type = tensor_slice<c2<T, D2, backend>, T2, D>;
    ASSERT(a.shape() == b.shape(), "Failed to compute hadamard product two objects.  The two objects do not have the same shape.");
    return hadamard_type<t1type, t2type>(hadamard_binary_type<t1type, t2type>(a, b), b.shape());
}

}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_HADAMARD_HPP//

