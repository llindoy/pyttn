#ifndef LINALG_ALGEBRA_OVERLOADS_TENSOR_PERMUTATION_DENSE_HPP
#define LINALG_ALGEBRA_OVERLOADS_TENSOR_PERMUTATION_DENSE_HPP

namespace linalg
{


template <typename T>
perm3_return_type<T, false> permute_dims(const T& a, typename traits<T>::size_type d1, typename traits<T>::size_type d2)
{
    using size_type = typename traits<T>::size_type;
    ASSERT(d1 == 2 || d2 == 2, "The permute_dims function currently only supports permutation of the final dimension with a given dimension.");
    size_type dim = (d1 == 2) ? d2 : d1;
    using rettype = perm3_type<T, false>;
    CALL_AND_RETHROW(return rettype(a, dim));
}

//tensor_permutation_3 of conjugate is conjugate tensor_permutation_3
template <typename T>
perm3_return_type<T, true> permute_dims(const conj_type<T>& a, typename traits<T>::size_type d1, typename traits<T>::size_type d2)
{
    using size_type = typename traits<T>::size_type;
    ASSERT(d1 == 2 || d2 == 2, "The permute_dims function currently only supports permutation of the final dimension with a given dimension.");
    size_type dim = (d1 == 2) ? d2 : d1;
    using rettype = perm3_type<T, true>;
    CALL_AND_RETHROW(return rettype(a.obj(), dim));
}

//tensor_permutation_3 of scalar times tensor
template <typename T1, typename T2>
scal_perm3_return_type<T1, T2, false> permute_dims(const scal_type<T1, T2>& a, typename traits<T2>::size_type d1, typename traits<T2>::size_type d2)
{
    using size_type = typename traits<T2>::size_type;
    ASSERT(d1 == 2 || d2 == 2, "The permute_dims function currently only supports permutation of the final dimension with a given dimension.");
    size_type dim = (d1 == 2) ? d2 : d1;
    using rettype = perm3_type<T2, false>;
    CALL_AND_RETHROW(return rettype(a.right(), dim, static_cast<typename traits<T2>::value_type>(a.left())));
}


template <typename T1, typename T2>
scal_perm3_return_type<T1, T2, true> permute_dims(const scalconj_type<T1, T2>& a, typename traits<T2>::size_type d1, typename traits<T2>::size_type d2)
{
    using size_type = typename traits<T2>::size_type;
    ASSERT(d1 == 2 || d2 == 2, "The permute_dims function currently only supports permutation of the final dimension with a given dimension.");
    size_type dim = (d1 == 2) ? d2 : d1;

    using rettype = perm3_type<T2, true>;
    CALL_AND_RETHROW(return rettype(a.right().obj(), dim, static_cast<typename T2::value_type>(a.left())));
}


}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_TENSOR_PERMUTATION_DENSE_HPP//

