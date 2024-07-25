#ifndef LINALG_ALGEBRA_OVERLOADS_TRANSPOSE_DENSE_HPP
#define LINALG_ALGEBRA_OVERLOADS_TRANSPOSE_DENSE_HPP

#include <initializer_list>

namespace linalg
{

//transpose of matrix
template <typename array_type>
trans_return_type<array_type, false> trans(const array_type& a){using rettype = trans_type<array_type, false>; CALL_AND_RETHROW(return rettype(a));}

//transpose of conjugate is conjugate transpose
template <typename array_type>
trans_return_type<array_type, true> trans(const conj_type<array_type>& a){using rettype = trans_type<array_type, true>; CALL_AND_RETHROW(return rettype(a.obj()));}

//transpose of transpose is object
template <typename array_type> 
typename std::enable_if<array_type::rank == 2, const array_type&>::type trans(const trans_type<array_type, false>& a){return a.matrix();}

template <typename array_type> 
typename std::enable_if<array_type::rank == 2, conj_type<array_type>>::type trans(const trans_type<array_type, true>& a){CALL_AND_RETHROW(return conj(a.matrix()));}

//conjugate transpose of matrix
template <typename array_type>
trans_return_type<array_type, true> adjoint(const array_type& a){using rettype = trans_type<array_type, true>; CALL_AND_RETHROW(return rettype(a));}

//conjugate transpose of conjugate transpose
template <typename array_type> 
typename std::enable_if<array_type::rank == 2, const array_type&>::type adjoint(const trans_type<array_type, true>& a){return a.matrix();}

//conjugate transpose of transpose
template <typename array_type> typename std::enable_if<array_type::rank == 2, conj_type<array_type> >::type adjoint(const trans_type<array_type, false>& a){CALL_AND_RETHROW(return conj(a.matrix()));}

//transpose of scalar times matrix
template <typename T, typename array_type>
scal_trans_return_type<T, array_type, false> trans(const scal_type<T, array_type>& a)
{
    using rettype = trans_type<array_type, false>;  
    CALL_AND_RETHROW(return rettype(a.right(), static_cast<typename array_type::value_type>(a.left())));
}

//conjugate transpose of scalar time matrix
template <typename T, typename array_type>
scal_trans_return_type<T, array_type, true> 
adjoint(const scal_type<T, array_type>& a)
{
    using std::conj;
    using rettype = trans_type<array_type, true>;  
    CALL_AND_RETHROW(return rettype(a.right(), conj(static_cast<typename array_type::value_type>(a.left()))));
}

//transpose of matrix matrix product - we need to figure out the best way of implementing this.
template <typename T1, typename T2> 
gemm_return_type<T1, T2> trans(const expression_templates::matrix_matrix_product<T1, T2>& r){using rettype = gemm_type<T1, T2>;  CALL_AND_RETHROW(return rettype(r, true));}

template <typename T1, typename T2> 
gemm_return_type<T1, T2> adjoint(const expression_templates::matrix_matrix_product<T1, T2>& r){using rettype = gemm_type<T1, T2>;  CALL_AND_RETHROW(return rettype(r, true, true));}

//transpose of scalar*conjugate
template <typename T1, typename T2>
trans_return_type<T2, true> trans(const scalconj_type<T1, T2>& a){using rettype = trans_type<T2, true>; CALL_AND_RETHROW(return rettype(a.right().obj(), static_cast<typename T2::value_type>(a.left())));}

//conjugate transpose of scalar*conjugate
template <typename T1, typename T2>
trans_return_type<T2, false> adjoint(const scalconj_type<T1, T2>& a){using std::conj; using rettype = trans_type<T2, false>; CALL_AND_RETHROW(return rettype(a.right().obj(), conj(static_cast<typename T2::value_type>(a.left()))));}


template <typename tensor_type, typename int_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
trans(const tensor_type& A, const std::vector<int_type>& order)
{
    return expression_templates::tensor_transpose_expression<tensor_type>(A, order);
}

template <typename tensor_type, typename int_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
trans(const tensor_type& A, const std::array<int_type, tensor_type::rank>& order)
{
    std::vector<int_type> vec(order.begin(), order.end());
    return expression_templates::tensor_transpose_expression<tensor_type>(A, vec);
}

template <typename tensor_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
trans(const tensor_type& A, const strict_array<int, tensor_type::rank>& order)
{
    std::vector<size_t> vec(order.begin(), order.end());
    return expression_templates::tensor_transpose_expression<tensor_type>(A, vec);
}


template <typename tensor_type, typename int_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
transpose(const tensor_type& A, const std::vector<int_type>& order)
{
    return expression_templates::tensor_transpose_expression<tensor_type>(A, order);
}

template <typename tensor_type, typename int_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
transpose(const tensor_type& A, const std::array<int_type, tensor_type::rank>& order)
{
    std::vector<int_type> vec(order.begin(), order.end());
    return expression_templates::tensor_transpose_expression<tensor_type>(A, vec);
}

template <typename tensor_type>
typename std::enable_if<is_linalg_object<tensor_type>::value,expression_templates::tensor_transpose_expression<tensor_type>>::type 
transpose(const tensor_type& A, const strict_array<int, tensor_type::rank>& order)
{
    std::vector<size_t> vec(order.begin(), order.end());
    return expression_templates::tensor_transpose_expression<tensor_type>(A, vec);
}

}   //namespace linalg


#endif  //LINALG_ALGEBRA_OVERLOADS_TRANSPOSE_DENSE_HPP//
