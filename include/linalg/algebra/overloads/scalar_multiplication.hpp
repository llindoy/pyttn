#ifndef LINALG_ALGEBRA_OVERLOADS_SCALAR_MULTIPLICATION_DENSE_HPP
#define LINALG_ALGEBRA_OVERLOADS_SCALAR_MULTIPLICATION_DENSE_HPP

namespace linalg
{

//now we add overloads for scalar multiplication
//tensor - scalar
template <template <typename, size_t, typename> class c1, typename T1, typename T2, size_t D, typename backend>
scal_return_type<T2, c1<T1, D, backend>> operator*(const c1<T1, D, backend>& a, const T2& b)
{
    using ttype = c1<T1, D, backend>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(b), a), a.shape());
}

//scalar - tensor
template <template <typename, size_t, typename> class c1, typename T1, typename T2, size_t D, typename backend>
scal_return_type<T2, c1<T1, D, backend>> operator*(const T2& a, const c1<T1, D, backend>& b)
{
    using ttype = c1<T1, D, backend>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(a), b), b.shape());
}

//tensor slice - scalar
template <template <typename, size_t, typename> class c1, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
scal_return_type<T2, tensor_slice<c1<T, D1, backend>, T1, D>> operator*(const tensor_slice<c1<T, D1, backend>, T1, D>& a, const T2& b)
{
    using ttype = tensor_slice<c1<T, D1, backend>, T1, D>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(b), a), a.shape());
}

//scalar - tensor slice
template <template <typename, size_t, typename> class c1, typename T, typename T1, typename T2, size_t D, size_t D1, typename backend>
scal_return_type<T2, tensor_slice<c1<T, D1, backend>, T1, D>> operator*(const T2& a, const tensor_slice<c1<T, D1, backend>, T1, D>& b)
{
    using ttype = tensor_slice<c1<T, D1, backend>, T1, D>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(a), b), b.shape());
}

//special tensor - scalar
template <template <typename, typename> class c1, typename T1, typename T2, typename backend>
scal_return_type<T2, c1<T1, backend>> operator*(const c1<T1, backend>& a, const T2& b)
{
    using ttype = c1<T1,  backend>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(b), a), a.shape());
}

//scalar - special tensor
template <template <typename, typename> class c1, typename T1, typename T2,typename backend>
scal_return_type<T2, c1<T1, backend>> operator*(const T2& a, const c1<T1, backend>& b)
{
    using ttype = c1<T1,  backend>;
    using lit_type = expression_templates::literal_type<T2, typename traits<ttype>::backend_type>;
    return scal_type<T2, ttype>(scal_binary_type<T2, ttype>(lit_type(a), b), b.shape());
}



//scalar - scalar*expression
template <typename T1, typename T2, typename arrtype>
scal_scal_return_type<T1, T2, arrtype> operator*(const T1& a, const scal_type<T2, arrtype>& b)
{
    using valtype = decltype(T1()*T2());
    using lit_type = expression_templates::literal_type<valtype, typename traits<arrtype>::backend_type>;
    return scal_type<valtype, arrtype>(scal_binary_type<valtype, arrtype>(lit_type(static_cast<valtype>(static_cast<T2>(b.left())*a)), b.right()), b.shape());
}

//scalar*expression - scalar
template <typename T1, typename T2, typename arrtype>
scal_scal_return_type<T1, T2, arrtype> operator*(const scal_type<T2, arrtype>& a, const T1& b)
{
    using valtype = decltype(T1()*T2());
    using lit_type = expression_templates::literal_type<valtype, typename traits<arrtype>::backend_type>;
    return scal_type<valtype, arrtype>(scal_binary_type<valtype, arrtype>(lit_type(static_cast<valtype>(static_cast<T2>(a.left())*b)), a.right()), a.shape());
}

//scalar - trans(matrix)
template <typename T1, typename T2, bool conjugate>
scal_trans_return_type<T2, T1, conjugate> operator*(const T2& l, const trans_type<T1, conjugate>& r){return trans_type<T1, conjugate>(r.matrix(), l*r.coeff());}

//trans(matrix) - scalar
template <typename T1, typename T2, bool conjugate>
scal_trans_return_type<T2, T1, conjugate> operator*(const trans_type<T1, conjugate>& l, const T2& r){return trans_type<T1, conjugate>(l.matrix(), r*l.coeff());}

//scalar - permute_dims(tensor)
template <typename T1, typename T2, bool conjugate>
scal_perm3_return_type<T2, T1, conjugate> operator*(const T2& l, const perm3_type<T1, conjugate>& r){return perm3_type<T1, conjugate>(r.tensor(), r.permutation_index(), l*r.coeff());}

//permute_dims(tensor) - scalar
template <typename T1, typename T2, bool conjugate>
scal_perm3_return_type<T2, T1, conjugate> operator*(const perm3_type<T1, conjugate>& l, const T2& r){return perm3_type<T1, conjugate>(l.tensor(), l.permutation_index(), r*l.coeff());}



//TODO fix these up later -  also maybe make all matrix-matrix product and such routines derive from a contraction class - and implement this solely for contraction objects rather than
//for each individual contraction object.
//matrix product * scalar
template <typename T1, typename T2, typename T3> scal_gemm_return_type<T3, T1, T2>  operator*(const T3& l, const expression_templates::matrix_matrix_product<T1, T2>& r){return expression_templates::matrix_matrix_product<T1, T2>(r, l);}
template <typename T1, typename T2, typename T3> scal_gemm_return_type<T3, T1, T2> operator*(const expression_templates::matrix_matrix_product<T1, T2>& l, const T3& r){return expression_templates::matrix_matrix_product<T1, T2>(l, r);}

template <typename T1, typename T2, typename T3> scal_gemv_return_type<T3, T1, T2>  operator*(const T3& l, const expression_templates::matrix_vector_product<T1, T2>& r){return expression_templates::matrix_vector_product<T1, T2>(r, l);}
template <typename T1, typename T2, typename T3> scal_gemv_return_type<T3, T1, T2> operator*(const expression_templates::matrix_vector_product<T1, T2>& l, const T3& r){return expression_templates::matrix_vector_product<T1, T2>(l, r);}


template <typename T1, typename T2, typename T3>
scal_mtc1_return_type<T3, T1, T2>  operator*(const mtc1_type<T1, T2>& l, const T3& r){return mtc1_type<T1, T2>(l, static_cast<typename mtc1_type<T1, T2>::value_type>(r));}

template <typename T1, typename T2, typename T3>
scal_mtc1_return_type<T1, T2, T3>  operator*(const T1& l, const mtc1_type<T2, T3>& r){return mtc1_type<T2, T3>(r, static_cast<typename mtc1_type<T2, T3>::value_type>(l));}

template <typename T1, typename T2, typename T3>
scal_ttc2_return_type<T1, T2, T3>
operator*(const ttc2_type<T2, T3>& l, const T1& r){return ttc2_type<T2, T3>(l, static_cast<typename ttc2_type<T2, T3>::value_type>(r));}

template <typename T1, typename T2, typename T3>
scal_ttc2_return_type<T1, T2, T3> 
operator*(const T1& l, const ttc2_type<T2, T3>& r){return ttc2_type<T2, T3>(r, static_cast<typename ttc2_type<T2, T3>::value_type>(l));}


//operator /
template <typename T1, typename T2, typename = typename std::enable_if< (is_tensor<T1>::value || is_expression<T1>::value) && is_number<T2>::value, void>::type>
auto operator/(const T1& t, const T2& o) -> decltype(t*(static_cast<typename traits<T1>::value_type>(1.0)/o)){return t*(static_cast<typename traits<T1>::value_type>(1.0)/o);}

}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_SCALAR_MULTIPLICATION_OVERLOAD_DENSE_HPP//

