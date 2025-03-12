#ifndef LINALG_OVERLOADS_EXPRESSION_TYPE_ALIASE_HPP
#define LINALG_OVERLOADS_EXPRESSION_TYPE_ALIASE_HPP

#include "../../linalg_forward_decl.hpp"

namespace linalg
{

//type aliases for the various different expressions that we have
template <typename T1, typename T2> using gemv_type = expression_templates::matrix_vector_product<T1, T2>;

template <typename T1, typename T2> 
using gemm_type = 
    typename std::conditional
    < 
        is_dense<T1>::value,
        typename std::conditional
        <
            is_dense<T2>::value, 
            expression_templates::matrix_matrix_product<T1, T2>,
            expression_templates::matrix_matrix_product<T2, T1>
        >::type,
        expression_templates::matrix_matrix_product<T1, T2>
    >::type;

template <typename T, bool conj> using trans_type = expression_templates::transpose_expression<T, conj>;

template <typename T, bool conj> using perm3_type = expression_templates::tensor_permutation_3_expression<T, conj>;

template <typename T1, typename T2>
struct validate_axpy_binary_type
{
    using axpy_binary_type = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value &&
        expression_templates::addition_allowed<T1>::value && expression_templates::addition_allowed<T2>::value,
        expression_templates::binary_expression<T1, T2, expression_templates::addition_op, typename traits<T2>::backend_type>>::type;
};
template <typename T1, typename T2> using axpy_binary_type = typename validate_axpy_binary_type<T1, T2>::axpy_binary_type;
template <typename T1, typename T2> using axpy_type = expression_templates::expression_tree<axpy_binary_type<T1, T2>, traits<T1>::rank, typename traits<T1>::backend_type>;


template <typename T1, typename T2>
struct validate_hadamard_binary_type
{
    using hadamard_binary_type = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value &&
        expression_templates::hadamard_allowed<T1>::value && expression_templates::hadamard_allowed<T2>::value,
        expression_templates::binary_expression<T1, T2, expression_templates::hadamard_op, typename traits<T2>::backend_type>>::type;
};
template <typename T1, typename T2> using hadamard_binary_type = typename validate_hadamard_binary_type<T1, T2>::hadamard_binary_type;
template <typename T1, typename T2> using hadamard_type = expression_templates::expression_tree<hadamard_binary_type<T1, T2>, traits<T1>::rank, typename traits<T1>::backend_type>;


template <typename T1, typename T2>
struct validate_complex_binary_type
{
    using complex_binary_type = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value &&
        expression_templates::complex_allowed<T1>::value && expression_templates::complex_allowed<T2>::value,
        expression_templates::binary_expression<T1, T2, expression_templates::complex_op, typename traits<T2>::backend_type>>::type;
};
template <typename T1, typename T2> using complex_binary_type = typename validate_complex_binary_type<T1, T2>::complex_binary_type;
template <typename T1, typename T2> using complex_type = expression_templates::expression_tree<complex_binary_type<T1, T2>, traits<T1>::rank, typename traits<T1>::backend_type>;

template <typename T1, typename T2>
struct validate_polar_binary_type
{
    using polar_binary_type = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value &&
        expression_templates::polar_allowed<T1>::value && expression_templates::polar_allowed<T2>::value,
        expression_templates::binary_expression<T1, T2, expression_templates::polar_op, typename traits<T2>::backend_type>>::type;
};
template <typename T1, typename T2> using polar_binary_type = typename validate_polar_binary_type<T1, T2>::polar_binary_type;
template <typename T1, typename T2> using polar_type = expression_templates::expression_tree<polar_binary_type<T1, T2>, traits<T1>::rank, typename traits<T1>::backend_type>;


template <typename T1, typename T2>
using scalconj_binary_type = expression_templates::binary_expression<expression_templates::literal_type<T1, typename traits<T2>::backend_type>, expression_templates::expression_tree<expression_templates::unary_expression<T2, expression_templates::conjugation_op, typename traits<T2>::backend_type>, traits<T2>::rank, typename traits<T2>::backend_type>, expression_templates::multiplication_op, typename traits<T2>::backend_type>;
template <typename T1, typename T2> using scalconj_type = expression_templates::expression_tree<scalconj_binary_type<T1, T2>, traits<T2>::rank, typename traits<T2>::backend_type>;

template <typename T1, typename T2> using scal_binary_type = expression_templates::binary_expression<expression_templates::literal_type<T1, typename traits<T2>::backend_type>, T2, expression_templates::multiplication_op, typename traits<T2>::backend_type>;
template <typename T1, typename T2> using scal_type = expression_templates::expression_tree<scal_binary_type<T1, T2>, traits<T2>::rank, typename traits<T2>::backend_type>;



template <typename T1, typename T2, typename = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value, void>::type >
struct is_valid_elemental
{
    using type = same_topology_type<T1, T2>;
    static constexpr bool value = type::value;
};

template <typename T1, typename T2> using addition_supported = is_valid_elemental<T1, T2>;
template <typename T1, typename T2> using hadamard_supported = is_valid_elemental<T1, T2>;
template <typename T1, typename T2> using complex_supported = is_valid_elemental<T1, T2>;
template <typename T1, typename T2> using polar_supported = is_valid_elemental<T1, T2>;

//type aliases that validate the input template parameters for the various operator overloads and give the resulting return type if the 
//input parameters are valid.
template <typename T1, typename T2> using axpy_return_type = typename std::enable_if<addition_supported<T1, T2>::value,axpy_type<T1, T2>>::type;
template <typename T1, typename T2> using hadamard_return_type = typename std::enable_if<hadamard_supported<T1, T2>::value,hadamard_type<T1, T2>>::type;
template <typename T1, typename T2> using complex_return_type = typename std::enable_if<complex_supported<T1, T2>::value,complex_type<T1, T2>>::type;
template <typename T1, typename T2> using polar_return_type = typename std::enable_if<polar_supported<T1, T2>::value, polar_type<T1, T2>>::type;

template <typename T1, typename T2>
using gemv_return_type = 
    typename std::enable_if
    <
        expression_templates::is_valid_gemv<T1, T2>::value, 
        gemv_type<T1, T2> 
    >::type;

template <typename T1, typename T2>
using gemm_return_type = 
    typename std::enable_if
    <
        expression_templates::is_valid_gemm<T1, T2>::value, 
        gemm_type<T1, T2> 
    >::type;

template <typename T, bool conjugate>
using trans_return_type = 
    typename std::enable_if
    <
        traits<T>::rank == 2,
        trans_type<T, conjugate> 
    >::type;

template <typename T, bool conjugate>
using perm3_return_type = 
    typename std::enable_if
    <
        traits<T>::rank == 3,
        perm3_type<T, conjugate> 
    >::type;

template <typename T1, typename T2>
using scalconj_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value, 
        scalconj_type<T1, T2> 
    >::type;

template <typename T1, typename T2, bool conjugate>
using scal_trans_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value && 
        std::is_convertible<T1, typename trans_type<T2, conjugate>::value_type>::value, 
        trans_type<T2, conjugate> 
    >::type;

template <typename T1, typename T2, bool conjugate>
using scal_perm3_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value && 
        std::is_convertible<T1, typename perm3_type<T2, conjugate>::value_type>::value, 
        perm3_type<T2, conjugate> 
    >::type;

template <typename T1, typename T2>
using scal_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value, 
        scal_type<T1, T2>
    >::type;

template <typename T1, typename T2>
using scal_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value, 
        scal_type<T1, T2>
    >::type;

template <typename T1, typename T2, typename T3>
using scal_scal_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value && 
        is_number<T2>::value, 
        scal_type<decltype(T1()*T2()), T3>
    >::type;

template <typename T1, typename T2, typename T3>
using scal_gemv_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the scalar type is a number
        expression_templates::is_valid_gemv<T2, T3>::value &&                                      //check that the two matrices form a valid matrix matrix product
        std::is_convertible<T1, typename gemv_type<T2, T3>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        gemv_type<T2, T3> 
    >::type;

template <typename T1, typename T2, typename T3, typename T4>
using scal_scal_gemv_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the first scalar type is a number
        is_number<T3>::value &&                                                                                     //check that the second scalar type is a number
        expression_templates::is_valid_gemv<T2, T4>::value &&                                      //check that the two matrices form a valid matrix matrix product
        std::is_convertible<decltype(T1()*T3()), typename gemv_type<T2, T4>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        gemv_type<T2, T4> 
    >::type;



template <typename T1, typename T2, typename T3>
using scal_gemm_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the scalar type is a number
        expression_templates::is_valid_gemm<T2, T3>::value &&                                      //check that the two matrices form a valid matrix matrix product
        std::is_convertible<T1, typename gemm_type<T2, T3>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        gemm_type<T2, T3> 
    >::type;

template <typename T1, typename T2, typename T3, typename T4>
using scal_scal_gemm_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the first scalar type is a number
        is_number<T3>::value &&                                                                                     //check that the second scalar type is a number
        expression_templates::is_valid_gemm<T2, T4>::value &&                                      //check that the two matrices form a valid matrix matrix product
        std::is_convertible<decltype(T1()*T3()), typename gemm_type<T2, T4>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        gemm_type<T2, T4> 
    >::type;


template <typename T1, typename T2> 
using mtc1_type = 
    typename std::conditional
    < 
        traits<T1>::rank == 3,
        expression_templates::tensor_contraction_1_mt<T1, T2>,
        expression_templates::tensor_contraction_1_mt<T2, T1>
    >::type;

template <typename T1, typename T2>
using mtc1_return_type = 
    typename std::enable_if
    <
        expression_templates::is_valid_mtc1<T1, T2>::value || expression_templates::is_valid_mtc1<T2, T1>::value, 
        mtc1_type<T1, T2> 
    >::type;


template <typename T1, typename T2, typename T3>
using scal_mtc1_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the scalar type is a number
        expression_templates::is_valid_mtc1<T2, T3>::value &&                                      //check that the two matrices form a valid matrix tensor product
        std::is_convertible<T1, typename mtc1_type<T2, T3>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        mtc1_type<T2, T3> 
    >::type;

template <typename T1, typename T2, typename T3, typename T4>
using scal_scal_mtc1_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the first scalar type is a number
        is_number<T3>::value &&                                                                                     //check that the second scalar type is a number
        expression_templates::is_valid_mtc1<T2, T4>::value &&                                      //check that the two matrices form a valid matrix tensor product
        std::is_convertible<decltype(T1()*T3()), typename mtc1_type<T2, T4>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        mtc1_type<T2, T4> 
    >::type;

template <typename T1, typename T2> 
using ttc2_type = expression_templates::tensor_contraction_332<T1, T2>;

template <typename T1, typename T2>
using ttc2_return_type = 
    typename std::enable_if
    <
        expression_templates::is_valid_ttc2<T1, T2>::value, 
        ttc2_type<T1, T2> 
    >::type;

template <typename T1, typename T2, typename T3>
using scal_ttc2_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the scalar type is a number
        expression_templates::is_valid_ttc2<T2, T3>::value &&                                      //check that the two matrices form a valid matrix tensor product
        std::is_convertible<T1, typename ttc2_type<T2, T3>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        ttc2_type<T2, T3> 
    >::type;

template <typename T1, typename T2, typename T3, typename T4>
using scal_scal_ttc2_return_type = 
    typename std::enable_if
    <
        is_number<T1>::value &&                                                                                     //check that the first scalar type is a number
        is_number<T3>::value &&                                                                                     //check that the second scalar type is a number
        expression_templates::is_valid_ttc2<T2, T4>::value &&                                      //check that the two matrices form a valid matrix tensor product
        std::is_convertible<decltype(T1()*T3()), typename ttc2_type<T2, T4>::value_type>::value,   //check that the scalar type is convertible to the matrix value type
        ttc2_type<T2, T4> 
    >::type;

//functions for treating complex numbers
template <typename T> using conj_unary_type = expression_templates::unary_expression<T, expression_templates::conjugation_op, typename traits<T>::backend_type>;
template <typename T> using conj_type = expression_templates::expression_tree<conj_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using real_unary_type = expression_templates::unary_expression<T, expression_templates::real_op, typename traits<T>::backend_type>;
template <typename T> using real_type = expression_templates::expression_tree<real_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using imag_unary_type = expression_templates::unary_expression<T, expression_templates::imag_op, typename traits<T>::backend_type>;
template <typename T> using imag_type = expression_templates::expression_tree<imag_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using norm_unary_type = expression_templates::unary_expression<T, expression_templates::norm_op, typename traits<T>::backend_type>;
template <typename T> using norm_type = expression_templates::expression_tree<norm_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using arg_unary_type = expression_templates::unary_expression<T, expression_templates::arg_op, typename traits<T>::backend_type>;
template <typename T> using arg_type = expression_templates::expression_tree<arg_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using unit_polar_unary_type = expression_templates::unary_expression<T, expression_templates::unit_polar_op, typename traits<T>::backend_type>;
template <typename T> using unit_polar_type = expression_templates::expression_tree<unit_polar_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;

template <typename T> using elemental_exp_unary_type = expression_templates::unary_expression<T, expression_templates::elemental_exp_op, typename traits<T>::backend_type>;
template <typename T> using elemental_exp_type = expression_templates::expression_tree<elemental_exp_unary_type<T>, traits<T>::rank, typename traits<T>::backend_type>;


template <typename T> using conj_return_type = typename std::enable_if<is_linalg_object<T>::value,conj_type<T>>::type;
template <typename T> using real_return_type = typename std::enable_if<is_linalg_object<T>::value,real_type<T>>::type;
template <typename T> using imag_return_type = typename std::enable_if<is_linalg_object<T>::value,imag_type<T>>::type;
template <typename T> using norm_return_type = typename std::enable_if<is_linalg_object<T>::value,norm_type<T>>::type;
template <typename T> using arg_return_type =  typename std::enable_if<is_linalg_object<T>::value, arg_type<T>>::type;
template <typename T> using unit_polar_return_type =  typename std::enable_if<is_linalg_object<T>::value, unit_polar_type<T>>::type;
template <typename T> using elemental_exp_return_type = typename std::enable_if<is_linalg_object<T>::value,elemental_exp_type<T>>::type;

//TODO introduce generic tensor contraction result type aliases

}   //namespace linalg

#endif


