#ifndef LINALG_TYPE_TRAITS_HPP
#define LINALG_TYPE_TRAITS_HPP

#include <type_traits>
#include <array>

#include "utils/linalg_utils.hpp"


namespace linalg
{

template<typename T, size_t N>
struct strict_array : std::array<T, N>
{
    template<typename ... V>
    strict_array(V... vals) // no &&/forward to simplify
        : std::array<T, N>( {vals...} )
    {
        static_assert(sizeof...(vals) == N, "Please provide exactly 5 values");
    }
}; 

//TODO: Add all of the type traits objects to another namespace to avoid namespace pollution

/////////////////////////////////////////////////////////////////////////////////////////////////
//           type traits for determining whether an object is a valid backend type             //
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename backend> struct is_valid_backend : public std::is_base_of<backend_base, backend>{};

template <typename T, typename res = void> using validate_value_type = typename std::enable_if<is_number<typename remove_cvref<T>::type >::value, res>::type;
template <typename backend, typename res = void> using validate_backend_type = typename std::enable_if<is_valid_backend<backend>::value, res>::type;
template <typename T, typename backend, typename res = void> using validate_value_backend_type = typename std::enable_if<is_number<typename remove_cvref<T>::type>::value && is_valid_backend<backend>::value, res>::type;

/////////////////////////////////////////////////////////////////////////////////////////////////
//type traits for determining whether an object is a tensor and if it what type of tensor it is//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename type> struct is_tensor : public std::is_base_of<generic_tensor_type, typename remove_cvref<type>::type>{};
template <typename type> struct is_dense : public std::is_base_of<dense_type, typename remove_cvref<type>::type>{};

template <typename type> struct is_expression : public std::is_base_of<generic_expression_type, typename remove_cvref<type>::type>{};
template <typename type> struct is_sparse : public std::is_base_of<sparse_type, typename remove_cvref<type>::type>{};

template <typename type> struct is_linalg_object{static constexpr bool value = is_expression<type>::value || is_tensor<type>::value;};

template <typename type> struct is_dense_tensor{static constexpr bool value = is_dense<type>::value && is_tensor<type>::value;};
template <typename type> struct is_sparse_tensor{static constexpr bool value = is_sparse<type>::value && is_tensor<type>::value;};

template <typename T, typename res = void> using validate_linalg_type = typename std::enable_if<is_linalg_object<T>::value, res>::type;
template <typename T, typename res = void> using validate_tensor_type = typename std::enable_if<is_tensor<T>::value, res>::type;
template <typename T, typename res = void> using validate_dense_tensor_type = typename std::enable_if<is_dense_tensor<T>::value, res>::type;
template <typename T, typename res = void> using validate_sparse_tensor_type = typename std::enable_if<is_sparse_tensor<T>::value, res>::type;

template <template <typename > class get_tens_type, typename type, size_t rank>
struct _is_rank_tensor : public std::conditional<get_tens_type<type>::value, typename std::conditional<traits<type>::rank == rank, std::true_type, std::false_type>::type, std::false_type>::type{};

template <template <typename > class get_tens_type, typename type> using _is_matrix = _is_rank_tensor<get_tens_type, type, 2>;
template <template <typename > class get_tens_type, typename type> using _is_vector = _is_rank_tensor<get_tens_type, type, 1>;

template <typename type> using is_vector = _is_vector<is_tensor, typename remove_cvref<type>::type>;
template <typename type> using is_matrix = _is_matrix<is_tensor, typename remove_cvref<type>::type>;
template <typename type> using is_dense_vector = _is_vector<is_dense_tensor, typename remove_cvref<type>::type>;
template <typename type> using is_dense_matrix = _is_matrix<is_dense_tensor, typename remove_cvref<type>::type>;
template <typename type> struct is_hermitian_matrix {static constexpr bool value = is_matrix<type>::value &&  std::is_base_of<hermitian_type, type>::value;};
template <typename type> struct is_upper_hessenberg_matrix {static constexpr bool value = is_matrix<type>::value &&  std::is_base_of<upper_hessenberg_type, type>::value;};

template <typename type> using is_sparse_matrix = _is_matrix<is_sparse_tensor, type>;

template <typename type> struct is_csr_type : public std::is_base_of<csr_matrix_type, typename remove_cvref<type>::type> {};
template <typename type> struct is_diagonal_type : public std::is_base_of<diagonal_matrix_type, typename remove_cvref<type>::type> {};
template <typename type> struct is_symtridiag_type : public std::is_base_of<symmetric_tridiagonal_matrix_type, typename remove_cvref<type>::type> {};

template <typename type> struct is_csr_matrix_type {static constexpr bool value = is_csr_type<type>::value && is_matrix<type>::value;};
template <typename type> struct is_diagonal_matrix_type{static constexpr bool value = is_diagonal_type<type>::value && is_matrix<type>::value;};
template <typename type> struct is_symtridiag_matrix_type{static constexpr bool value = is_symtridiag_type<type>::value && is_matrix<type>::value;};

template <typename type> using is_tensor_rank_3 = _is_rank_tensor<is_tensor, type, 3>;
template <typename type> using is_dense_tensor_rank_3 = _is_rank_tensor<is_dense_tensor, type, 3>;

template <typename T1, typename T2> struct is_same_rank{static constexpr bool value = (traits<typename remove_cvref<T1>::type>::rank == traits<typename remove_cvref<T2>::type>::rank);};
template <typename T1, typename T2> struct is_same_backend : public std::is_same<typename traits<typename remove_cvref<T1>::type>::backend_type, typename traits<typename remove_cvref<T2>::type>::backend_type>{};
template <typename T1, typename T2> struct has_backend : public std::is_same<typename traits<typename remove_cvref<T1>::type>::backend_type, T2>{};

template <typename T1, typename T2> struct has_same_backend : public std::is_same<typename traits<typename remove_cvref<T1>::type>::backend_type, typename traits<typename remove_cvref<T2>::type>::backend_type>{};
template <typename T1, typename T2> struct is_same_value : public std::is_same<typename std::remove_cv<typename traits<typename remove_cvref<T1>::type>::value_type>::type, typename std::remove_cv<typename traits<typename remove_cvref<T2>::type>::value_type>::type>{};
template <typename T1, typename T2> struct is_real_to_complex_value{static constexpr bool value = (!is_same_value<T1, T2>::value && std::is_same<typename traits<T1>::value_type, typename get_real_type<typename traits<T2>::value_type>::type>::value);};

template <typename T1, typename T2> struct is_sparse_multiply_value {static constexpr bool value = (is_same_value<T1, T2>::value || is_real_to_complex_value<T1, T2>::value || is_real_to_complex_value<T2, T1>::value);};

template <typename T1, typename T2> struct same_sparsity_type :
    public std::conditional
    <
        is_linalg_object<T1>::value && is_linalg_object<T2>::value,
        typename std::conditional
        <
            is_sparse<T1>::value && is_sparse<T2>::value,   //check to make sure that both T1 and T2 are sparse 
            typename std::conditional                           //if they are both sparse then we can start checking that they have the same topology
            <
                (is_csr_type<T1>::value && is_csr_type<T2>::value) ||               //check to see if both objects are csr matrices
                (is_diagonal_type<T1>::value && is_diagonal_type<T2>::value) ||     //or they are both diagonal matrices
                (is_symtridiag_type<T1>::value && is_symtridiag_type<T2>::value),   //or they are both symmetric tridiagonal matrices
                std::true_type,
                std::false_type
            >::type,
            std::false_type                                     //otherwise they don't have the same sparsity type
        >::type,
        std::false_type
    >::type
{};


template <typename T1, typename T2> struct same_topology_type :
    public std::conditional
    <
        is_linalg_object<T1>::value && is_linalg_object<T2>::value,
        typename std::conditional
        <
            is_dense<T1>::value,                    //check if T1 is a dense object
            typename std::conditional                   //if T1 is dense
            <
                is_dense<T2>::value,                        //we check if T2 is a dense object
                is_same_rank<T1, T2>,                           //and the topology is the same only if the two dense objects have the same rank
                std::false_type                                 //otherwise if T2 is not dense then it is not the same topology
            >::type,
            typename std::conditional                   //if T2 is not dense
            <       
                is_dense<T2>::value,                        //we check if T2 is dense 
                std::false_type,                                //and if it is then the objects don't have the same topology
                same_sparsity_type<T1, T2>                      //then both objects should be sparse so we check that they both have the same sparse topology
            >::type
        >::type,
        std::false_type
    >::type
{};

template <typename src, typename dest>
struct copyable_traits : public std::conditional<is_same_rank<src,dest>::value && (is_same_value<src,dest>::value || (is_real_to_complex_value<src, dest>::value && is_same_backend<src, dest>::value)), std::true_type, std::false_type>::type{};

template <typename src, typename dest>
struct compatible_traits : public std::conditional<is_same_rank<src,dest>::value && is_same_value<src,dest>::value && is_same_backend<src, dest>::value, std::true_type, std::false_type>::type{};

template <typename T> struct is_const_lvalue_assignable : public std::conditional<std::is_base_of<is_const_expression_type<true>, T>::value, std::false_type, std::true_type>::type{};

template <typename src, typename dest>
struct is_dense_copy_assignable : 
    public std::conditional                               //if dest is a dense tensor
    <        
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_dense_tensor<dest>::value && traits<dest>::is_mutable,   
        typename std::conditional 
        <   
            is_tensor<src>::value,                                  //check if the src is also a tensor
            typename std::conditional                               //if src is also a tensor
            <
                is_dense<src>::value,                                   //check if src is dense
                copyable_traits<src, dest>,
                std::false_type//compatible_traits<src, dest>           //we currently don't support assignment of dense tensors from sparse tensors
            >::type,
            compatible_traits<src, dest>
        >::type,
        std::false_type
    >::type
{};


template <typename src, typename dest>
struct is_sparse_copy_assignable : 
    public std::conditional 
    <
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_sparse_tensor<dest>::value && traits<dest>::is_mutable,   
        typename std::conditional
        <
            is_sparse<src>::value, 
            typename std::conditional
            <
                same_topology_type<src, dest>::value,
                typename std::conditional
                <
                    is_tensor<src>::value,
                    copyable_traits<src, dest>, 
                    compatible_traits<src, dest>
                >::type, 
                std::false_type
            >::type,
            std::false_type
        >::type,
        std::false_type
    >::type
{};

template <typename type> struct is_copy_assignable<type>{static constexpr bool value = traits<type>::is_mutable;};

//check if a dense tensor object can be assigned from another object
template <typename src, typename dest>
struct is_copy_assignable<src, dest> :
    public std::conditional
    <
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_tensor<dest>::value && traits<dest>::is_mutable,   //first we check that dest is a tensor that is mutable
        typename std::conditional                               //if dest is a tensor
        <
            is_dense<dest>::value,                                  //first we check if dest is a dense
            is_dense_copy_assignable<src, dest>,                    //and if it is we check that it is dense copy assignable
            is_sparse_copy_assignable<src, dest>                    //otherwise we check 
        >::type, 
        std::false_type                                         //if dest is not a tensor then we cannot copy assign it in this way
    >::type
{};

template <typename src, typename dest>
struct is_dense_move_assignable :
    public std::conditional                               //if dest is a dense tensor
    <
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_dense_tensor<dest>::value && traits<dest>::is_mutable,   
        typename std::conditional 
        <   
            is_expression<src>::value,                                  //check if the src is also a tensor
            typename std::conditional                               //if src is also a tensor
            <
                is_dense<src>::value,                                   //check if src is dense
                compatible_traits<src, dest>,
                std::false_type
            >::type,
            std::false_type
        >::type,
        std::false_type
    >::type
{};

template <typename src, typename dest>
struct is_sparse_move_assignable : 
    public std::conditional 
    <
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_sparse_tensor<dest>::value && traits<dest>::is_mutable,   
        typename std::conditional
        <
            is_sparse<src>::value, 
            typename std::conditional
            <
                same_topology_type<src, dest>::value,
                typename std::conditional
                <
                    is_expression<src>::value,
                    compatible_traits<src, dest>,
                    std::false_type
                >::type, 
                std::false_type
            >::type,
            std::false_type
        >::type,
        std::false_type
    >::type
{};

template <typename src, typename dest>
struct is_move_assignable<src, dest> :
    public std::conditional
    <
        is_linalg_object<src>::value && is_linalg_object<dest>::value &&
        is_tensor<dest>::value && traits<dest>::is_mutable,   //first we check that dest is a tensor that is mutable
        typename std::conditional                               //if dest is a tensor
        <
            is_dense<dest>::value,                                  //first we check if dest is a dense
            is_dense_move_assignable<src, dest>,                    //and if it is we check that it is dense copy assignable
            is_sparse_move_assignable<src, dest>                    //otherwise we check 
        >::type, 
        std::false_type                                         //if dest is not a tensor then we cannot copy assign it in this way
    >::type
{};


template <typename type, typename res = void> using mutable_type = typename std::enable_if<traits<type>::is_mutable, res>::type;
template <typename U, typename type, typename res = type& > using value_update_type = typename std::enable_if<is_number<U>::value && std::is_convertible<U, typename traits<type>::value_type>::value && traits<type>::is_mutable, res>::type;

template <typename src, typename dest, typename res = dest&> using copy_assignable_type = typename std::enable_if<is_copy_assignable<src, dest>::value && !is_number<src>::value && is_const_lvalue_assignable<src>::value, res>::type;
template <typename src, typename dest, typename res = dest&> using other_copy_assignable_type = typename std::enable_if<is_copy_assignable<src, dest>::value && !std::is_same<src, dest>::value && !is_number<src>::value && is_const_lvalue_assignable<src>::value, res>::type;

template <typename src, typename dest, typename res = dest&> using move_assignable_type = typename std::enable_if<is_move_assignable<src, dest>::value && !is_number<src>::value, res>::type;
template <typename src, typename dest, typename res = dest&> using other_move_assignable_type = typename std::enable_if<is_move_assignable<src, dest>::value && !std::is_same<src, dest>::value && !is_number<src>::value, res>::type;


template <typename src, typename dest> using other_copy_constructable_type = other_copy_assignable_type<src, dest, void> ;
template <typename src, typename dest> using copy_constructable_type = copy_assignable_type<src, dest, void> ;

template <typename src, typename dest> using other_move_constructable_type = other_move_assignable_type<src, dest, void> ;
template <typename src, typename dest> using move_constructable_type = move_assignable_type<src, dest, void> ;




//check if a dense tensor object can have its buffer assigned from another object.
template <typename src, typename dest>
struct is_buffer_copyable_dense :
    public std::conditional
    <
        is_dense_tensor<dest>::value && is_dense_tensor<src>::value && traits<dest>::is_mutable,       //first we check that if dest is a dense tensor
        std::true_type,
        std::false_type 
    >::type
{};


/////////////////////////////////////////////////////////////////////////////////////////////////
//type traits for determining whether the specified types result in a valid expression object  //
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace expression_templates
{
template <typename T> struct addition_allowed : 
    public std::conditional
    <   
        is_expression<T>::value,
        std::false_type,
        typename std::conditional
        <
            is_csr_type<T>::value,
            std::false_type,
            std::true_type
        >::type
    >::type
{};

template <typename T, typename backend> struct addition_allowed<csr_matrix<T, backend> > : public std::false_type{};
template <typename expr, size_t D, typename backend> struct addition_allowed<expression_tree<expr, D, backend> > : 
    public std::conditional
    <
        is_csr_type<expression_tree<expr, D, backend> >::value,
        std::false_type,
        std::true_type
    >::type
{};


template <typename T> struct hadamard_allowed : 
    public std::conditional
    <   
        is_expression<T>::value,
        std::false_type,
        typename std::conditional
        <
            is_csr_type<T>::value,
            std::false_type,
            std::true_type
        >::type
    >::type
{};

template <typename T, typename backend> struct hadamard_allowed<csr_matrix<T, backend> > : public std::false_type{};
template <typename expr, size_t D, typename backend> struct hadamard_allowed<expression_tree<expr, D, backend> > : 
    public std::conditional
    <
        is_csr_type<expression_tree<expr, D, backend> >::value,
        std::false_type,
        std::true_type
    >::type
{};

template <typename T> struct complex_allowed : 
    public std::conditional
    <   
        is_expression<T>::value,
        std::false_type,
        typename std::conditional
        <
            is_csr_type<T>::value,
            std::false_type,
            std::true_type
        >::type
    >::type
{};

template <typename T, typename backend> struct complex_allowed<csr_matrix<T, backend> > : public std::false_type{};
template <typename expr, size_t D, typename backend> struct complex_allowed<expression_tree<expr, D, backend> > : 
    public std::conditional
    <
        is_csr_type<expression_tree<expr, D, backend> >::value,
        std::false_type,
        std::true_type
    >::type
{};

template <typename T> struct polar_allowed : 
    public std::conditional
    <   
        is_expression<T>::value,
        std::false_type,
        typename std::conditional
        <
            is_csr_type<T>::value,
            std::false_type,
            std::true_type
        >::type
    >::type
{};

template <typename T, typename backend> struct polar_allowed<csr_matrix<T, backend> > : public std::false_type{};
template <typename expr, size_t D, typename backend> struct polar_allowed<expression_tree<expr, D, backend> > : 
    public std::conditional
    <
        is_csr_type<expression_tree<expr, D, backend> >::value,
        std::false_type,
        std::true_type
    >::type
{};

template <typename T1, typename T2> struct is_valid_ddgemv
    : public std::conditional
    <
        is_dense_matrix<T1>::value && is_dense_vector<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct is_valid_sdgemv
    : public std::conditional
    <
        is_sparse_matrix<T1>::value && is_dense_vector<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_sparse_multiply_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct _is_valid_gemv : public std::conditional<is_valid_ddgemv<T1, T2>::value || is_valid_sdgemv<T1, T2>::value, std::true_type, std::false_type>::type{};

template <typename T1, typename T2, typename = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value, void>::type >
struct is_valid_gemv
{
    using type = typename _is_valid_gemv<T1, T2>::type;
    static constexpr bool value = type::value;
};


template <typename T1, typename T2> struct is_valid_ddgemm 
    : public std::conditional
    <
        is_dense_matrix<T1>::value && is_dense_matrix<T2>::value && !is_expression<T1>::value && !is_expression<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct is_valid_sdgemm 
    : public std::conditional
    <
        is_sparse_matrix<T1>::value && is_dense_matrix<T2>::value && !is_expression<T1>::value && !is_expression<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_sparse_multiply_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};


template <typename T1, typename T2> struct _is_valid_gemm : public std::conditional
                                                                  <
                                                                    is_linalg_object<T1>::value && is_linalg_object<T2>::value, 
                                                                    typename std::conditional<is_valid_ddgemm<T1, T2>::value || is_valid_sdgemm<T1, T2>::value || is_valid_sdgemm<T2, T1>::value, std::true_type, std::false_type>::type,
                                                                    std::false_type
                                                                  >::type{};

template <typename T1, typename T2, typename = typename std::enable_if<is_linalg_object<T1>::value && is_linalg_object<T2>::value, void>::type >
struct is_valid_gemm
{
    using type = typename _is_valid_gemm<T1, T2>::type;
    static constexpr bool value = type::value;
};



//valid rank 2 matrix rank 3 matrix to rank 3 matrix contraction.
template <typename T1, typename T2> struct is_valid_ddmtc1
    : public std::conditional
    <
        is_dense_matrix<T1>::value && is_dense_tensor_rank_3<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct is_valid_sdmtc1
    : public std::conditional
    <
        is_sparse_matrix<T1>::value && is_dense_tensor_rank_3<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct is_valid_mtc1 : public std::conditional<is_valid_ddmtc1<T1, T2>::value || is_valid_sdmtc1<T1, T2>::value, std::true_type, std::false_type>::type{};

//valid rank 2 matrix rank 3 matrix to rank 3 matrix contraction.
template <typename T1, typename T2> struct is_valid_ddttc2
    : public std::conditional
    <
        is_dense_tensor_rank_3<T1>::value && is_dense_tensor_rank_3<T2>::value,
        typename std::conditional<is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, std::true_type, std::false_type >::type, 
        std::false_type
    >::type 
{};

template <typename T1, typename T2> struct is_valid_ttc2 : public std::conditional<is_valid_ddttc2<T1, T2>::value, std::true_type, std::false_type>::type{};

}   //namespace expression_templates


}   //namespace linalg

#endif  //LINALG_TYPE_TRAITS_HPP//

