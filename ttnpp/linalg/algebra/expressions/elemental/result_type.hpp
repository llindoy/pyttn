#ifndef LINALG_ALGEBRA_RESULT_TYPE_HPP
#define LINALG_ALGEBRA_RESULT_TYPE_HPP

#include "../../../linalg_forward_decl.hpp"

//TODO: comment this file

namespace linalg
{
namespace expression_templates
{
//result type of the expression literal
template <typename T, typename backend> struct result_type<literal_type<T, backend>>
{
    using type = literal_type<T, backend>; 
    using value_type = T;   
    using backend_type = backend;
    static constexpr size_t rank = 0;
};

//result type of the dense tensor type
template <typename T, size_t D, typename backend> struct result_type<tensor<T, D, backend> >
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = D;
    using type = dense_tensor_type<rank>;   
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

template <typename ArrType, typename T, size_t D> struct result_type<tensor_slice<ArrType, T, D> >
{
    using value_type = T;     
    using backend_type = typename traits<ArrType>::backend_type;     
    static constexpr size_t rank = D;
    using type = dense_tensor_type<rank>;   
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

template <typename T, size_t D, typename backend> struct result_type<reinterpreted_tensor<T, D, backend> >
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = D;
    using type = dense_tensor_type<rank>;   
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

template <typename T, typename backend> struct result_type<hermitian_matrix<T, backend>>
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = 2;
    using type = dense_tensor_type<rank>;   
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

template <typename T, typename backend> struct result_type<upper_hessenberg_matrix<T, backend>>
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = 2;
    using type = dense_tensor_type<rank>;   
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

//result types of the sparse matrix types
template <typename T, typename backend> struct result_type<csr_matrix<T, backend>>
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = 2;
    using type = csr_matrix_type;
    using shape_type = const csr_topology_type<backend>&;
    using const_shape_reference = shape_type;
};

template <typename T, typename backend> struct result_type<symmetric_tridiagonal_matrix<T, backend>>
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = 2;
    using type = symmetric_tridiagonal_matrix_type;
    using shape_type = std::array<typename backend_type::size_type, 2>;
    using const_shape_reference = const shape_type&;
};

template <typename T, typename backend> struct result_type<diagonal_matrix<T, backend>>
{
    using value_type = T;     
    using backend_type = backend;     
    static constexpr size_t rank = 2;
    using type = diagonal_matrix_type;
    using shape_type = std::array<typename backend_type::size_type, 2>;
    using const_shape_reference = const shape_type&;
};

//type alias to allow for easy conditional compilation of results of scalar of a dense object
template <typename T, typename arr, typename backend> 
struct result_type<literal_type<T, backend>, arr, multiplication_op<backend> >
{
    using rtraits = result_type<arr>;
    using vt = typename rtraits::value_type;
    using value_type = decltype(vt()*T());  

    static_assert(is_number<T>::value && is_number<vt>::value, "Invalid result type.  Cannot perform multiplication between scalar and tensor if the underlying data types are not valid number types.");
    static_assert(is_number<value_type>::value, "Invalid result type.  The resultant value_type of the multiplication between scalar and tensor is not a valid number type.");

    using type = typename rtraits::type;
    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;

    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};


//type alias to allow for easy conditional compilation of results of conjugation of a dense object
template <typename arr, typename backend> 
struct result_type<arr, conjugation_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename rtraits::value_type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, real_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename get_real_type<typename rtraits::value_type>::type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, imag_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename get_real_type<typename rtraits::value_type>::type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, arg_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename get_real_type<typename rtraits::value_type>::type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, elemental_exp_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename rtraits::value_type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, unit_polar_op<backend> >
{
    using rtraits = result_type<arr>;
    using vt = typename rtraits::value_type;
    static_assert(is_number<vt>::value, "Failed to construct unit_polar_op result type, input type is not real.");
    static_assert(!is_complex<vt>::value, "Failed to construct unit_polar_op result type, input type is not real.");
    using value_type = complex<vt>;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

template <typename arr, typename backend> 
struct result_type<arr, norm_op<backend> >
{
    using rtraits = result_type<arr>;
    using value_type = typename get_real_type<typename rtraits::value_type>::type;  
    using type = typename rtraits::type;

    static_assert(std::is_same< typename rtraits::backend_type, backend>::value, "Invalid backend.");
    using backend_type = backend;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    static constexpr size_t rank = rtraits::rank;
};

//type alias to allow for easy conditional compilation of results of addition between two dense objects
template <typename arr1, typename arr2, typename backend> 
struct result_type<arr1, arr2, addition_op<backend> > 
{
    //the rank
    static_assert(result_type<arr1>::rank == result_type<arr2>::rank, "Invalid rank.  Cannot perform addition between two objects with different ranks.");
    static constexpr size_t rank = result_type<arr1>::rank;

    //the value type
    using vt1 = typename result_type<arr1>::value_type;    using vt2 = typename result_type<arr2>::value_type;
    static_assert(is_number<vt1>::value && is_number<vt2>::value, "Invalid result type.  Cannot perform addition of two objects if the underlying data types are not valid number types.");
    using value_type = decltype(vt1()+vt2());  
    static_assert(is_number<value_type>::value, "Invalid result type.  The resultant value type of the addition of two objects is not a valid number type.");

    //the result type tag
    static_assert(std::is_same< typename result_type<arr1>::type, typename result_type<arr2>::type>::value, "Invalid result type. Cannot perform addition between two objects with different result types.");

    using type = typename result_type<arr1>::type;
    static_assert(addition_allowed<type>::value, "Invalid result type.  The addition operation is not supported for the specified result type.");

    //the backend type
    static_assert(std::is_same< typename result_type<arr1>::backend_type, backend>::value && std::is_same< typename result_type<arr2>::backend_type, backend>::value , "Invalid result type.  Cannot perform addition between two objects with different backend types.");
    using backend_type = backend;

    static_assert(std::is_same< typename result_type<arr1>::shape_type, typename result_type<arr2>::shape_type>::value, "Invalid shape type.  Cannot perform addition between two objects with different shape types.");
    using shape_type = typename result_type<arr1>::shape_type;
    using const_shape_reference = typename result_type<arr1>::const_shape_reference;
};

//type alias to allow for easy conditioanl compilation of results of hadamard operations between two tensors
template <typename arr1, typename arr2, typename backend> 
struct result_type<arr1, arr2, hadamard_op<backend> > 
{
    //the rank
    static_assert(result_type<arr1>::rank == result_type<arr2>::rank, "Invalid rank.  Cannot perform hadamard product between two objects with different ranks.");
    static constexpr size_t rank = result_type<arr1>::rank;

    //the value type
    using vt1 = typename result_type<arr1>::value_type;    using vt2 = typename result_type<arr2>::value_type;
    static_assert(is_number<vt1>::value && is_number<vt2>::value, "Invalid result type.  Cannot perform hadamard product of two objects if the underlying data types are not valid number types.");
    using value_type = decltype(vt1()*vt2());  
    static_assert(is_number<value_type>::value, "Invalid result type.  The resultant value type of the hadamard product of two objects is not a valid number type.");

    //the result type tag
    static_assert(std::is_same< typename result_type<arr1>::type, typename result_type<arr2>::type>::value, "Invalid result type. Cannot perform hadamard product between two objects with different result types.");

    using type = typename result_type<arr1>::type;
    static_assert(hadamard_allowed<type>::value, "Invalid result type.  The hadamard product operation is not supported for the specified result type.");

    //the backend type
    static_assert(std::is_same< typename result_type<arr1>::backend_type, backend>::value && std::is_same< typename result_type<arr2>::backend_type, backend>::value , "Invalid result type.  Cannot perform hadamard product between two objects with different backend types.");
    using backend_type = backend;

    static_assert(std::is_same< typename result_type<arr1>::shape_type, typename result_type<arr2>::shape_type>::value, "Invalid shape type.  Cannot perform hadamard product between two objects with different shape types.");
    using shape_type = typename result_type<arr1>::shape_type;
    using const_shape_reference = typename result_type<arr1>::const_shape_reference;
};

//type alias to allow for easy conditioanl compilation of results of operations forming a complex tensor from two real tensors
template <typename arr1, typename arr2, typename backend> 
struct result_type<arr1, arr2, complex_op<backend> > 
{
    //the rank
    static_assert(result_type<arr1>::rank == result_type<arr2>::rank, "Invalid rank.  Cannot form complex between two objects with different ranks.");
    static constexpr size_t rank = result_type<arr1>::rank;

    //the value type
    using vt1 = typename result_type<arr1>::value_type;    using vt2 = typename result_type<arr2>::value_type;
    static_assert(is_number<vt1>::value && is_number<vt2>::value, "Invalid result type.  Cannot form complex of two objects if the underlying data types are not valid number types.");
    static_assert(!is_complex<vt1>::value && !is_complex<vt2>::value, "Invalid result type.  Cannot form complex of two objects if the underlying data types are not valid number types.");
    using value_type = complex<decltype(vt1()+vt2())>;  
    static_assert(is_number<value_type>::value, "Invalid result type.  The resultant value type of the addition of two objects is not a valid number type.");

    //the result type tag
    static_assert(std::is_same< typename result_type<arr1>::type, typename result_type<arr2>::type>::value, "Invalid result type. Cannot form complex between two objects with different result types.");

    using type = typename result_type<arr1>::type;
    static_assert(addition_allowed<type>::value, "Invalid result type.  The addition operation is not supported for the specified result type.");

    //the backend type
    static_assert(std::is_same< typename result_type<arr1>::backend_type, backend>::value && std::is_same< typename result_type<arr2>::backend_type, backend>::value , "Invalid result type.  Cannot form complex between two objects with different backend types.");
    using backend_type = backend;

    static_assert(std::is_same< typename result_type<arr1>::shape_type, typename result_type<arr2>::shape_type>::value, "Invalid shape type.  Cannot form complex between two objects with different shape types.");
    using shape_type = typename result_type<arr1>::shape_type;
    using const_shape_reference = typename result_type<arr1>::const_shape_reference;
};

//type alias to allow for easy conditioanl compilation of results of operations forming a complex tensor from the real argument and modulus tensors
template <typename arr1, typename arr2, typename backend> 
struct result_type<arr1, arr2, polar_op<backend> > 
{
    //the rank
    static_assert(result_type<arr1>::rank == result_type<arr2>::rank, "Invalid rank.  Cannot form polar complex between two objects with different ranks.");
    static constexpr size_t rank = result_type<arr1>::rank;

    //the value type
    using vt1 = typename result_type<arr1>::value_type;    using vt2 = typename result_type<arr2>::value_type;
    static_assert(is_number<vt1>::value && is_number<vt2>::value, "Invalid result type.  Cannot form polar complex of two objects if the underlying data types are not valid number types.");
    static_assert(!is_complex<vt1>::value && !is_complex<vt2>::value, "Invalid result type.  Cannot form polar complex of two objects if the underlying data types are not valid number types.");
    using value_type = complex<decltype(vt1()+vt2())>;  
    static_assert(is_number<value_type>::value, "Invalid result type.  The resultant value type of the addition of two objects is not a valid number type.");

    //the result type tag
    static_assert(std::is_same< typename result_type<arr1>::type, typename result_type<arr2>::type>::value, "Invalid result type. Cannot form polar complex between two objects with different result types.");

    using type = typename result_type<arr1>::type;
    static_assert(addition_allowed<type>::value, "Invalid result type.  The addition operation is not supported for the specified result type.");

    //the backend type
    static_assert(std::is_same< typename result_type<arr1>::backend_type, backend>::value && std::is_same< typename result_type<arr2>::backend_type, backend>::value , "Invalid result type.  Cannot form polar complex between two objects with different backend types.");
    using backend_type = backend;

    static_assert(std::is_same< typename result_type<arr1>::shape_type, typename result_type<arr2>::shape_type>::value, "Invalid shape type.  Cannot form polar complex between two objects with different shape types.");
    using shape_type = typename result_type<arr1>::shape_type;
    using const_shape_reference = typename result_type<arr1>::const_shape_reference;
};

//result type of unary expression
template <typename A, template <typename> class op, typename backend>
struct result_type<unary_expression<A, op, backend> >
{
    using rtype = result_type<A, op<backend>>;
    using type = typename rtype::type;    
    using backend_type = typename rtype::backend_type;    
    using value_type = typename rtype::value_type;
    using shape_type = typename rtype::shape_type;
    using const_shape_reference = typename rtype::const_shape_reference;
    static constexpr size_t rank = rtype::rank;
};

//result type of binary expression
template <typename A, typename B, template <typename> class op, typename backend>
struct result_type<binary_expression<A, B, op, backend> >
{
    using rtype = result_type<A, B, op<backend>>;
    using type = typename rtype::type;    
    using backend_type = typename rtype::backend_type;    
    using value_type = typename rtype::value_type;
    using shape_type = typename rtype::shape_type;
    using const_shape_reference = typename rtype::const_shape_reference;
    static constexpr size_t rank = rtype::rank;
};

template <typename expr, size_t D, typename backend>
struct result_type<expression_tree<expr, D, backend> >
{
    using rtype = result_type<expr>;
    using type = typename rtype::type;    
    using backend_type = typename rtype::backend_type;    
    using value_type = typename rtype::value_type;
    using shape_type = typename rtype::shape_type;
    using const_shape_reference = typename rtype::const_shape_reference;
    static constexpr size_t rank = rtype::rank;
};

}   //namespace expression_templates
}   //namespace linalg

#endif  //LINALG_ALGEBRA_RESULT_TYPE_HPP//


