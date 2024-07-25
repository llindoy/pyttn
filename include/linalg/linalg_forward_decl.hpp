#ifndef LINALG_FORWARD_DECL_HPP
#define LINALG_FORWARD_DECL_HPP

namespace linalg{class backend_base{};}   //namespace linalg

#include <common/exception_handling.hpp>

#include "utils/linalg_utils.hpp"
#include "backends/blas_backend.hpp"
#include "backends/cuda_backend.hpp"
#include "utils/memory_helper.hpp"
#include "utils/serialisation.hpp"

namespace linalg
{

//forward declaration of the traits types.  All linalg types have a traits object that provides additional information about the type
template <typename Container, typename enabled = void> struct traits;


/////////////////////////////////////////////////////////////////////////////////////////////////
//                         forward declaration of tensor type flags                            //
/////////////////////////////////////////////////////////////////////////////////////////////////
class linalg_type{};

class generic_tensor_type : linalg_type{}; template <size_t D> class tensor_type : public generic_tensor_type{static constexpr size_t rank = D;};
class generic_expression_type : public linalg_type{}; 
template <bool copyassignable> class is_const_expression_type : public generic_expression_type{};
template <size_t D, bool copyassignable> class expression_type : public is_const_expression_type<copyassignable>{static constexpr size_t rank = D;};

class dense_type{}; 
class sparse_type{};
class view_type{};

template <size_t D> class dense_tensor_type : public dense_type, public tensor_type<D> {};

class hermitian_type{};
class upper_hessenberg_type{};

class csr_matrix_type : public sparse_type, public tensor_type<2>{};
class diagonal_matrix_type : public tensor_type<2>, public sparse_type{};
class symmetric_tridiagonal_matrix_type : public tensor_type<2>, public sparse_type{};


enum MATRIX_ORDERING 
{
    ROW_MAJOR = 0,  
    COLUMN_MAJOR = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                        forward declaration of dense tensor types                            //
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ArrRef> class tensor_base;
template <typename T, size_t D, typename backend=blas_backend> class tensor;

//dense tensor views
template <typename SliceRef> class tensor_slice_base;
template <typename Arr, typename data_type, size_t D> class tensor_slice;
template <typename Arr, typename data_type, size_t D> struct tensor_slice_traits;

template <typename ViewRef> class tensor_view_base;
template <typename T, size_t D, typename backend> class tensor_view;
template <typename T, size_t D, typename backend = blas_backend> class reinterpreted_tensor;

//now we have additional special views for the dense tensors.  These are just additional functions that inherit from 
template <typename T, typename backend = blas_backend> class hermitian_matrix;
template <typename T, typename backend = blas_backend> class upper_hessenberg_matrix;

//sparse views of dense matrices
template <typename ArrRef> class diagonal_matrix_view_base;
template <typename Arr, typename data_type> class diagonal_matrix_view;

//additional details for dense tensor objects providing additional functionality
template < class ArrRef, size_t D = traits<ArrRef>::rank, bool is_mutable = traits<ArrRef>::is_mutable, typename backend_type = typename traits<ArrRef>::backend_type> class tensor_details{};

template <typename ... Args> struct is_copy_assignable;
template <typename ... Args> struct is_move_assignable;


/////////////////////////////////////////////////////////////////////////////////////////////////
//                       forward declaration of sparse matrix types                            //
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename backend> class csr_topology_type;
template <typename ArrRef> class csr_matrix_base;
template <typename T, typename backend=blas_backend> class csr_matrix;

template <typename SliceRef> class bcsr_matrix_base;
template <typename T, typename backend=blas_backend> class bcsr_matrix;

template <typename ViewRef> class special_matrix_base;

template <typename ViewRef> class diagonal_matrix_base;
template <typename T, typename backend=blas_backend> class diagonal_matrix;

template <typename ViewRef> class symmetric_tridiagonal_matrix_base;
template <typename T, typename backend=blas_backend> class symmetric_tridiagonal_matrix;

template <typename ViewRef> class triangular_matrix_base;
template <typename T, typename backend=blas_backend> class triangular_matrix;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         forward declaration of expression types                             //
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace expression_templates
{

template <typename T> struct storage_traits;
template <typename T, typename backend> class literal_type;

template <typename vtype, template <typename > class operation, typename backend> class unary_expression;
template <typename ltype, typename rtype, template <typename > class operation, typename backend> class binary_expression;

template <typename ... Args> struct result_type;

template <typename derived, bool has_buffers = true> class expression_base;
template <typename expr, size_t rank, typename backend> class expression_tree;

//declaration of tensor index permutation expression objects
template <typename arrtype, bool conjugate, typename enabler = void> class transpose_expression;
template <typename arrtype, typename enabler = void> class tensor_transpose_expression;
template <typename arrtype, bool conjugate> class tensor_permutation_3_expression;

///declaration of tensor contraction expression objects
class tensor_contraction_expression{};
template <typename impl> class matrix_vector_product_base;
template <typename T1, typename T2, class enabled = void> class matrix_vector_product;
template <typename impl> class matrix_matrix_product_base;
template <typename T1, typename T2, class enabled = void> class matrix_matrix_product;
template <typename a_tensor_type, typename b_tensor_type, size_t D, class enabled=void> class tensordot_expr;


template <typename T1, typename T2> class tensor_contraction_1_mt;
template <typename T1, typename T2> class tensor_contraction_332;

//declare the different operation types we support through expression trees
template <typename backend> class addition_op;
template <typename backend> class multiplication_op;
template <typename backend> class conjugation_op;
template <typename backend> class hadamard_op;
template <typename backend> class complex_op;
template <typename backend> class unit_polar_op;
template <typename backend> class polar_op;
template <typename backend> class real_op;
template <typename backend> class imag_op;
template <typename backend> class norm_op;
template <typename backend> class arg_op;
template <typename backend> class elemental_exp_op;

}   //namespace expression_templates



/////////////////////////////////////////////////////////////////////////////////////////////////
//                         forward declaration of decomposition types                          //
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename matrix_type, typename enabled = void> class generalised_eigensolver;
template <typename matrix_type, typename enabled = void> class eigensolver;
template <typename matrix_type, typename enabled = void> class tridiagonalisation;
template <typename matrix_type, bool use_divide_and_conquer = true, typename enabled = void> class singular_value_decomposition;
template <typename matrix_type, typename enabled = void> class qr;
template <typename matrix_type, typename enabled = void> class lq;
template <typename matrix_type, typename enabler = void> class lu_decomposition;
template <typename matrix_type, bool use_lu = true, typename enabler = void> class determinant;
template <typename matrix_type, bool use_lu = true, typename enabler = void> class linear_solver;

template <typename T, typename backend = blas_backend> class arnoldi_iteration;


}   //namespace linalg

#include "linalg_type_traits.hpp"

#endif  //LINALG_FORWARD_DECL_HPP


