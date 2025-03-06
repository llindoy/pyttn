#ifndef LINALG_ALGEBRA_SPARSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP
#define LINALG_ALGEBRA_SPARSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

#include "matrix_multiplication_base.hpp"

#include <type_traits>

namespace linalg
{

namespace expression_templates
{

//diagonal matrix dense matrix product
template <typename sparse_type, typename dense_type>
class matrix_matrix_product<sparse_type, dense_type, 
            typename std::enable_if<is_dense_matrix<dense_type>::value && is_diagonal_matrix_type<sparse_type>::value && is_same_backend<sparse_type, dense_type>::value && is_sparse_multiply_value<sparse_type, dense_type>::value, void>::type> : 
    public matrix_matrix_product_base<matrix_matrix_product<sparse_type, dense_type, void>>
{
public:
    static_assert(dense_type::rank == 2 , "Failed to construct matrix_matrix_product object.  The two input tensor must both be rank 2.");
    using backend_type = typename traits<dense_type>::backend_type;    using size_type = typename backend_type::size_type;
    using left_type = sparse_type; using right_type = dense_type;
    using lvalue_type = typename traits<left_type>::value_type;    using rvalue_type = typename traits<right_type>::value_type;
    using lvalue_ptr = typename std::add_pointer<typename std::add_const<lvalue_type>::type>::type;
    using rvalue_ptr = typename std::add_pointer<typename std::add_const<rvalue_type>::type>::type;
    using value_type = decltype(lvalue_type()*rvalue_type());    
    using self_type = matrix_matrix_product<left_type, right_type>;            using base_type = matrix_matrix_product_base<self_type>;

    static constexpr size_t rank = 2;
protected:
    lvalue_ptr m_Abuffer;
    rvalue_ptr m_Bbuffer;

    size_type m_Asize, m_Bsize;
    size_type m_incA, m_ldB;
    bool m_sparse_left;

    using base_type::m_m; using base_type::m_n; using base_type::m_k; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opB; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.shape(), B.shape(), transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.size()), m_Bsize(B.size()), m_incA(A.incx()), m_ldB(B.shape(1)), m_sparse_left(true){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const value_type& coeff, const right_type& B, const left_type& A, bool transB = false, bool transA = false, bool conjB = false, bool conjA = false) 
    try : base_type(coeff, B.shape(), A.shape(), transB, transA, conjB, conjA), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_incA(A.incx()), m_ldB(B.shape(1)), m_sparse_left(false){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    template <typename working_type>
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, working_type& working_buffer, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.shape(), B.shape(), working_buffer, transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_incA(A.incx()), m_ldB(B.shape(1)), m_sparse_left(true){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    template <typename working_type>
    matrix_matrix_product(const value_type& coeff, const right_type& B, const left_type& A, working_type& working_buffer, bool transB = false, bool transA = false, bool conjB = false, bool conjA = false) 
    try : base_type(coeff, B.shape(), A.shape(), working_buffer, transB, transA, conjB, conjA), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_incA(A.incx()), m_ldB(B.shape(1)), m_sparse_left(false){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const matrix_matrix_product& o, const value_type& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_Bbuffer(o.m_Bbuffer), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize), m_incA(o.m_incA), m_ldB(o.m_ldB), m_sparse_left(o.m_sparse_left){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const matrix_matrix_product& o, bool make_transpose = false, bool conjugate = false) 
    try : base_type(o, make_transpose, conjugate), m_Abuffer(o.m_Abuffer), m_Bbuffer(o.m_Bbuffer), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize), m_incA(o.m_incA), m_ldB(o.m_ldB)
    {if(make_transpose){m_sparse_left = !o.m_sparse_left;} else{m_sparse_left = o.m_sparse_left;}}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    //This routine expects dense matrices in row major order.  And as such there is no need to perform the reordering of operations required for the dense dense matrix products.
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = 0.0, value_type coeff_scale = 1.0)
    {
        try
        {
            ASSERT(res.buffer() != m_Bbuffer, "The matrix matrix product does not support inplace products.");
            value_type coeff = m_coeff*coeff_scale;
            size_type ldc = res.size(1);
            CALL_AND_HANDLE(backend_type::dgmm(m_sparse_left, m_opA, m_opB, m_n, m_m, m_k, coeff, m_Abuffer, m_incA, m_Bbuffer, m_ldB, beta, res.buffer(), ldc), "dgmm call failed.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate diagonal matrix matrix product.");
        }
    }
};  //matrix_matrix_product


}   //namespace expression_templates

}   //namespace linalg

#endif  //LINALG_ALGEBRA_SPARSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

