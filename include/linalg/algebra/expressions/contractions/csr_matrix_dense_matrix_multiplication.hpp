#ifndef LINALG_ALGEBRA_CSR_MATRIX_DENSE_MATRIX_MATRIX_CONTRACTION_HPP
#define LINALG_ALGEBRA_CSR_MATRIX_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

#include "matrix_multiplication_base.hpp"

#include <type_traits>

namespace linalg
{

namespace expression_templates
{

template <typename sparse_type, typename dense_type>
class matrix_matrix_product<sparse_type, dense_type, 
            typename std::enable_if<is_dense_matrix<dense_type>::value && 
                                    is_csr_matrix_type<sparse_type>::value && 
                                    has_same_backend<sparse_type, dense_type>::value && 
                                    is_sparse_multiply_value<sparse_type, dense_type>::value
          , void>::type> : 
    public matrix_matrix_product_base<matrix_matrix_product<sparse_type, dense_type, void>>
{
public:
    static_assert(dense_type::rank == 2 , "Failed to construct matrix_matrix_product object.  The two input tensor must both be rank 2.");

    using backend_type = typename traits<dense_type>::backend_type;    using size_type = typename backend_type::size_type;
    using index_type = typename backend_type::index_type;
    using left_type = sparse_type; using right_type = dense_type;
    using lvalue_type = typename traits<left_type>::value_type; using rvalue_type = typename traits<right_type>::value_type;

    using lvalue_ptr = typename std::add_pointer<typename std::add_const<lvalue_type>::type>::type;
    using rvalue_ptr = typename std::add_pointer<typename std::add_const<rvalue_type>::type>::type;
    using index_ptr = typename std::add_pointer<typename std::add_const<index_type>::type>::type;

    using value_type = decltype(lvalue_type()*rvalue_type());    
    using self_type = matrix_matrix_product<left_type, right_type>;            using base_type = matrix_matrix_product_base<self_type>;
    using ttype = typename backend_type::transform_type;

    static constexpr size_t rank = 2;
protected:
    lvalue_ptr m_Abuffer;
    index_ptr m_rowptr;
    index_ptr m_colind;
    rvalue_ptr m_Bbuffer;

    size_type m_Asize, m_Bsize;
    size_type m_ldB, m_ldBt;    
    bool m_transpose_result;
    

    using base_type::m_m; using base_type::m_n; using base_type::m_k; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opB; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.dims(), B.shape(), transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Bbuffer(B.buffer()), m_Asize(A.size()), m_Bsize(B.size()), m_ldB(B.shape(1)), m_ldBt(B.shape(0)), m_transpose_result(false){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const value_type& coeff, const right_type& B, const left_type& A, bool transB = false, bool transA = false, bool conjB = false, bool conjA = false) 
    try : base_type(coeff, B.shape(), A.dims(), transB, transA, conjB, conjA), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_ldB(B.shape(1)), m_ldBt(B.shape(0)), m_transpose_result(true)
    {
        RAISE_EXCEPTION("Currently doesn't work.");
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    template <typename working_type>
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, working_type& working_buffer, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.dims(), B.shape(), working_buffer, transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_ldB(B.shape(1)), m_ldBt(B.shape(0)), m_transpose_result(false){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    template <typename working_type>
    matrix_matrix_product(const value_type& coeff, const right_type& B, const left_type& A, working_type& working_buffer, bool transB = false, bool transA = false, bool conjB = false, bool conjA = false) 
    try : base_type(coeff, B.shape(), A.dims(), working_buffer, transB, transA, conjB, conjA), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Bbuffer(B.buffer()), m_Asize(A.nnz()), m_Bsize(B.size()), m_ldB(B.shape(1)), m_ldBt(B.shape(0)), m_transpose_result(true)
    {
        RAISE_EXCEPTION("Currently doesn't work.");
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const matrix_matrix_product& o, const value_type& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Bbuffer(o.m_Bbuffer), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize), m_ldB(o.m_ldB), m_ldBt(o.m_ldBt), m_transpose_result(o.m_transpose_result){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    matrix_matrix_product(const matrix_matrix_product& o, bool make_transpose = false, bool conjugate = false) 
    try : base_type(o, false, conjugate), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Bbuffer(o.m_Bbuffer), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize), m_ldB(o.m_ldB), m_ldBt(o.m_ldBt)
    {
        if(make_transpose)
        {
            m_transpose_result = !o.m_transpose_result;
            this->m_shape[0] = o.m_shape[1];
            this->m_shape[1] = o.m_shape[0];
            this->m_m = o.m_n;    this->m_n = o.m_m;    this->m_k = o.m_k;
        }
        else{m_transpose_result = o.m_transpose_result;}
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix matrix product object.");}

    //This routine expects dense matrices in row major order.  And as such there is no need to perform the reordering of operations required for the dense dense matrix products.
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = 0.0, value_type coeff_scale = 1.0)
    {
        try
        {
            ASSERT(res.buffer() != m_Bbuffer, "matrix matrix product does not support inplace products.");
            value_type coeff = m_coeff*coeff_scale;
            size_type ldc = res.size(1);

            if(m_opA == backend_type::op_n || m_opA == backend_type::op_c)
            {
                CALL_AND_HANDLE(backend_type::csrmm(m_transpose_result, m_opA, m_opB, m_n, m_k, m_m, coeff, m_Abuffer, m_rowptr, m_colind, m_Bbuffer, m_ldB, beta, res.buffer(), ldc), "csrmm call failed.");
            }
            else
            {
                RAISE_EXCEPTION("The requested contraction is currently not supported.");
                //ttype opA = (m_opA == backend_type::op_h) ? backend_type::op_c : backend_type::op_n;
                //CALL_AND_HANDLE(backend_type::cscmm(m_transpose_result, m_opA, m_opB, m_n, m_m, coeff, m_Abuffer, m_rowptr, m_colind, m_Bbuffer, m_ldB, beta, res.buffer(), ldc), "csrmm call failed.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate csr matrix dense matrix product.");
        }
    }
};  //matrix_matrix_product

}   //namespace expression_templates

}   //namespace linalg

#endif  //LINALG_ALGEBRA_CSR_MATRIX_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

