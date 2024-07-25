#ifndef LINALG_ALGEBRA_CSR_MATRIX_VECTOR_CONTRACTION_HPP
#define LINALG_ALGEBRA_CSR_MATRIX_VECTOR_CONTRACTION_HPP

#include "matrix_vector_multiplication_base.hpp"

#include <type_traits>

namespace linalg
{

namespace expression_templates
{
//blas csr matrix dense vector product
template <typename sparse_type, typename dense_type>
class matrix_vector_product<sparse_type, dense_type, 
            typename std::enable_if<is_dense_vector<dense_type>::value && 
                                    is_csr_matrix_type<sparse_type>::value && 
                                    has_same_backend<sparse_type, dense_type>::value && 
                                    has_backend<sparse_type, blas_backend>::value && 
                                    is_sparse_multiply_value<sparse_type, dense_type>::value
          , void>::type> : 
    public matrix_vector_product_base<matrix_vector_product<sparse_type, dense_type, void>>
{
public:
    using backend_type = typename traits<dense_type>::backend_type;    using size_type = typename backend_type::size_type;
    using index_type = typename backend_type::index_type;
    using left_type = sparse_type; using right_type = dense_type;
    using lvalue_type = typename traits<left_type>::value_type; using rvalue_type = typename traits<right_type>::value_type;

    using lvalue_ptr = typename std::add_pointer<typename std::add_const<lvalue_type>::type>::type;
    using rvalue_ptr = typename std::add_pointer<typename std::add_const<rvalue_type>::type>::type;
    using index_ptr = typename std::add_pointer<typename std::add_const<index_type>::type>::type;

    using value_type = decltype(rvalue_type()*lvalue_type());    
    using self_type = matrix_vector_product<left_type, right_type>;            using base_type = matrix_vector_product_base<self_type>;
    using ttype = typename backend_type::transform_type;

    static constexpr size_t rank = 1;
protected:
    lvalue_ptr m_Abuffer;
    index_ptr m_rowptr;
    index_ptr m_colind;
    rvalue_ptr m_Xbuffer;

    size_type m_Asize, m_Xsize;
    size_type m_incX;

    using base_type::m_m; using base_type::m_n; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opX; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.dims(), X.shape(), transA, conjA, conjX), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Xbuffer(X.buffer()), m_Asize(A.size()), m_Xsize(X.size()), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    template <typename working_type>
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, working_type& working_buffer, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.dims(), X.shape(), working_buffer, transA, conjA, conjX), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Xbuffer(X.buffer()), m_Asize(A.nnz()), m_Xsize(X.size()), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    matrix_vector_product(const matrix_vector_product& o, const value_type& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incX(o.m_incX){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    matrix_vector_product(const matrix_vector_product& o, bool conjugate = false) 
    try : base_type(o, conjugate), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incX(o.m_incX){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    //This routine expects dense matrices in row major order.  And as such there is no need to perform the reordering of operations required for the dense dense matrix products.
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = 0.0, value_type coeff_scale = 1.0)
    {
        try
        {
            ASSERT(res.buffer() != m_Xbuffer, "The matrix vector product does not support inplace products.");
            value_type coeff = m_coeff*coeff_scale;
            size_type incc = res.incx();
            
            ASSERT(m_opA == backend_type::op_h || m_opA == backend_type::op_t, "Transposed csr matrices are currently not supported.");
            ttype opA = (m_opA == backend_type::op_h) ? backend_type::op_c : backend_type::op_n;
            bool conjB = (m_opX == backend_type::op_c || m_opX == backend_type::op_h);

            CALL_AND_HANDLE(backend_type::csrmv(opA, conjB, m_n, m_m, coeff, m_Abuffer, m_rowptr, m_colind, m_Xbuffer, m_incX, beta, res.buffer(), incc), "Error when making call to backend::csrmv");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute csr matrix vector product.");
        }
    }
};  //matrix_vector_product

#ifdef __NVCC__
//cuda csr matrix dense vector product
template <typename sparse_type, typename dense_type>
class matrix_vector_product<sparse_type, dense_type, 
            typename std::enable_if<is_dense_vector<dense_type>::value && 
                                    is_csr_matrix_type<sparse_type>::value && 
                                    has_same_backend<sparse_type, dense_type>::value && 
                                    has_backend<sparse_type, cuda_backend>::value && 
                                    is_sparse_multiply_value<sparse_type, dense_type>::value
          , void>::type> : 
    public matrix_vector_product_base<matrix_vector_product<sparse_type, dense_type, void>>
{
public:
    using backend_type = typename traits<dense_type>::backend_type;    using size_type = typename backend_type::size_type;
    using index_type = typename backend_type::index_type;
    using left_type = sparse_type; using right_type = dense_type;
    using lvalue_type = typename traits<left_type>::value_type; using rvalue_type = typename traits<right_type>::value_type;

    using lvalue_ptr = typename std::add_pointer<typename std::add_const<lvalue_type>::type>::type;
    using rvalue_ptr = typename std::add_pointer<typename std::add_const<rvalue_type>::type>::type;
    using index_ptr = typename std::add_pointer<typename std::add_const<index_type>::type>::type;

    using value_type = decltype(rvalue_type()*lvalue_type());    
    using self_type = matrix_vector_product<left_type, right_type>;            using base_type = matrix_vector_product_base<self_type>;
    using ttype = typename backend_type::transform_type;

    static constexpr size_t rank = 1;
protected:
    lvalue_ptr m_Abuffer;
    index_ptr m_rowptr;
    index_ptr m_colind;
    rvalue_ptr m_Xbuffer;

    size_type m_Asize, m_Xsize;
    size_type m_incX;

    using base_type::m_m; using base_type::m_n; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opX; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.dims(), X.shape(), transA, conjA, conjX), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Xbuffer(X.buffer()), m_Asize(A.size()), m_Xsize(X.size()), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    template <typename working_type>
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, working_type& working_buffer, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.dims(), X.shape(), working_buffer, transA, conjA, conjX), m_Abuffer(A.buffer()), m_rowptr(A.rowptr()), m_colind(A.colind()), m_Xbuffer(X.buffer()), m_Asize(A.nnz()), m_Xsize(X.size()), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    matrix_vector_product(const matrix_vector_product& o, const value_type& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incX(o.m_incX){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    matrix_vector_product(const matrix_vector_product& o, bool conjugate = false) 
    try : base_type(o, conjugate), m_Abuffer(o.m_Abuffer), m_rowptr(o.m_rowptr), m_colind(o.m_colind), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incX(o.m_incX){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");}

    //This routine expects dense matrices in row major order.  And as such there is no need to perform the reordering of operations required for the dense dense matrix products.
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = 0.0, value_type coeff_scale = 1.0)
    {
        RAISE_EXCEPTION("cuda csr matrix product has not yet been implemented.");
    }
};  //matrix_vector_product
#endif

}   //namespace expression_templates

}   //namespace linalg

#endif  //LINALG_ALGEBRA_CSR_MATRIX_VECTOR_CONTRACTION_HPP

