#ifndef LINALG_ALGEBRA_DENSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP
#define LINALG_ALGEBRA_DENSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

#include "matrix_multiplication_base.hpp"
#include <type_traits>

namespace linalg
{

namespace expression_templates
{

template <typename T1, typename T2>
class matrix_matrix_product<T1, T2, typename std::enable_if<is_dense_matrix<T1>::value && is_dense_matrix<T2>::value && is_same_backend<T1, T2>::value && is_same_value<T1, T2>::value, void>::type> 
    : public matrix_matrix_product_base<matrix_matrix_product<T1, T2, typename std::enable_if<is_dense_matrix<T1>::value && is_dense_matrix<T2>::value, void>::type>>
{
public:
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;    using backend_type = typename traits<T1>::backend_type;    using size_type = typename backend_type::size_type;
    using self_type = matrix_matrix_product<T1, T2>;    using base_type = matrix_matrix_product_base<self_type>;
    using left_type = T1;    using right_type = T2;
    using value_ptr = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
    static constexpr size_t rank = 2;

    using ttype = typename base_type::ttype;

protected:
    value_ptr m_Abuffer;
    value_ptr m_Bbuffer;
    size_type m_Asize, m_Bsize;
    size_type m_ldA, m_ldB;
    using base_type::m_m; using base_type::m_n; using base_type::m_k; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opB; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.shape(), B.shape(), transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.size()), m_Bsize(B.size()), m_ldA(A.shape(1)), m_ldB(B.shape(1)){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix matrix product object.");}

    template <typename wbuff>
    matrix_matrix_product(const value_type& coeff, const left_type& A, const right_type& B, wbuff& working, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false) 
    try : base_type(coeff, A.shape(), B.shape(), working, transA, transB, conjA, conjB), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_Asize(A.size()), m_Bsize(B.size()), m_ldA(A.shape(1)), m_ldB(B.shape(1)){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix matrix product object.");}

    template <typename V, typename = typename std::enable_if<std::is_convertible<V, value_type>::value, void>::type > 
    matrix_matrix_product(const matrix_matrix_product& o, const V& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_Bbuffer(o.m_Bbuffer), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize), m_ldA(o.m_ldA), m_ldB(o.m_ldB){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix matrix product object.");}

    matrix_matrix_product(const matrix_matrix_product& o, bool make_transpose = false, bool conjugate = false) 
    try : base_type(o, make_transpose, conjugate)
    {
        if(make_transpose)
        {
            m_Abuffer = o.m_Bbuffer;    m_Bbuffer = o.m_Abuffer;
            m_Asize = o.m_Bsize;        m_Bsize = o.m_Asize;
            m_ldA = o.m_ldB;            m_ldB = o.m_ldA;
        }
        else
        {
            m_Abuffer = o.m_Abuffer;    m_Bbuffer = o.m_Bbuffer;
            m_Asize = o.m_Asize;        m_Bsize = o.m_Bsize;
            m_ldA = o.m_ldA;            m_ldB = o.m_ldB;
        }
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix matrix product object.");}

    //performs the dense matrix matrix product.  This uses gemm which expect column major order matrices and so has been written to take that into account.  Rather than performing the operation 
    //C = op(A) op(B) we perform C^T = op(B^T) op(A^T)
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = value_type(0.0), value_type coeff_scale = value_type(1.0))
    {
        try
        {
            ASSERT(res.buffer() != m_Abuffer && res.buffer() != m_Bbuffer, "The matrix matrix product does not support inplace products.");

            value_type coeff = m_coeff*coeff_scale;
            if(m_opA == backend_type::op_c && m_opB == backend_type::op_c)
            {
                if(is_complex<value_type>::value)
                {
                    coeff = conj(coeff);    
                    if(beta != value_type(0.0))
                    {
                        beta = conj(beta);
                        CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                    }
                    CALL_AND_HANDLE(backend_type::gemm(backend_type::op_n, backend_type::op_n, m_m, m_n, m_k, coeff, m_Bbuffer, m_ldB, m_Abuffer, m_ldA, beta, res.buffer(), res.shape(1)), "Failed to compute matrix product.  Error when calling gemm.");
                    CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                }
                else{CALL_AND_HANDLE(backend_type::gemm(backend_type::op_n, backend_type::op_n, m_m, m_n, m_k, coeff, m_Bbuffer, m_ldB, m_Abuffer, m_ldA, beta, res.buffer(), res.shape(1)), "Failed to compute matrix product.  Error when calling gemm.");}
            }
            else
            {
                ttype opA = m_opA;
                ttype opB = m_opB;
                value_ptr Abuffer = m_Abuffer;
                value_ptr Bbuffer = m_Bbuffer;

                if(m_opA == backend_type::op_c && m_opB != backend_type::op_c)
                {
                    opA = backend_type::op_n;
                    if(is_complex<value_type>::value)
                    {
                        ASSERT(m_working != nullptr, "The workspace array has not been bound.");
                        ASSERT(m_working_size >= m_Asize, "The workspace array is not large enough to store temporary objects.");
                        CALL_AND_HANDLE(backend_type::complex_conjugate(m_Asize, m_Abuffer, m_working), "Failed to compute complex conjugate array value.");
                        Abuffer = static_cast<value_ptr>(m_working);
                    }
                }
                else if(m_opA != backend_type::op_c && m_opB == backend_type::op_c)
                {
                    opB = backend_type::op_n;
                    if(is_complex<value_type>::value)
                    {
                        ASSERT(m_working != nullptr, "The workspace array has not been bound.");
                        ASSERT(m_working_size >= m_Bsize, "The workspace array is not large enough to store temporary objects.");
                        CALL_AND_HANDLE(backend_type::complex_conjugate(m_Bsize, m_Bbuffer, m_working), "Failed to compute complex conjugate array value.");
                        Bbuffer = static_cast<value_ptr>(m_working);
                    }
                }

                CALL_AND_HANDLE
                (
                    backend_type::gemm(opB, opA, m_m, m_n, m_k, coeff, Bbuffer, m_ldB, Abuffer, m_ldA, beta, res.buffer(), res.shape(1)), 
                    "Error when calling gemm."
                );
            }
        }   
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute dense matrix - dense matrix product.");
        }
    }
};  //matrix_matrix_product

}   //namespace expression_templates

template <typename T1, typename T2>
struct traits<expression_templates::matrix_matrix_product<T1, T2> >
{
    using lvalue_type = typename traits<T1>::value_type;
    using rvalue_type = typename traits<T2>::value_type;
    using value_type = decltype(lvalue_type()*rvalue_type());
    using backend_type = typename traits<T1>::backend_type;
    using shape_type = std::array<typename backend_type::size_type, 2>;
    using const_shape_reference = const shape_type&;
    static constexpr size_t rank = 2;
};

}   //namespace linalg

#endif  //LINALG_ALGEBRA_DENSE_DENSE_MATRIX_MATRIX_CONTRACTION_HPP

