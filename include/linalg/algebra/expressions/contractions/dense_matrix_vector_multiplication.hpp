#ifndef LINALG_ALGEBRA_DENSE_MATRIX_vector_CONTRACTION_HPP
#define LINALG_ALGEBRA_DENSE_MATRIX_vector_CONTRACTION_HPP

#include "matrix_vector_multiplication_base.hpp"
#include <type_traits>

namespace linalg
{

namespace expression_templates
{

template <typename T1, typename T2>
class matrix_vector_product<T1, T2, typename std::enable_if<is_dense_matrix<T1>::value && is_dense_vector<T2>::value, void>::type> 
    : public matrix_vector_product_base<matrix_vector_product<T1, T2, typename std::enable_if<is_dense_matrix<T1>::value && is_dense_vector<T2>::value, void>::type>>
{
public:
    static_assert(is_same_value<T1, T2>::value, "Failed to construct contraction_332 object.  The two input tensors must have the same value_type.");
    static_assert(is_same_backend<T1, T2>::value, "Failed to construct contraction_332 object.  The two input tensors must have the same backend type.");

    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;    
    using backend_type = typename traits<T1>::backend_type;    using size_type = typename traits<T1>::size_type;
    using self_type = matrix_vector_product<T1, T2>;    using base_type = matrix_vector_product_base<self_type>;
    using left_type = T1;    using right_type = T2;
    using value_ptr = typename std::add_pointer<typename std::add_const<value_type>::type>::type;


    using ttype = typename backend_type::transform_type;
    static constexpr size_t rank = 1;

protected:
    value_ptr m_Abuffer;
    value_ptr m_Xbuffer;
    size_type m_Asize, m_Xsize;
    size_type m_ldA, m_incX;
    using base_type::m_m; using base_type::m_n; using base_type::m_coeff; using base_type::m_opA; using base_type::m_opX; using base_type::m_working; using base_type::m_working_size;
public:
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.shape(), X.shape(), transA, conjA, conjX), m_Abuffer(A.buffer()), m_Xbuffer(X.buffer()), m_Asize(A.size()), m_Xsize(X.size()), m_ldA(A.shape(1)), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix vector product object.");}

    template <typename working_type>
    matrix_vector_product(const value_type& coeff, const left_type& A, const right_type& X, working_type& working_buffer, bool transA = false, bool conjA = false, bool conjX = false) 
    try : base_type(coeff, A.shape(), X.shape(), working_buffer, transA, conjA, conjX), m_Abuffer(A.buffer()), m_Xbuffer(X.buffer()), m_Asize(A.size()), m_Xsize(X.size()), m_ldA(A.shape(1)), m_incX(X.incx()){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix vector product object.");}

    template <typename V, typename = typename std::enable_if<std::is_convertible<V, value_type>::value, void>::type > 
    matrix_vector_product(const matrix_vector_product& o, const V& factor)
    try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_ldA(o.m_ldA), m_incX(o.m_incX){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix vector product object.");}

    matrix_vector_product(const matrix_vector_product& o, bool conjugate = false) 
    try : base_type(o, conjugate)
    {
        m_Abuffer = o.m_Abuffer;    m_Xbuffer = o.m_Xbuffer;
        m_Asize = o.m_Asize;        m_Xsize = o.m_Xsize;
        m_ldA = o.m_ldA;            m_incX = o.m_incX;
    }
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl; RAISE_EXCEPTION("Failed to construct dense dense matrix vector product object.");}

    //performs the dense matrix vector product.  This uses gemv which expect column major order matrices and so has been written to take that into account.  Rather than performing the operation 
    //C = op(A) op(X) we perform C^T = op(X^T) op(A^T)
    template <typename T3>
    void applicative_impl(T3& res, value_type beta = value_type(0.0), value_type coeff_scale = value_type(1.0))
    {
        try
        {
            ASSERT(res.buffer() != m_Abuffer && res.buffer() != m_Xbuffer, "Failed to compute matrix vector product.  The matrix vector product does not support inplace products.");

            value_type coeff = m_coeff*coeff_scale;
            
            if(m_opX == backend_type::op_c)
            {
                if(m_opA == backend_type::op_c || m_opA == backend_type::op_h)
                {
                    ttype opA;
                    if(m_opA == backend_type::op_c){opA = backend_type::op_n;}
                    else{opA = backend_type::op_t;} //if (m_opA == backend_type::op_h)
                
                    if(is_complex<value_type>::value)
                    {
                        coeff = conj(coeff);
                        if(beta != value_type(0.0))
                        {
                            beta = conj(beta);
                            CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                        }
                        CALL_AND_HANDLE(backend_type::gemv(opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_Xbuffer, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");
                        CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                    }
                    else{CALL_AND_HANDLE(backend_type::gemv(opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_Xbuffer, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");}
                }
                else
                {
                    //now we need to explicitly conjugate Xbuffer
                    if(is_complex<value_type>::value)
                    {
                        ASSERT(m_working != nullptr, "The working space buffer has not been bound.");
                        ASSERT(m_working_size >= m_Xsize, "The working space array is not large enough to store temporary objects.");
                        CALL_AND_HANDLE(backend_type::complex_conjugate(m_Xsize, m_Xbuffer, m_working), "Failed to compute temporary complex conjugate array value.");
                        CALL_AND_HANDLE(backend_type::gemv(m_opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_working, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");
                    }
                    else{CALL_AND_HANDLE(backend_type::gemv(m_opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_Xbuffer, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");}
                }
            }
            else 
            {
                if(m_opA == backend_type::op_c)
                {
                    ttype opA = backend_type::op_n;
                    if(is_complex<value_type>::value)
                    {
                        ASSERT(m_working != nullptr, "The working space array is not large enough to store temporary objects.");
                        ASSERT(m_working_size >= m_Xsize, "The working space array is not large enough to store temporary objects.");
                        //conjugate X and the coefficients
                        CALL_AND_HANDLE(backend_type::complex_conjugate(m_Xsize, m_Xbuffer, m_working), "Failed to compute temporary complex conjugate array value.");
                        coeff = conj(coeff);
                        if(beta != value_type(0.0))
                        {
                            beta = conj(beta);
                            CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                        }
                        CALL_AND_HANDLE(backend_type::gemv(opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_working, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");
                        CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                    }
                    else{CALL_AND_HANDLE(backend_type::gemv(opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_Xbuffer, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");}
                    
                }
                else{CALL_AND_HANDLE(backend_type::gemv(m_opA, m_m, m_n, coeff, m_Abuffer, m_ldA, m_Xbuffer, m_incX, beta, res.buffer(), res.incx()), "Error when calling gemv.");}
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate matrix vector product.");
        }
    }
};  //matrix_vector_product

}   //namespace expression_templates

template <typename T1, typename T2>
struct traits<expression_templates::matrix_vector_product<T1, T2> >
{
    using lvalue_type = typename traits<T1>::value_type;
    using rvalue_type = typename traits<T2>::value_type;
    using value_type = decltype(lvalue_type()*rvalue_type());
    using backend_type = typename traits<T1>::backend_type;
    using shape_type = std::array<typename backend_type::size_type, 1>;
    using const_shape_reference = const shape_type&;
    static constexpr size_t rank = 1;
};

}   //namespace linalg

#endif  //LINALG_ALGBXRA_DENSE_MATRIX_VECTOR_CONTRACTION_HPP

