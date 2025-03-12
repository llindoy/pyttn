#ifndef LINALG_ALGEBRA_MATRIX_VECTOR_MULTIPLICATION_CRTP_BASE_HPP
#define LINALG_ALGEBRA_MATRIX_VECTOR_MULTIPLICATION_CRTP_BASE_HPP

#include "../../../linalg_forward_decl.hpp"

namespace linalg
{

namespace expression_templates
{

template <typename impl>
class matrix_vector_product_base : public tensor_contraction_expression, public expression_base<matrix_vector_product_base<impl>, true>, public dense_type
{
public:
    using base_type = expression_base<matrix_vector_product_base<impl>, true>;
    using value_type = typename std::remove_cv<typename traits<impl>::value_type>::type;
    using lvalue_type = typename traits<impl>::lvalue_type;
    using rvalue_type = typename traits<impl>::rvalue_type;
    using backend_type = typename traits<impl>::backend_type;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, 1>;
    using Ashape_type = std::array<size_type, 2>;
    using ttype = typename backend_type::transform_type;

protected:
    value_type m_coeff; 

    //working buffer
    value_type* m_working;
    size_type m_working_size;

    ttype m_opA;    ttype m_opX;
    size_type m_m, m_n;
    
protected:
    void initialise_state(const Ashape_type& As, const shape_type& Xs, bool transA, bool conjA, bool conjX)
    {
        m_m = As[1];
        m_n = As[0];
        if(transA)
        {
            m_opA = conjA ? backend_type::op_c : backend_type::op_n;
            ASSERT(Xs[0] == m_n, "Failed to initialise matrix vector product.  The matrix and vector do not have compatible sizes.");
            this->m_shape[0] = m_m;   
        }
        else
        {
            m_opA = conjA ? backend_type::op_h : backend_type::op_t;
            ASSERT(Xs[0] == m_m, "Failed to initialise matrix vector product.  The matrix and vector do not have compatible sizes.");
            this->m_shape[0] = m_n;   
        }

        m_opX = conjX ? backend_type::op_c : backend_type::op_n;
    }

public:
    matrix_vector_product_base(const value_type& coeff, const Ashape_type& As, const shape_type& Xs, bool transA = false, bool conjA = false, bool conjX = false) 
        : base_type(shape_type{{0}}), m_coeff(coeff), m_working(nullptr), m_working_size(0)
    {
        //we start by determining all of the required parameters for the gemm call (and determine if the requested matrix-matrix product actually maps onto a gemm call.
        CALL_AND_HANDLE(initialise_state(As, Xs, transA, conjA, conjX), "Failed to construct matrix_vector_product object.  Failed to correctly initialise the matrix state.");
    }

    template <typename working_type>
    matrix_vector_product_base(const value_type& coeff, const Ashape_type& As, const shape_type& Xs, working_type& working_buffer, bool transA = false, bool conjA = false, bool conjX = false) 
        : base_type(shape_type{{0}}), m_coeff(coeff), m_working(nullptr), m_working_size(0)
    {
        //we start by determining all of the required parameters for the gemm call (and determine if the requested matrix-matrix product actually maps onto a gemm call.
        CALL_AND_HANDLE(initialise_state(As, Xs, transA, conjA, conjX), "Failed to construct matrix_vector_product object.  Failed to correctly initialise the matrix state.");
        CALL_AND_HANDLE(bind_conjugate_working(working_buffer), "Failed to bind working buffer.");
    }

    template <typename V, typename = typename std::enable_if<std::is_convertible<V, value_type>::value, void>::type > 
    matrix_vector_product_base(const matrix_vector_product_base& o, const V& factor)
        : base_type(o.shape()), m_coeff(o.m_coeff*factor), m_working(o.m_working), m_working_size(o.m_working_size), m_opA(o.m_opA), m_opX(o.m_opX), m_m(o.m_m), m_n(o.m_n) {}

    matrix_vector_product_base(const matrix_vector_product_base& o, bool conjugate = false) : base_type(shape_type{{0}}), m_working(o.m_working), m_working_size(o.m_working_size)
    {
        using std::conj;
        m_coeff = conjugate ? conj(o.m_coeff) : o.m_coeff;
        this->m_shape = o.m_shape;  m_m = o.m_m;    m_n = o.m_n;

        switch(o.m_opA)
        {
            case(backend_type::op_n):{m_opA = conjugate ? backend_type::op_c : backend_type::op_n; break;};
            case(backend_type::op_t):{m_opA = conjugate ? backend_type::op_h : backend_type::op_t; break;};
            case(backend_type::op_c):{m_opA = conjugate ? backend_type::op_n : backend_type::op_c; break;};
            case(backend_type::op_h):{m_opA = conjugate ? backend_type::op_t : backend_type::op_h; break;};
        };
        switch(o.m_opX)
        {
            case(backend_type::op_n):{m_opX = conjugate ? backend_type::op_c : backend_type::op_n; break;};
            case(backend_type::op_c):{m_opX = conjugate ? backend_type::op_n : backend_type::op_c; break;};
        }
    }

    template <typename conj_type>
    impl& bind_conjugate_workspace(conj_type& conj)
    {
        static_assert(is_dense<conj_type>(), "Failed to construct contraction_332 object.  The input conjorary object type is not a dense tensor.");
        static_assert(std::is_same<value_type, typename conj_type::value_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same value_type as the input tensors.");
        static_assert(std::is_same<backend_type, typename conj_type::backend_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same backend_type as the input tensors.");
        try
        {
            ASSERT(m_working == nullptr, "A working buffer has already been bound.");
            m_working = conj.buffer();
            m_working_size = conj.size();
            return *(static_cast<impl*>(this));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to bind conj array.");
        }
    }

    void unbind_conjugate_workspace()
    {
        m_working = nullptr;
        m_working_size = 0;
    }

    template <typename T3>
    void applicative(T3& res, value_type beta = 0.0, value_type coeff_scale = 1.0)
    {
        static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible value_type.");
        static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible backend type.");
        static_assert(is_dense_vector<T3>::value , "Failed to construct evaluate operator for matrix_vector_product object.  The result type must be a dense vector.");
        CALL_AND_RETHROW(static_cast<impl*>(this)->applicative_impl(res, beta, coeff_scale));
    }

    template <typename T3>
    void addition_applicative(T3& res, value_type coeff_scale = 1.0)
    {
        static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible value_type.");
        static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible backend type.");
        static_assert(is_dense_vector<T3>::value , "Failed to construct evaluate operator for matrix_vector_product object.  The result type must be a dense vector.");
        CALL_AND_RETHROW(static_cast<impl*>(this)->applicative_impl(res, 1.0, coeff_scale));
    }


    template <typename T3>
    void subtraction_applicative(T3& res, value_type coeff_scale = 1.0)
    {
        static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible value_type.");
        static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_vector_product object.  The result_type must have a compatible backend type.");
        static_assert(is_dense_vector<T3>::value , "Failed to construct evaluate operator for matrix_vector_product object.  The result type must be a dense vector.");
        CALL_AND_RETHROW(static_cast<impl*>(this)->applicative_impl(res, -1.0, coeff_scale));
    }
};  //matrix_vector_product_base

}   //namespace expression_templates
template <typename impl>
struct traits<expression_templates::matrix_vector_product_base<impl> >
{
    using value_type = typename traits<impl>::value_type;
    using lvalue_type = typename traits<impl>::lvalue_type;
    using rvalue_type = typename traits<impl>::rvalue_type;
    using backend_type = typename traits<impl>::backend_type;
    using shape_type = std::array<typename backend_type::size_type, 1>;
    using const_shape_reference = const shape_type&;
    static constexpr size_t rank = 1;
};

}   //namespace linalg

#endif  //LINALG_ALGEBRA_MATRIX_VECTOR_MULTIPLICATION_CRTP_BASE_HPP//


