#ifndef TTNS_OPERATORS_SEQUENTIAL_PRODUCT_OPERATORS_HPP
#define TTNS_OPERATORS_SEQUENTIAL_PRODUCT_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
namespace ops
{

//need to implement the kronecker product operator object
template <typename T, typename backend = linalg::blas_backend> 
class sequential_product_operator : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;

public:
    sequential_product_operator()  : base_type() {}

    sequential_product_operator(const std::vector<std::shared_ptr<base_type>>& ops) try : base_type(), m_operators(ops)   
    {
        ASSERT(ops.size() > 0, "Cannot create sequential product operator without an element.");
        size_type size = ops[0]->size();
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            ASSERT(m_operators[i]->size() == size, "All operators in the sequential product operator must have the same size.");
        }
        base_type::m_size = size;
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    sequential_product_operator(const sequential_product_operator& o) = default;
    sequential_product_operator(sequential_product_operator&& o) = default;

    sequential_product_operator& operator=(const sequential_product_operator& o) = default;
    sequential_product_operator& operator=(sequential_product_operator&& o) = default;

    void append_operator(std::shared_ptr<base_type> op)
    {
        m_operators.insert(m_operators.begin(), op);
    }

    template <typename OpType>
    void append_operator(const OpType& op)
    {
        m_operators.insert(m_operators.begin(), std::make_shared<OpType>(op));
    }

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    void update(real_type t, real_type dt) final
    {
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            CALL_AND_RETHROW(m_operators[i]->update(t, dt));
        }
    }  
    std::shared_ptr<base_type> clone() const{return std::make_shared<sequential_product_operator>(m_operators);}

    void apply(const_matrix_ref A, matrix_ref HA) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        __apply_internal(A, HA);
    }  
    void apply(const_matrix_ref A, matrix_ref HA, real_type t, real_type dt) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        this->update(t, dt);
        __apply_internal(A, HA);
    } 

    void apply(const_vector_ref A , vector_ref HA) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        __apply_internal(A, HA);
    }  

    void apply(const_vector_ref A, vector_ref HA, real_type t, real_type dt) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        this->update(t, dt);
        __apply_internal(A, HA);
    }

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "sequential product operator: " << std::endl;
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            oss << m_operators[i]->to_string() << std::endl;
        }
        return oss.str();
    }

protected:
    std::vector<std::shared_ptr<base_type>> m_operators;
    linalg::vector<T, backend> m_temp;

    template <typename T1, typename T2>
    void __apply_internal(const T1& A, T2& HA)
    {
        m_temp.resize(A.size());
        auto t = m_temp.reinterpret_shape(A.shape());
        t = A;
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            size_t ind = m_operators.size() - (i+1);
            m_operators[ind]->apply(t, HA);
            t = HA;
        }
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise sequential_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_operators)), "Failed to serialise sequential_product operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise sequential_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_operators)), "Failed to serialise sequential_product operator object.  Error when serialising the matrix.");
    }
#endif
};


}
}

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::sequential_product_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_SEQUENTIAL_PRODUCT_OPERATOR_HPP
