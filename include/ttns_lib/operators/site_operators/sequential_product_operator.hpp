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
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

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
        ASSERT(op.size() == base_type::m_size, "Failed to append operator invalid size.");
        m_operators.insert(m_operators.begin(), op);
    }

    template <typename OpType>
    void append_operator(const OpType& op)
    {
        ASSERT(op.size() == base_type::m_size, "Failed to append operator invalid size.");
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


    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  
    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}


protected:
    template <typename T1, typename T2>
    void apply_rank_2(const T1& A, T2& HA)
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        try
        {
            m_temp.resize(A.size());
            resview t = m_temp.reinterpret_shape(A.shape(0), A.shape(1));
            resview _HA = HA.reinterpret_shape(A.shape(0), A.shape(1));
            for(size_t i = 0; i < m_operators.size(); ++i)
            {
                if(i == 0){t.set_buffer(A);}
                else{t.set_buffer(_HA);}
                size_t ind = m_operators.size() - (i+1);
                m_operators[ind]->apply(t, _HA);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply sequential product operator.");
        }
    }

    template <typename T1, typename T2>
    void apply_rank_3(const T1& A, T2& HA)
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");
        try
        {
            m_temp.resize(A.size());
            restensview t = m_temp.reinterpret_shape(A.shape(0), A.shape(1), A.shape(2));
            restensview _HA = HA.reinterpret_shape(A.shape(0), A.shape(1), A.shape(2));
            for(size_t i = 0; i < m_operators.size(); ++i)
            {
                if(i == 0){t.set_buffer(A);}
                else{t.set_buffer(_HA);}
                size_t ind = m_operators.size() - (i+1);
                m_operators[ind]->apply(t, _HA);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply sequential product operator.");
        }
    }
public:
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
