#ifndef TTNS_OPERATORS_DIRECT_PRODUCT_OPERATORS_HPP
#define TTNS_OPERATORS_DIRECT_PRODUCT_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
namespace ops
{

//need to implement the kronecker product operator object
template <typename T, typename backend = linalg::blas_backend> 
class direct_product_operator : public primitive<T, backend>
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
    direct_product_operator()  : base_type() {}

    direct_product_operator(const std::vector<matrix_type>& ops) try : base_type()   
    {
        CALL_AND_RETHROW(initialise(ops));
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    template <typename ... Args>
    direct_product_operator(const matrix_type& m, Args&& ... args) try : base_type()
    {
        m_operators.resize(sizeof...(args)+1);
        set_operators(0, m, std::forward<Args>(args)...);
        size_type size = 1;
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            ASSERT(m_operators[i].shape(0) == m_operators[i].shape(1), "The operator to be bound must be a square matrix.");
            size *= m_operators[i].shape(0);
        }
        base_type::m_size = size;
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    void initialise(const std::vector<matrix_type>& ops)
    {
        m_operators=ops;        
        size_type size = 1;
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            ASSERT(m_operators[i].shape(0) == m_operators[i].shape(1), "The operator to be bound must be a square matrix.");
            size *= m_operators[i].shape(0);
        }
        base_type::m_size = size;
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }

    direct_product_operator(const direct_product_operator& o) = default;
    direct_product_operator(direct_product_operator&& o) = default;

    direct_product_operator& operator=(const direct_product_operator& o) = default;
    direct_product_operator& operator=(direct_product_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    void update(real_type /*t*/, real_type /*dt*/) final{}  
    std::shared_ptr<base_type> clone() const{return std::make_shared<direct_product_operator>(m_operators);}
    void apply(const_matrix_ref A, matrix_ref HA) final
    {
        CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
        
        bool HA_set = __apply_internal(A, HA);
        if(!HA_set){CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.shape(0), HA.shape(1)), "Failed to copy temp array.");}
    }  
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const_vector_ref A , vector_ref HA) final
    {
        CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
        ASSERT(m_operators.size() > 0, "Invliad operator object.");

        bool HA_set = __apply_internal(A, HA);
        if(!HA_set){CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.size()), "Failed to copy temp array.");}
    }  
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  
    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "direct product operator: " << std::endl;
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            oss << m_operators[i] << std::endl;
        }
        return oss.str();
    }

protected:
    std::vector<matrix_type> m_operators;
    vector_type m_temp;

    template <typename T1, typename T2>
    bool __apply_internal(const T1& A, T2& HA)
    {
        std::array<size_type, 3> mdims = {{1,1,A.size()}};
        
        bool HA_set = false;
        for(size_type i = 0; i < m_operators.size(); ++i)
        {
            mdims[0] *= mdims[1];
            mdims[1] = m_operators[i].shape(0);
            mdims[2] /= mdims[1];

            auto At = A.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto HAt = HA.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto Tt = m_temp.reinterpret_shape(mdims[0], mdims[1], mdims[2]);

            if(i == 0)
            {
                CALL_AND_HANDLE(HAt = contract(m_operators[i], 1, At, 1), "Failed to compute kronecker product contraction.");      
                HA_set = true;
            }
            else
            {
                if(HA_set)
                {
                    CALL_AND_HANDLE(Tt = contract(m_operators[i], 1, HAt, 1), "Failed to compute kronecker product contraction.");      
                    HA_set = false;
                }
                else
                {
                    CALL_AND_HANDLE(HAt = contract(m_operators[i], 1, Tt, 1), "Failed to compute kronecker product contraction.");      
                    HA_set = true;
                }
            }
        }
        return HA_set;
    }

protected:

    void set_operators(size_t i, const matrix_type& m)
    {
        m_operators[i] = m;
    }

    template <typename ... Args>
    void set_operators(size_t i, const matrix_type& m, Args&& ... args)
    {
        m_operators[i] = m;
        set_operators(i+1, std::forward<Args>(args)...);
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise direct_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise direct_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
    }
#endif
};


}
}

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::direct_product_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_DIRECT_PRODUCT_OPERATOR_HPP
