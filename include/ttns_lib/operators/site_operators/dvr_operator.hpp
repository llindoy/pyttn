#ifndef TTNS_OPERATORS_DVR_OPERATORS_HPP
#define TTNS_OPERATORS_DVR_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
namespace ops
{

//need to implement the kronecker product operator object
template <typename T, typename backend = linalg::blas_backend> 
class dvr_operator : public primitive<T, backend>
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
    dvr_operator()  : base_type() {}

    dvr_operator(const linalg::diagonal_matrix<T, backend>& V, const std::vector<matrix_type>& ops) try : base_type(), m_V(V), m_operators(ops)
    {
        size_type size = 1;
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            ASSERT(m_operators[i].shape(0) == m_operators[i].shape(1), "The operator to be bound must be a square matrix.");
            size *= m_operators[i].shape(0);
        }
        base_type::m_size = size;
        ASSERT(m_V.shape(0) == size, "The diagonal operator and Hamiltonian operators are not compatible sizes.")
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    template <typename ... Args>
    dvr_operator(const linalg::diagonal_matrix<T, backend>& V, const matrix_type& m, Args&& ... args) try : base_type(), m_V(V)
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
        ASSERT(m_V.shape(0) == size, "The diagonal operator and Hamiltonian operators are not compatible sizes.")
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }
    dvr_operator(const dvr_operator& o) = default;
    dvr_operator(dvr_operator&& o) = default;

    dvr_operator& operator=(const dvr_operator& o) = default;
    dvr_operator& operator=(dvr_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    void update(real_type /*t*/, real_type /*dt*/) final{}  
    std::shared_ptr<base_type> clone() const{return std::make_shared<dvr_operator>(m_V, m_operators);}

    void apply(const_matrix_ref A, matrix_ref HA) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");

        std::array<size_type, 3> mdims = {{1,1,A.size()}};
        
        HA = m_V*A;
        for(size_type i = 0; i < m_operators.size(); ++i)
        {
            mdims[0] *= mdims[1];
            mdims[1] = m_operators[i].shape(0);
            mdims[2] /= mdims[1];

            auto At = A.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto HAt = HA.reinterpret_shape(mdims[0], mdims[1], mdims[2]);

            CALL_AND_HANDLE(HAt += contract(m_operators[i], 1, At, 1), "Failed to compute kronecker product contraction.");      
        }
    }  
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const_vector_ref A , vector_ref HA) final
    {
        ASSERT(m_operators.size() > 0, "Invalid operator object.");

        std::array<size_type, 3> mdims = {{1,1,A.size()}};
        
        HA = m_V*A;
        for(size_type i = 0; i < m_operators.size(); ++i)
        {
            mdims[0] *= mdims[1];
            mdims[1] = m_operators[i].shape(0);
            mdims[2] /= mdims[1];

            auto At = A.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto HAt = HA.reinterpret_shape(mdims[0], mdims[1], mdims[2]);

            CALL_AND_HANDLE(HAt += contract(m_operators[i], 1, At, 1), "Failed to compute kronecker product contraction.");      
        }
    }  
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "dvr operator: " << std::endl;
        oss << "T: " << std::endl;
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            oss << m_operators[i] << std::endl;
        }
        oss << "V: " << std::endl << m_V << std::endl;
        return oss.str();
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
    linalg::diagonal_matrix<T, backend> m_V;
    std::vector<matrix_type> m_operators;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise dvr operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise dvr operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("potential", m_V)), "Failed to serialise dvr operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise dvr operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise dvr operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("potential", m_V)), "Failed to serialise dvr operator object.  Error when serialising the matrix.");
    }
#endif
};


}
}

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::dvr_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_DVR_OPERATOR_HPP
