#ifndef TTNS_OPERATORS_PURIFICATION_OPERATOR_HPP
#define TTNS_OPERATORS_PURIFICATION_OPERATOR_HPP

#include <complex>
#include <linalg/linalg.hpp>

#include "serialisation_helper.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>
#endif

namespace ttns
{
namespace ops
{

template <template <typename, typename> class op, typename T, typename backend = linalg::blas_backend> 
class purification_operator : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;
    using mat_type = op<T, backend>;

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
    purification_operator()  : base_type() {}
    template <typename ... Args>
    purification_operator(Args&& ... args) try : base_type(), m_operator(std::forward<Args>(args)...)
    {
        ASSERT(m_operator.shape(0) == m_operator.shape(1), "The operator to be bound must be a square matrix.");
        base_type::m_size = m_operator.shape(0)*m_operator.shape(0);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct commutator operator object.");
    }
    purification_operator(const purification_operator& o) = default;
    purification_operator(purification_operator&& o) = default;

    purification_operator& operator=(const purification_operator& o) = default;
    purification_operator& operator=(purification_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /* n */){ASSERT(false, "This shouldn't be called.");}
    std::shared_ptr<base_type> clone() const{return std::make_shared<purification_operator>(m_operator);}
    void apply(const_matrix_ref A, matrix_ref HA)
    {
        try
        {
            //reinterpret the arrays as rank 3 tensors 
            auto HAt = HA.reinterpret_shape(m_operator.shape(0), m_operator.shape(1)*HA.shape(1));
            auto At = A.reinterpret_shape(m_operator.shape(0), m_operator.shape(1)*HA.shape(1));
            CALL_AND_HANDLE(HAt  = m_operator*At, "Failed to apply Hamiltonian on the left.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate dense commutator operator.");
        }
    }  
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/){CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const_vector_ref A, vector_ref HA)
    {
        try
        {
            //reinterpret the arrays as rank 3 tensors 
            auto HAt = HA.reinterpret_shape(m_operator.shape(0), m_operator.shape(1));
            auto At = A.reinterpret_shape(m_operator.shape(0), m_operator.shape(1));
            CALL_AND_HANDLE(HAt  = m_operator*At, "Failed to apply Hamiltonian on the left.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate dense commutator operator.");
        }
    }  
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/){CALL_AND_RETHROW(this->apply(A, HA));}  
    void update(real_type /*t*/, real_type /*dt*/) final{}  

    const matrix_type& mat()const{return m_operator;}

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "purification operator: " << std::endl;
        oss << "H: " << std::endl;
        oss << m_operator << std::endl;
        return oss.str();
    }
protected:
    mat_type m_operator;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise commutator operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise commutator operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise commutator operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise commutator operator object.  Error when serialising the matrix.");
    }
#endif
};


template <typename T, typename backend = linalg::blas_backend> 
using dense_purification_operator = purification_operator<linalg::matrix, T, backend>;

template <typename T, typename backend = linalg::blas_backend> 
using sparse_purification_operator = purification_operator<linalg::csr_matrix, T, backend>;

template <typename T, typename backend = linalg::blas_backend> 
using diagonal_purification_operator = purification_operator<linalg::diagonal_matrix, T, backend>;

}   //namespace ttns
}   //namespace ops

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::dense_purification_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::sparse_purification_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::diagonal_purification_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_PURIFICATION_OPERATOR_HPP//

