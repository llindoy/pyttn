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

template <typename T, typename backend = linalg::blas_backend> 
class purification_operator : public primitive<T, backend>
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

public:
    purification_operator()  : base_type() {}

    purification_operator(std::shared_ptr<base_type> op) try : base_type(), m_operator(op->clone())
    {
        base_type::m_size = op->size()*op->size();
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct commutator operator object.");
    }

    template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
    purification_operator(const OpType& op) try : purification_operator(std::make_shared<OpType>(op)) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct purification operator.");}

    purification_operator(const purification_operator& o) = default;
    purification_operator(purification_operator&& o) = default;

    purification_operator& operator=(const purification_operator& o) = default;
    purification_operator& operator=(purification_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /* n */){ASSERT(false, "This shouldn't be called.");}
    std::shared_ptr<base_type> clone() const{return std::make_shared<purification_operator>(m_operator);}


    void apply(const resview& /* A */, resview& /* HA */)
    {
        RAISE_EXCEPTION("Cannot have nested purifications.");
    }  
    void apply(const resview& A, resview& HA, real_type t, real_type dt){this->update(t, dt); CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const matview& /* A */, resview& /* HA */)
    {
        RAISE_EXCEPTION("Cannot have nested purifications.");
    }  
    void apply(const matview& A, resview& HA, real_type t, real_type dt){this->update(t, dt); CALL_AND_RETHROW(this->apply(A, HA));}  

    void apply(const_matrix_ref A, matrix_ref HA)
    {
        HA.resize(A.shape(0), A.shape(1));
        CALL_AND_RETHROW(apply_internal(A, HA));
    }  
    void apply(const_matrix_ref A, matrix_ref HA, real_type t, real_type dt){this->update(t, dt); CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const_vector_ref A, vector_ref HA)
    {
        try
        {
            HA.resize(A.shape());
            //reinterpret the arrays as rank 3 tensors 
            auto HAt = HA.reinterpret_shape(m_operator->size(), m_operator->size());
            auto At = A.reinterpret_shape(m_operator->size(), m_operator->size());
            CALL_AND_HANDLE(m_operator->apply(At, HAt), "Failed to apply Hamiltonian on the left.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate purification operator.");
        }
    }  
    void apply(const_vector_ref A, vector_ref HA, real_type t, real_type dt){this->update(t, dt); CALL_AND_RETHROW(this->apply(A, HA));}  
    void update(real_type t, real_type dt) final{m_operator->update(t, dt);}  

    const matrix_type& mat()const{return m_operator;}

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "purification operator: " << std::endl;
        oss << "H: " << std::endl;
        oss << m_operator->to_string() << std::endl;
        return oss.str();
    }
protected:
    template <typename Atype, typename Rtype>
    void apply_internal(const Atype& A, Rtype& HA)
    {
        try
        {
            //reinterpret the arrays as rank 3 tensors 
            auto HAt = HA.reinterpret_shape(m_operator->size(), (HA.shape(0)*HA.shape(1))/m_operator->size());
            auto At = A.reinterpret_shape(m_operator->size(), (HA.shape(0)*HA.shape(1))/m_operator->size());
            CALL_AND_HANDLE(m_operator->apply(At, HAt), "Failed to apply Hamiltonian on the left.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate purification operator.");
        }
    }

protected:
    std::shared_ptr<base_type> m_operator;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise commutator operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("op", m_operator)), "Failed to serialise commutator operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise commutator operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("op", m_operator)), "Failed to serialise commutator operator object.  Error when serialising the matrix.");
    }
#endif
};

}   //namespace ttns
}   //namespace ops

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::purification_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_PURIFICATION_OPERATOR_HPP//

