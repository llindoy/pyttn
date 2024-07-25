#ifndef TTNS_OPERATORS_PRIMITIVE_OPERATORS_HPP
#define TTNS_OPERATORS_PRIMITIVE_OPERATORS_HPP

#include <complex>
#include <string>
#include <memory>

#include <linalg/linalg.hpp>
#include <common/tmp_funcs.hpp>

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
class primitive
{
public:
    using vector_type = linalg::vector<T, backend>;
    using matrix_type = linalg::matrix<T, backend>;
    using size_type = typename backend::size_type;
    using matrix_ref = matrix_type&;
    using const_matrix_ref = const matrix_type&;
    using vector_ref = vector_type&;
    using const_vector_ref = const vector_type&;
    using real_type = typename tmp::get_real_type<T>::type;

protected:
    size_type m_size;
    bool m_is_identity;

public:
    primitive() : m_size(0), m_is_identity(false) {}
    primitive( size_type _size) : m_size(_size), m_is_identity(false) {}
    primitive( size_type _size, bool _is_identity) : m_size(_size), m_is_identity(_is_identity) {}
    primitive(const primitive& o) = default;
    primitive(primitive&& o) = default;
    virtual ~primitive() {}

    primitive& operator=(const primitive& o) = default;
    primitive& operator=(primitive&& o) = default;


    virtual void apply(const_matrix_ref A, matrix_ref working) = 0;
    virtual void apply(const_matrix_ref A, matrix_ref working, real_type t, real_type dt) = 0;
    virtual void apply(const_vector_ref A, vector_ref working) = 0;
    virtual void apply(const_vector_ref A, vector_ref working, real_type t, real_type dt) = 0;

    //function for allowing you to update time-dependent Hamiltonians
    virtual void update(real_type t, real_type dt) = 0;     

    virtual bool is_resizable() const{return true;}
    virtual void resize(size_type n){m_size = n;}
    virtual std::shared_ptr<primitive> clone() const = 0;
    virtual std::string to_string() const = 0;

    size_type size() const{return m_size;}
    bool is_identity() const{return m_is_identity;}


#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("size", m_size)), 
            "Failed to primitive operator.  Failed to serialise its size."
        );

        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("is_identity", m_is_identity)),
            "Failed to primitive operator.  Failed to serialise whether or not it is the identity operator."
        );
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("size", m_size)), 
            "Failed to primitive operator.  Failed to serialise its size."
        );
        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("is_identity", m_is_identity)), 
            "Failed to primitive operator.  Failed to serialise whether or not it is the identity operator."
        );
    }
#endif
};


//implementation of the identity ops ops
template <typename T, typename backend = linalg::blas_backend> 
class identity final : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;
    
    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::vector_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::real_type;

public:
    identity() : base_type() {}
    identity(size_type size) : base_type(size, true) {}
    identity(const identity& o) = default;
    identity(identity&& o) = default;
    ~identity() {}
    void apply(const_matrix_ref A, matrix_ref working) final {working = A;}
    void apply(const_matrix_ref A, matrix_ref working, real_type /*t*/, real_type /*dt*/) final {working = A;}
    void apply(const_vector_ref A, vector_ref working) final {working = A;}
    void apply(const_vector_ref A, vector_ref working, real_type /*t*/, real_type /*dt*/) final {working = A;}
    void update(real_type /* t */, real_type /* dt */) final{};    

    std::shared_ptr<base_type> clone() const{return std::make_shared<identity>(base_type::m_size);}
    std::string to_string() const final{return std::string("I_")+std::to_string(this->m_size);}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise identity operator object.  Error when serialising the base object."
        );
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE
        (   
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise identity operator object.  Error when serialising the base object."
        );
    }
#endif
};

}   //namespace ops
}   //namespace ttns



#ifdef CEREAL_LIBRARY_FOUND
#ifdef __NVCC__
#define SERIALIZE_CUDA_TYPES 
#endif
TTNS_REGISTER_SERIALIZATION(ttns::ops::identity, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::test_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_PRIMITIVE_OPERATORS_HPP//

