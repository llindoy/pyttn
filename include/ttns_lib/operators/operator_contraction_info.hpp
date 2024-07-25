#ifndef TTNS_OPERATOR_NODE_CONTRACTION_INFO_HPP
#define TTNS_OPERATOR_NODE_CONTRACTION_INFO_HPP

#include <common/exception_handling.hpp>
#include "mf_index.hpp"

namespace ttns
{

template <typename T> 
class operator_contraction_info 
{
public:
    using size_type = size_t;
    using real_type = typename tmp::get_real_type<T>::type; 

    using accum_coeff_type = std::vector<T>;

    using spf_index_type = std::vector<std::vector<std::array<size_type, 2>>>;
    using mf_index_type = std::vector<mf_index<size_type>>;

    template <typename Y, typename V> friend class operator_info;
public:
    operator_contraction_info() : m_is_identity_spf(false), m_is_identity_mf(false) {}
    operator_contraction_info(const operator_contraction_info& o) = default;
    operator_contraction_info(operator_contraction_info&& o) = default;


    operator_contraction_info(const spf_index_type& spf, const mf_index_type& mf, const T& c, const accum_coeff_type& spf_c, const accum_coeff_type& mf_c, bool idspf, bool idmf) 
      :  m_is_identity_spf(idspf), m_is_identity_mf(idmf), m_spf_index(spf), m_mf_index(mf), m_coeff(c), m_spf_coeff(spf_c), m_mf_coeff(mf_c)
    {
        ASSERT(m_spf_index.size() == m_spf_coeff.size() || m_mf_index.size() == m_mf_coeff.size(), 
             "Failed to construct operator contraction info object.");


    }

    operator_contraction_info& operator=(const operator_contraction_info& o) = default;
    operator_contraction_info& operator=(operator_contraction_info&& o) = default;

    void resize_indexing(size_type nspfterms, size_type nmfterms)
    {
        m_spf_index.resize(nspfterms);
        m_mf_index.resize(nmfterms);  
    }
    void resize_indexing(const std::vector<size_type>& spfsize, const std::vector<size_type>& mfsize)
    {
        m_spf_index.resize(spfsize.size());   for(size_type i=0; i<spfsize.size(); ++i){m_spf_index.resize(spfsize[i]);}
        m_mf_index.resize(mfsize.size());   for(size_type i=0; i<mfsize.size(); ++i){m_mf_index[i].resize(mfsize[i]);}
    }

    void clear() 
    {
        CALL_AND_HANDLE(m_spf_coeff.clear(), "Failed to clear the coefficient array.");
        CALL_AND_HANDLE(m_mf_coeff.clear(), "Failed to clear the coefficient array.");
        CALL_AND_HANDLE(m_spf_index.clear(), "Failed to clear spf object.");
        CALL_AND_HANDLE(m_mf_index.clear(), "Failed to clear mf object.");
        m_is_identity_spf = false;
        m_is_identity_mf = false;
    }

    const T& coeff() const{return m_coeff;}

    const accum_coeff_type& spf_coeff() const{return m_spf_coeff;}
    const T& spf_coeff(size_type i) const{return m_spf_coeff[i];}

    const accum_coeff_type& mf_coeff() const{return m_mf_coeff;}
    const T& mf_coeff(size_type i) const{return m_mf_coeff[i];}

    const spf_index_type& spf_indexing() const{return m_spf_index;}
    const mf_index_type& mf_indexing() const{return m_mf_index;}
    
    size_type nspf_terms() const{return m_spf_index.size();}
    size_type nmf_terms() const{return m_mf_index.size();}
    size_type nterms() const{return m_spf_index.size() > m_mf_index.size() ? m_spf_index.size() : m_mf_index.size();}

    bool is_identity_spf() const{return m_is_identity_spf;}
    bool is_identity_mf() const{return m_is_identity_mf;}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("coeff", m_coeff)), "Failed to serialise operator term object.  Error when serialising the coefficient.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("spf_coeff", m_spf_coeff)), "Failed to serialise operator term object.  Error when serialising the coefficient.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf_coeff", m_mf_coeff)), "Failed to serialise operator term object.  Error when serialising the coefficient.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("is_identity_spf", m_is_identity_spf)), "Failed to serialise operator term object.  Error when serialising whether the spf matrix is the identity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("is_identity_mf", m_is_identity_mf)), "Failed to serialise operator term object.  Error when serialising whether the mf matrix is the identity.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("spf_index", m_spf_index)), "Failed to serialise operator term object.  Error when serialising the spf indexing info.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf_index", m_mf_index)), "Failed to serialise operator term object.  Error when serialising the mf indexing info.");
    }
#endif

protected:
    bool m_is_identity_spf;
    bool m_is_identity_mf;

    spf_index_type m_spf_index;
    mf_index_type m_mf_index;

    T m_coeff;

    accum_coeff_type m_spf_coeff;
    accum_coeff_type m_mf_coeff;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const operator_contraction_info<T>& t)
{
    os << "cr: " << t.coeff() << std::endl;
    for(size_t i = 0; i < t.spf_coeff().size(); ++i)
    {
        os << "c: " << t.spf_coeff(i) << " " << t.mf_coeff(i) << std::endl;
    }
    for(size_t i = 0; i < t.spf_indexing().size(); ++i)
    {
        os << "spf: ";
        for(size_t j = 0; j < t.spf_indexing()[i].size(); ++j)
        {
            os << "(" << t.spf_indexing()[i][j][0] << ", " << t.spf_indexing()[i][j][1] << ")" << " ";
        }   
        os << std::endl;
    }
    for(size_t i = 0; i < t.mf_indexing().size(); ++i)
    {
        os << "mf: " << t.mf_indexing()[i] << std::endl;
    }
    return os;
}

}   //namespace ttns

#endif  //TTNS_OPERATOR_NODE_DATA_HPP

