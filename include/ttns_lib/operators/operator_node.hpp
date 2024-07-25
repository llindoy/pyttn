#ifndef TTNS_OPERATOR_NODE_DATA_HPP
#define TTNS_OPERATOR_NODE_DATA_HPP

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "operator_contraction_info.hpp"
#include "../sop/autoSOP_node.hpp"

namespace ttns
{

template <typename T, typename B> class operator_node_data;

template <typename T, typename B> 
class operator_term 
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 
    using triad = std::vector<linalg::matrix<T, B>>;

    using tree_type = tree<operator_node_data<T, B>>;
    using node_type = typename tree_type::node_type;

    using accum_coeff_type = std::vector<T>;

    using spf_index_type = std::vector<std::vector<std::array<size_type, 2>>>;
    using mf_index_type = std::vector<mf_index<size_type>>;

    template <typename Y, typename V> friend class operator_container;
public:
    operator_term() {}
    operator_term(const operator_term& o) = default;
    operator_term(operator_term&& o) = default;
    operator_term& operator=(const operator_term& o) = default;

    operator_term(const operator_contraction_info<T> & o)  
    {
        CALL_AND_HANDLE(m_oci = o, "Failed to construct operator contraction term from operator_contraction_info.");
    }

    void set_operator_contraction_info(const operator_contraction_info<T> & o)
    {
        CALL_AND_HANDLE(m_oci = o, "Failed to set operator_contraction_info.");
    }


    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, operator_term&>::type operator=(const operator_term<T, be> & o) 
    {
        try
        {
            CALL_AND_HANDLE(m_spf = o.spf(), "Failed to copy spf matrix.");
            CALL_AND_HANDLE(m_mf = o.mf(), "Failed to copy mf matrix.");
            
            m_oci = o.m_oci;
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    void reallocate_matrices(size_type capacity)
    {
        if(!m_exploit_identity_opt || !m_oci.is_identity_spf())
        {
            CALL_AND_HANDLE(m_spf.reallocate(capacity), "Failed to resize spf matrix.");
        }
        if(!m_exploit_identity_opt || !m_oci.is_identity_mf())
        {
            CALL_AND_HANDLE(m_mf.reallocate(capacity), "Failed to resize spf matrix.");
        }
    }

    size_t get_memory_requirements(size_type capacity)
    {
        size_t ret = 0;

        if(!m_exploit_identity_opt || !m_oci.is_identity_spf())
        {
            ret += capacity;
        }
        if(!m_exploit_identity_opt || !m_oci.is_identity_mf())
        {
            ret += capacity;
        }
        return ret;
    }

    void resize_matrices(size_type n, size_type m)
    {
        m_matsize[0] = n;   m_matsize[1] = m;
        if(!m_exploit_identity_opt || !m_oci.is_identity_spf())
        {
            CALL_AND_HANDLE(m_spf.resize(n, m), "Failed to resize spf matrix.");
        }
        if(!m_exploit_identity_opt || !m_oci.is_identity_mf())
        {
            CALL_AND_HANDLE(m_mf.resize(n, m), "Failed to resize spf matrix.");
        }
    }

    const size_type& matrix_size(size_type i) const
    {
        ASSERT(i < 2, "Index out of bounds.");
        return m_matsize[i];
    }

    void clear() 
    {
        CALL_AND_HANDLE(m_oci.clear(), "Failed to clear the coefficient array.");
        CALL_AND_HANDLE(m_mf.clear(), "Failed to clear mf matrix object.");
        CALL_AND_HANDLE(m_spf.clear(), "Failed to clear spf matrix object.");
    }

    const linalg::matrix<T, B>& spf() const{return m_spf;}
    linalg::matrix<T, B>& spf(){return m_spf;}

    const linalg::matrix<T, B>& mf() const{return m_mf;}
    linalg::matrix<T, B>& mf(){return m_mf;}

    bool is_identity_spf() const{return m_exploit_identity_opt && m_oci.is_identity_spf();}
    bool is_identity_mf() const{return m_exploit_identity_opt && m_oci.is_identity_mf();}

    const bool& exploit_identity_opt() const{return m_exploit_identity_opt;}
    bool& exploit_identity_opt(){return m_exploit_identity_opt;}

    const T& coeff() const{return m_oci.coeff();}

    const accum_coeff_type& spf_coeff() const{return m_oci.spf_coeff();}
    const T& spf_coeff(size_type i) const{return m_oci.spf_coeff(i);}

    const accum_coeff_type& mf_coeff() const{return m_oci.mf_coeff();}
    const T& mf_coeff(size_type i) const{return m_oci.mf_coeff(i);}

    const spf_index_type& spf_indexing() const{return m_oci.spf_indexing();}
    const mf_index_type& mf_indexing() const{return m_oci.mf_indexing();}
    
    size_type nspf_terms() const{return m_oci.nspf_terms();}
    size_type nmf_terms() const{return m_oci.nmf_terms();}

    const operator_contraction_info<T>& contraction_info() const{return m_oci;}


    T energy() const
    {
        if(m_oci.is_identity_spf() && m_oci.is_identity_mf())
        {
            return coeff();
        }
        else if(m_oci.is_identity_spf())
        {
            return linalg::trace(m_mf)*coeff();
        }
        else if(m_oci.is_identity_mf())*coeff()
        {
            return linalg::trace(m_spf)*coeff();
        }
        else
        {
            auto spf = m_spf.reinterpret_shape(m_spf.shape(0)*m_spf.shape(1));
            auto mf = m_mf.reinterpret_shape(m_mf.shape(0)*m_mf.shape(1));
            return linalg::dot_product(spf, mf)*coeff();
        }
    }
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("spf", m_spf)), "Failed to serialise operator term object.  Error when serialising the spf matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf", m_mf)), "Failed to serialise operator term object.  Error when serialising the mf matrix.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("contraction_info", m_oci)), "Failed to serialise operator term object.  Error when serialising the contraction info.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("use_idopt", m_exploit_identity_opt)), "Failed to serialise operator term object.  Error when serialising the contraction info.");
    }
#endif

protected:
    linalg::matrix<T, B> m_spf;
    linalg::matrix<T, B> m_mf; 

    std::array<size_type, 2> m_matsize;
    bool m_exploit_identity_opt = true;

    operator_contraction_info<T> m_oci;
};

template <typename T, typename backend>
std::ostream& operator<<(std::ostream& os, const operator_term<T, backend>& t)
{
    os << t.contraction_info();
    return os;
}

template <typename T, typename B>
class operator_node_data
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 

    template <typename Y, typename V> friend class operator_container;
public:
    operator_node_data() : m_has_idmat(false){}

    operator_node_data(const operator_node_data& o) = default;
    operator_node_data(operator_node_data&& o) = default;
    operator_node_data& operator=(const operator_node_data& o) = default;
    operator_node_data& operator=(operator_node_data&& o) = default;

    void set_contraction_info(const std::vector<operator_contraction_info<T> >& o)
    {
        m_term.clear();
        m_term.resize(o.size());
        for(size_t i = 0; i < o.size(); ++i)
        {
            m_term[i] = o[i];
        }
    }

    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, operator_node_data&>::type operator=(const operator_node_data<T, be> & o) 
    {
        try
        {
            m_term.resize(o.nterms());
            m_idmatspf = o.m_idmatspf;
            m_idmatmf = o.m_idmatmf;

            for(size_type i = 0; i < o.nterms(); ++i){m_term[i] = o.terms(i);}
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    ~operator_node_data(){}

    void resize_matrices(size_type n, size_type m)
    {
        for(size_type i=0; i < m_term.size(); ++i)
        {
            CALL_AND_HANDLE(m_term[i].resize_matrices(n, m), "Failed to setup matrices for operator node.");
        }

        if(m_has_idmat)
        {
            CALL_AND_HANDLE(m_idmatspf.resize(n, m), "Failed to resize id operator matrices needed when dealing with matrix elements between tensor networks of different sizes.");
            CALL_AND_HANDLE(m_idmatmf.resize(n, m), "Failed to resize id operator matrices needed when dealing with matrix elements between tensor networks of different sizes.");
        }
    }

    const size_type& matrix_size(size_type ind) const
    {
        ASSERT(m_term.size() > 0, "Index out of bounds.");
        CALL_AND_RETHROW(return m_term[0].matrix_size(ind));
    }

    size_t get_memory_requirements(size_type capacity)
    {
        size_t ret = 0;
        for(size_type i=0; i < m_term.size(); ++i)
        {
            CALL_AND_HANDLE(ret += m_term[i].get_memory_requirements(capacity), "Failed to get memory requirements for  operator node.");
        }
        return ret;
    }

    void reallocate_matrices(size_type capacity)
    {
        for(size_type i=0; i < m_term.size(); ++i)
        {
            CALL_AND_HANDLE(m_term[i].reallocate_matrices(capacity), "Failed to setup matrices for operator node.");
        }
        if(m_has_idmat)
        {
            CALL_AND_HANDLE(m_idmatspf.reallocate(capacity), "Failed to resize id operator matrices needed when dealing with matrix elements between tensor networks of different sizes.");
            CALL_AND_HANDLE(m_idmatmf.reallocate(capacity), "Failed to resize id operator matrices needed when dealing with matrix elements between tensor networks of different sizes.");
        }
    }

    T energy() const
    {
        T e(0);
        for(size_t i = 0; i < m_term.size(); ++i)
        {
            e += m_term[i].energy();
        }
        return e;
    }

    void clear() 
    {
        try
        {
            for(size_type i = 0; i < m_term.size(); ++i){m_term[i].clear();}
            m_term.clear();

            m_idmatspf.clear();
            m_idmatmf.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear operator node object.");
        }
    }

    size_type nterms() const{return m_term.size();}

    const operator_term<T, B>& term(size_type i) const
    {
        ASSERT(i  < m_term.size(), "Index out of bounds.");
        return m_term[i];
    }

    operator_term<T, B>& term(size_type i)
    {
        ASSERT(i < m_term.size(), "Index out of bounds.");
        return m_term[i];
    }

    const operator_term<T, B>& operator[](size_type i) const {return m_term[i];}
    operator_term<T, B>& operator[](size_type i){return m_term[i];}

    /* 
     *  Accessor functions for the identity matrices.  These are only used whenever bra and ket are different
     */
    const bool& has_id_matrices() const{return m_has_idmat;}
    bool& has_id_matrices(){return m_has_idmat;}

    const bool& exploit_identity_opt() const{return m_exploit_identity_opt;}
    void set_exploit_identity_optimisations(bool use_id = true)
    {
        m_exploit_identity_opt = use_id;
        for(auto& t : m_term)
        {
            t.exploit_identity_opt() = use_id;
        }
    }


    const linalg::matrix<T, B>& id_spf() const
    {
        ASSERT(m_has_idmat, "Failed to access id matrix modes out of bounds.");
        return m_idmatspf;
    }

    linalg::matrix<T, B>& id_spf()
    {
        ASSERT(m_has_idmat, "Failed to access id matrix modes out of bounds.");
        return m_idmatspf;
    }
    
    const linalg::matrix<T, B>& id_mf(size_t i) const
    {
        ASSERT(m_has_idmat, "Failed to access id matrix modes out of bounds.");
        return m_idmatmf;
    }

    linalg::matrix<T, B>& id_mf()
    {
        ASSERT(m_has_idmat, "Failed to access id matrix modes out of bounds.");
        return m_idmatmf;
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {       
        CALL_AND_HANDLE(ar(cereal::make_nvp("terms", m_term)), "Failed to serialise operator node object.  Error when serialising the terms.");
    }
#endif

protected:
    linalg::matrix<T, B> m_idmatspf;
    linalg::matrix<T, B> m_idmatmf;
    std::vector<operator_term<T, B>> m_term;
    bool m_has_idmat;
    bool m_exploit_identity_opt;
};  //operator_node_data


template <typename T, typename backend>
std::ostream& operator<<(std::ostream& os, const operator_node_data<T, backend>& t)
{
    for(size_t i = 0; i < t.nterms(); ++i)
    {
        os << t[i] << std::endl;
    }
    return os;
}

//implementation of the sparse_multiset_operator_node_data type
template <typename T, typename B>
using multiset_operator_node_data = std::vector<std::vector<operator_node_data<T, B>>>;

namespace node_data_traits
{
    //clear traits for the operator node data object
    template <typename T, typename backend>
    struct clear_traits<operator_node_data<T, backend> > 
    {
        void operator()(operator_node_data<T, backend>& t){CALL_AND_RETHROW(t.clear());}
    };

    template <typename T, typename backend>
    struct clear_traits<multiset_operator_node_data<T, backend> > 
    {
        void operator()(multiset_operator_node_data<T, backend>& t)
        {
            for(auto& a : t)
            {
                for(auto& b : a)
                {
                    CALL_AND_RETHROW(b.clear());
                }
                a.clear();
            }
            CALL_AND_RETHROW(t.clear());
        }
    };
}   //namespace node_data_traits
}   //namespace ttns

#endif  //TTNS_OPERATOR_NODE_DATA_HPP

