#ifndef TTNS_SWEEPING_ALGORITHM_SOP_ENV_NODE_NODE_HPP
#define TTNS_SWEEPING_ALGORITHM_SOP_ENV_NODE_NODE_HPP

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "observable_node.hpp"
#include "../ttn/tree/tree_node.hpp"


namespace ttns
{

template <typename T, typename B> 
class sop_env_node_data
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 

    sop_env_node_data() {}
    sop_env_node_data(const sop_env_node_data& o) = default;
    sop_env_node_data(sop_env_node_data&& o) = default;

    ~sop_env_node_data(){}

    sop_env_node_data& operator=(const sop_env_node_data& o) = default;
    sop_env_node_data& operator=(sop_env_node_data&& o) = default;

    template <typename hnode>
    void initialise(const sttn_node_data<T>& h, const hnode& a)
    {
        if(h.nterms() > m_spf.size())
        {
            CALL_AND_HANDLE(resize(h.nterms()), "Failed to resize matrix element container.");
        }
        CALL_AND_HANDLE(reallocate_matrices(a.maxhrank(true)*a.maxhrank(true)), "Failed to reallocate mel tensors.");
        CALL_AND_HANDLE(resize_matrices(a.maxhrank(), a.maxhrank()), "Failed to reallocate mel tensors.");
    }

    void store_identity()
    {   
        m_spf.store_identity();
        m_mf.store_identity();
    }

    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, sop_env_node_data&>::type operator=(const sop_env_node_data<T, be> & o) 
    {
        try
        {
            m_spf = o.spf_data();
            m_mf = o.mf_data();
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    void resize_matrices(size_type n, size_type m)
    {
        CALL_AND_RETHROW(m_spf.resize_matrices(n,m));
        CALL_AND_RETHROW(m_mf.resize_matrices(n,m));
    }

    void reallocate_matrices(size_type capacity)
    {
        CALL_AND_RETHROW(m_spf.reallocate_matrices(capacity));
        CALL_AND_RETHROW(m_mf.reallocate_matrices(capacity));
    }


    void resize(size_type r)
    {
        m_spf.resize(r);
        m_mf.resize(r);
    }
public:
    void clear() 
    {
        try
        {
            m_spf.clear();
            m_mf.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear observable node object.");
        }
    }

    size_type size() const{return m_spf.size();}

    const linalg::matrix<T, B>& spf(size_type i) const{return m_spf[i];}
    linalg::matrix<T, B>& spf(size_type i){return m_spf[i];}

    const linalg::matrix<T, B>& mf(size_type i) const{return m_mf[i];}
    linalg::matrix<T, B>& mf(size_type i){return m_mf[i];}

    const linalg::matrix<T, B>& spf_id() const{return m_spf.id();}
    linalg::matrix<T, B>& spf_id(){return m_spf.id();}

    const linalg::matrix<T, B>& mf_id() const{return m_mf.id();}
    linalg::matrix<T, B>& mf_id(){return m_mf.id();}

    const observable_node_data<T, B>& spf_data() const{return m_spf;}
    observable_node_data<T, B>& spf_data(){return m_spf;}
    const observable_node_data<T, B>& mf_data() const{return m_mf;}
    observable_node_data<T, B>& mf_data(){return m_mf;}

    bool has_identity() const{return m_spf.has_identity();}
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {       
        CALL_AND_HANDLE(ar(cereal::make_nvp("spf", m_spf)), "Failed to serialise observable node object.  Error when serialising the terms.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf", m_mf)), "Failed to serialise observable node object.  Error when serialising the terms.");
    }
#endif

protected:
    observable_node_data<T, B> m_spf;
    observable_node_data<T, B> m_mf;
};

namespace node_data_traits
{
    //assignment traits for the tensor and matrix objects
    template <typename T, typename U, typename be1, typename be2>
    struct assignment_traits<sop_env_node_data<T, be1>, sop_env_node_data<U, be2> >
    {
        using is_applicable = std::is_convertible<U, T>;

        inline void operator()(sop_env_node_data<T, be1>& o,  const sop_env_node_data<U, be2>& i){CALL_AND_RETHROW(o = i);}
    };

    template <typename T, typename be>
    struct clear_traits<sop_env_node_data<T, be>> 
    {
        void operator()(sop_env_node_data<T, be>& t){CALL_AND_RETHROW(t.clear());}
    };

}   //namespace node_data_traits


}   //namespace ttns

#endif  //TTNS_SWEEPING_ALGORITHM_SOP_ENV_NODE_NODE_HPP

