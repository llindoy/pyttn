#ifndef TTNS_SWEEPING_ALGORITHM_MULTISET_SOP_ENV_NODE_NODE_HPP
#define TTNS_SWEEPING_ALGORITHM_MULTISET_SOP_ENV_NODE_NODE_HPP

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "sop_env_node.hpp"
#include "../ttn/tree/tree_node.hpp"


namespace ttns
{

template <typename T, typename B> 
class ms_sop_env_node_data
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 

    ms_sop_env_node_data() {}
    ms_sop_env_node_data(const ms_sop_env_node_data& o) = default;
    ms_sop_env_node_data(ms_sop_env_node_data&& o) = default;

    ~ms_sop_env_node_data(){}

    ms_sop_env_node_data& operator=(const ms_sop_env_node_data& o) = default;
    ms_sop_env_node_data& operator=(ms_sop_env_node_data&& o) = default;

    template <typename hnode>
    void initialise(const multiset_sttn_node_data<T>& h, const hnode& a)
    {
        m_data.clear();
        m_data.resize(h.size());
        for(size_t row = 0; row < h.size(); ++row)
        {
            m_data[row].resize(h[row].size());
            for(size_t ci = 0; ci < h[row].size(); ++ci)
            {
                size_t col = h[row][ci].col();
                CALL_AND_RETHROW(m_data[row][ci].resize(h[row][ci].nterms()));
                CALL_AND_RETHROW(m_data[row][ci].reallocate_matrices(a(col).hrank(true)*a(col).hrank(true)));
                CALL_AND_RETHROW(m_data[row][ci].resize_matrices(a(col).hrank(), a(col).hrank()));
                if(row != col)
                {
                    m_data[row][ci].store_identity();
                }
            }
        }
    }

    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, ms_sop_env_node_data&>::type operator=(const ms_sop_env_node_data<T, be> & o) 
    {
        try
        {
            m_data = o.data();
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    sop_env_node_data<T, B>& operator()(size_t i, size_t c)
    {
        ASSERT(i < m_data.size() && c  < m_data[i].size(), "Index out of bounds.");
        return m_data[i][c];
    }

    const sop_env_node_data<T, B>& operator()(size_t i , size_t c) const
    {
        ASSERT(i < m_data.size() && c  < m_data[i].size(), "Index out of bounds.");
        return m_data[i][c];
    }

    std::vector<sop_env_node_data<T, B>>& operator[](size_t i)
    {
        ASSERT(i < m_data.size(), "Index out of bounds.");
        return m_data[i];
    }

    const std::vector<sop_env_node_data<T, B>>& operator[](size_t i) const
    {
        ASSERT(i < m_data.size(), "Index out of bounds.");
        return m_data[i];
    }

public:
    void clear() 
    {
        try
        {
            m_data.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear observable node object.");
        }
    }

    const std::vector<std::vector<sop_env_node_data<T, B>>>& data() const{return m_data;}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {       
        CALL_AND_HANDLE(ar(cereal::make_nvp("data", m_data)), "Failed to serialise observable node object.  Error when serialising the terms.");
    }
#endif

protected:
    std::vector<std::vector<sop_env_node_data<T, B>>> m_data;
};

namespace node_data_traits
{
    //assignment traits for the tensor and matrix objects
    template <typename T, typename U, typename be1, typename be2>
    struct assignment_traits<ms_sop_env_node_data<T, be1>, ms_sop_env_node_data<U, be2> >
    {
        using is_applicable = std::is_convertible<U, T>;

        inline void operator()(ms_sop_env_node_data<T, be1>& o,  const ms_sop_env_node_data<U, be2>& i){CALL_AND_RETHROW(o = i);}
    };

    template <typename T, typename be>
    struct clear_traits<ms_sop_env_node_data<T, be>> 
    {
        void operator()(ms_sop_env_node_data<T, be>& t){CALL_AND_RETHROW(t.clear());}
    };

}   //namespace node_data_traits

template <typename T, typename B> 
class ms_sop_env_slice 
{
    using obj_type = typename tree_base<ms_sop_env_node_data<T, B>>::node_type&;
public:
    ms_sop_env_slice() = delete;
    ms_sop_env_slice(obj_type o, size_t i, size_t c) : m_obj(o), m_i(i), m_c(c) {}

    sop_env_node_data<T, B>& operator()() const{return m_obj()(m_i, m_c);}

    ms_sop_env_slice parent() const {return ms_sop_env_slice(m_obj.parent(), m_i, m_c);}
    ms_sop_env_slice operator[](size_t i) const {return ms_sop_env_slice(m_obj[i], m_i, m_c);}

    size_t size() const{return m_obj.size();}
protected:
    obj_type m_obj;
    size_t m_i;
    size_t m_c;
};

template <typename T, typename B> 
ms_sop_env_slice<T, B> ms_slice(typename tree_base<ms_sop_env_node_data<T, B>>::node_type& obj, size_t i, size_t j)
{
    return ms_sop_env_slice(obj, i, j);
}


}   //namespace ttns

#endif  //TTNS_SWEEPING_ALGORITHM_MULTISET_SOP_ENV_NODE_NODE_HPP


