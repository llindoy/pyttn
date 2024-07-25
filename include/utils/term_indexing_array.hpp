#ifndef TTNS_UTILS_MINIMAL_INDEX_ARRAY_HPP
#define TTNS_UTILS_MINIMAL_INDEX_ARRAY_HPP

#include <linalg/linalg.hpp>
#include <common/exception_handling.hpp>
#include <common/zip.hpp>

#include <memory>
#include <list>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <set>
#include <tuple>
#include <memory>
#include <utility>
#include <initializer_list>
#include <type_traits>

namespace utils
{
template <typename size_type> class term_indexing_array;

template <typename size_type> std::ostream& operator<<(std::ostream& os, const term_indexing_array<size_type>& op);

//TODO: Implement an iterator for this object.
template <typename size_type> 
class term_indexing_array
{
public:
    friend std::ostream& operator<< <size_type>(std::ostream& os, const term_indexing_array<size_type>& op);
    static_assert(std::is_integral<size_type>::value, "Cannot create minimal array for non-integral type.");
      
    //TODO: Currently the complement iterator does not work at all.  We need to fix this.
    class indexing_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;

        using pointer = const size_t*;
        using reference = const size_t&;
        using setiter = typename std::set<size_t>::const_iterator;
        using self_type = indexing_iterator;

    protected:
        setiter m_it;
        setiter m_itend;
        bool m_complement;
        size_t m_maxr;
        size_t m_curr_value;

    protected:
        void get_initial_value()
        { 
            if(!m_complement){m_curr_value = *m_it;}
            else
            {
                setiter it = m_it; 
                for(size_t i = 0; i < m_maxr; ++i)
                {
                    if(it != m_itend)
                    {
                        if(i != *it)
                        {   
                            m_curr_value = i;    
                            return;
                        }
                        ++it;
                    }
                    else
                    {
                        m_curr_value = i;
                        return;
                    }
                }
            }
        }

    public:
        indexing_iterator() : m_complement(false), m_maxr(0), m_curr_value(0){}
        indexing_iterator(const indexing_iterator& o) = default;
        indexing_iterator(setiter it, setiter itend, bool complement, size_t maxr)  : 
          m_it(it), m_itend(itend), m_complement(complement), m_maxr(maxr)
        {
            get_initial_value();
        }
        indexing_iterator(setiter it, setiter itend, bool complement, size_t maxr, size_t fval)  : 
          m_it(it), m_itend(itend), m_complement(complement), m_maxr(maxr), m_curr_value(fval){}

        self_type& operator=(const self_type& other) = default;

        bool operator==(const self_type& other) const
        {
            bool first = (m_it == other.m_it && m_complement == other.m_complement);
            if(!first){return false;}
            else
            {
                if(m_complement)
                {
                    return m_curr_value == other.m_curr_value;
                }
                else
                {
                    return m_it == other.m_it;
                }
            }
            return true;
        }
        bool operator!=(const self_type& other) const{return !(this->operator==(other));}

        reference operator*() const
        {
            if(m_complement){return m_curr_value;}
            else
            {
                return *m_it;
            }
        }
        pointer operator->() const
        {
            if(m_complement)
            {
                return &m_curr_value;
            }
            else
            {
                return m_it.operator->();
            }
        }

        self_type& operator++()
        {
            //if we aren't working with the complement then we just iterate the set index.
            if(!m_complement){++m_it;}
            else
            {
                ++m_curr_value;
                if(m_it != m_itend)
                {
                    while(m_curr_value > *m_it)
                    {
                        ++m_it;
                        if(m_it == m_itend)
                        {
                            return *this;
                        }
                    }

                    while(m_curr_value == *m_it)
                    {
                        ++m_curr_value;
                        ++m_it;
                        //if we reach the end from this increment then curr_value is a new point and we continue incrementing.
                        if(m_it == m_itend)
                        {
                            return *this;
                        }
                    }
                }
            }
            return *this;
        }
    };

public:
    using const_iterator = indexing_iterator;
public:
    term_indexing_array() : m_store_complement(false), m_maxr(0){}
    term_indexing_array(size_type r) : m_store_complement(false), m_maxr(r) {}
    term_indexing_array(const std::vector<size_type>& r, size_type maxr) try : m_store_complement(false), m_maxr(maxr){set(r);}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct term_indexing_array object.");
    }
    term_indexing_array(const term_indexing_array& o) = default;
    term_indexing_array(term_indexing_array&& o) = default;

    term_indexing_array& operator=(const term_indexing_array& o) = default;
    term_indexing_array& operator=(term_indexing_array&& o) = default;
    term_indexing_array& operator=(const std::vector<size_type>& o)
    {
        set(o);
        return *this;
    }

    bool operator==(const term_indexing_array& o) const
    {
        if(o.m_store_complement == m_store_complement)
        {
            return o.m_r == m_r && o.m_maxr == m_maxr;
        }
        else
        {
            if(o.size() != this->size() || m_maxr != o.m_maxr){return false;}
            for(auto z : common::zip(o, *this))
            {
                if(std::get<0>(z) != std::get<1>(z)){return false;}
            }
            return true;
        }
    }
    
    bool operator!=(const term_indexing_array& o) const
    {
        return !this->operator==(o);
    }

    void reserve(size_type r) {m_maxr = r;}
    size_type capacity()const {return m_maxr;}

    void insert(size_type i)
    {
        ASSERT(i < m_maxr, "Cannot insert element index out of bounds.");
        //if we are storing the complement then we just attempt to remove the element
        if(m_store_complement){m_r.erase(i);}
        //otherwise we attempt to insert the element and if it gets too large we need to invert and reset 
        else
        {
            if(m_r.count(i) == 0)
            {
                m_r.insert(i);
                //check if the new element is in the array
                //if inserting the new element will make the array larger than half then we invert the storage and attempt to insert
            }
        }
        test_inversion();
        ASSERT(this->nelems() <= m_maxr, "Array index out of bound.");
    }

    template <typename Iter>
    void insert(Iter start, Iter end)
    {
        if(m_store_complement)
        {
            for(Iter iter = start; iter != end; ++iter)
            {
                ASSERT(*iter < m_maxr, "Cannot insert element index out of bounds.");
                m_r.erase(*iter);
            }
        }
        else
        {
            for(Iter iter = start; iter != end; ++iter)
            {
                ASSERT(*iter < m_maxr, "Cannot insert element index out of bounds.");
                m_r.insert(*iter);
            }

        }
        test_inversion();
    }

    void insert(const term_indexing_array& o)
    {
        ASSERT(o.m_maxr == m_maxr, "Cannot insert term indexing array object unless they are both the same size.");
        //if both of the arrays are inverted
        if(m_store_complement && o.m_store_complement)
        {
            for(auto iter = o.m_r.begin(); iter != o.m_r.end(); ++iter)
            {
                m_r.erase(*iter);
            }
        }
        else if(!m_store_complement && !o.m_store_complement)
        {
            for(auto iter = o.m_r.begin(); iter != o.m_r.end(); ++iter)
            {
                m_r.insert(*iter);
            }
        }
        else
        {
            if(m_store_complement)
            {
                for(auto iter = o.m_r.begin(); iter != o.m_r.end(); ++iter)
                {
                    m_r.erase(*iter);
                }
            }
            else
            {
                typename std::set<size_type>::const_iterator in_iter = o.m_r.begin();
                for(size_type i = 0; i < o.m_maxr; ++i)
                {
                    if(in_iter != o.m_r.end())
                    {
                        if(*in_iter  != i){ m_r.insert(i);}
                        else{++in_iter;}
                    }
                    else{ m_r.insert(i);}
                }
            }
        }
        test_inversion();
    }
    
    void erase(size_type i)
    {
        ASSERT(i < m_maxr, "Cannot remvoe element index out of bounds.");
        if(m_store_complement)
        {
            m_r.insert(i);
        }
        else
        {
            m_r.erase(i);
        }
        test_inversion();
        ASSERT(this->nelems() <= m_maxr, "Array index out of bound.");
    }

    void set(const std::vector<size_type>& l)
    {
        ASSERT(l.size() <= m_maxr, "Failed to set index out of bounds.");

        m_r.clear();
        m_store_complement = false;    
        m_r = std::set<size_t>(l.begin(), l.end());
        test_inversion();
    }

    void set(const std::vector<size_type>& l, size_type maxr)
    {
        m_maxr = maxr;
        set(l);
    }

    std::vector<size_type> get() const
    {
        if(m_store_complement)
        {
            std::vector<size_type> ret; ret.resize(m_maxr - m_r.size());
            invert(m_r, ret, m_maxr);
            return ret;
        }
        else
        {
            std::vector<size_type> ret; ret.reserve(m_r.size());
            std::copy(m_r.begin(), m_r.end(), std::back_inserter(ret));
            return ret;
        }
    }

    std::vector<size_type>& get(std::vector<size_type>& ret) const
    {
        ret.clear();
        if(m_store_complement)
        {
            ret.reserve(m_maxr - m_r.size());
            invert(m_r, ret, m_maxr);
            return ret;
        }
        else
        {
            ret.reserve(m_r.size());
            std::copy(m_r.begin(), m_r.end(), std::back_inserter(ret));
            return ret;
        }
    }


    size_type nelems() const
    {
        return this->size();
    }

    size_type size() const
    {
        if(m_store_complement){return m_maxr - m_r.size();}
        else{return m_r.size();}
    }

public:
    bool contains(size_type v) const
    {
        if(v > m_maxr){return false;}
        if(!m_store_complement){return m_r.count(v) == 1;}
        else{return m_r.count(v) == 0;}
    }

    void clear()
    {
        m_store_complement = false;
        m_r.clear();
    }

    //THE ITERATOR CLASS CURRENTLY DOESN"T WORK CORRECTLY
    const_iterator begin() const
    {
        return const_iterator(m_r.begin(), m_r.end(), m_store_complement, m_maxr);
    }
    const_iterator end() const
    {
        return const_iterator(m_r.end(), m_r.end(), m_store_complement, m_maxr, m_maxr);
    }
protected:
    void test_inversion()
    {
        if(m_r.size() >= m_maxr/2)
        {
            //if we are exactly at the midpoint then we don't attempt to flip this unless we are not storing the complement
            if(m_r.size() == m_maxr/2 && !m_store_complement){return;}

            std::set<size_type> temp = m_r; 
            m_r.clear();
            invert(temp, m_r, m_maxr);
            m_store_complement = !m_store_complement;
        }
    }

protected:
    std::set<size_type> m_r;                                   
    bool m_store_complement = false;
    size_type m_maxr;

public:
    const std::set<size_type>& r() const{return m_r;}
    bool store_complement() const{return m_store_complement;}
    void complement(){m_store_complement = !m_store_complement;}

protected:
    template <typename T1, typename T2> 
    static typename std::enable_if<!std::is_same<T2, std::set<size_type>>::value, void>::type invert(const T1& in, T2& out, size_type maxr)
    {
        typename T1::const_iterator in_iter = in.begin();
        for(size_type i = 0; i < maxr; ++i)
        {
            if(in_iter != in.end())
            {
                if(*in_iter  != i){out.push_back(i);}
                else{++in_iter;}
            }
            else{out.push_back(i);}
        }
    }

    template <typename T1> 
    static void invert(const T1& in, std::set<size_type>& out, size_type maxr)
    {
        typename T1::const_iterator in_iter = in.begin();
        for(size_type i = 0; i < maxr; ++i)
        {
            if(in_iter != in.end())
            {
                if(*in_iter != i){out.insert(i);}
                else{++in_iter;}
            }
            else{out.insert(i);}
        }
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("r", m_r)), "Failed to serialise term_indexing_array object.  Error when serialising the r array.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("store_complement", m_store_complement)), "Failed to serialise term_indexing_array object.  Error when serialising the store_complement variable.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxr", m_maxr)), "Failed to serialise term_indexing_array object.  Error when serialising the maxr.");
    }
#endif

public:
    static void set_intersection(const term_indexing_array& a, const term_indexing_array& b, term_indexing_array& c)
    {
        c.clear();
        ASSERT(a.m_maxr == b.m_maxr, "Cannot compute intersection of two term_indexing_arrays that act on different spaces.");
        c.m_maxr = a.m_maxr;
        if(!a.m_store_complement && !b.m_store_complement)
        {
            std::set_intersection(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            c.m_store_complement = false;
        }
        else if(a.m_store_complement && b.m_store_complement)
        {
            //if each array stores the complement then we should compute the union of the two sets
            std::set_union(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));

            //if the resultant union is too large, then we should invert the array and store the result into c, which will no longer be an complement array
            c.m_store_complement = true;
        }
        //now if one is storing the complement and the other isn't, then we will invert the one storing the complement and compute the intersection
        else
        {
            if(a.m_store_complement)
            {
                std::set_difference(b.m_r.begin(), b.m_r.end(), a.m_r.begin(), a.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            }
            else
            {
                std::set_difference(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            }
            c.m_store_complement = false;
        }
        c.test_inversion();
    }

    static void set_union(const term_indexing_array& a, const term_indexing_array& b, term_indexing_array& c)
    {
        c.clear();
        ASSERT(a.m_maxr == b.m_maxr, "Cannot compute intersection of two term_indexing_arrays that act on different spaces.");
        c.m_maxr = a.m_maxr;
        if(!a.m_store_complement && !b.m_store_complement)
        {
            std::set_union(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            c.m_store_complement = false;
        }
        else if(a.m_store_complement && b.m_store_complement)
        {
            //if each array stores the complement then we should compute the union of the two sets
            std::set_intersection(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));

            //if the resultant union is too large, then we should invert the array and store the result into c, which will no longer be an complement array
            c.m_store_complement = true;
        }
        //now if one is storing the complement and the other isn't, then we will invert the one storing the complement and compute the intersection
        else
        {
            if(a.m_store_complement)
            {
                std::set_difference(a.m_r.begin(), a.m_r.end(), b.m_r.begin(), b.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            }
            else
            {
                std::set_difference(b.m_r.begin(), b.m_r.end(), a.m_r.begin(), a.m_r.end(), std::inserter(c.m_r, c.m_r.begin()));
            }
            c.m_store_complement = true;
        }
        c.test_inversion();
    }

    static void complement(const term_indexing_array& a, term_indexing_array& o)
    {
        o.m_r = a.m_r;
        o.m_maxr = a.m_maxr;
        o.m_store_complement = !a.m_store_complement;
        o.test_inversion();
    }
};


template <typename size_type>
std::ostream& operator<<(std::ostream& os, const term_indexing_array<size_type>& op)
{
    //for(const size_type& i : op){os << i << " ";}
    if(!op.m_store_complement)
    {
        for(auto& i : op.m_r){os << i <<  " ";}
    }
    else
    {
        typename std::set<size_type>::const_iterator in_iter = op.m_r.begin();
        for(size_type i = 0; i < op.m_maxr; ++i)
        {
            if(in_iter != op.m_r.end())
            {
                if(*in_iter  != i){ os << i << " ";}
                else{++in_iter;}
            }
            else{ os << i << " ";}
        }
    }
    return os;
}
}   //namespace utils

#endif  //TTNS_UTILS_MINIMAL_INDEX_ARRAY_HPP

