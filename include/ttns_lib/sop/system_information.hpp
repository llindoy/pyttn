#ifndef TTNS_SYSTEM_INFORMATION_HPP
#define TTNS_SYSTEM_INFORMATION_HPP

#include <iostream>
#include <utils/occupation_number_basis_indexing.hpp>

namespace ttns
{

enum mode_type
{
    FERMION_MODE,
    BOSON_MODE,
    SPIN_MODE,
    QUBIT_MODE,
    GENERIC_MODE
};

class primitive_mode_data
{
public:
    primitive_mode_data() : m_lhd(1), m_type(mode_type::GENERIC_MODE){}
    primitive_mode_data(size_t d) : m_lhd(d), m_type(mode_type::GENERIC_MODE){}
    primitive_mode_data(size_t d, mode_type type) : m_lhd(d), m_type(type){}

    primitive_mode_data(const primitive_mode_data& o) = default;
    primitive_mode_data(primitive_mode_data&& o) = default;

    primitive_mode_data& operator=(const primitive_mode_data& o) = default;
    primitive_mode_data& operator=(primitive_mode_data&& o) = default;

    const mode_type& type() const{return m_type;}
    mode_type& type(){return m_type;}

    bool fermionic() const{return m_type == mode_type::FERMION_MODE;}

    const size_t& lhd() const{return m_lhd;}
    size_t& lhd(){return m_lhd;}

protected:
    size_t m_lhd;           //local hilbert space dimension
    mode_type m_type;
};

class mode_data
{
public:
    mode_data(){}
    mode_data(size_t d)
    {
        m_modes.resize(1);
        m_modes[0] = primitive_mode_data(d);
    }
    mode_data(size_t d, mode_type type)
    {
        m_modes.resize(1);
        m_modes[0] = primitive_mode_data(d, type);
    }

    mode_data(const mode_data& o) = default;
    mode_data(mode_data&& o) = default;

    mode_data(const std::vector<primitive_mode_data>& m) : m_modes(m){}

    mode_data(const primitive_mode_data& o) 
    {
        m_modes.resize(1);
        m_modes[0] = o;
    }
    mode_data(primitive_mode_data&& o) 
    {
        m_modes.resize(1);
        m_modes[0] = std::move(o);
    }

    mode_data& operator=(const mode_data& o) = default;
    mode_data& operator=(mode_data&& o) = default;

    mode_data& operator=(const std::vector<primitive_mode_data>& m) 
    {
        m_modes = m;
        return *this;
    }

    mode_data& operator=(const primitive_mode_data& o)
    {
        m_modes.resize(1);
        m_modes[0] = o;
        return *this;
    }

    mode_data& operator=(primitive_mode_data&& o)
    {
        m_modes.resize(1);
        m_modes[0] = std::move(o);
        return *this;
    }

    primitive_mode_data& operator[](size_t i)
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }

    const primitive_mode_data& operator[](size_t i) const
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }

    void append(const primitive_mode_data& o)
    {
        m_modes.push_back(o);
    }

    void append(primitive_mode_data&& o)
    {
        m_modes.push_back(std::move(o));
    }

    void append(const mode_data& o)
    {
        for(size_t i = 0; i < o.nmodes(); ++i)
        {
            m_modes.push_back(o.m_modes[i]);
        }
    }

    void clear(){m_modes.clear();}

    void resize(size_t n)
    {
        m_modes.resize(n);
    }

    size_t nmodes() const{return m_modes.size();}
    size_t lhd() const
    {
        size_t _lhd = 1;
        for(const auto& m : m_modes)
        {
            _lhd *= m.lhd();
        }
        return _lhd;
    }

    mode_data liouville_space() const
    {
        std::vector<primitive_mode_data> mr;
        mr.reserve(m_modes.size()*2);
        for(const auto& mode : m_modes)
        {
            mr.push_back(mode);
            mr.push_back(mode);
        }
        return mode_data(mr);
    }

    bool contains_fermion() const
    {
        for(const auto& pm : m_modes)
        {
            if(pm.fermionic())
            {
                return true;
            }
        }

        return false;
    }
public:
    using iterator = typename std::vector<primitive_mode_data>::iterator;
    using const_iterator = typename std::vector<primitive_mode_data>::const_iterator;
    using reverse_iterator = typename std::vector<primitive_mode_data>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<primitive_mode_data>::const_reverse_iterator;

    iterator begin() {  return iterator(m_modes.begin());  }
    iterator end() {  return iterator(m_modes.end());  }
    const_iterator begin() const {  return const_iterator(m_modes.begin());  }
    const_iterator end() const {  return const_iterator(m_modes.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_modes.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_modes.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_modes.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_modes.rend());  }
protected:
    std::vector<primitive_mode_data> m_modes;
};



inline primitive_mode_data fermion_mode(){return primitive_mode_data(2, mode_type::FERMION_MODE);}
inline primitive_mode_data boson_mode(size_t N){return primitive_mode_data(N, mode_type::BOSON_MODE);}
inline primitive_mode_data qubit_mode(){return primitive_mode_data(2, mode_type::QUBIT_MODE);}
inline primitive_mode_data spin_mode(size_t N){return primitive_mode_data(N, mode_type::SPIN_MODE);}
inline primitive_mode_data generic_mode(size_t N){return primitive_mode_data(N, mode_type::GENERIC_MODE);}

class system_modes
{
public:
    friend system_modes combine_systems(const system_modes& a, const system_modes& b);
public:
    using iterator = typename std::vector<mode_data>::iterator;
    using const_iterator = typename std::vector<mode_data>::const_iterator;
    using reverse_iterator = typename std::vector<mode_data>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<mode_data>::const_reverse_iterator;

    void set_default_mode_ordering()
    {
        for(size_t i = 0; i < m_tree_leaf_indices.size(); ++i){m_tree_leaf_indices[i] = i;}
    }

public:
    system_modes(){}
    system_modes(size_t N) : m_modes(N), m_tree_leaf_indices(N)
    {
        set_default_mode_ordering();
    }
    system_modes(size_t N, size_t d) : m_modes(N), m_tree_leaf_indices(N)
    {
        set_default_mode_ordering();
        for(auto& mode : m_modes)
        {   
            mode.resize(1);
            mode[0].lhd() = d;
        }
    }

    system_modes(const primitive_mode_data& o) : m_tree_leaf_indices(1)
    {
        m_modes.resize(1);  m_modes[0] = o;
        set_default_mode_ordering();
    }

    system_modes(const mode_data& o) : m_tree_leaf_indices(1)
    {
        m_modes.resize(1);  m_modes[0] = o;
        set_default_mode_ordering();
    }

    system_modes(const std::vector<mode_data>& o) : m_modes(o) , m_tree_leaf_indices(o.size())
    {
        set_default_mode_ordering();
    }

    system_modes(size_t N, size_t d, const std::vector<size_t>& ordering) : m_modes(N), m_tree_leaf_indices(ordering)
    {
        ASSERT(ordering.size() == N, "Failed to construct system modes object ordering size incorrect.");
        for(auto& mode : m_modes)
        {   
            mode.resize(1);
            mode[0].lhd() = d;
        }
    }
    system_modes(const std::vector<mode_data>& o, const std::vector<size_t>& ordering) : m_modes(o) , m_tree_leaf_indices(ordering)
    {
        ASSERT(ordering.size() == o.size(), "Failed to construct system modes object ordering size incorrect.");
    }

    system_modes(const system_modes& o) = default;
    system_modes(system_modes&& o) = default;

    system_modes& operator=(const system_modes& o) = default;
    system_modes& operator=(system_modes&& o) = default;


    system_modes liouville_space() const
    {
        system_modes ret(nmodes());
        ret.m_tree_leaf_indices = m_tree_leaf_indices;
        for(size_t i = 0; i < nmodes(); ++i)
        {
            ret[i] = m_modes[i].liouville_space();
        }
        return ret;
    }

    size_t nmodes() const{return m_modes.size();}
    size_t nprimitive_modes() const
    {
        size_t nm=0;
        for(size_t i = 0; i < m_modes.size(); ++i)
        {
            nm += m_modes[i].nmodes();
        }
        return nm;
    }

    void resize(size_t N)
    {
        if(N >= nmodes())
        {
            m_modes.resize(N);
            m_tree_leaf_indices.resize(N);
            set_default_mode_ordering();
        }
        else
        {
            clear();
            m_modes.resize(N);
            m_tree_leaf_indices.resize(N);
            set_default_mode_ordering();
        }
    }
    
    //functions for adding additional modes to the system information
    void add_mode(const primitive_mode_data& m)
    {
        CALL_AND_RETHROW(add_mode(mode_data(m)));
    }

    void add_mode(const primitive_mode_data& m, size_t tree_index)
    {
        CALL_AND_RETHROW(add_mode(mode_data(m), tree_index));
    }

    void add_mode(const mode_data& m)
    {
        CALL_AND_RETHROW(add_mode(m, m_modes.size()));
    }

    void add_mode(const mode_data& m, size_t i)
    {
        if(std::find(m_tree_leaf_indices.begin(), m_tree_leaf_indices.end(), i) != m_tree_leaf_indices.end())
        {
            RAISE_EXCEPTION("Failed to add mode to system information.  The tree leaf index specified has already been bound.");
        }
        m_modes.push_back(m);
        m_tree_leaf_indices.push_back(i);
    }
  
    //append another system to the end of this one.  This will keep the internal ordering of each system but have it so that
    //the second system variables are appended
    void append_system(const system_modes& o)
    {
        size_t nmodes = m_modes.size();
        m_modes.reserve(nmodes + o.nmodes());
        for(size_t i = 0; i < o.nmodes(); ++i)
        {
            //add a new mode which is the input mode object and points to the tree leaf index object indexed specified in o
            //but incremented by the number of modes in the current tree to skip over all of these indices.
            this->add_mode(o.m_modes[i], o.m_tree_leaf_indices[i]+nmodes);
        }
    }

    //functions for accessing the composite mode information
    mode_data& operator[](size_t i)
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }

    const mode_data& operator[](size_t i) const
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }

    mode_data& mode(size_t i)
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }

    const mode_data& mode(size_t i) const
    {
        ASSERT(i < m_modes.size(), "Index out of bounds.");
        return m_modes[i];
    }


    std::pair<size_t, size_t> primitive_mode_index(size_t i) const
    {
        size_t counter = 0;
        for(size_t ind = 0; ind <= i; ++ind)
        {
            if(counter + m_modes[ind].nmodes() > i)
            {
                return std::make_pair(ind, i-counter);
            }
            else
            {
                counter += m_modes[ind].nmodes();
            }
        }
        RAISE_EXCEPTION("Index out of bounds.");
    }

    //functions for accessing the underlying primitive modes
    primitive_mode_data& primitive_mode(size_t i)
    {
        size_t counter = 0;
        for(size_t ind = 0; ind <= i; ++ind)
        {
            if(counter + m_modes[ind].nmodes() > i)
            {
                return m_modes[ind][i-counter];
            }
            else
            {
                counter += m_modes[ind].nmodes();
            }
        }
        RAISE_EXCEPTION("Index out of bounds.");
    }

    const primitive_mode_data& primitive_mode(size_t i) const
    {
        size_t counter = 0;
        for(size_t ind = 0; ind <= i; ++ind)
        {
            if(counter + m_modes[ind].nmodes() > i)
            {
                return m_modes[ind][i-counter];
            }
            else
            {
                counter += m_modes[ind].nmodes();
            }
        }
        RAISE_EXCEPTION("Index out of bounds.");
    }

    const std::vector<size_t>& mode_indices()const
    {
        return m_tree_leaf_indices;
    }

    void set_mode_indices(const std::vector<size_t>& inds)
    {
        ASSERT(inds.size() == m_tree_leaf_indices.size(), "Failed to set mode indices.");
        m_tree_leaf_indices = inds;
    }

    size_t& mode_index(size_t i){return m_tree_leaf_indices[i];}
    const size_t& mode_index(size_t i) const {return m_tree_leaf_indices[i];}

    void clear() noexcept
    {
        m_modes.clear();
        m_tree_leaf_indices.clear();
    }

      
    //flatten the entire system mode to a single composite mode
    mode_data as_combined_mode() const
    {
        std::vector<primitive_mode_data> md;
        md.reserve(this->nprimitive_modes());

        for(const auto& cm : m_modes)
        {
            for(const auto& pm : cm)
            {
                md.push_back(pm);
            }
        }
        return mode_data(md);
    }

    bool contains_fermion() const
    {
        for(const auto& cm : m_modes)
        {
            if(cm.contains_fermion())
            {
                return true;
            }
        }

        return false;
    }
public:
    iterator begin() {  return iterator(m_modes.begin());  }
    iterator end() {  return iterator(m_modes.end());  }
    const_iterator begin() const {  return const_iterator(m_modes.begin());  }
    const_iterator end() const {  return const_iterator(m_modes.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_modes.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_modes.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_modes.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_modes.rend());  }

protected:
    std::vector<mode_data> m_modes;
    std::vector<size_t> m_tree_leaf_indices;
};


inline system_modes combine_systems(const system_modes& a, const system_modes& b)
{
    system_modes ret(a);
    ret.append_system(b);
    return ret;
}

}

inline std::ostream& operator<<(std::ostream& o, const ttns::mode_type& m)
{
    switch(m)
    {
        case ttns::mode_type::FERMION_MODE:
            o << "fermion";
            break;
        case ttns::mode_type::BOSON_MODE:
            o << "boson";
            break;
        case ttns::mode_type::SPIN_MODE:
            o << "spin";
            break;
        case ttns::mode_type::QUBIT_MODE:
            o << "tls";
            break;
        case ttns::mode_type::GENERIC_MODE:
            o << "generic";
            break;
    }
    return o;
}

inline std::ostream& operator<<(std::ostream& o, const ttns::primitive_mode_data& m)
{
    return o << m.type() << " mode (" << m.lhd() << ") ";
}
inline std::ostream& operator<<(std::ostream& o, const ttns::mode_data& m)
{
    o << "( ";
    for(const auto& i : m)
    {
        o << i;
    }
    o << ") ";
    return o;
}
inline std::ostream& operator<<(std::ostream& o, const ttns::system_modes& m)
{
    for(const auto& i : m)
    {
        o << i;
    }
    return o;
}

#endif

