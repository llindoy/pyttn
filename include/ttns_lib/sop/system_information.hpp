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

class mode_data
{
public:
    mode_data() : m_lhd(1), m_type(mode_type::GENERIC_MODE){}
    mode_data(size_t d) : m_lhd(d), m_type(mode_type::GENERIC_MODE){}
    mode_data(size_t d, mode_type type) : m_lhd(d), m_type(type){}

    mode_data(const mode_data& o) = default;
    mode_data(mode_data&& o) = default;

    mode_data& operator=(const mode_data& o) = default;
    mode_data& operator=(mode_data&& o) = default;

    const mode_type& type() const{return m_type;}
    mode_type& type(){return m_type;}

    bool fermionic() const{return m_type == mode_type::FERMION_MODE;}

    const size_t& lhd() const{return m_lhd;}
    size_t& lhd(){return m_lhd;}

protected:
    size_t m_lhd;           //local hilbert space dimension
    mode_type m_type;
};


inline mode_data fermion_mode(){return mode_data(2, mode_type::FERMION_MODE);}
inline mode_data boson_mode(size_t N){return mode_data(N, mode_type::BOSON_MODE);}
inline mode_data qubit_mode(){return mode_data(2, mode_type::QUBIT_MODE);}
inline mode_data spin_mode(size_t N){return mode_data(N, mode_type::SPIN_MODE);}
inline mode_data generic_mode(size_t N){return mode_data(N, mode_type::GENERIC_MODE);}

class composite_mode
{   
public:
    composite_mode(){}
    composite_mode(const composite_mode& o) = default;
    composite_mode(composite_mode&& o) = default;

    composite_mode& operator=(const composite_mode& o) = default;
    composite_mode& operator=(composite_mode&& o) = default;
protected:
    std::vector<size_t> m_primitive_mode_indices;
    std::shared_ptr<utils::occupation_number_basis> m_composite_basis;
};

class system_modes
{
public:
    using iterator = typename std::vector<mode_data>::iterator;
    using const_iterator = typename std::vector<mode_data>::const_iterator;
    using reverse_iterator = typename std::vector<mode_data>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<mode_data>::const_reverse_iterator;

    void set_default_mode_ordering()
    {
        for(size_t i = 0; i < m_mode_ordering.size(); ++i){m_mode_ordering[i] = i;}
    }

public:
    system_modes(){}
    system_modes(size_t N) : m_primitive_modes(N), m_mode_ordering(N)
    {
        set_default_mode_ordering();
    }
    system_modes(size_t N, size_t d) : m_primitive_modes(N), m_mode_ordering(N)
    {
        set_default_mode_ordering();
        for(auto& mode : m_primitive_modes){mode.lhd() = d;}
    }
    system_modes(const std::vector<mode_data>& o) : m_primitive_modes(o) , m_mode_ordering(o.size())
    {
        set_default_mode_ordering();
    }

    system_modes(size_t N, size_t d, const std::vector<size_t>& ordering) : m_primitive_modes(N), m_mode_ordering(ordering)
    {
        ASSERT(ordering.size() == N, "Failed to construct system modes object ordering size incorrect.");
        for(auto& mode : m_primitive_modes){mode.lhd() = d;}
    }
    system_modes(const std::vector<mode_data>& o, const std::vector<size_t>& ordering) : m_primitive_modes(o) , m_mode_ordering(ordering)
    {
        ASSERT(ordering.size() == o.size(), "Failed to construct system modes object ordering size incorrect.");
    }

    system_modes(const system_modes& o) = default;
    system_modes(system_modes&& o) = default;

    system_modes& operator=(const system_modes& o) = default;
    system_modes& operator=(system_modes&& o) = default;

    size_t nmodes() const{return m_primitive_modes.size();}

    void resize(size_t N)
    {
        if(N >= nmodes())
        {
            m_primitive_modes.resize(N);
            m_mode_ordering.resize(N);
            set_default_mode_ordering();
        }
        else
        {
            clear();
            m_primitive_modes.resize(N);
            m_mode_ordering.resize(N);
            set_default_mode_ordering();
        }
    }

    mode_data& operator[](size_t i)
    {
        ASSERT(i < m_primitive_modes.size(), "Index out of bounds.");
        return m_primitive_modes[i];
    }

    const mode_data& operator[](size_t i) const
    {
        ASSERT(i < m_primitive_modes.size(), "Index out of bounds.");
        return m_primitive_modes[i];
    }

    mode_data& mode(size_t i)
    {
        ASSERT(i < m_primitive_modes.size(), "Index out of bounds.");
        return m_primitive_modes[i];
    }

    const mode_data& mode(size_t i) const
    {
        ASSERT(i < m_primitive_modes.size(), "Index out of bounds.");
        return m_primitive_modes[i];
    }

    const std::vector<size_t>& mode_indices()const
    {
        return m_mode_ordering;
    }

    void set_mode_indices(const std::vector<size_t>& inds)
    {
        ASSERT(inds.size() == m_mode_ordering.size(), "Failed to set mode indices.");
        m_mode_ordering = inds;
    }

    size_t& mode_index(size_t i){return m_mode_ordering[i];}
    const size_t& mode_index(size_t i) const {return m_mode_ordering[i];}

    void clear() noexcept
    {
        m_primitive_modes.clear();
        m_mode_ordering.clear();
        m_composite_modes.clear();
        m_bound_primitive_modes.clear();
    }

public:
    iterator begin() {  return iterator(m_primitive_modes.begin());  }
    iterator end() {  return iterator(m_primitive_modes.end());  }
    const_iterator begin() const {  return const_iterator(m_primitive_modes.begin());  }
    const_iterator end() const {  return const_iterator(m_primitive_modes.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_primitive_modes.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_primitive_modes.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_primitive_modes.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_primitive_modes.rend());  }

protected:
    std::vector<mode_data> m_primitive_modes;
    std::vector<size_t> m_mode_ordering;
    std::vector<composite_mode> m_composite_modes;
    std::list<size_t> m_bound_primitive_modes;
};



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
            o << "qubit";
            break;
        case ttns::mode_type::GENERIC_MODE:
            o << "generic";
            break;
    }
    return o;
}
inline std::ostream& operator<<(std::ostream& o, const ttns::mode_data& m)
{
    return o << m.type() << " mode (" << m.lhd() << ") ";
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

