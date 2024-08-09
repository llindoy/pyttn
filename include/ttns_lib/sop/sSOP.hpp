#ifndef TTNS_OPERATOR_GEN_LIB_SSOP_HPP
#define TTNS_OPERATOR_GEN_LIB_SSOP_HPP

#include <iostream>
#include <list>
#include <regex>

#include <utils/io/input_wrapper.hpp>
#include <common/zip.hpp>
#include "coeff_type.hpp"

namespace ttns
{


class sOP
{
public:
    sOP() : m_fermionic(false){}
    sOP(const std::string& op, size_t mode) : m_op_data(op), m_mode(mode), m_fermionic(false){} 
    sOP(const std::string& op, size_t mode, bool fermionic) : m_op_data(op), m_mode(mode), m_fermionic(fermionic){} 

    sOP(const sOP& o) = default;
    sOP(sOP&& o) = default;

    sOP& operator=(const sOP& o) = default;
    sOP& operator=(sOP&& o) = default;

    void clear(){m_mode = 0; m_op_data.clear(); m_fermionic = false;}
    
    const std::string& op() const{return m_op_data;}
    std::string& op(){return m_op_data;}

    const size_t& mode() const{return m_mode;}
    size_t& mode(){return m_mode;}

    const bool& fermionic() const{return m_fermionic;}
    bool& fermionic(){return m_fermionic;}

    friend std::ostream& operator<<(std::ostream& os, const sOP& op);
    friend bool operator==(const sOP& A, const sOP& B);
    friend bool operator!=(const sOP& A, const sOP& B);

    operator std::string() const
    {
        if(m_fermionic)
        {
            return std::string("fermi_") + m_op_data + std::string("_") + std::to_string(m_mode);
        }
        else
        {
            return  m_op_data + std::string("_") + std::to_string(m_mode);
        }
    }
protected:
    std::string m_op_data;
    size_t m_mode;
    bool m_fermionic;
};

inline std::ostream& operator<<(std::ostream& os, const ttns::sOP& op)
{
    if(op.m_fermionic)
    {
        os << "fermi_" << op.m_op_data << "_" << op.m_mode;
    }
    else
    {
        os << op.m_op_data << "_" << op.m_mode; 
    }
    return os;
}

inline bool operator==(const sOP& A, const sOP& B)
{
    return (A.m_op_data == B.m_op_data && A.m_mode == B.m_mode && A.m_fermionic == B.m_fermionic);
}

inline bool operator!=(const sOP& A, const sOP& B){return !(A == B);}

static inline sOP fermion_operator(const std::string& op, size_t mode)
{
    std::string label(op);
    utils::io::remove_whitespace_and_to_lower(label);

    return sOP(op, mode, true);
}

inline bool operator<(const sOP& A, const sOP& B)
{   
    if(A.mode() == B.mode()){return A.op() < B.op();}
    else{return A.mode() < B.mode();}
}

}


namespace ttns
{

//product of string operators
class sPOP
{
public:
    using iterator = typename std::list<sOP>::iterator;
    using const_iterator = typename std::list<sOP>::const_iterator;
    using reverse_iterator = typename std::list<sOP>::reverse_iterator;
    using const_reverse_iterator = typename std::list<sOP>::const_reverse_iterator;

public:
    sPOP(){}

    sPOP(const sOP& ops) {m_ops.push_back(ops);}
    sPOP(sOP&& ops)  {m_ops.push_back(std::forward<sOP>(ops));}

    template <typename ... Args>
    sPOP(const sOP& ops, Args&& ... args) 
    {
        m_ops.push_back(ops);
        unpack_args(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    sPOP(sOP&& ops, Args&& ... args) 
    {
        m_ops.push_back(std::forward<sOP>(ops));
        unpack_args(std::forward<Args>(args)...);
    }

    sPOP( const std::list<sOP>& ops) : m_ops(ops){}
    sPOP( std::list<sOP>&& ops) : m_ops(std::forward<std::list<sOP>>(ops)){}

    sPOP(const sPOP& o) = default;
    sPOP(sPOP&& o) = default;

    sPOP& operator=(const sPOP& o) = default;
    sPOP& operator=(sPOP&& o) = default;

    const std::list<sOP>& ops() const{return m_ops;}
    std::list<sOP>& ops(){return m_ops;}
    void clear(){m_ops.clear();}
    void append(const sOP& o){m_ops.push_back(o);}
    void prepend(const sOP& o){m_ops.push_front(o);}

    sPOP& operator*=(const sOP& b)
    {
        m_ops.push_back(b);
        return *this;
    }
    sPOP& operator*=(const sPOP& b)
    {
        for(const auto& op : b.ops())
        {
            m_ops.push_back(op);
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const sPOP& op);

    friend bool operator==(const sPOP& A, const sPOP& B);
    friend bool operator!=(const sPOP& A, const sPOP& B);

    size_t size() const
    {
        return m_ops.size();
    }
    size_t nmodes() const
    {
        size_t mode = 0;
        for(const auto& sop : m_ops)
        {
            if(sop.mode()+1 > mode){mode = sop.mode()+1;}
        }
        return mode;
    }

    operator std::string() const
    {
        const auto separator = " ";    const auto* sep = "";
        std::string ret;
        for(const auto& t : m_ops)
        {
            ret += sep+std::string(t);
            sep = separator;
        }
        return ret;
    }
protected: 
    template <typename ... Args>
    void unpack_args(const sOP& ops, Args&& ... args)
    {
        m_ops.push_back(ops);
        unpack_args(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    void unpack_args(sOP&& ops, Args&& ... args)
    {
        m_ops.push_back(std::forward<sOP>(ops));
        unpack_args(std::forward<Args>(args)...);
    }

    void unpack_args(){}


public:
    iterator begin() {  return iterator(m_ops.begin());  }
    iterator end() {  return iterator(m_ops.end());  }
    const_iterator begin() const {  return const_iterator(m_ops.begin());  }
    const_iterator end() const {  return const_iterator(m_ops.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_ops.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_ops.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_ops.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_ops.rend());  }
protected:
    std::list<sOP> m_ops;
};




inline std::ostream& operator<<(std::ostream& os, const ttns::sPOP& op)
{
    const auto separator = " ";    const auto* sep = "";
    for(const auto& t : op.m_ops)
    {
        os << sep << t;
        sep = separator;
    }
    return os;
}

inline bool operator==(const sPOP& A, const sPOP& B)
{
    if(A.m_ops.size() == B.m_ops.size())
    {
        for(auto z : common::zip(A.m_ops, B.m_ops))
        {
            if(std::get<0>(z) != std::get<1>(z)){return false;}
        }
        return true;
    }
    return false;
}

inline bool operator!=(const sPOP& A, const sPOP& B){return !(A == B);}
//compare two sop objects.  This assumes that the objects are sorted
inline bool operator<(const sPOP& a, const sPOP& b)
{
    //iterate through each of the terms in a,b and if they aren't equal return which is larger
    for(auto z : common::zip(a, b))
    {
        if(std::get<0>(z) != std::get<1>(z))
        {
            return std::get<0>(z) < std::get<1>(z);
        }
    }

    //if they have all been equal then we return whether or not the size of the first one is smaller than the second one.  If it is then the above loop terminated because a is too small.
    return a.ops().size() < b.ops().size();
}

}

template <>
struct std::hash<ttns::sPOP>
{
    std::size_t operator()(const ttns::sPOP& k) const
    {
        return std::hash<std::string>()(std::string(k));
    }
};
  
inline ttns::sPOP operator*(const ttns::sOP& a, const ttns::sOP& b)
{
    ttns::sPOP ret{{a, b}};
    return ret;
}

inline ttns::sPOP operator*(const ttns::sPOP& a, const ttns::sOP& b)
{
    ttns::sPOP ret(a);
    ret.append(b);
    return ret;
}

inline ttns::sPOP operator*(const ttns::sOP& a, const ttns::sPOP& b)
{
    ttns::sPOP ret(b);
    ret.prepend(a);
    return ret;
}


inline ttns::sPOP operator*(const ttns::sPOP& a, const ttns::sPOP& b)
{
    ttns::sPOP ret(a);
    for(const auto& op : b.ops())
    {
        ret.append(op);
    }
    return ret;
}


#include "coeff_type.hpp"

namespace ttns
{

template <typename T> class sNBO;
template <typename T> std::ostream& operator<<(std::ostream& os, const sNBO<T>& op);

//TODO: Allow for mapping of sNBO operator to a vector of site_operators.
template <typename T> 
class sNBO
{
public:
    using iterator = typename sPOP::iterator;
    using const_iterator = typename sPOP::const_iterator;
    using reverse_iterator = typename sPOP::reverse_iterator;
    using const_reverse_iterator = typename sPOP::const_reverse_iterator;
    using function_type = typename literal::coeff<T>::function_type;
public:
    sNBO(){}

    sNBO(const sPOP& p) : m_coeff(T(1.0)), m_ops(p) {}
    sNBO(sPOP&& p) : m_coeff(T(1.0)), m_ops(std::forward<sPOP>(p)) {}

    template <typename ... Args>
    sNBO(const sOP& p, Args&&... args) : m_coeff(T(1.0)), m_ops(p, std::forward<Args>(args)...) {}
    template <typename ... Args>
    sNBO(sOP&& p, Args&&... args) : m_coeff(T(1.0)), m_ops(std::forward<sOP>(p), std::forward<Args>(args)...) {}

    sNBO(const T& coeff, const sPOP& p) : m_coeff(coeff), m_ops(p) {}
    sNBO(const T& coeff, sPOP&& p) : m_coeff(coeff), m_ops(std::forward<sPOP>(p)) {}

    sNBO(const literal::coeff<T>& coeff, const sPOP& p) : m_coeff(coeff), m_ops(p) {}
    sNBO(const literal::coeff<T>& coeff, sPOP&& p) : m_coeff(coeff), m_ops(std::forward<sPOP>(p)) {}

    sNBO(const function_type& coeff, const sPOP& p) : m_coeff(coeff), m_ops(p) {}
    sNBO(const function_type& coeff, sPOP&& p) : m_coeff(coeff), m_ops(std::forward<sPOP>(p)) {}

    sNBO(function_type&& coeff, const sPOP& p) : m_coeff(std::move(coeff)), m_ops(p) {}
    sNBO(function_type&& coeff, sPOP&& p) : m_coeff(std::move(coeff)), m_ops(std::forward<sPOP>(p)) {}

    template <typename ... Args>
    sNBO(const T& coeff, const sOP& p, Args&&... args) : m_coeff(coeff), m_ops(p, std::forward<Args>(args)...) {}
    template <typename ... Args>
    sNBO(const T& coeff, sOP&& p, Args&&... args) : m_coeff(coeff), m_ops(std::forward<sOP>(p), std::forward<Args>(args)...) {}

    sNBO(const sNBO& o) = default;
    sNBO(sNBO&& o) = default;

    template <typename U>
    sNBO(const sNBO<U>& o) : m_coeff(o.coeff()), m_ops(o.ops()){}

    sNBO& operator=(const sNBO& o) = default;
    sNBO& operator=(sNBO&& o) = default;

    template <typename U>
    sNBO& operator=(const sNBO<U>& o)
    {
        m_coeff = o.coeff();
        m_ops = o.pop();
        return *this;
    }

    const literal::coeff<T>& coeff() const{return m_coeff;}
    literal::coeff<T>& coeff(){return m_coeff;}

    const std::list<sOP>& ops() const{return m_ops.ops();}
    std::list<sOP>& ops(){return m_ops.ops();}

    const sPOP& pop() const{return m_ops;}
    sPOP& pop(){return m_ops;}

    void clear(){m_coeff = T(1);    m_ops.clear();}
    void append(const sOP& o){m_ops.append(o);}
    void prepend(const sOP& o){m_ops.prepend(o);}

    sNBO<T>& operator*=(const sOP& b)
    {
        m_ops.append(b);
        return *this;
    }

    sNBO<T>& operator*=(const sPOP& b)
    {
        for(const auto& op : b.ops())
        {
            m_ops.append(op);
        }
        return *this;
    }

    template <typename U>
    sNBO<T>& operator*=(const sNBO<U>& b)
    {
        m_coeff *= b.coeff();
        for(const auto& op : b.ops())
        {
            m_ops.append(op);
        }
        return *this;
    }

    sNBO<T>& operator*=(const T& b)
    {
        m_coeff*=b;
        return *this;
    }

    friend std::ostream& operator<< <T>(std::ostream& os, const sNBO<T>& op);

    operator std::string() const
    {
        std::ostringstream oss;
        oss << m_coeff << static_cast<std::string>(m_ops);
        return oss.str();
    }

    size_t nmodes() const
    {
        return m_ops.nmodes();
    }
public:
    iterator begin() {  return iterator(m_ops.begin());  }
    iterator end() {  return iterator(m_ops.end());  }
    const_iterator begin() const {  return const_iterator(m_ops.begin());  }
    const_iterator end() const {  return const_iterator(m_ops.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_ops.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_ops.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_ops.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_ops.rend());  }

protected:
    literal::coeff<T> m_coeff;
    sPOP m_ops;
};

template <typename T> 
std::ostream& operator<<(std::ostream& os, const ttns::sNBO<T>& op)
{
    os << op.coeff() << " " << op.pop();
    return os;
}

}   //namespace ttns

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const T& a, const ttns::sOP& b)
{
    ttns::sNBO<T> ret(a, {b});
    return ret;
}


template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::sOP& a, const T& b)
{
    ttns::sNBO<T> ret(b, {a});
    return ret;
}


template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const T& a, const ttns::sPOP& b)
{
    ttns::sNBO<T> ret(a, b);
    return ret;
}


template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::sPOP& a, const T& b)
{
    ttns::sNBO<T> ret(b, a);
    return ret;
}

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::literal::coeff<T>& a, const ttns::sOP& b)
{
    ttns::sNBO<T> ret(a, {b});
    return ret;
}


template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::sOP& a, const ttns::literal::coeff<T>& b)
{
    ttns::sNBO<T> ret(b, {a});
    return ret;
}


template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::literal::coeff<T>& a, const ttns::sPOP& b)
{
    ttns::sNBO<T> ret(a, b);
    return ret;
}

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sNBO<T> operator*(const ttns::sPOP& a, const ttns::literal::coeff<T>& b)
{
    ttns::sNBO<T> ret(b, a);
    return ret;
}




template <typename T, typename U>
ttns::sNBO<decltype(T()*U())> operator*(const ttns::sNBO<T>& a, const U& b)
{
    ttns::sNBO<decltype(T()*U())> ret(a);
    ret.coeff()*=b;
    return ret;
}

template <typename T, typename U>
ttns::sNBO<decltype(T()*U())> operator*(const U& a, const ttns::sNBO<T>& b)
{
    ttns::sNBO<decltype(T()*U())> ret(b);
    ret.coeff()*=a;
    return ret;
}





template <typename T>
ttns::sNBO<T> operator*(const ttns::sNBO<T>& a, const ttns::sOP& b)
{
    ttns::sNBO<T> ret(a);
    ret.pop().append(b);
    return ret;
}


template <typename T>
ttns::sNBO<T> operator*(const ttns::sOP& a, const ttns::sNBO<T>& b)
{
    ttns::sNBO<T> ret(b);
    ret.pop().prepend(a);
    return ret;
}


template <typename T>
ttns::sNBO<T> operator*(const ttns::sNBO<T>& a, const ttns::sPOP& b)
{
    ttns::sNBO<T> ret(a);
    for(const auto& op : b.ops())
    {
        ret.pop().append(op);
    }
    return ret;
}

template <typename T>
ttns::sNBO<T> operator*(const ttns::sPOP& a, const ttns::sNBO<T>& b)
{
    ttns::sNBO<T> ret(a);
    ret.coeff() = b.coeff();
    for(const auto& op : b.ops())
    {
        ret.pop().append(op);
    }
    return ret;
}

template <typename T, typename U>
ttns::sNBO<decltype(T()*U())> operator*(const ttns::sNBO<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sNBO<decltype(T()*U())> ret(a);
    ret.coeff() = a.coeff()*b.coeff();
    for(const auto& op : b.ops())
    {
        ret.pop().append(op);
    }
    return ret;
}

namespace ttns
{

template <typename T> class sSOP;
template <typename T> std::ostream& operator<<(std::ostream& os, const sSOP<T>& op); 

//the string sum of product operator class used for storing the representation of the Hamiltonian of interest.
template <typename T> 
class sSOP
{
public:
    using container_type = std::vector<sNBO<T>>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

public:
    sSOP(){}
    sSOP(size_t nt){m_terms.reserve(nt);}
    sSOP(const std::string& label) : m_label(label){}
    sSOP(std::string&& label) : m_label(std::forward<std::string>(label)){}
    sSOP(const sOP& str){m_terms.push_back(sNBO<T>(T(1.0), {str}) );}
    sSOP(const sPOP& str){m_terms.push_back(sNBO<T>(T(1.0), str) );}
    sSOP(const sNBO<T>& str){m_terms.push_back(str);}
    template <typename U>
    sSOP(const sNBO<U>& str){m_terms.push_back(str);}
    sSOP(const container_type& str) : m_terms(str){}
    sSOP(sNBO<T>&& str){m_terms.push_back(std::forward(str));}
    sSOP(container_type&& str) : m_terms(std::forward(str)){}

    sSOP(const sSOP& o) = default;

    template <typename U>
    sSOP(const sSOP<U>& o)
    {
        m_terms.reserve(o.nterms());
        for(const auto& op : o)
        {
            m_terms.push_back(op);
        }
    }
    sSOP(sSOP&& o) = default;

    void clear(){m_terms.clear();}
    void reserve(size_t nterms){m_terms.reserve(nterms);}

    sSOP& operator=(const sSOP& o) = default;

    template <typename U>
    sSOP& operator=(const sSOP<U>& o)
    {
        clear();
        m_terms.reserve(o.nterms());
        for(const auto& op : o)
        {
            m_terms.push_back(op);
        }
        return *this;
    }

    sSOP& operator=(sSOP&& o) = default;

    template <typename U>
    sSOP<T>& operator+=(const sSOP<U>& a)
    {
        for(auto& t : a.terms())
        {
            m_terms.push_back(t);
        }
        return *this;
    }

    sSOP<T>& operator+=(const sPOP& a)
    {
        m_terms.push_back({T(1), a});
        return *this;
    }

    sSOP<T>& operator+=(const sOP& a)
    {
        m_terms.push_back({T(1), {a}});
        return *this;
    }

    template <typename U>
    sSOP<T>& operator+=(const sNBO<U>& a)
    {
        m_terms.push_back(a);
        return *this;
    }


    template <typename U>
    sSOP<T>& operator-=(const sSOP<U>& a)
    {
        for(auto& t : a.terms())
        {
            m_terms.push_back({T(-1.0)*t.coeff(), t.pop()});
        }
        return *this;
    }

    sSOP<T>& operator-=(const sPOP& a)
    {
        m_terms.push_back({T(-1.0), a});
        return *this;
    }

    sSOP<T>& operator-=(const sOP& a)
    {
        m_terms.push_back({T(-1.0), {a}});
        return *this;
    }

    template <typename U>
    sSOP<T>& operator-=(const sNBO<U>& a)
    {
        m_terms.push_back({T(-1.0)*a.coeff(), a.pop()});
        return *this;
    }

    template <typename U>
    sSOP<T>& operator*=(const U& a)
    {
        for(auto& op : m_terms)
        {
            op *= a;
        }
        return *this;
    }


    template <typename U>
    sSOP<T>& operator*=(const sSOP<U>& a)
    {
        container_type terms;   terms.reserve(m_terms.size() * a.terms().size());

        for(const auto& _b : m_terms)
        {
            for(const auto& _a : a.terms())
            {
                terms.push_back(_a*_b);
            }
        }
        m_terms = terms;

        return *this;
    }

    template <typename ... Args>
    sSOP<T>& emplace_back(Args&& ... args)
    {
        m_terms.emplace_back(std::forward<Args>(args)...);
        return *this;
    }


    friend std::ostream& operator<< <T>(std::ostream& os, const sSOP<T>& op);

    size_t nterms() const{return m_terms.size();}
    size_t nmodes() const
    {
        size_t mode = 0;
        for(const auto& sop : m_terms)
        {
            if(sop.nmodes() > mode){mode = sop.nmodes();}
        }
        return mode;
    }

    const std::string& label() const{return m_label;}
    std::string& label(){return m_label;}

    container_type& terms() {return m_terms;}
    const container_type& terms() const {return m_terms;}


    sNBO<T>& operator[](size_t r){return m_terms[r];}
    const sNBO<T>& operator[](size_t r) const{return m_terms[r];}
public:
    iterator begin() {  return iterator(m_terms.begin());  }
    iterator end() {  return iterator(m_terms.end());  }
    const_iterator begin() const {  return const_iterator(m_terms.begin());  }
    const_iterator end() const {  return const_iterator(m_terms.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_terms.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_terms.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_terms.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_terms.rend());  }
protected:
    container_type m_terms;
    std::string m_label;
};


template <typename T> 
std::ostream& operator<<(std::ostream& os, const ttns::sSOP<T>& op)
{
    if(!op.label().empty()){os << op.label() << ": " << std::endl;}
    const auto separator = "";    const auto* sep = "";
    const auto plus = "+";
    for(const auto& t : op)
    {
        sep = t.coeff().is_positive() ? plus : separator;
        os << sep << t << std::endl;
    }
    return os;
}
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator+(const ttns::sSOP<T>& a, const ttns::sSOP<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret;
    ret.reserve(a.nterms() + b.nterms());
    for(auto& t : a.terms())
    {
        ret.terms().push_back(t);
    }

    for(auto& t : b.terms())
    {
        ret.terms().push_back(t);
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator+(const ttns::sNBO<T>& a, const ttns::sSOP<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(b);
    ret.terms().push_back(a);
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator+(const ttns::sSOP<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(a);
    ret.terms().push_back(b);
    return  ret;
}


template <typename T>
ttns::sSOP<T> operator+(const ttns::sPOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret(b);
    ret.terms().push_back({T(1), a});
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sSOP<T>& a, const ttns::sPOP& b)
{
    ttns::sSOP<T> ret(a);
    ret.terms().push_back({T(1), b});
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret(b);
    ret.terms().push_back({1.0, a});
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sSOP<T>& a, const ttns::sOP& b)
{
    ttns::sSOP<T> ret(a);
    ret.terms().push_back({1.0, b});
    return  ret;
}


template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator+(const ttns::sNBO<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret;     ret.reserve(2);
    ret += a;
    ret += b;
    return  ret;
}


template <typename T>
ttns::sSOP<T> operator+(const ttns::sPOP& a, const ttns::sNBO<T>& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret += b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sNBO<T>& a, const ttns::sPOP& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret += b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sOP& a, const ttns::sNBO<T>& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret += b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator+(const ttns::sNBO<T>& a, const ttns::sOP& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret += b;
    return  ret;
}

//functions for adding other types
inline ttns::sSOP<double> operator+(const ttns::sOP& a, const ttns::sOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({1.0, b});
    return ret;
}

inline ttns::sSOP<double> operator+(const ttns::sPOP& a, const ttns::sOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({1.0, b});
    return ret;
}

inline ttns::sSOP<double> operator+(const ttns::sOP& a, const ttns::sPOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({1.0, b});
    return ret;
}


inline ttns::sSOP<double> operator+(const ttns::sPOP& a, const ttns::sPOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({1.0, b});
    return ret;
}


//subtraction functions
template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator-(const ttns::sSOP<T>& a, const ttns::sSOP<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret;
    ret.reserve(a.nterms() + b.nterms());
    for(auto& t : a.terms())
    {
        ret.terms().push_back(t);
    }

    for(auto& t : b.terms())
    {
        ret.terms().push_back({T(-1.0)*t.coeff(), t.pop()});
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator-(const ttns::sNBO<T>& a, const ttns::sSOP<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret;
    ret.reserve(b.nterms() + 1);
    ret.terms().push_back(a);
    for(auto& t : b.terms())
    {
        ret.terms().push_back({T(-1.0)*t.coeff(), t.pop()});
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator-(const ttns::sSOP<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(a);
    ret.terms().push_back({T(-1.0)*b.coeff(), b.pop()});
    return  ret;
}


template <typename T>
ttns::sSOP<T> operator-(const ttns::sPOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret;
    ret.reserve(b.nterms() + 1);
    ret.terms().push_back({T(1.0), a});
    for(auto& t : b.terms())
    {
        ret.terms().push_back({T(-1.0)*t.coeff(), t.pop()});
    }
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sSOP<T>& a, const ttns::sPOP& b)
{
    ttns::sSOP<T> ret(a);
    ret.terms().push_back({T(-1.0), b});
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret;  ret.reserve(b.nterms()+1);
    ret.terms().push_back({1.0, a});
    for(auto& t : b.terms())
    {
        ret.terms().push_back({T(-1.0)*t.coeff(), t.pop()});
    }
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sSOP<T>& a, const ttns::sOP& b)
{
    ttns::sSOP<T> ret(a);
    ret.terms().push_back({T(-1.0), b});
    return  ret;
}


template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator-(const ttns::sNBO<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret;     ret.reserve(2);
    ret += a;
    ret -= b;
    return  ret;
}


template <typename T>
ttns::sSOP<T> operator-(const ttns::sPOP& a, const ttns::sNBO<T>& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret -= b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sNBO<T>& a, const ttns::sPOP& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret -= b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sOP& a, const ttns::sNBO<T>& b)
{
    ttns::sSOP<T> ret;     ret.reserve(2);
    ret += a;
    ret -= b;
    return  ret;
}

template <typename T>
ttns::sSOP<T> operator-(const ttns::sNBO<T>& a, const ttns::sOP& b)
{
    ttns::sSOP<T> ret;  ret.reserve(2);
    ret += a;
    ret -= b;
    return  ret;
}

//functions for adding other types
inline ttns::sSOP<double> operator-(const ttns::sOP& a, const ttns::sOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({-1.0, b});
    return ret;
}

inline ttns::sSOP<double> operator-(const ttns::sPOP& a, const ttns::sOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({-1.0, b});
    return ret;
}

inline ttns::sSOP<double> operator-(const ttns::sOP& a, const ttns::sPOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({-1.0, b});
    return ret;
}


inline ttns::sSOP<double> operator-(const ttns::sPOP& a, const ttns::sPOP& b)
{
    ttns::sSOP<double> ret;     ret.reserve(2);
    ret.terms().push_back({1.0, a});
    ret.terms().push_back({-1.0, b});
    return ret;
}



//MULTIPLICATION FUNCTIONS
template <typename T, typename U> 
ttns::sSOP<decltype(T()*U())> operator*(const ttns::sSOP<T>& a, const U& b)
{
    ttns::sSOP<decltype(T()*U())> ret(a);
    for(auto& op : ret.terms())
    {
        op *= b;
    }
    return  ret;
}


template <typename T, typename U> 
ttns::sSOP<decltype(T()*U())> operator*(const U& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(b);
    for(auto& op : ret.terms())
    {
        op *= a;
    }
    return  ret;
}


template <typename T> 
ttns::sSOP<T> operator*(const ttns::sOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret(b.nterms());
    for(const auto& op : b)
    {
        ret.terms().push_back(a*op);
    }
    return  ret;
}

template <typename T> 
ttns::sSOP<T> operator*(const ttns::sSOP<T>& a, const ttns::sOP& b)
{
    ttns::sSOP<T> ret(a);
    for(auto& op : ret.terms())
    {
        op *= b;
    }
    return  ret;
}

template <typename T> 
ttns::sSOP<T> operator*(const ttns::sPOP& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<T> ret(b.nterms());
    for(const auto& op : b)
    {
        ret.terms().push_back(a*op);
    }
    return  ret;
}

template <typename T> 
ttns::sSOP<T> operator*(const ttns::sSOP<T>& a, const ttns::sPOP& b)
{
    ttns::sSOP<T> ret(a);
    for(auto& op : ret.terms())
    {
        op *= b;
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator*(const ttns::sNBO<U>& a, const ttns::sSOP<T>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(b.nterms());
    for(auto& op : b.terms())
    {
        ret.terms().push_back(a*op);
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator*(const ttns::sSOP<T>& a, const ttns::sNBO<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(a);
    for(auto& op : ret.terms())
    {
        op *= b;
    }
    return  ret;
}

template <typename T, typename U>
ttns::sSOP<decltype(T()*U())> operator*(const ttns::sSOP<T>& a, const ttns::sSOP<U>& b)
{
    ttns::sSOP<decltype(T()*U())> ret(a.nterms()*b.nterms());
    for(const auto& _a : a.terms())
    {
        for(const auto& _b : b.terms())
        {
            ret.terms().push_back(_a*_b) ;
        }
    }
    return  ret;
}



#endif  //TTNS_OPERATOR_GEN_LIB_SSOP_HPP

