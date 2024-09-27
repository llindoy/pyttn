#ifndef TTNS_OPERATOR_GEN_LIB_SOP_HPP
#define TTNS_OPERATOR_GEN_LIB_SOP_HPP

#include "coeff_type.hpp"
#include "sSOP.hpp"
#include "system_information.hpp"
#include "operator_dictionaries/default_operator_dictionaries.hpp"

#include <tuple>
#include <vector>
#include <algorithm>

namespace ttns
{

class prodOP
{
public:
    using elem_type = std::tuple<size_t, size_t, bool>; 
    using container_type = std::vector<elem_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

public:
    prodOP() {}
    prodOP(size_t nelem){m_ops.resize(nelem);}
    prodOP(const container_type& o) : m_ops(o){}
    prodOP(const prodOP& o) = default;
    prodOP(prodOP&& o) = default;

    prodOP& operator=(const prodOP& o) = default;
    prodOP& operator=(prodOP&& o) = default;

    void clear(){m_ops.clear();}
    void append(const elem_type& o){m_ops.push_back(o);}
    void resize(size_t nelem){m_ops.resize(nelem);}

    prodOP& operator*=(const elem_type& b)
    {
        m_ops.push_back(b);
        return *this;
    }

    size_t size() const{return m_ops.size();}
    size_t nmodes() const
    {
        size_t mode = 0;
        for(const auto& sop : m_ops)
        {
            if(std::get<1>(sop)+1 > mode){mode = std::get<1>(sop)+1;}
        }
        return mode;
    }


    elem_type& operator[](size_t i){return m_ops[i];}    
    const elem_type& operator[](size_t i) const{return m_ops[i];}    

    container_type& operator()(){return m_ops;}
    const container_type& operator()() const{return m_ops;}

    bool hasHash() const{return m_has_hash;}
    std::size_t hash() const{return m_hash;}
    void get_hash()
    {
        std::hash<std::string> hashobj;
        m_hash = hashobj(this->hash_string());
        m_has_hash = true;
    }

public:
    std::string hash_string() const
    {
        container_type o(m_ops);
        std::stable_sort(o.begin(), o.end(), [](const elem_type& a, const elem_type& b){return std::get<1>(a) < std::get<1>(b);});
        std::string ret;
        for(const auto& t : o)
        {
            ret += std::to_string(std::get<0>(t))
                  +std::string("_")
                  +std::to_string(std::get<1>(t))
                  +(std::get<2>(t) ? std::string("_0") : std::string("_1"));
        }
        return ret;
    }

    operator std::string() const
    {
        std::string ret;
        for(const auto& t : m_ops)
        {
            ret += std::to_string(std::get<0>(t))
                  +std::string("_")
                  +std::to_string(std::get<1>(t))
                  +(std::get<2>(t) ? std::string("_0") : std::string("_1"));
        }
        return ret;
    }

    std::ostream& label(std::ostream& os, const std::vector<std::vector<std::string>>& opdict) const
    {
        if(m_mapped)
        {
            const auto separator = " ";    const auto* sep = "";
            size_t opind = 0;

            for(size_t mode_ind = 0; mode_ind < opdict.size(); ++mode_ind)
            {
                while(opind < m_ops.size() && std::get<1>(m_ops[opind]) == mode_ind)
                {   
                    const auto& t = m_ops[opind];
                    os << sep << opdict[std::get<1>(t)][std::get<0>(t)] << "_" << std::get<1>(t);
                    sep = separator;
                    ++opind;
                }

                if(m_prepend_jw[mode_ind])
                { 
                    os << sep << "jw_" << mode_ind;
                    sep = separator;
                }
            }
        }
        else
        {
            const auto separator = " ";    const auto* sep = "";
            for(const auto& t : m_ops)
            {
                if(std::get<2>(t))
                {
                    os << sep << "fermi_" << opdict[std::get<1>(t)][std::get<0>(t)] << "_" << std::get<1>(t);
                }
                else
                {
                    os << sep << opdict[std::get<1>(t)][std::get<0>(t)] << "_" << std::get<1>(t);
                }
                sep = separator;
            }
        }
        return os;
    }

    sPOP as_prod_op(const std::vector<std::vector<std::string>>& opdict) const
    {
        sPOP ret;
        for(const auto& t : m_ops)
        {
            if(opdict[std::get<1>(t)][std::get<0>(t)] != std::string("id"))
            {
                ret *= sOP(opdict[std::get<1>(t)][std::get<0>(t)],std::get<1>(t) );
            }
        }
        return ret;
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


public:
    //a function to reorder the mode indices in the object in ascending order.  This function assumes commutation of operators acting
    //on different nodes and so will lead to errors if dealing with fermionic systems.
    void order_modes()
    {
        //ensure that we have stored the current hash before we edit the form of the object.
        //here we use stable sort to preserve order of elements acting on a common mode
        std::stable_sort(m_ops.begin(), m_ops.end(), [](const elem_type& a, const elem_type& b){return std::get<1>(a) < std::get<1>(b);});
        if(!m_has_hash){get_hash();}
    }

    //a function for reordering the modes in the object in ascending order.  This function will take into account fermionic
    //exchange statistics
    bool fermion_order_modes(const std::vector<std::vector<std::string>>& opdict)
    {
        if(m_ops.size() == 1){return false;}

        std::vector<bool> changes_parity(m_ops.size());
        using fermi_dict = fermion::default_fermion_operator_dictionary<double>;
        
        //iterate through the operator list and get the number of fermionic operators present in the term
        for(size_t i = 0; i < m_ops.size(); ++i)
        {   
            if(std::get<2>(m_ops[i]))
            {
                std::string label = opdict[std::get<1>(m_ops[i])][std::get<0>(m_ops[i])];
                CALL_AND_HANDLE(changes_parity[i] = fermi_dict::query(label)->changes_parity(), "Failed to order fermionic modes.");
            }
            else
            {
                changes_parity[i] = false;
            }
        }

        //now we go through and sort the fermion operators picking up and sign change due to fermion exchange statistics.
        //Here for simplicity we implement this using a bubble sort
        bool flip_sign = false;
        for(size_t i = 0; i < m_ops.size()-1; ++i)
        {
            bool swapped = false;   
            for(size_t j = 0; j < m_ops.size() - i - 1; ++j)
            {
                //if the current operator has a larger mode index than its next 
                if(std::get<1>(m_ops[j]) > std::get<1>(m_ops[j+1]))
                {
                    //if the two operators change the parity of the fermionic state 
                    if(changes_parity[j] && changes_parity[j+1]){flip_sign = !flip_sign;}

                    std::swap(m_ops[j], m_ops[j+1]);
                    std::swap(changes_parity[j], changes_parity[j+1]);
                    swapped = true;
                }
            }
            if(!swapped){return flip_sign;}
        }

        if(!m_has_hash){get_hash();}
        return flip_sign;
    }

    bool jordan_wigner(const std::vector<std::vector<std::string>>& opdict, const std::vector<bool>& is_fermion_mode) const
    {
        //ensure that we ahve stored the current hash before we edit the form of the object.
        m_prepend_jw.resize(is_fermion_mode.size());    std::fill(m_prepend_jw.begin(), m_prepend_jw.end(), false);
        using fermi_dict = fermion::default_fermion_operator_dictionary<double>;

        bool flip_sign = false;
        using op_iterator = typename container_type::const_iterator;
        for(op_iterator iter = m_ops.begin(); iter != m_ops.end(); ++iter)
        {
            const auto& op = *iter;
            if(std::get<2>(op))
            {
                std::string label = opdict[std::get<1>(op)][std::get<0>(op)];
                bool do_jw;
                CALL_AND_HANDLE(do_jw = fermi_dict::query(label)->changes_parity(), "Failed to jordan_wigner map fermionic mode.");

                if(do_jw)
                {
                    op_iterator jiter = m_ops.begin();
                    //iterate over all the nodes
                    for(size_t i = 0; i < std::get<1>(op); ++i)
                    {
                        //now take the starting operator in the product operator and if it has mode less than the current mode
                        while(std::get<1>(*jiter) < i && jiter != iter)
                        {
                            ++jiter;
                        }

                        //if the current operator is acting on i and it isn't the operator we are jw mapping
                        if(std::get<1>(*jiter) == i && jiter != iter)
                        {
                            //while we are currently acting on the same node and we haven't reached iter we just keep incrementing through the operator objects and figure out whether they flip sign
                            while(std::get<1>(*jiter) == i && jiter != iter)
                            {
                                std::string _label = opdict[std::get<1>(*jiter)][std::get<0>(*jiter)];
                                bool jw_flips;
                                CALL_AND_HANDLE(jw_flips = fermi_dict::query(_label)->jw_sign_change(), "Failed to query fermionic mode information.");
                                if(jw_flips){flip_sign = !flip_sign;}

                                ++jiter;
                            }
                        }
                        //otherwise we are at a point where there is no operator acting on this mode and we just figure out whether or not we need to prepend the jordan-wigner string
                        else
                        {
                            m_prepend_jw[i] = !m_prepend_jw[i];
                        }
                    }
                }
            }
        }

        m_mapped=true;
        return flip_sign;
    }

    bool contains_jordan_wigner_string() const{return m_prepend_jw.size() != 0;}
    bool prepend_jordan_wigner_string(size_t i) const{return m_prepend_jw[i];}
protected:
    container_type m_ops;       //the ops object has been made mutable so that we can edit it while using it as a key in a unordered_map.
                                        //additionally we have ensured that the 
    bool m_has_hash = false;
    std::size_t m_hash;

    mutable std::vector<bool> m_prepend_jw;
    mutable bool m_mapped = false;
};

inline bool operator==(const prodOP& A, const prodOP& B)
{
    if(A.size() != B.size()){return false;}
    for(size_t i = 0; i < A.size(); ++i)
    {
        if(std::get<0>(A[i]) != std::get<0>(B[i]) || std::get<1>(A[i]) != std::get<1>(B[i]) || std::get<2>(A[i]) != std::get<2>(B[i]))
        {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const prodOP& A, const prodOP& B){return !(A == B);}

}


template <>
struct std::hash<ttns::prodOP>
{
    std::size_t operator()(const ttns::prodOP& k) const
    {
        if(k.hasHash()){return k.hash();}
        else{return std::hash<std::string>()(k.hash_string());}
    }
};


#include <unordered_map>
namespace ttns
{

template <typename T> class SOP;
template <typename T> std::ostream& operator<<(std::ostream& os, const SOP<T>& op);

//the string sum of product operator class used for storing the representation of the Hamiltonian of interest.
//TODO: Ensure correctness of the Jordan-Wigner Mapping code
template <typename T> 
class SOP
{
public:
    using operator_dictionary_type = std::vector<std::vector<std::string>>;
    using container_type = std::unordered_map<prodOP, literal::coeff<T>>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using function_type = typename literal::coeff<T>::function_type;

protected:
    iterator last_insert;
    literal::coeff<T> m_Eshift = literal::coeff<T>(T(0));

public:
    SOP(){}
    SOP(size_t nmodes) : m_opdict(nmodes){}
    SOP(size_t nmodes, const std::string& label) : m_opdict(nmodes), m_label(label){}

    SOP(const SOP& o) = default;
    SOP(SOP&& o) = default;

    SOP& operator=(const SOP& o) = default;
    SOP& operator=(SOP&& o) = default;

    void resize(size_t nmodes){m_opdict.resize(nmodes);}
    void reserve(size_t nt){m_terms.reserve(nt);}
    void clear()
    {
        m_opdict.clear();
        m_terms.clear(); 
        m_label.clear();
        m_allow_insertion = true;
    }


    template <typename U>
    SOP<T>& operator*=(const U& a)
    {
        this->Eshift() *= a;
        for(auto& t : m_terms)
        {
            t.second *= a;
        }
        return *this;
    }

    template <typename U>
    SOP<T>& operator/=(const U& a)
    {
        this->Eshift() /= a;
        for(auto& t : m_terms)
        {
            t.second /= a;
        }
        return *this;
    }

    template <typename U>
    SOP<T>& operator+=(const sSOP<U>& a)
    {
        for(const auto& t : a)
        {
            insert(t);
        }
        return *this;
    }
    
    SOP<T>& operator+=(const sOP& a)
    {
        insert(T(1), {a});
        return *this;
    }

    SOP<T>& operator+=(const sPOP& a)
    {
        insert(T(1), a);
        return *this;
    }

    template <typename U>
    SOP<T>& operator+=(const sNBO<U>& a)
    {
        insert(a);
        return *this;
    }

    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, SOP<T>&>::type operator+=(const U& a)
    {
        this->Eshift() += a;
        return *this;
    }

    SOP<T>& operator+=(const literal::coeff<T>& a)
    {
        this->Eshift() += a;
        return *this;
    }

    template <typename U>
    SOP<T>& operator-=(const sSOP<U>& a)
    {
        for(const auto& t : a)
        {
            insert({T(-1.0)*t.coeff(), t.pop()});
        }
        return *this;
    }
    
    SOP<T>& operator-=(const sOP& a)
    {
        insert(T(-1), {a});
        return *this;
    }

    SOP<T>& operator-=(const sPOP& a)
    {
        insert(T(-1), a);
        return *this;
    }

    SOP<T>& operator-=(const sNBO<T>& a)
    {
        insert(T(-1.0)*a.coeff(), a.pop());
        return *this;
    }
    
    template <typename U>
    typename std::enable_if<linalg::is_number<U>::value, SOP<T>&>::type operator-=(const U& a)
    {
        this->Eshift() -= a;
        return *this;
    }

    SOP<T>& operator-=(const literal::coeff<T>& a)
    {
        this->Eshift() -= a;
        return *this;
    }

    void set_operator_dictionary(const SOP& o)
    {
        m_opdict = o.m_opdict;
    }

    const operator_dictionary_type& operator_dictionary() const{return m_opdict;}

    friend std::ostream& operator<< <T>(std::ostream& os, const SOP<T>& op);

    size_t nterms() const{return m_terms.size();}
    size_t nmodes() const{return m_opdict.size();}

    const std::string& label() const{return m_label;}
    std::string& label(){return m_label;}

    const literal::coeff<T>& Eshift() const{return m_Eshift;}
    literal::coeff<T>& Eshift(){return m_Eshift;}

    template <typename U>
    void insert(const U& ac, const sPOP& a)
    {
        ASSERT(m_allow_insertion, "Additional terms cannot be added to Hamiltonian following jordan-wigner mapping.");\
        prodOP temp(a.size());
        size_t counter = 0;

        bool contains_fermionic = false;
        for(const auto& m : a)
        {
            ASSERT(m.mode() < m_opdict.size(), "Cannot add element.  Matrix element out of bounds.");
            auto& mlabels = m_opdict[m.mode()];
            auto it = std::find(mlabels.begin(), mlabels.end(), m.op());
            size_t ind = 0;
            if(it == mlabels.end())
            {
                ind = mlabels.size();
                mlabels.push_back(m.op());
            }
            else
            {
                ind = static_cast<size_t>(it - mlabels.begin());
            }

            if(m.fermionic()){contains_fermionic=true;}
            temp[counter] = std::make_tuple(ind, m.mode(), m.fermionic());
            ++counter;
        }
        U v = ac;
        if(!contains_fermionic)
        {
            temp.order_modes();
        }
        else
        {
            bool flip_sign = temp.fermion_order_modes(m_opdict);
            if(flip_sign){v = U(-1.0)*ac;}
        }
        m_terms[temp] += v;
    }

    void insert(const sNBO<T>& a){insert(a.coeff(), a.pop());}
    template <typename U>
    void insert(const U& v, const prodOP& a)
    {
        ASSERT(m_allow_insertion, "Additional terms cannot be added to Hamiltonian following jordan-wigner mapping.");\
        bool contains_fermionic = false;
        for(const auto& o : a)
        {
            if(std::get<2>(o)){contains_fermionic=true;}
        }

        prodOP b = a;
        U v2 = v;
        if(contains_fermionic)
        {
            bool flip_sign = b.fermion_order_modes(m_opdict);
            if(flip_sign){v2 = U(-1.0)*v;}
        }
        else
        {
            b.order_modes();
        }
        m_terms[b] += v2;
    }
public:
    iterator begin() {  return iterator(m_terms.begin());  }
    iterator end() {  return iterator(m_terms.end());  }
    const_iterator begin() const {  return const_iterator(m_terms.begin());  }
    const_iterator end() const {  return const_iterator(m_terms.end());  }

    size_t jordan_wigner_index(size_t i) const{ASSERT(i < m_jordan_wigner_indices.size(), "Failed to access jordan wigner index.") return m_jordan_wigner_indices[i];}
protected:
    operator_dictionary_type m_opdict;
    container_type m_terms;
    std::string m_label;
    bool m_allow_insertion = true;
    std::vector<size_t> m_jordan_wigner_indices;

public:
    inline bool set_is_fermionic_mode(std::vector<bool>& is_fermion_mode) const
    {
        std::fill(is_fermion_mode.begin(), is_fermion_mode.end(), false);
        std::vector<bool> mode_properties_set(is_fermion_mode.size());  std::fill(mode_properties_set.begin(), mode_properties_set.end(), false);

        //first verify that the input sSOP is valid
        for(const auto& pop : m_terms)
        {
            for(const auto& op : pop.first)
            {
                size_t mode = std::get<1>(op);
                //if we have set the mode properties of this mode already ensure that the new properties we have obtained from this operator are consistent.
                if(mode_properties_set[mode])
                {
                    if(std::get<2>(op) != is_fermion_mode[mode]){return false;}
                }
                else
                {
                    if(std::get<2>(op) ){is_fermion_mode[mode] = true;}
                    mode_properties_set[mode] = true;
                }
            }
        }
        return true;
    }

protected:
    void jordan_wigner(const std::vector<bool>& is_fermion_mode, double tol)
    {
        //first iterate through each term of the array and if there are any elements where the value 
        this->prune_zeros(tol);

        bool contains_fermionic_operator = false;
        for(size_t i = 0; i < this->nmodes(); ++i)
        {
            if(is_fermion_mode[i]){contains_fermionic_operator = true;}
        }

        if(contains_fermionic_operator)
        {
            m_jordan_wigner_indices.resize(this->nmodes());
            for(size_t mode = 0; mode < this->nmodes(); ++mode)
            {
                if(is_fermion_mode[mode])
                {   
                    m_jordan_wigner_indices[mode] = m_opdict[mode].size();
                    m_opdict[mode].push_back(std::string("jw"));
                }
            }
            for(auto& a : m_terms)
            {
                bool flip_sign = a.first.jordan_wigner(m_opdict, is_fermion_mode);
                if(flip_sign){a.second *= T(-1.);}

            }
        }
    }

public:
    void prune_zeros(double tol = 1e-15)
    {
        for(auto it = m_terms.begin(); it != m_terms.end(); )
        {
            if(it->second.is_zero(tol)){it = m_terms.erase(it);}
            else{it++;}
        }
    }

    SOP& jordan_wigner(const system_modes& sys_info, double tol = 1e-15)
    {
        size_t nmodes = sys_info.nprimitive_modes();
        ASSERT(nmodes == this->nmodes(), "Failed to simplify sum of product operator object operator sys_info object does not have the correct size.");
        std::vector<bool> is_fermion_mode(nmodes);      std::fill(is_fermion_mode.begin(), is_fermion_mode.end(), false);
        for(size_t i = 0; i < nmodes; ++i)
        {
            is_fermion_mode[i] = sys_info.primitive_mode(i).fermionic();
        }
        this->jordan_wigner(is_fermion_mode, tol);
        return *this;
    }

    //SOP& jordan_wigner(double tol = 1e-15)
    //{
    //    size_t nmodes = this->nmodes();
    //    std::vector<bool> is_fermion_mode(nmodes);      std::fill(is_fermion_mode.begin(), is_fermion_mode.end(), false);
    //    this->set_is_fermionic_mode(is_fermion_mode);
    //    this->jordan_wigner(is_fermion_mode, tol);
    //    return *this;
    //}
    
    
    sSOP<T> expand() const
    {
        sSOP<T> ret;
        for(const auto& t : *this)
        {
            ret += t.second*t.first.as_prod_op(this->m_opdict);
        }
        return ret;
    }
};


template <typename T> 
std::ostream& operator<<(std::ostream& os, const ttns::SOP<T>& op)
{
    if(!op.label().empty()){os << op.label() << ": " << std::endl;}
    os << op.Eshift() << std::endl;
    const auto separator = "";    const auto* sep = "";
    const auto plus = "+";
    for(const auto& t : op)
    {
        sep = t.second.is_positive() ? plus : separator;
        os << sep << t.second << " ";   t.first.label(os, op.m_opdict) << std::endl;
    }
    return os;
}
}


/* 
template <typename T>
ttns::SOP<T> operator+(const ttns::SOP<T>& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(a);
    for(auto& t : b){ret.insert(t.second, t.first);}
    return  ret;
}*/


template <typename T, typename U>
typename std::enable_if<linalg::is_number<T>::value, ttns::SOP<decltype(T()*U())>>::type operator+(const T& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.Eshift() += a;
    return  ret;
}

template <typename T, typename U>
typename std::enable_if<linalg::is_number<T>::value, ttns::SOP<decltype(T()*U())>>::type operator+(const ttns::SOP<U>& b, const T& a)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.Eshift() += a;
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator+(const ttns::sSOP<T>& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    for(const auto& _a : a)
    {
        ret.insert(_a);
    }
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator+(const ttns::SOP<T>& b, const ttns::sSOP<U>& a)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    for(const auto& _a : a)
    {
        ret.insert(_a);
    }
    return  ret;
}


template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator+(const ttns::sNBO<T>& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.insert(a);
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator+(const ttns::SOP<T>& b, const ttns::sNBO<U>& a)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.insert(a);
    return  ret;
}


template <typename T>
ttns::SOP<T> operator+(const ttns::sPOP& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(b);
    ret.insert(1.0, a);
    return  ret;
}

template <typename T>
ttns::SOP<T> operator+(const ttns::SOP<T>& b, const ttns::sPOP& a)
{
    ttns::SOP<T> ret(b);
    ret.insert(1.0, a);
    return  ret;
}

template <typename T>
ttns::SOP<T> operator+(const ttns::sOP& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(b);
    ret.insert(1.0, {a});
    return  ret;
}

template <typename T>
ttns::SOP<T> operator+(const ttns::SOP<T>& b, const ttns::sOP& a)
{
    ttns::SOP<T> ret(b);
    ret.insert(1.0, {a});
    return  ret;
}

/* 
template <typename T>
ttns::SOP<T> operator-(const ttns::SOP<T>& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(a);
    for(auto& t : b)
    {
        ret.insert(-t.second, t.first);
    }
    return ret;
}*/


template <typename T, typename U>
typename std::enable_if<linalg::is_number<T>::value, ttns::SOP<decltype(T()*U())>>::type operator-(const T& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret *= T(-1.0);
    ret.Eshift() += a;
    return  ret;
}

template <typename T, typename U>
typename std::enable_if<linalg::is_number<T>::value, ttns::SOP<decltype(T()*U())>>::type operator-(const ttns::SOP<U>& b, const T& a)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.Eshift() -= a;
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator-(const ttns::sSOP<T>& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    for(const auto& _a : a)
    {
        ret.insert(-1.0*_a.coeff(), _a.pop());
    }
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator-(const ttns::SOP<T>& b, const ttns::sSOP<U>& a)
{
    ttns::SOP<T> ret(b);
    for(const auto& _a : a)
    {
        ret.insert(-1.0*_a.coeff(), _a.pop());
    }
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator-(const ttns::sNBO<T>& a, const ttns::SOP<U>& b)
{
    ttns::SOP<decltype(T()*U())> ret(b);
    ret.insert(-1.0*a.coeff(), a.pop());
    return  ret;
}

template <typename T, typename U>
ttns::SOP<decltype(T()*U())> operator-(const ttns::SOP<T>& b, const ttns::sNBO<U>& a)
{
    ttns::SOP<T> ret(b);
    ret.insert(-1.0*a.coeff(), a.pop());
    return  ret;
}


template <typename T>
ttns::SOP<T> operator-(const ttns::sPOP& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(b);
    ret.insert(-1.0, a);
    return  ret;
}

template <typename T>
ttns::SOP<T> operator-(const ttns::SOP<T>& b, const ttns::sPOP& a)
{
    ttns::SOP<T> ret(b);
    ret.insert(-1.0, a);
    return  ret;
}

template <typename T>
ttns::SOP<T> operator-(const ttns::sOP& a, const ttns::SOP<T>& b)
{
    ttns::SOP<T> ret(b);
    ret.insert(-1.0, {a});
    return  ret;
}

template <typename T>
ttns::SOP<T> operator-(const ttns::SOP<T>& b, const ttns::sOP& a)
{
    ttns::SOP<T> ret(b);
    ret.insert(-1.0, {a});
    return  ret;
}


#endif
