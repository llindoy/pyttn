#ifndef TTNS_LIB_OPERATOR_GEN_COMPRESSED_SOP_HPP
#define TTNS_LIB_OPERATOR_GEN_COMPRESSED_SOP_HPP

#include <linalg/linalg.hpp>
#include <utils/term_indexing_array.hpp>
#include <linalg/linalg.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>


#include "coeff_type.hpp"
#include "SOP.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

template <typename T> class compressedSOP;
template <typename T> std::ostream& operator<<(std::ostream& os, const compressedSOP<T>& op);

class compositeSiteOperator
{
public:
    compositeSiteOperator(){}
    compositeSiteOperator(const std::vector<size_t>& o) : m_op_indices(o) {}

    const std::vector<size_t>& operator_indices() const{return m_op_indices;}
    std::vector<size_t>& operator_indices(){return m_op_indices;}

    void push_back(size_t i) {m_op_indices.push_back(i);}
    size_t size() const{return m_op_indices.size();}

    size_t operator[](size_t i) const{return m_op_indices[i];}
    size_t& operator[](size_t i){return m_op_indices[i];}

    std::string label(const std::vector<std::string>& opdict) const
    {
        std::string ret;
        for(auto ind : m_op_indices)
        {
            ASSERT(ind  < opdict.size(), "failed to get label for compositeSiteOperator.");
            ret += opdict[ind];
        }
        return ret;
    }

    sPOP as_product_operator(const std::vector<std::string>& opdict, size_t mode) const
    {
        sPOP ret;
        for(auto ind : m_op_indices)
        {
            ASSERT(ind  < opdict.size(), "failed to get label for compositeSiteOperator.");
            ret *= sOP(opdict[ind], mode);
        }
        return ret;
    }

    using container_type = std::vector<size_t>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    iterator begin(){return m_op_indices.begin();}
    iterator end(){return m_op_indices.end();}

    const_iterator begin() const{return m_op_indices.begin();}
    const_iterator end() const{return m_op_indices.end();}
protected:
    container_type m_op_indices;
};

inline bool operator==(const compositeSiteOperator& A, const compositeSiteOperator& B)
{
    if(A.size() != B.size()){return false;}
    for(size_t i = 0; i < A.size(); ++i)
    {
        if(A[i] != B[i])
        {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const compositeSiteOperator& A, const compositeSiteOperator& B){return !(A == B);}
}

template <>
struct std::hash<ttns::compositeSiteOperator>
{
    std::size_t operator()(const ttns::compositeSiteOperator& k) const
    {
        std::size_t seed = k.size();
        for(auto x : k.operator_indices())
        {
            x =  ((x >> 16) ^ x) *0x45d9f3b;
            x =  ((x >> 16) ^ x) *0x45d9f3b;
            x =  (x >> 16) ^ x;
            seed ^= x + 0x9ef779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

namespace ttns
{

//a generic sum of product operator object.
template <typename T>
class compressedSOP
{
public:
    using element_container_type = utils::term_indexing_array<size_t>;
    using element_type = std::unordered_map<compositeSiteOperator, element_container_type>;
    using container_type = std::vector<element_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    using dict_type = std::vector<std::string>;
    using dict_container = std::vector<dict_type>;

    using site_ops_type = std::vector<std::vector<sPOP>>;
public:

    compressedSOP() : m_nterms(0){}
    compressedSOP(size_t nterms, size_t nmodes) 
    try : m_mode_operators(nmodes), m_op_dict(nmodes), m_nterms(nterms), m_coeff(nterms){}
    //try : m_mode_operators(nmodes), m_op_dict(nmodes), m_composite_operators(nmodes), m_nterms(nterms), m_coeff(nterms){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct sum of product operator object.");
    }    

    compressedSOP(const compressedSOP& o) = default;
    compressedSOP(compressedSOP&& o) = default;

    compressedSOP(const SOP<T>& op) 
    {
        try
        {
            initialise(op);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sum of product operator object.");
        }    
    }


    compressedSOP(const SOP<T>& op, const std::vector<size_t>& inds) 
    {
        try
        {
            initialise(op, inds);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sum of product operator object.");
        }    
    }

    compressedSOP& operator=(const compressedSOP& o) = default;
    compressedSOP& operator=(compressedSOP&& o) = default;

    compressedSOP& operator=(const SOP<T>& o)
    {
        try
        {
            initiailise(o);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sum of product operator object.");
        }    
    }

    void initialise(const SOP<T>& o, const std::vector<size_t>& inds, size_t nmodes = 0)
    {
        try
        {
            m_label = o.label();
            resize(o.nterms(), nmodes > o.nmodes() ? nmodes : o.nmodes());

            //start by copying the base operator dictionary object across
            m_op_dict = o.reordered_operator_dictionary(inds);

            size_t r = 0;
            
            //iterate over each term in the SOP 
            for(const auto& pop : o)
            {
                //and extract the operators acting on each mode.  Here we need to take into acconut
                //that we can have multiple operators all acting on the same mode and need to form a new
                //operator label corresponding to these
                m_coeff[r] = pop.second;
        
                std::unordered_map<size_t, compositeSiteOperator> terms;    
                //now we iterate over the prodOP object and get the compressed 
                
                //iterate over each term and store the mode and the compositeSiteOperator.  Here we are using the fact
                //that the SOP object enforces ordering of operators
                for(const auto& op : pop.first)
                {
                    terms[inds[std::get<1>(op)]].push_back(std::get<0>(op));
                }
                if(pop.first.contains_jordan_wigner_string())
                {
                    for(size_t i = 0; i < o.nmodes(); ++i)
                    {
                        if(pop.first.prepend_jordan_wigner_string(i))
                        {
                            terms[inds[i]].push_back(o.jordan_wigner_index(i));
                        }
                    }
                }

                //now attempt to insert these terms into the mode_operator object
                for(const auto& kv : terms)
                {
                    this->_insert(kv.second, r, kv.first);
                }
                ++r;
            }

            CALL_AND_HANDLE(insert_identities(), "Failed to insert identity operators into array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise compressed sum of product operator object.");
        }    
    }

    void initialise(const SOP<T>& o, size_t nmodes = 0)
    {
        try
        {
            m_label = o.label();
            resize(o.nterms(), nmodes > o.nmodes() ? nmodes : o.nmodes());

            //start by copying the base operator dictionary object across
            m_op_dict = o.operator_dictionary();

            size_t r = 0;
            
            //iterate over each term in the SOP 
            for(const auto& pop : o)
            {
                //and extract the operators acting on each mode.  Here we need to take into acconut
                //that we can have multiple operators all acting on the same mode and need to form a new
                //operator label corresponding to these
                m_coeff[r] = pop.second;
        
                std::unordered_map<size_t, compositeSiteOperator> terms;    
                //now we iterate over the prodOP object and get the compressed 
                
                //iterate over each term and store the mode and the compositeSiteOperator.  Here we are using the fact
                //that the SOP object enforces ordering of operators
                for(const auto& op : pop.first)
                {
                    terms[std::get<1>(op)].push_back(std::get<0>(op));
                }
                if(pop.first.contains_jordan_wigner_string())
                {
                    for(size_t i = 0; i < o.nmodes(); ++i)
                    {
                        if(pop.first.prepend_jordan_wigner_string(i))
                        {
                            terms[i].push_back(o.jordan_wigner_index(i));
                        }
                    }
                }

                //now attempt to insert these terms into the mode_operator object
                for(const auto& kv : terms)
                {
                    this->_insert(kv.second, r, kv.first);
                }
                ++r;
            }

            CALL_AND_HANDLE(insert_identities(), "Failed to insert identity operators into array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise compressed sum of product operator object.");
        }    
    }

    void insert_identities()
    {
        try
        {
            m_identity_index.resize(m_mode_operators.size());
            for(size_t nu = 0; nu < m_mode_operators.size(); ++nu)
            {
                element_container_type rinds, tval;   
                rinds.reserve(m_coeff.size());
                tval.reserve(m_coeff.size());
                size_t count = 0;
                bool rinds_set = false;

                //check if the operator dictionary contains the m_op
                std::string op("id");
                auto& mlabels = m_op_dict[nu];
                auto it = std::find(mlabels.begin(), mlabels.end(), op);
                size_t ind = 0;
                if(it == mlabels.end())
                {
                    ind = mlabels.size();
                    mlabels.push_back(op);
                }
                else
                {
                    ind = static_cast<size_t>(it - mlabels.begin());
                }

                compositeSiteOperator id({ind});

                for(const auto& combop : m_mode_operators[nu])
                {
                    if(count == 0)
                    {
                        tval = combop.second;
                        ++count;
                    }
                    else
                    {
                        if(rinds_set)
                        {
                            element_container_type::set_union(rinds, combop.second, tval);
                            rinds_set = false;
                        }
                        else
                        {
                            element_container_type::set_union(tval, combop.second, rinds);
                            rinds_set = true;
                        }
                    }
                }
                if(!rinds_set){element_container_type::complement(tval, m_mode_operators[nu][id]);}
                else{element_container_type::complement(rinds, m_mode_operators[nu][id]);}

                auto id_iter = m_mode_operators[nu].find(id);
                ASSERT(id_iter != m_mode_operators[nu].end(), "Id operator has not been added.");
                m_identity_index[nu] = std::distance(m_mode_operators[nu].begin(), id_iter);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert identity operators into object.");
        }    
    }

    void resize(size_t nterms, size_t nmodes)
    {
        try
        {
            clear();
            m_coeff.resize(nterms); std::fill(m_coeff.begin(), m_coeff.end(), T(1));
            m_mode_operators.resize(nmodes);
            //m_composite_operators.resize(nmodes);
            m_nterms = nterms;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize sp hamiltonian object.");
        }
    }
   
    size_t identity_index(size_t nu) const
    {
        ASSERT(nu < m_identity_index.size(), "Failed to access identity index.  Index out of bounds.");
        return m_identity_index[nu];
    }

protected:



    void _insert(const compositeSiteOperator& op, const element_container_type& r, size_t nu)
    {
        try
        {
            auto& v = m_mode_operators[nu][op];
            std::cerr << "inserting at mode: " << nu << " term: " << r << " label:" << op.label(m_op_dict[nu]) << std::endl;
            v.reserve(m_nterms);
            v.insert(r);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert operator into compressedSOP");
        }
    } 
    
    void _insert(const compositeSiteOperator& op, const std::vector<size_t>& r, size_t nu)
    {
        try
        {
            auto& v = m_mode_operators[nu][op];
            v.reserve(m_nterms);
            v.insert(std::begin(r), std::end(r));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert operator into compressedSOP");
        }
    } 

    void _insert(const compositeSiteOperator& op, size_t r, size_t nu)
    {
        try
        {
            auto& v = m_mode_operators[nu][op];
            v.reserve(m_nterms);
            CALL_AND_HANDLE(v.insert(r), "Failed to push mode operator term to list.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert operator into compressedSOP");
        }
    } 

    template <typename rtype>
    void _insert(const std::string& op, rtype&& r, size_t nu)
    {
        try
        {
            auto& mlabels = m_op_dict[nu];
            auto it = std::find(mlabels.begin(), mlabels.end(), op);
            size_t ind = 0;
            if(it == mlabels.end())
            {
                ind = mlabels.size();
                mlabels.push_back(op);
            }
            else
            {
                ind = static_cast<size_t>(it - mlabels.begin());
            }

            compositeSiteOperator cOP({ind});
            CALL_AND_RETHROW(_insert(cOP, r, nu));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert operator into cSSOP");
        }
    } 


public:
    void clear()
    {
        m_mode_operators.clear();
        //m_composite_operators.clear();
        m_op_dict.clear();
        m_coeff.clear();
        m_nterms = 0;
    }

    const element_type& operators(size_t nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator[](size_t nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator()(size_t nu) const
    {
        return m_mode_operators[nu];
    }

    const std::vector<literal::coeff<T>>& coeff() const{return m_coeff;}
    const literal::coeff<T>& coeff(size_t r)const{return m_coeff[r];}
    literal::coeff<T>& coeff(size_t r){return m_coeff[r];}

    size_t nterms() const{return m_nterms;}
    size_t nterms(size_t nu) const{return m_mode_operators[nu].size();}
    size_t nmodes() const{return m_mode_operators.size();}

    iterator begin() {  return iterator(m_mode_operators.begin());  }
    iterator end() {  return iterator(m_mode_operators.end());  }
    const_iterator begin() const {  return const_iterator(m_mode_operators.begin());  }
    const_iterator end() const {  return const_iterator(m_mode_operators.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_mode_operators.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_mode_operators.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_mode_operators.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_mode_operators.rend());  }

    const std::string& operator_label(size_t nu, size_t id) const
    {
        ASSERT(nu < m_op_dict.size() && id < m_op_dict[nu].size(), "Unable to access operator label.  Index out of bounds.");
        return m_op_dict[nu][id];
    }

    size_t nops(size_t nu) const
    {
        ASSERT(nu < m_op_dict.size(), "Unable to get operator index.  Index out of bounds.");
        return m_op_dict[nu].size();
    }

    friend std::ostream& operator<< <T>(std::ostream& os, const compressedSOP<T>& op);


    site_ops_type site_operators() const
    {
        site_ops_type ret(m_mode_operators.size());
        for(size_t i = 0; i < m_mode_operators.size(); ++i)
        {
            for(const auto& op : m_mode_operators[i])
            {
                ret[i].push_back(op.first.as_product_operator(m_op_dict[i], i));
            }
        }
        return ret;
    }

    const std::string& label() const{return m_label;}
    std::string& label(){return m_label;}
protected:
    //check whether any of the new r values are already bound for this mode.
    bool valid_rvals(const element_container_type& r, size_t nu)
    {
        //iterate over all elements in the map associated with mode nu
        for(const auto& combop : m_mode_operators[nu])
        {
            element_container_type rtemp(m_coeff.size());
            element_container_type::set_intersection(combop.second, r, rtemp);
            if(rtemp.size() != 0){return false;}
        }
        return true;
    }

    bool valid_rvals(size_t r, size_t nu)
    {
        //iterate over all elements in the map associated with mode nu
        for(const auto& combop : m_mode_operators[nu])
        {
            if(std::get<1>(combop).contains(r)){return false;}
        }
        return true;
    }

public:
    //create the SOP representation of the compressedSOP.  This is done by going through an intermediate sSOP class and so is not super efficient
    SOP<T> sop() const
    {

        std::vector<sPOP > string_obj(m_nterms);
        std::vector<size_t> rinds(m_nterms);
        for(size_t i = 0; i < m_mode_operators.size(); ++i)
        {   
            for(const auto& op : m_mode_operators[i])
            {
                sPOP top = op.first.as_product_operator(m_op_dict[i], i);

                //if this operator is the identity operator we don't actually add it to the sop
                if(top.size() == 1 && top.ops().front() == sOP("id", i)){}
                else
                {
                    op.second.get(rinds);
                    for(const auto& r : rinds)
                    {
                        string_obj[r] *= top;
                    }
                }
            }
        }

        SOP<T> sop(m_nterms);
        for(size_t i = 0; i < m_nterms; ++i)
        {
            sop += m_coeff[i] * string_obj[i];
        }
        return sop;
    }

protected:
    container_type m_mode_operators;
    dict_container m_op_dict;

    size_t m_nterms;
    std::vector<literal::coeff<T>> m_coeff;
    std::string m_label;
    std::vector<size_t> m_identity_index;
};  //class compressedSOP


template <typename T>
std::ostream& operator<<(std::ostream& os, const compressedSOP<T>& op)
{
    if(!op.label().empty()){os << op.label() << ": " << std::endl;}


    for(size_t i = 0; i < op.m_coeff.size(); ++i)
    {
        os << "(" << i << "," << op.m_coeff[i] << ")" << " ";
    }
    os << std::endl;
    //print out the sparse mode operators info info
    for(size_t nu = 0; nu < op.m_mode_operators.size(); ++nu)
    {
        os << "mode: " << nu << std::endl;
        os << "labels: { ";
        for(const auto& id : op.m_op_dict[nu])
        {
            os << id << " ";
        }
        os <<"}" << std::endl;

        size_t i = 0;
        for( auto& mop : op.m_mode_operators[nu])
        {
            os << i << ": " << std::get<0>(mop).label(op.m_op_dict[nu]) << " " << std::get<1>(mop) << std::endl;
            ++i;
        }
        os << std::endl;
    }
    
    return os;
}


}   //namespace ttns

#endif  //UTILS_OPERATOR_GEN_COMPRESSED_SOP_HPP


