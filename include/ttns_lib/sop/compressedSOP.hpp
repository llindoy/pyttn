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

    //std::string label(const std::vector<std::string>& opdict) const
    //{
    //    std::string ret;
    //    for(auto ind : m_op_indices)
    //    {
    //        ASSERT(ind  < opdict.size(), "failed to get label for compositeSiteOperator.");
    //        ret += opdict[ind];
    //    }
    //    return ret;
    //}

    std::string label(const std::vector<std::pair<size_t, size_t>>& copdict, const std::vector<std::vector<std::string>>& opdict) const
    {
        std::string ret;
        for(auto ind : m_op_indices)
        {
            ASSERT(ind  < opdict.size(), "failed to get label for compositeSiteOperator.");
            auto rv = copdict[ind];
            ret += opdict[std::get<0>(rv)][std::get<1>(rv)];
        }
        return ret;
    }

    //sPOP as_product_operator(const std::vector<std::string>& opdict, size_t mode) const
    //{
    //    sPOP ret;
    //    for(auto ind : m_op_indices)
    //    {
    //        ASSERT(ind  < opdict.size(), "failed to get label for compositeSiteOperator.");
    //        ret *= sOP(opdict[ind], mode);
    //    }
    //    return ret;
    //}

    sPOP as_product_operator(const std::vector<std::pair<size_t, size_t>>& copdict, const std::vector<std::vector<std::string>>& opdict) const
    {
        sPOP ret;
        for(auto ind : m_op_indices)
        {
            ASSERT(ind < copdict.size(), "failed to get label for compositeSiteOperator.");
            auto rv = copdict[ind];
            ret *= sOP(opdict[std::get<0>(rv)][std::get<1>(rv)], std::get<0>(rv));
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

    using composite_dict_type = std::vector<std::pair<size_t, size_t>>;
    using composite_dict_container = std::vector<composite_dict_type>;

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


    compressedSOP(const SOP<T>& op, const system_modes& inds) 
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

    void initialise(const SOP<T>& o, const system_modes& sysinf, size_t nmodes = 0)
    {
        try
        {
            m_label = o.label();
            resize(o.nterms(), nmodes > o.nmodes() ? nmodes : o.nmodes());

            //an object for mapping the composite mode indices back onto the primitive mode indices
            m_composite_op_dict.resize(sysinf.nmodes());

            //start by copying the base operator dictionary object across
            m_op_dict = o.operator_dictionary();

            //an object for mapping the primitive mode indices back on the composite mode indices.  This is only 
            //needed locally as it is used to construct the m_mode_operators infor for composite modes.
            std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> prim_to_comp;

            //now iterate over the primitive modes 
            for(size_t i = 0; i < m_op_dict.size(); ++i)
            {
                //get the composite mode info associated with the mode
                auto cminf = sysinf.primitive_mode_index(i);
                size_t mode = std::get<0>(cminf);

                //now map the composite mode index from the underlying hamiltonian ordering to the tree ordering
                size_t cmode = sysinf.mode_index(mode);

                for(size_t j = 0; j < m_op_dict[i].size(); ++j)
                {
                    //and add in a pair pointing to the correct primitive mode in the primitive mode ordering of the Hamiltonian as well as the index in the op_dict.
                    m_composite_op_dict[cmode].push_back(std::make_pair(i, j));

                    //now map the primitive mode index and composite mode index 
                    prim_to_comp[std::make_pair(i, j)] = std::make_pair(cmode, m_composite_op_dict[cmode].size()-1);
                }
            }

            size_t r = 0;
            //iterate over each term in the SOP 
            for(const auto& pop : o)
            {
                //and extract the operators acting on each mode.  Here we need to take into account
                //that we can have multiple operators all acting on the same mode and need to form a new
                //operator label corresponding to these
                m_coeff[r] = pop.second;
        
                std::unordered_map<size_t, compositeSiteOperator> terms;    
                //now we iterate over the prodOP object and get the compressed 
                
                for(const auto& op : pop.first)
                {
                    std::pair<size_t, size_t> prim_data = std::make_pair(std::get<1>(op), std::get<0>(op));
                    ASSERT(prim_to_comp.find(prim_data) != prim_to_comp.end(), "Primitive mode term is not in the map.");
                    auto mp = prim_to_comp[prim_data];
                    terms[std::get<0>(mp)].push_back(std::get<1>(mp));
                }

                if(pop.first.contains_jordan_wigner_string())
                {
                    for(size_t i = 0; i < o.nmodes(); ++i)
                    {
                        if(pop.first.prepend_jordan_wigner_string(i))
                        {
                            std::pair<size_t, size_t> prim_data = std::make_pair(i, o.jordan_wigner_index(i));
                            ASSERT(prim_to_comp.find(prim_data) != prim_to_comp.end(), "Primitive mode term is not in the map.");
                            auto mp = prim_to_comp[prim_data];
                            
                            terms[std::get<0>(mp)].push_back(std::get<1>(mp));
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
    //Insert additional identity elements into the network.  Here we start by searching through the mode object and checking if
    //and identity elements have already been bound reusing it if present.  And only bind a new one if no identity terms are 
    //present.  We also note that in the event of a composite mode with multiple identities bound we do not combine these together
    //which may lead to slightly larger bond dimensions in these case, this can be avoided by not explicitly binding identity operators 
    //unless absolutely necessary.
    void insert_identities()
    {
        try
        {
            //resize that holds the mode label corresponding to the identity element
            m_identity_index.resize(m_mode_operators.size());
            for(size_t mode = 0; mode < m_mode_operators.size(); ++mode)
            {
                element_container_type rinds, tval;   
                rinds.reserve(m_coeff.size());
                tval.reserve(m_coeff.size());
                size_t count = 0;
                bool rinds_set = false;

                //get the current modes composite operator dictionary.
                auto& mcompdict = m_composite_op_dict[mode];

                //now set up the spin label associated with the identity term.
                std::string op("id");

                //now we iterate over each of the terms in the composite operator dictionary and check to see if any of the 
                //corresponding mode operators are the identity.  If they are we will set that to be the identity index.
                //Otherwise we inset an identity operator at the end of the composite operator dictionary.
                
                //the index of the identity element
                size_t idind = 0;

                //the mode that we will store this on.  
                size_t idnu = 0;

                //iterate over all terms in the composite operator dictionary associated with this mode
                bool identity_found = false;
                for(size_t index = 0; index < mcompdict.size() && !identity_found; ++index)
                {
                    auto ind = mcompdict[index];

                    //extract the op_dict mode and term index from this composite site operator
                    size_t nu = std::get<0>(ind);
                    size_t tindex = std::get<1>(ind);

                    //now if the label associated with this mode is the identity we have found the identity
                    //term store the current mode and the index associated with this term and set identity_found
                    //to terminate the iteration
                    if(m_op_dict[idnu][tindex] == op)
                    {   
                        idnu = nu;
                        idind = index;
                        identity_found = true;
                    }
                    else
                    {
                        idnu = nu;
                    }
                }
                
                if(identity_found)
                {
                    compositeSiteOperator id({idind});
                    auto id_iter = m_mode_operators[mode].find(id);
                    ASSERT(id_iter != m_mode_operators[mode].end(), "Id operator has not been added.");
                    m_identity_index[mode] = std::distance(m_mode_operators[mode].begin(), id_iter);

                    for(const auto& combop : m_mode_operators[mode])
                    {
                        if(count != m_identity_index[mode])
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
                        ++count;
                    }
                    if(!rinds_set){element_container_type::complement(tval, m_mode_operators[mode][id]);}
                    else{element_container_type::complement(rinds, m_mode_operators[mode][id]);}
                }
                else
                {
                    idind = mcompdict.size();
                    mcompdict.push_back(std::make_pair(idnu, m_op_dict[idnu].size()));
                    m_op_dict[idnu].push_back(op);
                    compositeSiteOperator id({idind});

                    //iterate through the mode_operator index and combine together all 
                    for(const auto& combop : m_mode_operators[mode])
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
                    if(!rinds_set){element_container_type::complement(tval, m_mode_operators[mode][id]);}
                    else{element_container_type::complement(rinds, m_mode_operators[mode][id]);}
                    auto id_iter = m_mode_operators[mode].find(id);
                    ASSERT(id_iter != m_mode_operators[mode].end(), "Id operator has not been added.");
                    m_identity_index[mode] = std::distance(m_mode_operators[mode].begin(), id_iter);
                }
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
            m_composite_op_dict.resize(nmodes);
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
    void _insert(const compositeSiteOperator& op, size_t r, size_t nu)
    {
        try
        {
            auto& v = m_mode_operators[nu][op];
            v.reserve(m_nterms);
            v.insert(r);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to insert operator into compressedSOP");
        }
    } 
    

public:
    void clear()
    {
        m_mode_operators.clear();
        m_composite_op_dict.clear();
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
                ret[i].push_back(op.first.as_product_operator(m_composite_op_dict[i], m_op_dict));
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
                sPOP top = op.first.as_product_operator(m_composite_op_dict[i], m_op_dict);

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
    composite_dict_container m_composite_op_dict;

    size_t m_nterms;
    std::vector<literal::coeff<T>> m_coeff;
    std::string m_label;
    std::vector<size_t> m_identity_index;
};  //class compressedSOP


template <typename T>
std::ostream& operator<<(std::ostream& os, const compressedSOP<T>& op)
{
    RAISE_EXCEPTION("Stream operator not currently working for compressedSOP operator.");
    //if(!op.label().empty()){os << op.label() << ": " << std::endl;}


    //for(size_t i = 0; i < op.m_coeff.size(); ++i)
    //{
    //    os << "(" << i << "," << op.m_coeff[i] << ")" << " ";
    //}
    //os << std::endl;
    ////print out the sparse mode operators info info
    //for(size_t nu = 0; nu < op.m_mode_operators.size(); ++nu)
    //{
    //    os << "mode: " << nu << std::endl;
    //    os << "labels: { ";
    //    for(const auto& id : op.m_op_dict[nu])
    //    {
    //        os << id << " ";
    //    }
    //    os <<"}" << std::endl;

    //    size_t i = 0;
    //    for( auto& mop : op.m_mode_operators[nu])
    //    {
    //        os << i << ": " << std::get<0>(mop).label(op.m_op_dict[nu]) << " " << std::get<1>(mop) << std::endl;
    //        ++i;
    //    }
    //    os << std::endl;
    //}
    //
    //return os;
}


}   //namespace ttns

#endif  //UTILS_OPERATOR_GEN_COMPRESSED_SOP_HPP
