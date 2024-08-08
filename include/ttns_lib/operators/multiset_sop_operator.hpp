#ifndef TTNS_MULTISET_SOP_OPERATOR_CONTAINER_HPP
#define TTNS_MULTISET_SOP_OPERATOR_CONTAINER_HPP

#include <linalg/linalg.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>
#include <map>

#include <linalg/linalg.hpp>
#include "site_operators/site_operator.hpp"
#include "../sop/system_information.hpp"
#include "../sop/autoSOP.hpp"
#include "../sop/operator_dictionaries/operator_dictionary.hpp"
#include "site_operators/sequential_product_operator.hpp"
#include "../ttn/ttn.hpp"
#include "../sop/coeff_type.hpp"


#include "../sop/multiset_SOP.hpp"
#include "sop_operator.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

template <typename T>
using multiset_sttn_node_data = std::vector<std::vector<sttn_node_data<T>>>;


namespace node_data_traits
{
    //clear traits for the operator node data object
    template <typename T>
    struct clear_traits<multiset_sttn_node_data<T> > 
    {
        void operator()(multiset_sttn_node_data<T>& t)
        {
            for(auto& i : t)
            {
                for(auto& j : i)
                {
                    j.clear();
                }
                i.clear();
            }
            CALL_AND_RETHROW(t.clear());
        }
    };

    //assignment traits for the tensor and matrix objects
    template <typename T>
    struct assignment_traits<multiset_sttn_node_data<T>, multiset_sttn_node_data<T> >
    {
        using is_applicable = std::true_type;
        inline void operator()(multiset_sttn_node_data<T>& o,  const multiset_sttn_node_data<T>& i){CALL_AND_RETHROW(o = i);}
    };
}   //namespace node_data_traits


//a generic sum of product operator object.  This stores all of the operator and indexing required for the evaluation of the sop on a TTN state but 
//doesn't store the buffers needed to perform the required contractions
template <typename T, typename backend = linalg::blas_backend>
class multiset_sop_operator
{
public:
    using size_type = typename backend::size_type;

    using op_type = ops::primitive<T, backend>;
    using element_type = site_operator<T, backend>;

    using tree_type = tree<multiset_sttn_node_data<T>>;
    using node_type = typename tree_type::node_type;

    using mode_terms_type = std::vector<element_type>;
    using real_type = typename tmp::get_real_type<T>::type; 

    using ttn_type = ms_ttn<T, backend>;

    using single_set_container = std::vector<mode_terms_type>;

    using container_type = std::vector<std::vector<single_set_container>>;

    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    using site_ops_type = typename autoSOP<T>::site_ops_type;
protected:
    std::vector<std::vector<literal::coeff<T>>> _m_Eshift;
    std::vector<std::vector<T>> m_Eshift;
    std::vector<std::vector<bool>> m_time_dependent_operators_set;
    std::vector<std::vector<bool>> m_time_dependent_coefficients_set;
    bool m_time_dependent = false;
    tree_type m_contraction_info;
    container_type m_mode_operators;
    std::vector<size_type> m_mode_dimension;
    std::vector<std::vector<size_t>> m_indices;
    size_t m_nset;

    bool m_time_dependent_operators = false;
    bool m_time_dependent_coefficients = false;

public:
    multiset_sop_operator(){}
    multiset_sop_operator(const multiset_sop_operator& o) = default;
    multiset_sop_operator(multiset_sop_operator&& o) = default;
    multiset_sop_operator(multiset_SOP<T>& sop, const ttn_type& A, const system_modes& sys, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        CALL_AND_HANDLE(initialise(sop, A, sys, compress, exploit_identity, use_sparse), "Failed to construct sop operator.");
    }
    multiset_sop_operator(multiset_SOP<T>& sop, const ttn_type& A, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        CALL_AND_HANDLE(initialise(sop, A, sys, opdict, compress, exploit_identity, use_sparse), "Failed to construct sop operator.");
    }

    multiset_sop_operator& operator=(const multiset_sop_operator& o) = default;
    multiset_sop_operator& operator=(multiset_sop_operator&& o) = default;

    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(multiset_SOP<T>& sop, const ttn_type& A, const system_modes& sys, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        m_nset = sop.nset();
        //first check that the system modes info object is suitable for the resize function that does not take a user defined 
        //operator dictionary.  That is none of the system modes can be generic modes
        ASSERT(sys.nmodes() == A.nleaves(), "Failed to construct sop operator.  The ttn object and sys object do not have compatible numbers of modes.");

        //allocate the contraction info object
        CALL_AND_HANDLE(m_contraction_info.construct_topology(A), "Failed to construct topology of multiset_sop_operator.");

        //and set up the outer vector for this
        for(auto& cinf : m_contraction_info)
        {
            cinf().resize(m_nset);
        }

        //resize the mode operators object
        m_mode_operators.resize(m_nset);
        _m_Eshift.resize(m_nset);
        m_Eshift.resize(m_nset);
        m_indices.resize(m_nset);
        m_time_dependent_coefficients_set.resize(m_nset);
        m_time_dependent_operators_set.resize(m_nset);

        //iterate over each of the sop objects in the multiset sop class
        for(auto& data : sop)
        {
            site_ops_type site_ops;

            //set up the operator indexing tree and bind it in the correct location
            size_t i = std::get<0>(data.first);
            size_t j = std::get<1>(data.first);
            m_indices[i].push_back(j);


            auto& _sop = data.second;

            setup_indexing_tree(_sop, i, j, A, compress, exploit_identity, site_ops);

            single_set_container mode_operators;
            //set the single set literal mode operators from the information provided
            using sop_op_type = sop_operator<T, backend>;
            CALL_AND_RETHROW(sop_op_type::set_primitive_operators(mode_operators, sys, site_ops, use_sparse, A.is_purification()));

            //and set the element as required.
            m_mode_operators[i].push_back(mode_operators);
            _m_Eshift[i].push_back(_sop.Eshift());
            m_Eshift[i].push_back(T(0.0));
            m_time_dependent_coefficients_set[i].push_back(false);
            m_time_dependent_operators_set[i].push_back(false);

            mode_operators.clear();
            mode_operators.shrink_to_fit();
            site_ops.clear();
            site_ops.shrink_to_fit();
        }
        setup_time_dependence();
        update_coefficients(real_type(0.0), true);
    }

    void initialise(multiset_SOP<T>& sop, const ttn_type& A, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        m_nset = sop.nset();
        //first check that the system modes info object is suitable for the resize function that does not take a user defined 
        //operator dictionary.  That is none of the system modes can be generic modes
        ASSERT(sys.nmodes() == A.nleaves(), "Failed to construct sop operator.  The ttn object and sys object do not have compatible numbers of modes.");
        ASSERT(opdict.nmodes() == sys.nmodes(), "Failed to construct sop operator.  opdict does not have a compatible dimension.");

        //allocate the contraction info object
        CALL_AND_HANDLE(m_contraction_info.construct_topology(A), "Failed to construct topology of multiset_sop_operator.");

        //and set up the outer vector for this
        for(auto& cinf : m_contraction_info)
        {
            cinf().resize(m_nset);
        }

        //resize the mode operators object
        m_mode_operators.resize(m_nset);
        _m_Eshift.resize(m_nset);
        m_Eshift.resize(m_nset);
        m_time_dependent_coefficients_set.resize(m_nset);
        m_time_dependent_operators_set.resize(m_nset);

        //iterate over each of the sop objects in the multiset sop class
        for(auto& data : sop)
        {
            site_ops_type site_ops;

            //set up the operator indexing tree and bind it in the correct location
            size_t i = std::get<0>(data.first);
            size_t j = std::get<1>(data.first);
            auto& _sop = data.second;

            setup_indexing_tree(_sop, i, j, A, compress, exploit_identity, site_ops);

            single_set_container mode_operators;

            using sop_op_type = sop_operator<T, backend>;
            //set the single set literal mode operators from the information provided
            CALL_AND_RETHROW(sop_op_type::set_primitive_operators(mode_operators, sys, opdict, site_ops, use_sparse, A.is_purification()));

            //and set the element as required.
            m_mode_operators[i].push_back(mode_operators);
            _m_Eshift[i].push_back(_sop.Eshift());
            m_Eshift[i].push_back(T(0.0));
            m_time_dependent_coefficients_set[i].push_back(false);
            m_time_dependent_operators_set[i].push_back(false);

            mode_operators.clear();
            mode_operators.shrink_to_fit();
            site_ops.clear();
            site_ops.shrink_to_fit();
        }
        setup_time_dependence();
        update_coefficients(real_type(0.0), true);
    }

    void clear()
    {
        m_contraction_info.clear();
        m_mode_operators.clear();
        m_mode_dimension.clear();       
        _m_Eshift.clear();
        m_Eshift.clear();
    }

    const tree_type& contraction_info() const{return m_contraction_info;}

    container_type& mode_operators(){return m_mode_operators;}
    const container_type& mode_operators() const{return m_mode_operators;}

    template <typename RT>
    void update(size_type nu, RT t, RT dt)
    {
        for(auto& m1 : m_mode_operators)
        {
            for(auto& m2 : m1)
            {
                for(size_t term = 0; term < m2[nu].size(); ++term)
                {
                    m2[nu][term].update(t, dt);
                }
            }
        }
        update_coefficients(t);
    }

    const single_set_container& mode_operators(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_mode_operators[i][j];
    }

    void update_coefficients(real_type t, bool force_update = false)
    {
        for(size_t i = 0; i < m_nset; ++i)
        {
            for(size_t j = 0; j < _m_Eshift[i].size(); ++j)
            {
                if(m_time_dependent_coefficients_set[i][j] || force_update)
                {
                    if(_m_Eshift[i][j].is_time_dependent() || force_update)
                    {
                        m_Eshift[i][j] = _m_Eshift[i][j](t);
                    }

                    for(auto& n : m_contraction_info)
                    {
                        n()[i][j].update_coefficients(t, force_update);
                    }
                }
            }
        }
    }

    bool is_scalar(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_contraction_info.root()()[i][j].nterms() == 0;
    }

    literal::coeff<T>& Eshift(size_t i, size_t j)
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return _m_Eshift[i][j];
    }

    const T& Eshift_val(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_Eshift[i][j];
    }

    const T& Eshift(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_Eshift[i][j];
    }
    size_t nrow(size_t i) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        return m_indices[i].size();
    }
    size_t column_index(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_indices[i][j];
    }

    size_type nmodes() const{return m_mode_operators.size();}

    iterator begin() {  return iterator(m_mode_operators.begin());  }
    iterator end() {  return iterator(m_mode_operators.end());  }
    const_iterator begin() const {  return const_iterator(m_mode_operators.begin());  }
    const_iterator end() const {  return const_iterator(m_mode_operators.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_mode_operators.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_mode_operators.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_mode_operators.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_mode_operators.rend());  }

    size_type nset() const{return m_nset;}


    bool is_time_dependent(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_time_dependent_coefficients_set[i][j] || m_time_dependent_operators_set[i][j];
    }

    bool has_time_dependent_coefficients(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_time_dependent_coefficients_set[i][j];
    }

    bool has_time_dependent_operators(size_t i, size_t j) const
    {
        ASSERT(i < m_nset, "Index out of bounds.");
        ASSERT(j < m_indices[i].size(), "Index out of bounds.");
        return m_time_dependent_operators_set[i][j];
    }

    bool is_time_dependent() const{return m_time_dependent_coefficients || m_time_dependent_operators;}
    bool has_time_dependent_coefficients() const{return m_time_dependent_coefficients;}
    bool has_time_dependent_operators() const{return m_time_dependent_operators;}
protected:
    void setup_time_dependence()
    {
        //iterate through the coefficient tree and set the node to be time dependent if either its coefficients are time dependent
        //or it is a parent of a time dependent node
        for(auto it = m_contraction_info.rbegin(); it != m_contraction_info.rend(); ++it)
        {
            auto& n = *it;
            auto& _cinf = n();

            for(size_t si = 0; si < m_nset; ++si)
            {
                for(size_t sj = 0; sj < m_indices[si].size(); ++sj)
                {
                    auto& cinf = _cinf[si][sj];
                    if(n.is_leaf())
                    {
                        for(size_type ind = 0; ind < cinf.nterms(); ++ind)
                        {
                            bool time_dependent = false;
                            for(size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                            {
                                if(cinf[ind].time_dependent_spf_coeff(i))
                                {
                                    time_dependent = true;
                                    m_time_dependent_operators_set[si][sj] = true;
                                    m_time_dependent_operators = true;
                                    m_time_dependent_coefficients_set[si][sj] = true;
                                    m_time_dependent_coefficients = true;
                                }
                            }

                            if(cinf[ind].time_dependent_coeff())
                            {
                                m_time_dependent_coefficients_set[si][sj] = true;
                                m_time_dependent_coefficients = true;
                            }
                            if(!m_time_dependent_coefficients)
                            {
                                for(size_type i = 0; i < cinf[ind].nmf_terms(); ++i)
                                {
                                    if(cinf[ind].time_dependent_mf_coeff(i))
                                    {
                                        m_time_dependent_coefficients = true;
                                        m_time_dependent_coefficients_set[si][sj] = true;
                                    }
                                }
                            }

                            cinf[ind].set_is_time_dependent(time_dependent);
                        }
                    }
                    else
                    {
                        for(size_type ind = 0; ind < cinf.nterms(); ++ind)
                        {
                            bool time_dependent = false;
                            for(size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                            {
                                //check its accumulation coefficients
                                if(cinf[ind].time_dependent_spf_coeff(i))
                                {
                                    time_dependent = true;
                                    m_time_dependent_operators_set[si][sj] = true;
                                    m_time_dependent_operators = true;
                                    m_time_dependent_coefficients_set[si][sj] = true;
                                    m_time_dependent_coefficients = true;
                                }

                                //check the children spfs
                                const auto& spinds = cinf[ind].spf_indexing()[i];
                                for(size_type ni=0; ni<spinds.size(); ++ni)
                                {
                                    size_type nu = spinds[ni][0];
                                    size_type cri = spinds[ni][1];
                                    if(n[nu]()[si][sj][cri].is_time_dependent())
                                    {
                                        time_dependent = true;
                                        m_time_dependent_operators_set[si][sj] = true;
                                        m_time_dependent_operators = true;
                                        m_time_dependent_coefficients_set[si][sj] = true;
                                        m_time_dependent_coefficients = true;
                                    }
                                }
                            }

                            if(cinf[ind].time_dependent_coeff())
                            {
                                m_time_dependent_coefficients_set[si][sj] = true;
                                m_time_dependent_coefficients = true;
                            }
                            if(!m_time_dependent_coefficients)
                            {
                                for(size_type i = 0; i < cinf[ind].nmf_terms(); ++i)
                                {
                                    if(cinf[ind].time_dependent_mf_coeff(i))
                                    {
                                        m_time_dependent_coefficients = true;
                                        m_time_dependent_coefficients_set[si][sj] = true;
                                    }
                                }
                            }
                            cinf[ind].set_is_time_dependent(time_dependent);
                        }
                    }
                }
            }

        }
        for(size_t i = 0; i < m_nset; ++i)
        {
            for(size_t j = 0; j < m_indices[i].size(); ++j)
            {
                if(_m_Eshift[i][j].is_time_dependent())
                {
                    m_time_dependent_coefficients_set[i][j] = true;
                    m_time_dependent_coefficients = true;
                }
            }
        }
    }

    void setup_indexing_tree(SOP<T>& sop, size_t i, size_t j, const ttn_type& A, bool compress, bool exploit_identity, site_ops_type& site_ops)
    {
        tree<auto_sop::node_op_info<T>> bp;

        CALL_AND_HANDLE(autoSOP<T>::construct(sop, A, bp, site_ops, compress), "Failed to construct spo_operator.  Failed to convert SOP to indexing tree.");

        //now iterate through all nodes in the node_op_info object created by the autoSOP class and the contraction info object
        for(auto z : common::zip(bp, m_contraction_info))
        {
            const auto& h = std::get<0>(z);
            auto& cinf = std::get<1>(z)()[i];
            std::vector<operator_contraction_info<T>> contraction_info;
            //and set the contraction info from the autoSOP data
            h.get_contraction_info(contraction_info, exploit_identity);  


            sttn_node_data<T> temp;
            temp.set_contraction_info(contraction_info);
            temp.row() = i;
            temp.col() = j;
            cinf.push_back(temp);
        }
        bp.clear();
    }


#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("contraction_info", m_contraction_info)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_mode_operators)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimension", m_mode_dimension)), "Failed to serialise sum of product operator.  Failed to serialise the number of modes.");
    }
#endif
};  //class multiset_sop_operator

}   //namespace ttns

#endif  //TTNS_SOP_OPERATOR_CONTAINER_HPP

