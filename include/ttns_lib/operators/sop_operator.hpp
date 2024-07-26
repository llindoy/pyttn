#ifndef TTNS_SOP_OPERATOR_CONTAINER_HPP
#define TTNS_SOP_OPERATOR_CONTAINER_HPP

#include <linalg/linalg.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>
#include <map>

#include <linalg/linalg.hpp>
#include "site_operators/site_operator.hpp"
#include "../sop/system_information.hpp"
#include "../sop/SOP.hpp"
#include "../sop/autoSOP.hpp"
#include "../sop/operator_dictionaries/operator_dictionary.hpp"
#include "site_operators/sequential_product_operator.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

template <typename T>
class sttn_node_data
{
public:
    using real_type = typename tmp::get_real_type<T>::type; 
    using elem_type = operator_contraction_info<T>;

    template <typename Y, typename V> friend class operator_container;
public:
    sttn_node_data() {}

    sttn_node_data(const sttn_node_data& o) = default;
    sttn_node_data(sttn_node_data&& o) = default;
    sttn_node_data& operator=(const sttn_node_data& o) = default;
    sttn_node_data& operator=(sttn_node_data&& o) = default;

    void set_contraction_info(const std::vector<operator_contraction_info<T> >& o)
    {
        m_terms.clear();
        m_terms.resize(o.size());
        for(size_t i = 0; i < o.size(); ++i)
        {
            m_terms[i] = o[i];
        }
    }

    ~sttn_node_data(){}

    void clear() 
    {
        try
        {
            for(size_t i = 0; i < m_terms.size(); ++i){m_terms[i].clear();}
            m_terms.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear operator node object.");
        }
    }

    size_t nterms() const{return m_terms.size();}

    const elem_type& term(size_t i) const
    {
        ASSERT(i  < m_terms.size(), "Index out of bounds.");
        return m_terms[i];
    }

    elem_type& term(size_t i)
    {
        ASSERT(i < m_terms.size(), "Index out of bounds.");
        return m_terms[i];
    }

    const elem_type& operator[](size_t i) const {return m_terms[i];}
    elem_type& operator[](size_t i){return m_terms[i];}

    const size_t& row() const{return m_i_index;}
    size_t& row(){return m_i_index;}

    const size_t& col() const{return m_j_index;}
    size_t& col(){return m_j_index;}


    size_t nnz() const
    {
        size_t ret = 0;
        for(const auto& t : m_terms)
        {
            ret += t.nspf_terms();
        }
        return ret;
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {       
        CALL_AND_HANDLE(ar(cereal::make_nvp("terms", m_terms)), "Failed to serialise operator node object.  Error when serialising the terms.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("i_index", m_i_index)), "Failed to serialise operator node object.  Error when serialising i index.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("j_index", m_j_index)), "Failed to serialise operator node object.  Error when serialising j index.");
    }
#endif

protected:
    std::vector<operator_contraction_info<T>> m_terms;
    size_t m_i_index = 0;
    size_t m_j_index = 0;
};  //sop_node_data


namespace node_data_traits
{
    //clear traits for the operator node data object
    template <typename T>
    struct clear_traits<sttn_node_data<T> > 
    {
        void operator()(sttn_node_data<T>& t){CALL_AND_RETHROW(t.clear());}
    };

    //assignment traits for the tensor and matrix objects
    template <typename T>
    struct assignment_traits<sttn_node_data<T>, sttn_node_data<T> >
    {
        using is_applicable = std::true_type;
        inline void operator()(sttn_node_data<T>& o,  const sttn_node_data<T>& i){CALL_AND_RETHROW(o = i);}
    };
}   //namespace node_data_traits


//a generic sum of product operator object.  This stores all of the operator and indexing required for the evaluation of the sop on a TTN state but 
//doesn't store the buffers needed to perform the required contractions
template <typename T, typename backend = linalg::blas_backend>
class sop_operator
{
public:
    using size_type = typename backend::size_type;

    using op_type = ops::primitive<T, backend>;
    using element_type = site_operator<T, backend>;

    using tree_type = tree<sttn_node_data<T>>;
    using node_type = typename tree_type::node_type;

    using mode_terms_type = std::vector<element_type>;

    using value_type = T;

    using ttn_type = ttn<T, backend>;

    using container_type = std::vector<mode_terms_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    using site_ops_type = typename autoSOP<T>::site_ops_type;
protected:
    tree_type m_contraction_info;
    container_type m_mode_operators;
    std::vector<size_type> m_mode_dimension;
    T m_Eshift ;

public:
    sop_operator() : m_Eshift(T(0.0)){}
    sop_operator(const sop_operator& o) = default;
    sop_operator(sop_operator&& o) = default;
    sop_operator(SOP<T>& sop, const ttn_type& A, const system_modes& sys, bool compress = true, bool exploit_identity = true, bool use_sparse = true) : m_Eshift(T(0.0))
    {
        CALL_AND_HANDLE(initialise(sop, A, sys, compress, exploit_identity, use_sparse), "Failed to construct sop operator.");
    }
    sop_operator(SOP<T>& sop, const ttn_type& A, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool compress = true, bool exploit_identity = true, bool use_sparse = true) : m_Eshift(T(0.0))
    {
        CALL_AND_HANDLE(initialise(sop, A, sys, opdict, compress, exploit_identity, use_sparse), "Failed to construct sop operator.");
    }

    sop_operator& operator=(const sop_operator& o) = default;
    sop_operator& operator=(sop_operator&& o) = default;

    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(SOP<T>& sop, const ttn_type& A, const system_modes& sys, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        //first check that the system modes info object is suitable for the resize function that does not take a user defined 
        //operator dictionary.  That is none of the system modes can be generic modes
        ASSERT(sys.nmodes() == A.nleaves(), "Failed to construct sop operator.  The ttn object and sys object do not have compatible numbers of modes.");

        //setup the autoSOP indexing object
        site_ops_type site_ops;
        setup_indexing_tree(sop, A, compress, exploit_identity, site_ops);

        CALL_AND_RETHROW(set_primitive_operators(m_mode_operators, sys, site_ops, use_sparse, A.is_purification()));
        m_Eshift = sop.Eshift();

        site_ops.clear();
        site_ops.shrink_to_fit();
    }

    void initialise(SOP<T>& sop, const ttn_type& A, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool compress = true, bool exploit_identity = true, bool use_sparse = true)
    {
        //first check that the system modes info object is suitable for the resize function that does not take a user defined 
        //operator dictionary.  That is none of the system modes can be generic modes
        ASSERT(sys.nmodes() == A.nleaves(), "Failed to construct sop operator.  The ttn object and sys object do not have compatible numbers of modes.");
        ASSERT(opdict.nmodes() == sys.nmodes(), "Failed to construct sop operator.  opdict does not have a compatible dimension.");

        //setup the autoSOP indexing object
        site_ops_type site_ops;
        setup_indexing_tree(sop, A, compress, exploit_identity, site_ops);

        CALL_AND_RETHROW(set_primitive_operators(m_mode_operators, sys, opdict, site_ops, use_sparse, A.is_purification()));
        m_Eshift = sop.Eshift();

        site_ops.clear();
        site_ops.shrink_to_fit();
    }

    void clear()
    {
        m_contraction_info.clear();
        m_mode_operators.clear();
        m_mode_dimension.clear();       
    }

    const tree_type& contraction_info() const{return m_contraction_info;}

    size_t nrow(size_t i) const
    {
        ASSERT(i == 0, "Index out of bounds.");
        return 1;
    }
    size_t column_index(size_t i, size_t j) const
    {
        ASSERT(i == 0 && j == 0, "Index out of bounds.");
        return 0;
    }



    const mode_terms_type& operators(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const mode_terms_type& operator[](size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const mode_terms_type& operator()(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator()(size_type nu, size_type k) const
    {
        return m_mode_operators[nu][k];
    }

    template <typename RT>
    void update(size_type nu, RT t, RT dt)
    {
        for(size_t term = 0; term < m_mode_operators[nu].size(); ++term)
        {
            m_mode_operators[nu][term].update(t, dt);
        }
    }

    container_type& mode_operators(){return m_mode_operators;}
    const container_type& mode_operators() const{return m_mode_operators;}

    size_type nterms(size_type nu) const{return m_mode_operators[nu].size();}
    size_type nmodes() const{return m_mode_operators.size();}

    iterator begin() {  return iterator(m_mode_operators.begin());  }
    iterator end() {  return iterator(m_mode_operators.end());  }
    const_iterator begin() const {  return const_iterator(m_mode_operators.begin());  }
    const_iterator end() const {  return const_iterator(m_mode_operators.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_mode_operators.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_mode_operators.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_mode_operators.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_mode_operators.rend());  }

public:
    size_type nset() const{return 1;}

protected:
    void setup_indexing_tree(SOP<T>& sop, const ttn_type& A, bool compress, bool exploit_identity, site_ops_type& site_ops)
    {
        tree<auto_sop::node_op_info<T>> bp;

        CALL_AND_HANDLE(m_contraction_info.construct_topology(A), "Failed to construct topology of sop_operator.");

        bool autosop_run = false;
        CALL_AND_HANDLE(autosop_run = autoSOP<T>::construct(sop, A, bp, site_ops, compress), "Failed to construct spo_operator.  Failed to convert SOP to indexing tree.");

        if(autosop_run)
        {
            //now iterate through all nodes in the node_op_info object created by the autoSOP class and the contraction info object
            for(auto z : common::zip(bp, m_contraction_info))
            {
                const auto& h = std::get<0>(z);
                auto& cinf = std::get<1>(z);
                std::vector<operator_contraction_info<T>> contraction_info;
                //and set the contraction info from the autoSOP data
                h.get_contraction_info(contraction_info, exploit_identity);  
                cinf().set_contraction_info(contraction_info);
            }
            bp.clear();
        }
        else
        {
            //now iterate through all nodes in the node_op_info object created by the autoSOP class and the contraction info object
            for(auto& cinf : m_contraction_info)
            {
                std::vector<operator_contraction_info<T>> contraction_info;
                cinf().set_contraction_info(contraction_info);
            }
        }
    }

public:
    static void set_primitive_operators(container_type& mode_operators, const system_modes& sys, site_ops_type& site_ops, bool use_sparse = true, bool use_purification = false)
    {
        //now go though and set up the system operator solely using the default dictionaries
        mode_operators.resize(sys.nmodes());
        //now we go through and attempt to create the site operators for this class.
        for(size_t nu = 0; nu < sys.nmodes(); ++nu)
        {
            size_t hilbert_space_dimension = sys[nu].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            for(size_t j = 0; j < site_ops[nu].size(); ++j)
            {
                sPOP t = site_ops[nu][j];
                for(const auto& op : t.ops())
                {
                    ASSERT(op.mode() == nu, "Invalid site operators.");
                }

                if(t.size() == 1)
                {
                    std::string label = t.ops().front().op();
                    CALL_AND_HANDLE(mode_operators[nu].push_back(site_operator<T, backend>(operator_from_default_dictionaries<T, backend>::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification)), "Failed to insert new element in mode operator.");
                }
                else
                {
                    std::vector<std::shared_ptr<ops::primitive<T, backend>>> ops;   ops.reserve(t.size());
                    for(const auto& op : t.ops())
                    {
                        std::string label = op.op();
                        CALL_AND_HANDLE(ops.push_back(operator_from_default_dictionaries<T, backend>::query(label, basis, sys[nu].type(), use_sparse)), "Failed to insert new element in mode operator.");
                    }
                    mode_operators[nu].push_back(site_operator<T, backend>(ops::sequential_product_operator<T, backend>{ops}, nu, use_purification));
                }
            }
        }
    }

    static void set_primitive_operators(container_type& mode_operators, const system_modes& sys, const operator_dictionary<T, backend>& opdict, site_ops_type& site_ops, bool use_sparse = true, bool use_purification = false)
    {
        //now go though and set up the system operator solely using the default dictionaries
        mode_operators.resize(sys.nmodes());

        //now we go through and attempt to create the site operators for this class.
        for(size_t nu = 0; nu < sys.nmodes(); ++nu)
        {
            size_t hilbert_space_dimension = sys[nu].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            for(size_t j = 0; j < site_ops[nu].size(); ++j)
            {
                sPOP t = site_ops[nu][j];
                for(const auto& op : t.ops())
                {
                    ASSERT(op.mode() == nu, "Invalid site operators.");
                }

                if(t.size() == 1)
                {
                    std::string label = t.ops().front().op();

                    //first start to access element from opdict
                    std::shared_ptr<op_type> op = opdict.query(nu, label);

                    if(op != nullptr)
                    {
                        ASSERT(op->size() == hilbert_space_dimension, "Failed to construct sop_operator.  Mode operator from operator dictionary has incorrect size.");
                        mode_operators[nu].push_back(site_operator<T, backend>(op, nu, use_purification));
                    }
                    else
                    {
                        CALL_AND_HANDLE(mode_operators[nu].push_back(site_operator<T, backend>(operator_from_default_dictionaries<T, backend>::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification)), "Failed to insert new element in mode operator.");
                    }
                }
                else
                {
                    std::vector<std::shared_ptr<ops::primitive<T, backend>>> ops;   ops.reserve(t.size());
                    for(const auto& op : t.ops())
                    {
                        std::string label = op.op();

                        std::shared_ptr<op_type> curr_op = opdict.query(nu, label);
                        if(curr_op != nullptr)
                        {
                            ASSERT(curr_op->size() == hilbert_space_dimension, "Failed to construct sop_operator.  Mode operator from operator dictionary has incorrect size.");
                            ops.push_back(curr_op);
                        }
                        else
                        {
                            CALL_AND_HANDLE(ops.push_back(operator_from_default_dictionaries<T, backend>::query(label, basis, sys[nu].type(), use_sparse)), "Failed to insert new element in mode operator.");
                        }
                    }
                    mode_operators[nu].push_back(site_operator<T, backend>(ops::sequential_product_operator<T, backend>{ops}, nu, use_purification));
                }
            }
        }
    }
    const T& Eshift() const{return m_Eshift;}
    T& Eshift() {return m_Eshift;}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("contraction_info", m_contraction_info)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_mode_operators)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimension", m_mode_dimension)), "Failed to serialise sum of product operator.  Failed to serialise the number of modes.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("Eshift", m_Eshift)), "Failed to serialise operator node object.  Error when serialising Eshift.");
    }
#endif
};  //class sop_operator

}   //namespace ttns

#endif  //TTNS_SOP_OPERATOR_CONTAINER_HPP

