#ifndef TTNS_PRODUCT_OPERATOR_CONTAINER_HPP
#define TTNS_PRODUCT_OPERATOR_CONTAINER_HPP


#include <linalg/linalg.hpp>
#include <common/zip.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>
#include <map>

#include <linalg/linalg.hpp>
#include "site_operators/site_operator.hpp"
#include "../sop/system_information.hpp"
#include "../sop/sSOP.hpp"
#include "../sop/operator_dictionaries/operator_dictionary.hpp"
#include "../sop/coeff_type.hpp"
#include "site_operators/sequential_product_operator.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

//a generic sum of product operator object.  This stores all of the operator and indexing required for the evaluation of the sop on a TTN state but 
//doesn't store the buffers needed to perform the required contractions
template <typename T, typename backend = linalg::blas_backend>
class product_operator
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 

    using op_type = ops::primitive<T, backend>;
    using element_type = site_operator<T, backend>;

    using container_type = std::vector<element_type>;

    using value_type = T;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

protected:
    container_type m_mode_operators;
    literal::coeff<T> _m_coeff = T(1.0);
    T m_coeff = T(1.0);
public:
    product_operator(){}
    product_operator(const product_operator& o) = default;
    product_operator(product_operator&& o) = default;
    product_operator(const sOP& op, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        CALL_AND_HANDLE(initialise(op, sys, use_sparse, use_purification), "Failed to construct sop operator.");
    }
    product_operator(const sOP& op, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false) 
    {
        CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse, use_purification), "Failed to construct sop operator.");
    }

    product_operator(const sPOP& op, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        CALL_AND_HANDLE(initialise(op, sys, use_sparse, use_purification), "Failed to construct sop operator.");
    }
    product_operator(const sPOP& op, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false) 
    {
        CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse, use_purification), "Failed to construct sop operator.");
    }
    product_operator(const sNBO<T>& op, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        CALL_AND_HANDLE(initialise(op, sys, use_sparse, use_purification), "Failed to construct sop operator.");
    }
    product_operator(const sNBO<T>& op, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false) 
    {
        CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse, use_purification), "Failed to construct sop operator.");
    }
    product_operator& operator=(const product_operator& o) = default;
    product_operator& operator=(product_operator&& o) = default;

protected:
    //function for unpacking a product operator into operators acting on the same modes, preserving the ordering of the modes.  
    //Note that this assumes that operators acting on different modes commute and so will generally give incorrect operators
    //if applied to a fermionic operator that has not already been mapped to qubit operators
    std::map<size_t, std::list<std::string>> unpack_pop(const sPOP& pop)
    {
        std::map<size_t, std::list<std::string>> ret;
        for(const auto& op : pop)
        {
            ret[op.mode()].push_back(op.op());
        }
        return ret;
    }
public:
    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(const sPOP& pop, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        m_coeff = T(1.0);
        _m_coeff = T(1.0);
        m_mode_operators.resize(pop.size());
        auto up = unpack_pop(pop);

        using dfop_dict = operator_from_default_dictionaries<T, backend>;
        size_t i = 0;
        for(const auto& x : up)
        {
            size_t nu = x.first;

            size_t hilbert_space_dimension = sys[nu].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            const auto& t = x.second;
            if(t.size() == 1)
            {
                std::string label = t.front();
                CALL_AND_HANDLE(m_mode_operators[i] = element_type(dfop_dict::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification), "Failed to insert new element in mode operator.");
            }
            else
            {
                std::vector<std::shared_ptr<ops::primitive<T, backend>>> ops;   ops.reserve(t.size());
                for(const auto& label : t)
                {
                    CALL_AND_HANDLE(ops.push_back(dfop_dict::query(label, basis, sys[nu].type(), use_sparse)), "Failed to insert new element in mode operator.");
                }
                CALL_AND_HANDLE(m_mode_operators[i] = element_type(ops::sequential_product_operator<T, backend>{ops}, nu, use_purification), "Failed to insert new element in mode operator");
            }
            ++i;
        }
    }

    void initialise(const sPOP& pop, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false)
    {
        m_coeff = T(1.0);
        _m_coeff = T(1.0);
        m_mode_operators.resize(pop.size());
        auto up = unpack_pop(pop);

        using dfop_dict = operator_from_default_dictionaries<T, backend>;
        size_t i = 0;
        for(const auto& x : up)
        {
            size_t nu = x.first;
            size_t hilbert_space_dimension = sys[nu].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            const auto& t = x.second;
            if(t.size() == 1)
            {
                std::string label = t.front();
                //first start to access element from opdict
                std::shared_ptr<op_type> op = opdict.query(nu, label);

                if(op != nullptr)
                {
                    ASSERT(op->size() == hilbert_space_dimension, "Failed to construct product_operator.  Mode operator from operator dictionary has incorrect size.");
                    CALL_AND_HANDLE(m_mode_operators[i] = element_type(op, nu, use_purification), "Failed to insert new element in mode operator.");
                }
                else
                {
                    CALL_AND_HANDLE(m_mode_operators[i] = element_type(dfop_dict::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification), "Failed to insert new element in mode operator.");
                }

            }
            else
            {
                std::vector<std::shared_ptr<ops::primitive<T, backend>>> ops;   ops.reserve(t.size());
                for(const auto& label : t)
                {
                    std::shared_ptr<op_type> curr_op = opdict.query(nu, label);
                    if(curr_op != nullptr)
                    {
                        ASSERT(curr_op->size() == hilbert_space_dimension, "Failed to construct product_operator.  Mode operator from operator dictionary has incorrect size.");
                        ops.push_back(curr_op);
                    }
                    else
                    {
                        CALL_AND_HANDLE(ops.push_back(dfop_dict::query(label, basis, sys[nu].type(), use_sparse)), "Failed to insert new element in mode operator.");
                    }
                }
                CALL_AND_HANDLE(m_mode_operators[i] = element_type(ops::sequential_product_operator<T, backend>{ops}, nu, use_purification), "Failed to insert new element in mode operator");
            }
            ++i;
        }
    }
    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(const sOP& _op, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        m_coeff = T(1.0);
        _m_coeff = T(1.0);
        m_mode_operators.resize(1);

        using dfop_dict = operator_from_default_dictionaries<T, backend>;
        size_t nu = _op.mode();
        size_t hilbert_space_dimension = sys[nu].lhd();
        std::string label = _op.op();

        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

        CALL_AND_HANDLE(m_mode_operators[0] = element_type(dfop_dict::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification), "Failed to insert new element in mode operator.");
    }

    void initialise(const sOP& _op, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false)
    {
        m_mode_operators.resize(1);
        m_coeff = T(1.0);
        _m_coeff = T(1.0);
        size_t nu = _op.mode();
        size_t hilbert_space_dimension = sys[nu].lhd();
        std::string label = _op.op();

        using dfop_dict = operator_from_default_dictionaries<T, backend>;
        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

        //first start to access element from opdict
        std::shared_ptr<op_type> op = opdict.query(nu, label);
        if(op != nullptr)
        {
            ASSERT(op->size() == hilbert_space_dimension, "Failed to construct product_operator.  Mode operator from operator dictionary has incorrect size.");
            CALL_AND_HANDLE(m_mode_operators[0] = element_type(op, nu, use_purification), "Failed to insert new element in mode operator.");
        }
        else
        {
            CALL_AND_HANDLE(m_mode_operators[0] = element_type(dfop_dict::query(label, basis, sys[nu].type(), use_sparse), nu, use_purification), "Failed to insert new element in mode operator.");
        }
    }

    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(const sNBO<T>& pop, const system_modes& sys, bool use_sparse = true, bool use_purification = false)
    {
        CALL_AND_RETHROW(initialise(pop.pop(), sys, use_sparse, use_purification));
        _m_coeff = pop.coeff();
        update_coefficients(0);
    }

    void initialise(const sNBO<T>& pop, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true, bool use_purification = false)
    {
        CALL_AND_RETHROW(initialise(pop.pop(), sys, opdict, use_sparse, use_purification));
        _m_coeff = pop.coeff();
        update_coefficients(0);
    }

    void clear()
    {
        m_mode_operators.clear();
    }

    const element_type& operators(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator[](size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator()(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    std::ostream& print(std::ostream& os) const
    {
        os << "prod_op: [" << m_coeff << " ";
        for(size_t i = 0; i<m_mode_operators.size(); ++i)
        {
            os << m_mode_operators[i].to_string() << (i+1 != this->nmodes() ? ", " : "]");
        }
        return os;
    }

    void update_coefficients(real_type t)
    {
        m_coeff = _m_coeff(t);   
    }

    const T& coeff() const{return m_coeff;}
    T& coeff() {return m_coeff;}

    const container_type& mode_operators() const{return m_mode_operators;}
    container_type& mode_operators() {return m_mode_operators;}

    size_type nmodes() const{return m_mode_operators.size();}

    iterator begin() {  return iterator(m_mode_operators.begin());  }
    iterator end() {  return iterator(m_mode_operators.end());  }
    const_iterator begin() const {  return const_iterator(m_mode_operators.begin());  }
    const_iterator end() const {  return const_iterator(m_mode_operators.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_mode_operators.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_mode_operators.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_mode_operators.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_mode_operators.rend());  }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_mode_operators)), "Failed to serialise sum of product operator.  Failed to serialise array of operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("coeff", m_coeff)), "Failed to serialise sum of product operator.  Failed to serialise array of operators.");
    }
#endif
};  //class product_operator


template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const product_operator<T, backend>& t)
{
    return t.print(os);
}

}   //namespace ttns


#endif  //TTNS_PRODUCT_OPERATOR_CONTAINER_HPP
